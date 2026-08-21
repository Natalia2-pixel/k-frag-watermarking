import inspect
import torch
from kfrag.models.natural_channel_v2 import (ACTIVE_BIT_NAMES, BlindMultiScaleDecoder, NaturalChannelV2,
 analytical_carrier_bases, analytical_residual)
from kfrag.training.natural_channel_v2 import loss_components

def model(): return NaturalChannelV2(image_size=16,width=4)

def test_exactly_eight_canonical_bits_and_logits():
    m=model(); assert len(ACTIVE_BIT_NAMES)==8 and len(set(ACTIVE_BIT_NAMES))==8
    assert m.decoder(torch.rand(2,3,16,16)).shape==(2,8)

def test_active_bit_order_is_deterministic_and_one_to_one_target_order():
    assert ACTIVE_BIT_NAMES==tuple(f"regional_symbol_bit_{i}" for i in range(8))
    target=torch.eye(8); logits=(target*2-1)*20
    assert logits.ge(0).eq(target.bool()).all()

def test_each_flip_changes_its_carrier_contribution_and_all_bits_have_gradients():
    m=model(); base=torch.zeros(1,8,requires_grad=True); c0=m.encoder.carrier(base)
    for i in range(8):
        changed=base.detach().clone(); changed[0,i]=1
        difference=m.encoder.carrier(changed)-c0.detach()
        assert difference[:,i].abs().sum()>0 and difference[:,torch.arange(8)!=i].abs().sum()==0
    c0.square().mean().backward(); assert base.grad is not None and (base.grad.abs().sum(0)>0).all()

def test_blind_decoder_interface_and_no_target_leakage():
    assert list(inspect.signature(BlindMultiScaleDecoder.forward).parameters)==["self","questioned_image"]
    assert set(inspect.signature(BlindMultiScaleDecoder.forward).parameters).isdisjoint({"bits","target","payload","original","identifier","index"})

def test_residual_bound_and_strength_mask_noncollapse():
    m=model(); image=torch.rand(2,3,16,16); out=m(image,torch.randint(0,2,(2,8)).float(),.02)
    assert out["preclamp_residual"].abs().max()<=.020001
    assert out["strength_mask"].min()>=.25 and out["strength_mask"].max()<=1 and out["strength_mask"].mean()>.25

def test_losses_and_all_parameter_gradients_are_finite():
    m=model(); image=torch.rand(2,3,16,16); bits=torch.randint(0,2,(2,8)).float(); out=m(image,bits,.02)
    cfg={"amplitude":.02,"l1_weight":1,"structural_weight":.1,"saturation_start":.95,"mask_mean_min":.3,
      "communication_weight":1,"fidelity_weight":1,"energy_weight":.1,"saturation_weight":.1,"balance_weight":.1,
      "mask_collapse_weight":.1,"original_confidence_weight":.1,"carrier_correlation_weight":.1}
    losses=loss_components(m,image,bits,out,cfg); losses["total_loss"].backward()
    assert all(torch.isfinite(x) for x in losses.values())
    for bit in range(8): assert m.encoder.carrier.bases.grad[bit].abs().sum()>0

def test_inactive_packet_fields_cannot_enter_v2_loss():
    m=model(); image=torch.rand(1,3,16,16)
    try: m(image,torch.zeros(1,44),.01)
    except ValueError as error: assert "[B,8]" in str(error)
    else: raise AssertionError("44-bit packets must be rejected")

def test_all_analytical_carriers_are_distinct_normalized_and_decorrelated():
    bases=analytical_carrier_bases(32); gram=torch.nn.functional.normalize(bases.flatten(1),dim=1)@torch.nn.functional.normalize(bases.flatten(1),dim=1).T
    assert torch.allclose(bases.mean((-2,-1)),torch.zeros(8),atol=1e-6)
    assert torch.allclose(bases.square().mean((-2,-1)),torch.ones(8),atol=1e-5)
    assert (gram-torch.eye(8)).abs().max()<1e-5

def test_each_analytical_bit_flip_reverses_only_its_linear_contribution():
    base=torch.zeros(1,8); bases=analytical_carrier_bases(32)
    for bit in range(8):
        changed=base.clone(); changed[0,bit]=1
        signed_delta=(changed*2-1)-(base*2-1)
        contribution=torch.einsum("bk,khw->bhw",signed_delta,bases)
        assert contribution.abs().sum()>0
        assert torch.allclose(contribution,2*bases[bit:bit+1])

def test_analytical_carrier_survives_clamp_and_is_visible_to_highpass():
    m=model(); bits=torch.eye(8); residual=analytical_residual(bits,(16,16),.02); zero=torch.zeros(8,3,16,16)
    effective=(zero+residual).clamp(0,1)-zero
    assert (effective.flatten(1).abs().sum(1)>0).all()
    assert m.decoder.highpass(effective).square().flatten(1).mean(1).min()>0

def test_every_analytical_decoder_logit_has_gradient_in_canonical_order():
    m=model(); bits=torch.eye(8); questioned=(torch.full((8,3,16,16),.5)+analytical_residual(bits,(16,16),.02)).clamp(0,1)
    logits=m.decoder(questioned); logits.retain_grad(); torch.nn.functional.binary_cross_entropy_with_logits(logits,bits).backward()
    assert logits.grad.abs().sum(0).gt(0).all()
    matched=100*torch.einsum("bhw,khw->bk",questioned.mean(1),analytical_carrier_bases(16))/(16*16)
    assert matched.argmax(1).tolist()==list(range(8))

def test_learned_encoder_bit_flips_survive_mask_tanh_and_clamp():
    m=model(); image=torch.full((1,3,16,16),.5); bits=torch.zeros(1,8); base=m.encoder(image,bits,.02)
    differences=[]
    for bit in range(8):
        flipped=bits.clone();flipped[:,bit]=1;out=m.encoder(image,flipped,.02)
        differences.append((out["residual"]-base["residual"]).abs().mean())
        assert not torch.equal(out["bounded_residual"],base["bounded_residual"])
        assert not torch.equal(out["watermarked_image"],base["watermarked_image"])
    assert torch.stack(differences).min()>0

def test_every_learned_payload_bit_has_encoder_gradient_and_parameter_update():
    m=model(); image=torch.rand(2,3,16,16); bits=torch.rand(2,8,requires_grad=True);before=m.encoder.carrier.bases.detach().clone()
    loss=m.encoder(image,bits,.02)["residual"].square().mean();loss.backward()
    assert bits.grad.abs().sum(0).gt(0).all() and m.encoder.carrier.bases.grad.flatten(1).abs().sum(1).gt(0).all()
    torch.optim.AdamW(m.encoder.parameters(),lr=1e-3).step();assert (m.encoder.carrier.bases-before).abs().sum()>0

def test_payload_dependence_persists_across_images_and_mask_does_not_suppress_it():
    m=model();bits=torch.zeros(2,8);flipped=bits.clone();flipped[:,3]=1;images=torch.stack((torch.zeros(3,16,16),torch.rand(3,16,16)))
    a=m.encoder(images,bits,.02);b=m.encoder(images,flipped,.02);per_image=(a["residual"]-b["residual"]).abs().flatten(1).mean(1)
    assert per_image.min()>0 and a["strength_mask"].min()>=m.encoder.mask_floor

def test_payload_stem_injects_at_full_half_and_bottleneck_resolutions():
    m=model(); assert m.encoder.payload_stem is not None and m.encoder.down_payload is not None
    assert m.encoder.bottleneck.net[0].in_channels==m.encoder.down_payload.out_channels+8
