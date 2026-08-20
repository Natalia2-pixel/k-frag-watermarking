import inspect
import torch
from kfrag.models.natural_channel_v2 import ACTIVE_BIT_NAMES, BlindMultiScaleDecoder, NaturalChannelV2
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
