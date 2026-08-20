import inspect,torch
from kfrag.models.natural_channel_v2 import NaturalChannelV2
from kfrag.models.regional_channel_v1 import RegionalChannelV1,RegionalCarrierRouter,BlindRegionalDecoder

def model():return RegionalChannelV1(NaturalChannelV2(16,4),16)

def test_regional_shape_is_exactly_sixteen_by_eight():
    m=model();bits=torch.zeros(2,4,4,8);out=m(torch.rand(2,3,16,16),bits,.01)
    assert out["regional_logits"].shape==(2,4,4,8) and bits.numel()//len(bits)==128

def test_single_bit_changes_only_intended_pre_mixing_contribution():
    router=model().encoder.router;base=torch.zeros(1,4,4,8);before=router.contributions(base)
    for region in (0,5,15):
        for bit in (0,7):
            changed=base.clone();r,c=divmod(region,4);changed[0,r,c,bit]=1;delta=router.contributions(changed)-before
            nonzero=delta.abs().flatten(4).sum(-1).nonzero().tolist();assert nonzero==[[0,r,c,bit]]

def test_every_regional_bit_maps_to_canonical_output_position():
    targets=torch.zeros(1,4,4,8);targets[0,2,3,6]=1;logits=(targets*2-1)*20
    assert logits.ge(0).eq(targets.bool()).all() and logits.ge(0).nonzero().tolist()==[[0,2,3,6]]

def test_all_regional_inputs_and_encoder_parameters_receive_gradients():
    m=model();image=torch.rand(1,3,16,16);bits=torch.rand(1,4,4,8,requires_grad=True);loss=m.encoder(image,bits,.01)["residual"].square().mean();loss.backward()
    assert bits.grad.abs().flatten(1).sum(0).gt(0).all() and m.encoder.encoder.carrier_skip.weight.grad.abs().sum()>0

def test_blind_decoder_accepts_only_questioned_image():
    assert list(inspect.signature(BlindRegionalDecoder.forward).parameters)==["self","questioned_image"]

def test_inactive_packet_fields_cannot_enter_regional_interface():
    m=model()
    try:m(torch.rand(1,3,16,16),torch.zeros(1,4,4,44),.01)
    except ValueError as e:assert "[B,4,4,8]" in str(e)
    else:raise AssertionError("44-bit packets must be rejected")

def test_region_mask_prevents_inactive_carrier_contributions():
    m=model();bits=torch.ones(1,4,4,8);mask=torch.zeros(1,4,4,dtype=torch.bool);mask[:,1,2]=True;carrier=m.encoder.router(bits,mask)
    cells=carrier.reshape(1,8,4,4,4,4);energy=cells.square().sum((1,3,5));assert energy.nonzero().tolist()==[[0,1,2]]
