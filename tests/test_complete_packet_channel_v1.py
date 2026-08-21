import inspect
import torch

from kfrag.models.complete_packet_channel_v1 import CompletePacketChannelV1, packet_cell_bases
from kfrag.models.natural_channel_v2 import NaturalChannelV2
from kfrag.models.regional_channel_v1 import RegionalChannelV1


def model():
    return CompletePacketChannelV1(RegionalChannelV1(NaturalChannelV2(64, 8), 64), 64)


def test_44_carriers_are_normalized_and_decorrelated():
    bases=packet_cell_bases(); assert bases.shape==(44,16,16)
    assert torch.allclose(bases.mean((-2,-1)),torch.zeros(44),atol=1e-6)
    assert torch.allclose(bases.square().mean((-2,-1)),torch.ones(44),atol=1e-5)
    gram=bases.flatten(1)@bases.flatten(1).T/256
    assert (gram-torch.eye(44)).abs().masked_fill(torch.eye(44).bool(),0).max()<1e-5


def test_packet_shape_blind_interface_and_inactive_exclusion():
    channel=model(); image=torch.rand(2,3,64,64); bits=torch.randint(0,2,(2,4,4,44)).float()
    assert list(inspect.signature(channel.decoder.forward).parameters)==["questioned_image"]
    out=channel(image,bits,.014,12); assert out["packet_logits"].shape==(2,4,4,44)
    changed=bits.clone();changed[...,12:]=1-changed[...,12:]
    assert torch.equal(channel.encoder.router(bits,12),channel.encoder.router(changed,12))


def test_one_bit_changes_only_intended_pre_mixing_contribution():
    channel=model(); bits=torch.zeros(1,4,4,44); base=channel.encoder.router.contributions(bits)
    changed=bits.clone();changed[:,2,1,17]=1;delta=channel.encoder.router.contributions(changed)-base
    nonzero=delta.abs().flatten(-3).sum(-1).ne(0)
    expected=torch.zeros_like(nonzero);expected[:,2,1,17]=True
    assert torch.equal(nonzero,expected)


def test_every_active_bit_has_encoder_gradient():
    channel=model(); image=torch.rand(1,3,64,64); bits=torch.rand(1,4,4,44,requires_grad=True)
    channel.encoder(image,bits,.014,44)["residual"].square().sum().backward()
    assert bits.grad.abs().sum((0,1,2)).gt(0).all()


def test_batch_validation_fails_fast():
    channel=model()
    try: channel(torch.rand(2,3,64,64),torch.zeros(1,4,4,44),.014)
    except ValueError: pass
    else: raise AssertionError("mismatched batches must fail")
