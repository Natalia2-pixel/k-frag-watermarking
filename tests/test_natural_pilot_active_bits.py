import torch
from kfrag.training.trainer import active_symbol_mask, configured_active_channels, fresh_payloads, select_active

def test_natural_image_active_mask_selects_exactly_eight_symbol_bits():
    mask=active_symbol_mask()
    assert mask.shape == (1,44,4,4) and mask.sum().item() == 8*4*4
    assert mask[0,:,0,0].nonzero().flatten().tolist() == list(range(4,12))
    assert configured_active_channels({"active_channels":list(range(4,12))}) == tuple(range(4,12))

def test_each_active_bit_maps_to_its_intended_decoder_output():
    payload=torch.zeros(1,44,4,4)
    for local_bit,packet_bit in enumerate(range(4,12)):
        payload.zero_(); payload[0,packet_bit,local_bit%4,local_bit//4]=1
        assert select_active(payload).nonzero().tolist() == [[0,local_bit,local_bit%4,local_bit//4]]

def test_fresh_payloads_leave_all_inactive_packet_bits_zero():
    payload=fresh_payloads(4,generator=torch.Generator().manual_seed(1))
    assert not payload[:,:4].any() and not payload[:,12:].any()
    assert set(payload[:,4:12].unique().tolist()) == {0.0,1.0}
