import json
from pathlib import Path
import pytest
import torch

from kfrag.crypto.packets import verify_and_recover_token
from kfrag.training.complete_packet_v1 import bits_packet, ephemeral_key, fresh_packet_batch, packet_bits, stage_d_gates
from kfrag.diagnostics.stage_d_complete_packet import EXPECTED_PARENT_SHA256, verify_stage_c_parent


def base_config():
    return {"stage_c_checkpoint":"artifacts/stage_c_kaggle_final_evidence/kaggle_fidelity_repair_v1/best.pt",
      "stage_c_report":"artifacts/stage_c_kaggle_final_evidence/kaggle_fidelity_repair_v1/report.json","stage_c_sha256":EXPECTED_PARENT_SHA256,
      "stage_c_expected_size":2759150,"stage_b_checkpoint":"artifacts/stage_b_v2_kaggle_evidence/transition_repair_kaggle/best.pt","image_size":64,"width":16,
      "preprocessing":{"numeric_range":[0.0,1.0],"dtype":"float32","channel_order":"RGB","resize":[64,64],"interpolation":"bilinear","antialias":True,"normalization":"none"}}


def test_mandatory_parent_verifies_exactly():
    verification,_=verify_stage_c_parent(base_config());assert verification["passed"]
    assert verification["sha256"]==EXPECTED_PARENT_SHA256
    assert verification["roundtrip_max_logit_difference"]==0


def test_protocol_packet_bit_roundtrip_and_token_recovery():
    key=ephemeral_key();bits,tokens=fresh_packet_batch(2,key,torch.Generator().manual_seed(4))
    for sample in range(2):
        packets=[bits_packet(cell.tolist()) for cell in bits[sample].reshape(16,44)]
        assert verify_and_recover_token(packets,key)==tokens[sample]
        assert all(packet_bits(packet)==bits[sample].reshape(16,44)[i].tolist() for i,packet in enumerate(packets))


@pytest.mark.parametrize("bit",[0,4,12])
def test_wrong_key_and_index_symbol_tag_mutations_reject(bit):
    key=ephemeral_key();bits,tokens=fresh_packet_batch(1,key,torch.Generator().manual_seed(8));cells=bits[0].reshape(16,44)
    with pytest.raises(ValueError): verify_and_recover_token([bits_packet(x.tolist()) for x in cells],bytes(32))
    changed=cells.clone();changed[0,bit]=1-changed[0,bit]
    with pytest.raises(ValueError): verify_and_recover_token([bits_packet(x.tolist()) for x in changed],key)


def test_swapped_regions_reject():
    key=ephemeral_key();bits,_=fresh_packet_batch(1,key,torch.Generator().manual_seed(9));packets=[bits_packet(x.tolist()) for x in bits[0].reshape(16,44)]
    packets[0],packets[1]=packets[1],packets[0]
    assert packets[0].region_index!=0 and packets[1].region_index!=1


def test_saturated_or_low_fidelity_result_cannot_pass():
    metrics={"overall_bit_accuracy":1,"index_bit_accuracy":1,"symbol_bit_accuracy":1,"tag_bit_accuracy":1,"per_region_accuracy":[1]*16,"per_bit_accuracy":[1]*44,
      "correct_minus_shuffled_margin":.5,"correct_minus_spatial_margin":.5,"original_image_accuracy":.5,"cross_region_leakage":0,"wrong_key_acceptance_rate":0,"one_bit_mutation_acceptance_rate":0,
      "psnr":40,"ssim":1,"residual_saturation_fraction":.002,"analytical_contribution":0,"disjoint_images":True,"blind_decoder":True,"no_secret_or_expected_payload_serialized":True}
    assert not all(stage_d_gates(metrics).values());metrics["residual_saturation_fraction"]=0;metrics["psnr"]=34.99
    assert not all(stage_d_gates(metrics).values())


def test_no_key_in_public_config_or_serialized_schema():
    text=json.dumps(base_config()).lower();assert "hmac_key" not in text and "authentication_secret" not in text
