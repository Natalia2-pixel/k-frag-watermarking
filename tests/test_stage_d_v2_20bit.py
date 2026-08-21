import json
import torch
import yaml

from kfrag.diagnostics.stage_d_tag_capacity import verify_12bit_parent, transplant_audit
from kfrag.models.stage_d_tag_capacity_v1 import StageDTagCapacityV1
from kfrag.training.distributed_auth_neural_v2 import (
    deterministic_scientific_key, evaluate_protocol_controls, fragments_from_logits,
    fresh_distributed_packet_batch,
)


def config():
    with open("configs/stage_d_v2_20bit_neural_local.yaml",encoding="utf-8") as handle:return yaml.safe_load(handle)


def test_real_distributed_protocol_generates_packet_targets():
    g=torch.Generator().manual_seed(3); key=deterministic_scientific_key(3); bits,meta=fresh_distributed_packet_batch(2,key,g)
    assert bits.shape==(2,4,4,20) and len(meta)==2
    decoded=fragments_from_logits((bits*2-1)*20)
    from kfrag.protocols.distributed_auth_v2 import JointFragmentCode
    assert all(JointFragmentCode().recover_and_verify(decoded[i],meta[i].source_id,key)["status"]=="valid" for i in range(2))


def test_p0_exact_and_parent_frozen_with_zero_gradients():
    verification,parent=verify_12bit_parent(config()); assert verification["passed"]
    model=StageDTagCapacityV1(parent); [p.requires_grad_(False) for p in model.parent.parameters()]
    g=torch.Generator().manual_seed(4); bits,_=fresh_distributed_packet_batch(1,deterministic_scientific_key(4),g); packet=torch.cat((bits,torch.zeros(1,4,4,24)),-1); image=torch.rand(1,3,64,64,generator=g)
    assert transplant_audit(model,image,packet)["passed"]
    model(image,packet,.014,8)["packet_logits"][...,12:20].sum().backward()
    assert all(p.grad is None or p.grad.count_nonzero()==0 for p in model.parent.parameters())


def test_protocol_cannot_credit_failed_neural_decode():
    g=torch.Generator().manual_seed(5); key=deterministic_scientific_key(5); bits,meta=fresh_distributed_packet_batch(2,key,g)
    good=evaluate_protocol_controls((bits*2-1)*20,meta,key); assert good["authenticated_identity_acceptance"]==1
    failed=torch.zeros_like(bits); bad=evaluate_protocol_controls(failed,meta,key)
    assert bad["authenticated_identity_acceptance"]==0 and bad["token_reconstruction_success"]==0


def test_protocol_controls_consume_thresholded_logits_and_reject_failures():
    g=torch.Generator().manual_seed(6); key=deterministic_scientific_key(6); bits,meta=fresh_distributed_packet_batch(3,key,g)
    metrics=evaluate_protocol_controls((bits*2-1)*9,meta,key)
    assert metrics["protocol_input"]=="thresholded_blind_decoder_logits"
    assert metrics["duplicate_index_rejection"]==metrics["mixed_identity_rejection"]==metrics["insufficient_evidence_rejection"]==1


def test_key_payloads_and_targets_are_not_in_configuration():
    text=json.dumps(config()).lower(); assert "hmac_key" not in text and "expected_payload" not in text and "final_evaluation_targets" not in text

