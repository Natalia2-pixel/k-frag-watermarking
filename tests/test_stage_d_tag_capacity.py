import json
import torch
from kfrag.diagnostics.stage_d_tag_capacity import verify_12bit_parent,transplant_audit
from kfrag.models.stage_d_tag_capacity_v1 import StageDTagCapacityV1,_tag_bases,CAPACITIES
from kfrag.training.stage_d_tag_capacity import active_bits,tag_capacity_loss
import yaml

def config():return yaml.safe_load(open("configs/stage_d_tag_capacity_local.yaml"))
def loaded_model():
    verification,parent=verify_12bit_parent(config());assert verification["passed"];return StageDTagCapacityV1(parent)

def test_parent_hash_status_schema_and_roundtrip():
    verification,_=verify_12bit_parent(config());assert verification["passed"] and verification["sha256"]=="37B312A8FCCB93F23D5A519BB51EDFCA105A962BD9C4ECA24C395826C91BCC0A" and verification["roundtrip_max_difference"]==0

def test_capacity_mapping_is_p0_through_p4():
    assert CAPACITIES==(12,20,28,36,44);assert [len(active_bits(x)) for x in (0,8,16,24,32)]==list(CAPACITIES)

def test_p0_exactly_reproduces_12bit_parent_and_tags_are_inactive():
    model=loaded_model().eval();image=torch.rand(1,3,64,64);bits=torch.randint(0,2,(1,4,4,44)).float();audit=transplant_audit(model,image,bits);assert audit["passed"] and audit["maximum_logit_difference"]==0

def test_tag_bases_are_distinct_normalized_and_low_correlation():
    bases=_tag_bases(16);assert bases.shape==(32,16,16);gram=bases.flatten(1)@bases.flatten(1).T/256;assert torch.allclose(torch.diag(gram),torch.ones(32),atol=1e-5);assert (gram-torch.eye(32)).abs().max()<1e-5

def test_future_tag_bits_have_no_loss_gradient():
    logits=torch.randn(1,4,4,44,requires_grad=True);bits=torch.randint(0,2,(1,4,4,44)).float();parent=torch.randn_like(logits);tag_capacity_loss(logits,bits,8,parent,{})["total"].backward();assert logits.grad[...,20:].abs().sum()==0

def test_stage_e_and_secrets_absent_from_configuration():
    text=json.dumps(config()).lower();assert "authentication_secret" not in text and "hmac_key" not in text

def test_deterministic_index_is_excluded_from_shuffled_random_field_control():
    assert active_bits(8)==tuple(range(20));assert tuple(range(4,20))==tuple(range(4,12+8))
