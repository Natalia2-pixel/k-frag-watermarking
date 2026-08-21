import torch,yaml
from pathlib import Path
from kfrag.diagnostics.stage_d_tag_capacity import verify_12bit_parent
from kfrag.diagnostics.stage_d_observation_tag import deterministic_targets_key,evaluate
from kfrag.models.stage_d_observation_tag_v1 import StageDObservationTagV1

def model():
    config=yaml.safe_load(Path("configs/stage_d_observation_tag_local.yaml").read_text());verification,parent=verify_12bit_parent(config);assert verification["passed"] and verification["sha256"]=="37B312A8FCCB93F23D5A519BB51EDFCA105A962BD9C4ECA24C395826C91BCC0A";return StageDObservationTagV1(parent)
def test_zero_tag_exact_parent_reproduction():
    m=model().eval();image=torch.rand(1,3,64,64);bits=torch.randint(0,2,(1,4,4,44)).float()
    with torch.no_grad():a=m.parent(image,bits,.014,4);b=m(image,bits,.014,0)
    assert all(torch.equal(a[k],b[k]) for k in ("residual","watermarked_image","packet_logits"))
def test_parent_frozen_gradients_and_updates_zero():
    m=model();
    for p in m.parent.parameters():p.requires_grad=False
    before={k:v.clone() for k,v in m.parent.state_dict().items()};out=m(torch.rand(1,3,64,64),torch.randint(0,2,(1,4,4,44)).float(),.014,8);out["packet_logits"][...,12:20].sum().backward();assert all(p.grad is None for p in m.parent.parameters());assert all(torch.equal(before[k],v) for k,v in m.parent.state_dict().items())
def test_inactive_tags_zero_effect_and_gradient():
    m=model();tags=torch.rand(1,4,4,32,requires_grad=True);value=m.carrier(tags,8);value.square().sum().backward();assert tags.grad[...,8:].abs().sum()==0;changed=tags.detach().clone();changed[...,8:]=1-changed[...,8:];assert torch.equal(m.carrier(tags.detach(),8),m.carrier(changed,8))
def test_observation_space_bound():assert model().carrier.observation_correlation().abs().max()<1e-5
def test_failed_p1_cannot_create_best_policy():
    source=Path("kfrag/diagnostics/stage_d_observation_tag.py").read_text();assert 'if passed:' in source and 'directory/"best.pt"' in source;assert 'output/"best.pt"' not in source
def test_deterministic_targets_and_configuration():
    assert deterministic_targets_key(9)==deterministic_targets_key(9);assert deterministic_targets_key(9)!=deterministic_targets_key(10);config=yaml.safe_load(Path("configs/stage_d_observation_tag_local.yaml").read_text());assert config["seed"]==2031
def test_combined_residual_bound():
    m=model();out=m(torch.rand(2,3,64,64),torch.randint(0,2,(2,4,4,44)).float(),.014,8);assert out["residual"].abs().max()<=.01400001
def test_repeated_evaluation_targets_and_metrics_are_identical():
    m=model();images=torch.rand(1,3,64,64,generator=torch.Generator().manual_seed(3));bits=torch.randint(0,2,(1,4,4,44),generator=torch.Generator().manual_seed(4)).float();a=evaluate(m,images,bits,torch.Generator().manual_seed(5));b=evaluate(m,images,bits,torch.Generator().manual_seed(5));assert a==b
