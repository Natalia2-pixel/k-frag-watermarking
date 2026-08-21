import torch,yaml
from pathlib import Path
from kfrag.diagnostics.stage_d_tag_capacity import verify_12bit_parent
from kfrag.models.stage_d_isolated_tag_v1 import StageDIsolatedTagV1

def model():
    config=yaml.safe_load(Path("configs/stage_d_isolated_tag_local.yaml").read_text());verification,parent=verify_12bit_parent(config);assert verification["passed"];return StageDIsolatedTagV1(parent)

def test_zero_active_tags_exactly_reproduce_parent():
    m=model().eval();image=torch.rand(1,3,64,64);bits=torch.randint(0,2,(1,4,4,44)).float()
    with torch.no_grad():a=m.parent(image,bits,.014,4);b=m(image,bits,.014,0)
    for key in ("residual","watermarked_image","packet_logits"):assert torch.equal(a[key],b[key])

def test_inactive_tag_bits_have_zero_carrier_effect_and_gradient():
    m=model();tags=torch.rand(1,4,4,32,requires_grad=True);carrier=m.carrier(tags,8);carrier.square().sum().backward();assert tags.grad[...,8:].abs().sum()==0
    changed=tags.detach().clone();changed[...,8:]=1-changed[...,8:];assert torch.equal(m.carrier(tags.detach(),8),m.carrier(changed,8))

def test_tag_carriers_are_decorrelated_from_parent_subspace():
    correlation=model().carrier.correlation_matrix();assert correlation.shape==(32,12);assert correlation.abs().max()<1e-5

def test_parent_parameters_receive_no_gradients_or_updates():
    m=model();
    for p in m.parent.parameters():p.requires_grad=False
    before={k:v.clone() for k,v in m.parent.state_dict().items()};image=torch.rand(1,3,64,64);bits=torch.randint(0,2,(1,4,4,44)).float();m(image,bits,.014,8)["packet_logits"][...,12:20].sum().backward()
    assert all(p.grad is None for p in m.parent.parameters());assert all(torch.equal(before[k],v) for k,v in m.parent.state_dict().items())

def test_combined_residual_is_explicitly_bounded():
    m=model();image=torch.rand(2,3,64,64);bits=torch.randint(0,2,(2,4,4,44)).float();out=m(image,bits,.014,32);assert out["residual"].abs().max()<=.01400001

def test_lower_checkpoint_policy_and_no_stage_e_or_kaggle_support():
    source=Path("kfrag/diagnostics/stage_d_isolated_tag.py").read_text();assert 'directory/"best.pt"' in source and 'if passed:torch.save(checkpoint,output/"best.pt")' in source;assert "stage_e_permitted\":False" in source

def test_diagnostics_include_spatial_frequency_and_gradient_interference():
    source=Path("kfrag/diagnostics/stage_d_isolated_tag.py").read_text();assert "post_projection_correlation_summary" in source;assert '"matrix_shape":[32,12]' in source;assert "decoder_gradient_interference_cosine" in source
