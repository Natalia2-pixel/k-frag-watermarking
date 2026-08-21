import torch
import yaml
from pathlib import Path
from kfrag.models.natural_channel_v2 import NaturalChannelV2
from kfrag.models.regional_channel_v1 import RegionalChannelV1
from kfrag.models.stage_d_12bit_transition_v1 import StageD12BitTransitionV1
from kfrag.training.stage_d_12bit_transition import active_mapping,transition_loss,select_best_gate_evaluation

def model():return StageD12BitTransitionV1(RegionalChannelV1(NaturalChannelV2(64,8),64))

def test_mapping_is_index_then_exact_stage_c_rs():
    assert active_mapping(0)==tuple(range(4,12));assert active_mapping(4)==tuple(range(4,12))+tuple(range(4))

def test_r0_is_numerically_identical_to_parent():
    m=model().eval();image=torch.rand(2,3,64,64);bits=torch.randint(0,2,(2,4,4,44)).float()
    with torch.no_grad():parent=m.stage_c(image,bits[...,4:12],.014);repair=m.reproduce_stage_c(image,bits,.014)
    assert torch.equal(parent["residual"],repair["residual"]);assert torch.equal(parent["watermarked_image"],repair["watermarked_image"]);assert torch.equal(parent["regional_logits"],repair["packet_logits"][...,4:12])

def test_stage_c_basis_and_decoder_are_not_copied_or_reordered():
    m=model();assert m.stage_c.encoder.router is m.stage_c.decoder.router
    assert m.stage_c.decoder.regional_output.out_features==8

def test_inactive_tag_bits_have_no_carrier_output_or_loss_gradient():
    m=model();image=torch.rand(1,3,64,64);bits=torch.rand(1,4,4,44,requires_grad=True);out=m(image,bits,.014,4);parent=m.stage_c(image,bits[...,4:12],.014)["regional_logits"]
    loss=transition_loss(out["packet_logits"],bits,4,parent)["total"];loss.backward();assert bits.grad[...,12:].abs().sum()==0
    changed=bits.detach().clone();changed[...,12:]=1-changed[...,12:]
    with torch.no_grad():a=m(image,bits.detach(),.014,4);b=m(image,changed,.014,4)
    assert torch.equal(a["residual"],b["residual"])

def test_each_new_index_bit_changes_only_its_carrier_contribution():
    m=model();bits=torch.zeros(1,4,4,4)
    for j in range(4):
        changed=bits.clone();changed[:,0,0,j]=1
        delta=m.index_carrier(changed,4)-m.index_carrier(bits,4);assert delta.abs().sum()>0

def test_kaggle_config_is_real_coco_and_stops_at_12_bits():
    config=yaml.safe_load(Path("configs/stage_d_12bit_transition_kaggle.yaml").read_text())
    assert config["synthetic_image_count"]==0 and config["train_images"]==128 and config["validation_images"]==64 and config["final_evaluation_samples"]==256
    assert "tag" not in config and config["preprocessing"]["resize"]==[64,64]

def _passing_metrics(index=.99):
    return {"overall_active_bit_accuracy":.98,"index_bit_accuracy":index,"rs_bit_accuracy":.975,"per_region_accuracy":[.96]*16,"per_active_bit_accuracy":[.95]*9,
      "correct_minus_shuffled_margin":.46,"correct_minus_spatial_margin":.44,"original_image_randomized_field_accuracy":.50,"psnr":39,"ssim":.998,
      "residual_saturation_fraction":0,"analytical_contribution":0,"blind_decoder":True,"disjoint_images":True}

def test_gate_aware_selection_retains_early_index_peak_after_degradation():
    early=_passing_metrics(.9931640625);late=_passing_metrics(.88671875);late["overall_active_bit_accuracy"]=.94
    selected=select_best_gate_evaluation([early,late]);assert selected["index"]==0;assert selected["metrics"]["index_bit_accuracy"]==.9931640625

def test_no_passing_checkpoint_is_promoted_when_any_mandatory_gate_fails():
    saturated=_passing_metrics();saturated["residual_saturation_fraction"]=.002
    assert select_best_gate_evaluation([saturated]) is None
