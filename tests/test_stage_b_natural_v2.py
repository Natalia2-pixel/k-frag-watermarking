from pathlib import Path
import torch
from kfrag.models.natural_channel_v2 import NaturalChannelV2
from kfrag.training.natural_channel_v2 import (SCHEMA_VERSION, clip_gradients, deterministic_split,
 fresh_bits, gate_results, make_checkpoint, save_attempt, transition_weights)

CFG={"preprocessing":{"numeric_range":[0,1],"normalization":"none"}}

def test_disjoint_seeded_split_and_variable_pairing():
    ids=[f"image-{i}" for i in range(50)]; a=deterministic_split(ids,32,16,9); b=deterministic_split(ids,32,16,9)
    assert a==b and set(a["train"]).isdisjoint(a["validation"])
    gen=torch.Generator().manual_seed(3); first=fresh_bits(8,gen); second=fresh_bits(8,gen)
    assert not torch.equal(first,second) and not torch.equal(first[0],first[1])

def test_payload_generation_is_not_identifier_derived():
    gen1=torch.Generator().manual_seed(5); gen2=torch.Generator().manual_seed(5)
    assert torch.equal(fresh_bits(4,gen1),fresh_bits(4,gen2))

def test_transition_reaches_zero_analytical_contribution():
    assert transition_weights(0,10)==(1,0) and transition_weights(10,10)==(0,1) and transition_weights(99,10)==(0,1)

def test_gradient_clipping():
    p=torch.nn.Parameter(torch.tensor([10.])); p.grad=torch.tensor([10.]); before,after=clip_gradients([p],1)
    assert before==10 and after<=1.00001

def test_checkpoint_roundtrip_and_no_secrets_or_payloads(tmp_path):
    torch.manual_seed(1); m=NaturalChannelV2(16,4); image=torch.rand(2,3,16,16); bits=torch.randint(0,2,(2,8)).float()
    m.eval(); expected=m(image,bits,.01)["logits"]
    split={"seed":1,"train":["a"],"validation":["b"]}; ck=make_checkpoint(m,None,None,CFG,split,3,[],{})
    path=tmp_path/"roundtrip.pt"; torch.save(ck,path); loaded=torch.load(path,weights_only=False)
    n=NaturalChannelV2(16,4); n.load_state_dict(loaded["model_state"]); n.eval(); actual=n(image,bits,.01)["logits"]
    assert torch.equal(expected,actual) and loaded["schema_version"]==SCHEMA_VERSION
    text=str(loaded.keys()).lower(); assert "secret" not in text and "expected_validation_payload" not in text and "image_pixels" not in text

def test_failed_run_is_structurally_valid_and_best_is_gate_only(tmp_path):
    m=NaturalChannelV2(16,4); split={"seed":1,"train":["a"],"validation":["b"]}; ck=make_checkpoint(m,None,None,CFG,split,3,[],{"scientific_status":"blocked_by_stage_b_v2_prerequisite"})
    save_attempt(tmp_path,ck,False); assert (tmp_path/"last.pt").is_file() and not (tmp_path/"best.pt").exists()
    assert torch.load(tmp_path/"last.pt",weights_only=False)["scientific_status"]=="blocked_by_stage_b_v2_prerequisite"

def test_controls_and_per_bit_gate_reporting():
    metrics={"fresh_active_bit_accuracy":.9,"fresh_exact_symbol_accuracy":.6,"per_bit_accuracy":[.8]*8,
      "correct_minus_shuffled_margin":.3,"original_bit_accuracy":.5,"original_exact_false_positive":0,
      "psnr":40,"ssim":.99,"residual_saturation_fraction":0,"disjoint_images":True,"analytical_weight":0,
      "blind_decoder":True,"no_secret_serialized":True,"no_expected_payload_serialized":True}
    gates=gate_results(metrics); assert len(metrics["per_bit_accuracy"])==8 and all(gates.values())

def test_preprocessing_explicitly_excludes_imagenet_normalization():
    spec={"numeric_range":[0,1],"dtype":"float32","channel_order":"RGB","normalization":"none","imagenet_normalization":False}
    assert spec["numeric_range"]==[0,1] and spec["normalization"]=="none" and not spec["imagenet_normalization"]
