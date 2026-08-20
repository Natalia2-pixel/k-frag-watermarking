import json,torch
from kfrag.models.natural_channel_v2 import NaturalChannelV2
from kfrag.models.regional_channel_v1 import RegionalChannelV1
from kfrag.training.regional_channel_v1 import (BIT_MAPPING,active_region_mask,fresh_regional_bits,regional_metrics,stage_c_gates,save_stage_c,
 preprocess_stage_c_image,load_stage_c_population,validate_model_batch)
from kfrag.diagnostics.stage_c_regional import verify_parent,evaluate_final

SPEC={"numeric_range":[0.0,1.0],"dtype":"float32","channel_order":"RGB","resize":[64,64],"interpolation":"bilinear","antialias":True,"normalization":"none"}

def test_mapping_has_128_unique_regional_active_bits():
    assert len(BIT_MAPPING)==128 and len(set(BIT_MAPPING))==128 and all("regional_symbol_bit" in x for x in BIT_MAPPING)

def test_fresh_payloads_and_active_masks_vary_independently():
    g=torch.Generator().manual_seed(3);a=fresh_regional_bits(4,g);b=fresh_regional_bits(4,g);mask=active_region_mask(4,4,g)
    assert not torch.equal(a,b) and mask.sum(1).sum(1).tolist()==[4]*4

def test_spatial_and_shuffled_controls_are_separate():
    bits=torch.randint(0,2,(4,4,4,8)).float();logits=(bits*2-1)*20
    correct=regional_metrics(logits,bits);shuffled=regional_metrics(logits,torch.roll(bits,1,0));spatial=regional_metrics(logits,torch.roll(bits,1,1))
    assert correct["regional_active_bit_accuracy"]==1 and shuffled["regional_active_bit_accuracy"]<.7 and spatial["regional_active_bit_accuracy"]<.7

def test_checkpoint_verification_rejects_wrong_hash(tmp_path):
    p=tmp_path/"best.pt";r=tmp_path/"report.json";m=NaturalChannelV2(16,4);torch.save({"model_state":m.state_dict()},p);r.write_text(json.dumps({"stage_b_v2_passed":True,"stage_c_permitted":True,"learned_only_metrics":{"analytical_weight":0,"learned_weight":1}}))
    result,_,_=verify_parent(p,r,"0"*64,16,4);assert not result["passed"] and not result["checks"]["sha256"]

def test_stage_c_gate_policy_and_best_checkpoint(tmp_path):
    metrics={"regional_active_bit_accuracy":.95,"exact_regional_symbol_accuracy":.8,"per_region_bit_accuracy":[[.9]*4 for _ in range(4)],"per_bit_accuracy":[.9]*8,
      "correct_minus_shuffled_margin":.4,"correct_minus_spatially_permuted_margin":.4,"original_image_bit_accuracy":.5,"original_exact_symbol_false_positive":0,
      "cross_region_leakage":.01,"cross_region_leakage_threshold":.1,"psnr":36,"ssim":.96,"residual_saturation_fraction":0,"disjoint_images":True,"blind_decoder":True,
      "analytical_contribution":0,"no_authentication_secret":True,"no_expected_payload":True}
    assert all(stage_c_gates(metrics).values())

def test_failed_checkpoint_policy_always_last_never_best(tmp_path):
    parent=NaturalChannelV2(16,4);m=RegionalChannelV1(parent,16);opt=torch.optim.AdamW(m.parameters());sch=torch.optim.lr_scheduler.StepLR(opt,1);cfg={"preprocessing":{}}
    split={"seed":1,"train":["a"],"validation":["b"]};save_stage_c(tmp_path,m,opt,sch,cfg,split,1,{"scientific_status":"blocked_by_single_region_routing"},"abc",False)
    assert (tmp_path/"last.pt").exists() and not (tmp_path/"best.pt").exists()

def test_shared_preprocessing_resizes_256_and_non_square_sources():
    square=preprocess_stage_c_image(torch.rand(3,256,256),SPEC);rectangle=preprocess_stage_c_image(torch.rand(3,180,320),SPEC)
    assert square.shape==rectangle.shape==(3,64,64) and square.dtype==rectangle.dtype==torch.float32
    assert 0<=square.min()<=square.max()<=1 and 0<=rectangle.min()<=rectangle.max()<=1

def test_executed_population_loader_returns_checkpoint_shape_for_real_style_sources():
    class Dataset:
        samples=[{"relative_path":"square.jpg","image":torch.rand(3,256,256)},{"relative_path":"wide.jpg","image":torch.rand(3,144,320)}]
        def __len__(self):return len(self.samples)
        def __getitem__(self,index):return self.samples[index]
    loaded=load_stage_c_population(Dataset(),["wide.jpg","square.jpg"],SPEC,64)
    assert loaded.shape==(2,3,64,64)

def test_exact_repeated_final_evaluation_path_has_no_spatial_mismatch():
    model=RegionalChannelV1(NaturalChannelV2(64,16),64);validation=preprocess_stage_c_image(torch.rand(16,3,256,256),SPEC)
    eval_images=preprocess_stage_c_image(validation.repeat(4,1,1,1),SPEC);bits=fresh_regional_bits(64,torch.Generator().manual_seed(4));validate_model_batch(eval_images,bits,64)
    with torch.no_grad():out=model(eval_images,bits,.01)
    assert out["regional_logits"].shape==(64,4,4,8)

def test_fail_fast_model_batch_checks_spatial_and_batch_mismatch():
    bits=torch.zeros(2,4,4,8)
    for image,bad_bits in ((torch.zeros(2,3,256,256),bits),(torch.zeros(2,3,64,64),bits[:1])):
        try:validate_model_batch(image,bad_bits,64)
        except ValueError:pass
        else:raise AssertionError("invalid Stage-C model batch must fail")
