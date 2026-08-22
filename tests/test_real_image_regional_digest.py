import inspect,json
import numpy as np
import torch
from kfrag.diagnostics.real_image_regional_digest import deterministic_three_split,benign_transform,malicious_transform,_select_threshold
from kfrag.protocols.regional_perceptual_digest_v1 import *
def test_four_by_four_regions_and_digest_sizes():
    image=torch.rand(3,256,256);assert regions(image).shape==(16,3,64,64);assert len(DCTPerceptualHash().digest_image(image)[0])==64;assert len(DifferenceHash().digest_image(image)[0])==64
def test_registry_binds_required_public_fields_and_wrong_key_rejects():
    image=torch.rand(3,256,256);records,_=create_registry("coco/1.jpg",image,DCTPerceptualHash(),b"k"*32);assert authenticate_registry(records,b"k"*32) and not authenticate_registry(records,b"x"*32);record=records[3];assert record.image_identifier=="coco/1.jpg" and record.protocol_version==2 and record.region_index==3 and record.digest_type=="dct_phash" and record.digest_version==1
def test_three_populations_are_deterministic_disjoint():
    a=deterministic_three_split([str(x) for x in range(12)],(4,4,4),2);b=deterministic_three_split([str(x) for x in range(12)],(4,4,4),2);assert a==b and not set(a["calibration"])&set(a["selection_validation"]) and not set(a["selection_validation"])&set(a["locked_final_test"])
def test_threshold_selection_has_no_final_population_argument():assert list(inspect.signature(_select_threshold).parameters)==["calibration_benign","validation_benign","validation_malicious","validation_truth"]
def test_benign_and_content_changing_transforms_are_separate():
    image=torch.rand(3,256,256);donor=torch.rand_like(image);assert benign_transform(image,"jpeg_q90").shape==image.shape;changed,mask=malicious_transform(image,donor,"replacement_full");assert mask.sum()==1 and not torch.equal(changed,image)
def test_four_state_rule_requires_identity_and_preserves_missing_uncertain():
    distances=[0.,.2,.3,.8];present=[True,False,True,True];assert classify_distances(distances,present,True,.1,.5)==("valid","missing","uncertain","manipulated");assert classify_distances(distances,present,False,.1,.5)==("uncertain","missing","uncertain","uncertain")
def test_runtime_key_is_not_part_of_record_payload():assert "key" not in RegionalDigestRecord.__annotations__
def test_digest_names_do_not_claim_cryptographic_collision_resistance():assert all("crypto" not in digest.name for digest in digest_candidates())
