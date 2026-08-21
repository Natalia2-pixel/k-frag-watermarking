"""Held-out decoder-only evaluation of the failed Stage-D-v2 neural candidate."""
from __future__ import annotations
import json, math, shutil, time
from pathlib import Path
from statistics import mean
import torch

from kfrag.diagnostics.stage_d_tag_capacity import verify_12bit_parent
from kfrag.diagnostics.stage_d_v2_20bit import _three_way_split
from kfrag.diagnostics.stage_c_regional import SyntheticStageCDataset
from kfrag.models.stage_d_tag_capacity_v1 import StageDTagCapacityV1
from kfrag.protocols.soft_fragment_decoder_v1 import SoftAuthenticatedFragmentDecoder,observations_from_logits
from kfrag.training.distributed_auth_neural_v2 import deterministic_scientific_key,fresh_distributed_packet_batch,evaluate_protocol_controls
from kfrag.training.regional_channel_v1 import load_stage_c_population,build_evaluation_population
from kfrag.protocols.distributed_auth_v2 import JointFragmentCode


def _upper_zero_false_accepts(trials,alpha=.05):return 1-alpha**(1/max(1,trials))


def _soft_population(decoder,logits,metadata,key,candidate_sources=None):
    results=[]
    for i in range(len(logits)):
        sources=candidate_sources[i] if candidate_sources else [metadata[i].source_id]
        results.append(decoder.decode(logits[i].reshape(16,20),key,sources))
    runtimes=sorted(x["runtime_ms"] for x in results);p95=runtimes[min(len(runtimes)-1,math.ceil(.95*len(runtimes))-1)]
    protocol=JointFragmentCode();auth_rate=mean(protocol._shares(metadata[i].token,metadata[i].source_id,key)[:8] in x["_authenticator_values"] for i,x in enumerate(results))
    return {"token_reconstruction_rate":mean(x["token"]==metadata[i].token for i,x in enumerate(results)),"authenticator_reconstruction_rate":auth_rate,"authenticated_identity_acceptance":mean(x["status"]=="authenticated" for x in results),"false_acceptance":0.0,"false_rejection":mean(x["status"]!="authenticated" for x in results),"average_candidate_count":mean(x["candidate_count"] for x in results),"worst_candidate_count":max(x["candidate_count"] for x in results),"average_search_attempts":mean(x["search_attempts"] for x in results),"worst_search_attempts":max(x["search_attempts"] for x in results),"average_runtime_ms":mean(runtimes),"p95_runtime_ms":p95,"search_budget_exhaustion_rate":mean(x["search_budget_exhausted"] for x in results),"state_counts":{state:sum(list(x["states"].values()).count(state) for x in results) for state in ("valid","missing","manipulated","uncertain")}}

def _classification_summary(results,expected):
    states={state:sum(list(x["states"].values()).count(state) for x in results) for state in ("valid","missing","manipulated","uncertain")}
    correct=total=0
    for result in results:
        for index,state in expected.items():correct+=result["states"][index]==state;total+=1
    return {"state_counts":states,"classification_accuracy":correct/max(1,total)}


def run_soft_decoder_experiment(config):
    output=Path(config["output_directory"]);output.mkdir(parents=True,exist_ok=True);candidate=Path(config["selected_candidate_checkpoint"]);baseline_report=json.loads(Path(config["neural_report"]).read_text())
    verification,parent=verify_12bit_parent(config);checks={"parent":verification["passed"],"candidate_exists":candidate.is_file(),"failed_status_preserved":baseline_report.get("scientific_status")=="blocked_by_stage_d_v2_20bit_neural_gate","selected_step":baseline_report.get("selection_metrics",{}).get("step")==450,"not_promoted":candidate.name=="last.pt" and not (candidate.parent/"best.pt").exists()}
    if not all(checks.values()):
        report={"scientific_status":"blocked_by_neural_candidate","neural_stage_passed":False,"decoder_feasibility_passed":False,"stage_e_permitted":False,"checks":checks};(output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
    checkpoint=torch.load(candidate,map_location="cpu",weights_only=False);model=StageDTagCapacityV1(parent);model.load_state_dict(checkpoint["model_state"],strict=True);model.eval();[p.requires_grad_(False) for p in model.parameters()]
    shutil.copyfile(candidate,output/"selected_step450_non_promoted_candidate.pt")
    seed=int(config.get("seed",2040));generator=torch.Generator().manual_seed(seed+9001);key=deterministic_scientific_key(seed);dataset=SyntheticStageCDataset(int(config["synthetic_image_count"]));ids=[dataset[i]["relative_path"] for i in range(len(dataset))];split=_three_way_split(ids,int(config["train_images"]),int(config["selection_images"]),int(config["final_test_images"]),seed);images=load_stage_c_population(dataset,split["final_test"],config["preprocessing"],64);images,pop=build_evaluation_population(images,int(config["evaluation_samples"]));bits,metadata=fresh_distributed_packet_batch(len(images),key,generator);packet=torch.cat((bits,bits.new_zeros((*bits.shape[:-1],24))),-1)
    with torch.no_grad():logits=model(images,packet,.014,8)["packet_logits"][...,:20];original=torch.cat((model.parent.index_head(images),model.parent.stage_c.decoder(images),model.tag_head(images)[...,:8]),-1)
    hard_times=[]
    for i in range(len(logits)):
        started=time.perf_counter();evaluate_protocol_controls(logits[i:i+1],metadata[i:i+1],key);hard_times.append((time.perf_counter()-started)*1000)
    hard=evaluate_protocol_controls(logits,metadata,key);hard_sorted=sorted(hard_times);hard.update({"token_reconstruction_rate":hard["token_reconstruction_success"],"authenticator_reconstruction_rate":hard["authenticator_reconstruction_success"],"false_acceptance":0.0,"false_rejection":1-hard["authenticated_identity_acceptance"],"average_candidate_count":1,"worst_candidate_count":1,"average_search_attempts":1,"worst_search_attempts":1,"average_runtime_ms":mean(hard_times),"p95_runtime_ms":hard_sorted[min(len(hard_sorted)-1,math.ceil(.95*len(hard_sorted))-1)],"search_budget_exhaustion_rate":0.0})
    decoder=SoftAuthenticatedFragmentDecoder(int(config["field_top_k"]),int(config["beam_width"]),int(config["search_budget"]),float(config["uncertain_confidence"]));soft=_soft_population(decoder,logits,metadata,key)
    controls={}
    def corrupt(x,count):
        changed=x.clone();changed[:count,4:20]=-changed[:count,4:20];return changed
    transforms={"all_16":lambda x:x,"shuffled":lambda x:x[torch.randperm(16,generator=generator)],"missing_4":lambda x:x[:12],"auth_8":lambda x:x[:8],"duplicate":lambda x:torch.cat((x[:15],x[:1])),"corrupt_1":lambda x:corrupt(x,1),"corrupt_2":lambda x:corrupt(x,2),"insufficient":lambda x:x[:7]}
    for name,transform in transforms.items():
        results=[decoder.decode(transform(logits[i].reshape(16,20)),key,[metadata[i].source_id]) for i in range(len(logits))];expected={i:"valid" for i in range(16)}
        if name=="missing_4":expected.update({i:"missing" for i in range(12,16)})
        if name=="corrupt_1":expected[0]="manipulated"
        if name=="corrupt_2":expected.update({0:"manipulated",1:"manipulated"})
        protocol=JointFragmentCode();auth_rate=mean(protocol._shares(metadata[i].token,metadata[i].source_id,key)[:8] in result["_authenticator_values"] for i,result in enumerate(results))
        controls[name]={"authenticated":mean(x["status"]=="authenticated" for x in results),"authenticator_reconstruction":auth_rate,"status_counts":{s:sum(x["status"]==s for x in results) for s in ("authenticated","rejected","insufficient","ambiguous","search_budget_exceeded")},**_classification_summary(results,expected)}
    mixed=[]
    for i in range(len(logits)):mixed.append(decoder.decode(torch.cat((logits[i].reshape(16,20)[:8],logits[(i+1)%len(logits)].reshape(16,20)[8:])),key,[metadata[i].source_id]))
    controls["mixed_identities"]={"false_acceptance":mean(x["status"]=="authenticated" for x in mixed)}
    wrong=[decoder.decode(logits[i].reshape(16,20),bytes(32),[metadata[i].source_id]) for i in range(len(logits))];controls["incorrect_key"]={"false_acceptance":mean(x["status"]=="authenticated" for x in wrong)}
    negative_trials=int(config["negative_control_trials"]);false_accepts=0;hard_false_accepts=0
    for i in range(negative_trials):
        random_logits=torch.randn(16,20,generator=generator);source=bytes(torch.randint(0,256,(8,),generator=generator,dtype=torch.uint8).tolist());result=decoder.decode(random_logits,key,[source]);false_accepts+=result["status"]=="authenticated"
        random_meta=[type(metadata[0])(metadata[0].token,source)]
        hard_false_accepts+=evaluate_protocol_controls(random_logits.reshape(1,4,4,20),random_meta,key)["authenticated_identity_acceptance"]>0
    unwatermarked=[decoder.decode(original[i].reshape(16,20),key,[metadata[i].source_id]) for i in range(len(original))];false_accepts_unwatermarked=sum(x["status"]=="authenticated" for x in unwatermarked)
    negatives={"random_trials":negative_trials,"random_false_accepts":false_accepts,"random_false_accept_rate":false_accepts/negative_trials,"hard_random_false_accepts":hard_false_accepts,"zero_false_accept_95_percent_upper_bound":_upper_zero_false_accepts(negative_trials) if false_accepts==0 else None,"unwatermarked_trials":len(unwatermarked),"unwatermarked_false_accepts":false_accepts_unwatermarked,"claim":"finite-sample confidence bound only; not empirical proof of 2^-64 security"};hard["false_acceptance"]=(hard_false_accepts+0)/(negative_trials+len(unwatermarked));soft["false_acceptance"]=(false_accepts+false_accepts_unwatermarked)/(negative_trials+len(unwatermarked))
    feasibility=soft["authenticated_identity_acceptance"]>hard["authenticated_identity_acceptance"] and controls["mixed_identities"]["false_acceptance"]==0 and controls["incorrect_key"]["false_acceptance"]==0 and false_accepts==0 and false_accepts_unwatermarked==0
    observations=[obs for grid in logits for obs in observations_from_logits(grid.reshape(16,20),int(config["field_top_k"]))];logit_diagnostics={"mean_packet_log_likelihood":mean(x.packet_log_likelihood for x in observations),"mean_index_confidence":mean(x.index_confidence for x in observations),"mean_rs_byte_confidence":mean(x.rs_confidence for x in observations),"mean_authentication_share_confidence":mean(x.auth_confidence for x in observations),"mean_bit_probability":mean(p for x in observations for p in x.bit_probabilities)}
    report={"schema_version":"soft_authenticated_fragment_decoder_v1.0","checks":checks,"candidate_label":"selected_step450_non_promoted_diagnostic_candidate","baseline_scientific_status":baseline_report["scientific_status"],"hard_decoder":hard,"soft_decoder":soft,"logit_diagnostics":logit_diagnostics,"controls":controls,"negative_controls":negatives,"evaluation_population":pop,"candidate_source_policy":"caller supplies public registry/candidate source identifiers; no expected packet or token is supplied","neural_stage_passed":False,"decoder_feasibility_passed":feasibility,"stage_e_permitted":False,"scientific_status":"passed_soft_decoder_feasibility_only" if feasibility else "blocked_by_soft_authenticated_decoder","novelty_status":"falsifiable hypothesis requiring literature comparison; not an established novelty claim"};(output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
