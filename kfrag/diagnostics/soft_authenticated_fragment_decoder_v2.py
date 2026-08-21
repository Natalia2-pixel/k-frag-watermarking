"""Three-population selection and locked-final evaluation for soft decoder v2."""
from __future__ import annotations
import json,math,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean
import torch
from kfrag.diagnostics.soft_authenticated_fragment_decoder import _upper_zero_false_accepts
from kfrag.diagnostics.stage_c_regional import SyntheticStageCDataset
from kfrag.diagnostics.stage_d_tag_capacity import verify_12bit_parent
from kfrag.models.stage_d_tag_capacity_v1 import StageDTagCapacityV1
from kfrag.protocols.distributed_auth_v2 import JointFragmentCode
from kfrag.protocols.soft_fragment_decoder_v2 import SoftAuthenticatedFragmentDecoderV2,calibrated_observations
from kfrag.training.distributed_auth_neural_v2 import deterministic_scientific_key,fresh_distributed_packet_batch,evaluate_protocol_controls
from kfrag.training.regional_channel_v1 import load_stage_c_population

CLASSES=("valid","missing","manipulated","uncertain")
def _split(ids,counts,seed):
    order=list(ids);import random;random.Random(seed).shuffle(order);a,b,c=counts
    if a+b+c>len(order):raise ValueError("decoder populations must be disjoint")
    return {"development":order[:a],"selection_validation":order[a:a+b],"locked_final_test":order[a+b:a+b+c]}
def _logits(model,images,bits):
    packet=torch.cat((bits,bits.new_zeros((*bits.shape[:-1],24))),-1)
    with torch.no_grad():return model(images,packet,.014,8)["packet_logits"][...,:20]
def _population(decoder,logits,metadata,key):
    results=[decoder.decode(logits[i].reshape(16,20),key,[metadata[i].source_id]) for i in range(len(logits))];times=sorted(x["runtime_ms"] for x in results);protocol=JointFragmentCode()
    return {"token_reconstruction_rate":mean(x["token"]==metadata[i].token for i,x in enumerate(results)),"authenticator_reconstruction_rate":mean(protocol._shares(metadata[i].token,metadata[i].source_id,key)[:8] in x["_authenticator_values"] for i,x in enumerate(results)),"authenticated_acceptance":mean(x["status"]=="authenticated" for x in results),"false_rejection":mean(x["status"]!="authenticated" for x in results),"average_candidate_count":mean(x["candidate_count"] for x in results),"worst_candidate_count":max(x["candidate_count"] for x in results),"average_attempts":mean(x["search_attempts"] for x in results),"worst_attempts":max(x["search_attempts"] for x in results),"average_runtime_ms":mean(times),"p95_runtime_ms":times[min(len(times)-1,math.ceil(.95*len(times))-1)],"budget_exhaustion_rate":mean(x["search_budget_exhausted"] for x in results),"results":results}
def _oracle_diagnosis(decoder,logits,bits,metadata,key):
    rows=[];protocol=JointFragmentCode()
    for sample in range(len(logits)):
        result=decoder.decode(logits[sample].reshape(16,20),key,[metadata[sample].source_id]);obs=calibrated_observations(logits[sample].reshape(16,20),decoder.field_top_k,decoder.temperatures);truth=bits[sample].reshape(16,20).int();index_cov=[];rs_cov=[];auth_cov=[];packet_cov=[]
        for i,o in enumerate(obs):
            idx=int("".join(map(str,truth[i,:4].tolist())),2);rs=int("".join(map(str,truth[i,4:12].tolist())),2);auth=int("".join(map(str,truth[i,12:20].tolist())),2);a=idx in [x.value for x in o.indices];b=rs in [x.value for x in o.symbols];c=auth in [x.value for x in o.shares];index_cov.append(a);rs_cov.append(b);auth_cov.append(c);packet_cov.append(a and b and c)
        expected_auth=protocol._shares(metadata[sample].token,metadata[sample].source_id,key)[:8];token_present=metadata[sample].token in result["_token_values"];auth_present=expected_auth in result["_authenticator_values"]
        failure="accepted" if result["status"]=="authenticated" else "true_index_absent" if not all(index_cov) else "correct_rs_absent" if not all(rs_cov) else "correct_auth_share_absent" if not all(auth_cov) else "correct_token_absent" if not token_present else "correct_authenticator_absent" if not auth_present else "search_budget_or_pruning" if result["search_budget_exhausted"] else "ambiguity" if result["status"]=="ambiguous" else "hmac_verification_failed"
        rows.append({"sample":sample,"accepted":result["status"]=="authenticated","failure_stage":failure,"index_coverage":mean(index_cov),"rs_coverage":mean(rs_cov),"auth_share_coverage":mean(auth_cov),"complete_packet_coverage":mean(packet_cov),"correct_token_candidate":token_present,"correct_authenticator_candidate":auth_present,"correct_token_reached_hmac_verification":token_present and result["hmac_candidates_checked"]>0})
    return {"samples":rows,"top_k_coverage":{"indices":mean(x["index_coverage"] for x in rows),"rs_symbols":mean(x["rs_coverage"] for x in rows),"authentication_shares":mean(x["auth_share_coverage"] for x in rows),"complete_packets":mean(x["complete_packet_coverage"] for x in rows),"final_token_candidates":mean(x["correct_token_candidate"] for x in rows)},"failure_counts":{name:sum(x["failure_stage"]==name for x in rows) for name in sorted({x["failure_stage"] for x in rows})},"oracle_use":"measurement only; no diagnostic value is passed back to decode()"}
def _confusion(results,expected):
    matrix={actual:{pred:0 for pred in CLASSES} for actual in CLASSES}
    for result,labels in zip(results,expected):
        for i,actual in labels.items():matrix[actual][result["states"][i]]+=1
    metrics={}
    for cls in CLASSES:
        tp=matrix[cls][cls];fp=sum(matrix[a][cls] for a in CLASSES if a!=cls);fn=sum(matrix[cls][p] for p in CLASSES if p!=cls);precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.;metrics[cls]={"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0.}
    return {"matrix":matrix,"per_class":metrics}
def _control(decoder,logits,metadata,key,kind):
    transformed=[];expected=[]
    for i in range(len(logits)):
        x=logits[i].reshape(16,20).clone();labels={j:"valid" for j in range(16)}
        if kind=="shuffled":x=x[torch.arange(15,-1,-1)]
        elif kind=="missing4":x=x[:12];labels.update({j:"missing" for j in range(12,16)})
        elif kind=="corrupt1":x[0,4:20]=-x[0,4:20];labels[0]="manipulated"
        elif kind=="corrupt2":x[:2,4:20]=-x[:2,4:20];labels.update({0:"manipulated",1:"manipulated"})
        elif kind=="duplicate":x=torch.cat((x[:15],x[:1]));labels={j:"uncertain" for j in range(16)}
        elif kind=="insufficient":x=x[:7];labels={j:("uncertain" if j<7 else "missing") for j in range(16)}
        transformed.append(x);expected.append(labels)
    results=[decoder.decode(x,key,[metadata[i].source_id]) for i,x in enumerate(transformed)]
    return {"authenticated_acceptance":mean(x["status"]=="authenticated" for x in results),"budget_exhaustion_rate":mean(x["search_budget_exhausted"] for x in results),"status_counts":{s:sum(x["status"]==s for x in results) for s in ("authenticated","rejected","insufficient","ambiguous","search_budget_exceeded")},"classification":_confusion(results,expected)}
def _random_negative_worker(arguments):
    parameters,key,seed,count=arguments;decoder=SoftAuthenticatedFragmentDecoderV2(**parameters);generator=torch.Generator().manual_seed(seed);false_accepts=0;exhaustions=0
    for _ in range(count):
        source=bytes(torch.randint(0,256,(8,),generator=generator,dtype=torch.uint8).tolist());result=decoder.decode(torch.randn(16,20,generator=generator),key,[source]);false_accepts+=result["status"]=="authenticated";exhaustions+=result["search_budget_exhausted"]
    return false_accepts,exhaustions
def run_soft_decoder_v2(config):
    output=Path(config["output_directory"]);output.mkdir(parents=True,exist_ok=True);verification,parent=verify_12bit_parent(config);checkpoint=torch.load(config["selected_candidate_checkpoint"],map_location="cpu",weights_only=False);model=StageDTagCapacityV1(parent);model.load_state_dict(checkpoint["model_state"],strict=True);model.eval();[p.requires_grad_(False) for p in model.parameters()];seed=int(config["seed"]);key=deterministic_scientific_key(seed);dataset=SyntheticStageCDataset(int(config["synthetic_image_count"]));ids=[dataset[i]["relative_path"] for i in range(len(dataset))];split=_split(ids,(int(config["development_images"]),int(config["selection_images"]),int(config["final_test_images"])),seed);generator=torch.Generator().manual_seed(seed+11001)
    populations={}
    for name in ("development","selection_validation"):
        images=load_stage_c_population(dataset,split[name],config["preprocessing"],64);bits,metadata=fresh_distributed_packet_batch(len(images),key,generator);populations[name]=(images,bits,metadata,_logits(model,images,bits))
    baseline=SoftAuthenticatedFragmentDecoderV2(**config["decoder_candidates"][0]);development_diagnosis=_oracle_diagnosis(baseline,populations["development"][3],populations["development"][1],populations["development"][2],key);selection=[]
    for parameters in config["decoder_candidates"]:
        decoder=SoftAuthenticatedFragmentDecoderV2(**parameters);clean=_population(decoder,populations["selection_validation"][3],populations["selection_validation"][2],key);missing=_control(decoder,populations["selection_validation"][3],populations["selection_validation"][2],key,"missing4");score=(clean["authenticated_acceptance"]+missing["authenticated_acceptance"],-clean["budget_exhaustion_rate"],-clean["average_runtime_ms"]);selection.append({"parameters":parameters,"clean":{k:v for k,v in clean.items() if k!="results"},"missing4":missing,"score":score})
    chosen=max(selection,key=lambda x:tuple(x["score"]));decoder=SoftAuthenticatedFragmentDecoderV2(**chosen["parameters"])
    # Locked final material is created and decoded only after selection is complete.
    final_images=load_stage_c_population(dataset,split["locked_final_test"],config["preprocessing"],64);final_bits,final_metadata=fresh_distributed_packet_batch(len(final_images),key,generator);final_logits=_logits(model,final_images,final_bits);final=_population(decoder,final_logits,final_metadata,key);final_diagnosis=_oracle_diagnosis(decoder,final_logits,final_bits,final_metadata,key);hard=evaluate_protocol_controls(final_logits,final_metadata,key);controls={name:_control(decoder,final_logits,final_metadata,key,name) for name in ("shuffled","missing4","corrupt1","corrupt2","duplicate","insufficient")}
    mixed=[decoder.decode(torch.cat((final_logits[i].reshape(16,20)[:8],final_logits[(i+1)%len(final_logits)].reshape(16,20)[8:])),key,[final_metadata[i].source_id]) for i in range(len(final_logits))];wrong=[decoder.decode(final_logits[i].reshape(16,20),bytes(32),[final_metadata[i].source_id]) for i in range(len(final_logits))]
    with torch.no_grad():original=torch.cat((model.parent.index_head(final_images),model.parent.stage_c.decoder(final_images),model.tag_head(final_images)[...,:8]),-1)
    unwatermarked=[decoder.decode(original[i].reshape(16,20),key,[final_metadata[i].source_id]) for i in range(len(original))];negative_trials=int(config["negative_control_trials"]);workers=int(config.get("negative_control_workers",4));counts=[negative_trials//workers+(i<negative_trials%workers) for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as pool:negative_results=list(pool.map(_random_negative_worker,[(chosen["parameters"],key,seed+50000+i,count) for i,count in enumerate(counts)]))
    random_false=sum(x[0] for x in negative_results);random_exhaustions=sum(x[1] for x in negative_results)
    false_counts={"mixed_identities":sum(x["status"]=="authenticated" for x in mixed),"wrong_key":sum(x["status"]=="authenticated" for x in wrong),"unwatermarked":sum(x["status"]=="authenticated" for x in unwatermarked),"random_logits":random_false};total_negative=len(mixed)+len(wrong)+len(unwatermarked)+negative_trials;upper=_upper_zero_false_accepts(total_negative) if sum(false_counts.values())==0 else None
    criteria={"clean_acceptance":final["authenticated_acceptance"]>=.90,"shuffled_matches":abs(controls["shuffled"]["authenticated_acceptance"]-final["authenticated_acceptance"])<=1/len(final_logits),"twelve_of_sixteen":controls["missing4"]["authenticated_acceptance"]>=.75,"zero_false_accepts":sum(false_counts.values())==0,"no_budget_exhaustion":final["budget_exhaustion_rate"]==0 and all(x["budget_exhaustion_rate"]==0 for x in controls.values()),"runtime":final["average_runtime_ms"]<100,"manipulated_recall_reported":controls["corrupt1"]["classification"]["per_class"]["manipulated"]["recall"]>0};passed=all(criteria.values())
    clean_safe={k:v for k,v in final.items() if k!="results"};report={"schema_version":"soft_authenticated_fragment_decoder_v2.0","parent_verification":verification,"population_split":{"identifiers":split,"disjoint":True,"locked_final_evaluations":1},"development_failure_diagnosis":development_diagnosis,"selection_validation":selection,"selected_parameters":chosen["parameters"],"selection_rule":"maximize clean plus four-missing authenticated acceptance, then minimize exhaustion and runtime; locked final excluded","hard_baseline":hard,"locked_final":clean_safe,"locked_final_oracle_diagnosis":final_diagnosis,"controls":controls,"false_acceptance":{"counts":false_counts,"total_negative_trials":total_negative,"random_search_budget_exhaustions":random_exhaustions,"observed_rate":sum(false_counts.values())/total_negative,"one_sided_95_percent_upper_bound":upper,"claim":"finite-sample bound only, not proof of 2^-64 security"},"criteria":criteria,"neural_stage_passed":False,"decoder_feasibility_passed":passed,"stage_e_permitted":False,"scientific_status":"passed_soft_authenticated_fragment_decoder_v2_feasibility" if passed else "blocked_by_soft_authenticated_fragment_decoder_v2","novelty_status":"falsifiable hypothesis requiring literature comparison; not established novelty"};(output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
