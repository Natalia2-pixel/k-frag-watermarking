"""Real-COCO grid-erasure and blind arbitrary-crop recovery feasibility."""
from __future__ import annotations
import hashlib,io,json,math,random,time
from pathlib import Path
from statistics import mean
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from kfrag.data import CocoImageDataset
from kfrag.diagnostics.stage_d_tag_capacity import verify_12bit_parent
from kfrag.models.stage_d_tag_capacity_v1 import StageDTagCapacityV1
from kfrag.protocols.blind_crop_search_v1 import CropSearchLimits,blind_crop_search,blind_regional_logits
from kfrag.protocols.soft_fragment_decoder_v2 import SoftAuthenticatedFragmentDecoderV2,calibrated_observations
from kfrag.training.distributed_auth_neural_v2 import deterministic_scientific_key,fresh_distributed_packet_batch
from kfrag.training.regional_channel_v1 import load_stage_c_population

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()
def _walk(value,ids,hashes):
    if isinstance(value,dict):
        for item in value.values():
            if isinstance(item,str) and item.lower().endswith((".jpg",".jpeg",".png")):ids.add(Path(item).name)
            if isinstance(item,str) and len(item)==64 and all(x in "0123456789abcdefABCDEF" for x in item):hashes.add(item.lower())
            _walk(item,ids,hashes)
    elif isinstance(value,list):
        for item in value:_walk(item,ids,hashes)
def _exclusions():
    ids=set();hashes=set();paths=list(Path("artifacts").glob("**/report.json"))+list(Path("docs/evidence").glob("*.json"))+list(Path("outputs").glob("**/report.json"))
    for path in paths:
        try:_walk(json.loads(path.read_text()),ids,hashes)
        except (OSError,json.JSONDecodeError):pass
    return ids,hashes,[str(x) for x in paths]
def _population(dataset,config):
    excluded_ids,excluded_hashes,sources=_exclusions();available=[]
    for index,path in enumerate(dataset.image_paths):
        digest=_sha(path).lower()
        if path.name not in excluded_ids and digest not in excluded_hashes:available.append({"dataset_index":index,"identifier":path.relative_to(dataset.image_directory).as_posix(),"sha256":digest})
    random.Random(int(config["seed"])).shuffle(available);counts=config["population_counts"];needed=sum(map(int,counts.values()))
    if len(available)<needed:raise RuntimeError(f"insufficient unseen real-COCO population: {len(available)} available, {needed} required")
    selected=available[:needed];a=int(counts["decoder_development"]);b=int(counts["selection_validation"])
    split={"decoder_development":selected[:a],"selection_validation":selected[a:a+b],"locked_final":selected[a+b:needed]}
    return split,{"available_unseen":len(available),"excluded_identifier_count":len(excluded_ids),"excluded_sha256_count":len(excluded_hashes),"selected":needed,"identifier_overlap":0,"sha256_overlap":0,"splits_disjoint":True,"exclusion_report_paths":sources}
def _load_model(config):
    if _sha(config["selected_candidate_checkpoint"])!=config["selected_candidate_sha256"]:raise RuntimeError("selected step-450 candidate SHA-256 mismatch")
    verification,parent=verify_12bit_parent(config);checkpoint=torch.load(config["selected_candidate_checkpoint"],map_location="cpu",weights_only=False);model=StageDTagCapacityV1(parent);model.load_state_dict(checkpoint["model_state"],strict=True);model.eval();[x.requires_grad_(False) for x in model.parameters()]
    report=json.loads(Path(config["soft_decoder_report"]).read_text());params=report["selected_parameters"]
    expected={"field_top_k":4,"beam_width":256,"search_budget":4096,"temperatures":[1.2,1.5,1.5],"uncertain_confidence":0.2}
    if params!=expected:raise RuntimeError("frozen soft-decoder v2 parameters changed")
    return verification,model,SoftAuthenticatedFragmentDecoderV2(**params),params
def _materials(dataset,items,config,key,offset):
    result=[]
    for i,item in enumerate(items):
        image=load_stage_c_population(dataset,[item["identifier"]],config["preprocessing"],64);gen=torch.Generator().manual_seed(int(config["seed"])+offset+i);bits,meta=fresh_distributed_packet_batch(1,key,gen);packet=torch.cat((bits,bits.new_zeros((*bits.shape[:-1],24))),-1)
        result.append((item,image,bits,meta[0],packet))
    return result
def _issue(model,value):
    item,image,bits,metadata,packet=value
    with torch.no_grad():water=model(image,packet,.014,8)["watermarked_image"];logits=blind_regional_logits(model,water)
    return {"item":item,"original":image[0],"issued64":water[0],"issued256":F.interpolate(water,(256,256),mode="bilinear",align_corners=False,antialias=True)[0],"bits":bits[0],"metadata":metadata,"clean_logits":logits[0]}
def _limits(config):
    x=config["search"];return CropSearchLimits(tuple(x["scales"]),tuple(x["aspect_ratios"]),tuple(tuple(v) for v in x["offsets"]),int(x["regional_top_k"]),int(x["token_beam_limit"]),int(x["hmac_attempt_limit"]),int(x["total_search_budget"]),int(x["hypothesis_decode_limit"]),float(x["runtime_limit_ms"]))
def _crop_box(area,mode,sample,seed):
    ratio=1. if "varied_aspect" not in mode and "random" not in mode else (.75,1.,4/3)[(sample+round(area*100)+seed)%3]
    h=min(256,max(1,round(256*math.sqrt(area/ratio))));w=min(256,max(1,round(256*math.sqrt(area*ratio))));rng=random.Random(seed+sample*1009+round(area*100));my=256-h;mx=256-w
    if mode.startswith("grid_aligned"):y=(rng.randrange(my//64+1)*64 if my else 0);x=(rng.randrange(mx//64+1)*64 if mx else 0)
    elif mode.startswith("boundary"):y=min(my,32);x=min(mx,32)
    elif mode.startswith("non_grid"):y=min(my,17);x=min(mx,29)
    else:y=rng.randrange(my+1) if my else 0;x=rng.randrange(mx+1) if mx else 0
    return x,y,w,h
def _survival(box):
    x,y,w,h=box;result=[]
    for r in range(16):
        ry=r//4*64;rx=r%4*64;ix=max(0,min(x+w,rx+64)-max(x,rx));iy=max(0,min(y+h,ry+64)-max(y,ry));result.append(ix*iy/4096)
    return result
def _represent(issued,box,mode):
    x,y,w,h=box;crop=issued[:,y:y+h,x:x+w]
    if mode.endswith("resize"):return F.interpolate(crop[None],(256,256),mode="bilinear",align_corners=False,antialias=True)[0]
    canvas=issued.new_full((3,256,256),.5);top=(256-h)//2;left=(256-w)//2;canvas[:,top:top+h,left:left+w]=crop;return canvas
def _compound(image,kind,donor):
    if kind.startswith("jpeg"):
        buffer=io.BytesIO();TF.to_pil_image(image).save(buffer,format="JPEG",quality=int(kind[-2:]));buffer.seek(0);return TF.pil_to_tensor(Image.open(buffer).convert("RGB")).float()/255
    if kind=="mild_resize":return F.interpolate(F.interpolate(image[None],(224,224),mode="bilinear",align_corners=False,antialias=True),(256,256),mode="bilinear",align_corners=False,antialias=True)[0]
    if kind=="mild_colour":return TF.adjust_saturation(TF.adjust_brightness(image,1.04),.96)
    changed=image.clone();y=x=64
    if kind=="regional_overlay":changed[:,y:y+64,x:x+64]=.6*donor[:,y:y+64,x:x+64]+.4*changed[:,y:y+64,x:x+64]
    else:changed[:,y:y+64,x:x+64]=donor[:,y:y+64,x:x+64]
    return changed.clamp(0,1)

def _field_metrics(logits,bits):
    hard=logits.ge(0);truth=bits.bool();return {"regional_index_accuracy":float(hard[...,:4].eq(truth[...,:4]).float().mean()),"rs_symbol_accuracy":float(hard[...,4:12].eq(truth[...,4:12]).float().mean()),"authentication_share_accuracy":float(hard[...,12:20].eq(truth[...,12:20]).float().mean())}
def _coverage(logits,bits,decoder):
    obs=calibrated_observations(logits.reshape(16,20),decoder.field_top_k,decoder.temperatures);truth=bits.reshape(16,20).int();covered=[]
    for i,o in enumerate(obs):
        values=[int("".join(map(str,truth[i,a:b].tolist())),2) for a,b in ((0,4),(4,12),(12,20))]
        covered.append(all(value in [x.value for x in candidates] for value,candidates in zip(values,(o.indices,o.symbols,o.shares))))
    return mean(covered)
def _missing(states,fractions):
    expected={i for i,x in enumerate(fractions) if x==0};predicted={i for i,x in states.items() if x=="missing"};tp=len(expected&predicted);precision=tp/len(predicted) if predicted else (1. if not expected else 0.);recall=tp/len(expected) if expected else 1.;f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    return precision,recall,f1
def _crop_case(questioned,fractions,issue,model,decoder,key,sources,limits,label,area,mode):
    decision=blind_crop_search(questioned,model,decoder,key,sources,limits);logits=decision.pop("_selected_logits",torch.zeros(4,4,20));fields=_field_metrics(logits,issue["bits"]);coverage=_coverage(logits,issue["bits"],decoder);exact=logits.ge(0).eq(issue["bits"].bool()).all(-1).flatten();decodable=sum(f>=.5 and bool(ok) for f,ok in zip(fractions,exact));mp,mr,mf=_missing(decision["states"],fractions)
    accepted=decision["status"]=="authenticated" and decision.get("token")==issue["metadata"].token and decision.get("source_id")==issue["metadata"].source_id
    return {"label":label,"retained_area":area,"mode":mode,"accepted":accepted,"status":decision["status"],**fields,"candidate_coverage":coverage,"token_reconstruction":decision.get("token")==issue["metadata"].token,"authenticator_reconstruction":accepted,"false_acceptance":decision["status"]=="authenticated" and not accepted,"false_rejection":not accepted,"fully_surviving_regions":sum(x==1 for x in fractions),"partially_surviving_regions":sum(0<x<1 for x in fractions),"genuinely_decodable_symbols":decodable,"eligibility":">=12" if decodable>=12 else "8-11" if decodable>=8 else "<8","missing_precision":mp,"missing_recall":mr,"missing_f1":mf,"valid_rate":mean(x=="valid" for x in decision["states"].values()),"uncertain_rate":mean(x=="uncertain" for x in decision["states"].values()),"candidate_count":decision.get("candidate_count",0),"hypothesis_count":decision.get("hypothesis_count",0),"runtime_ms":decision["runtime_ms"],"search_budget_exhausted":decision.get("search_budget_exhausted",False),"regional_map":[decision["states"][i] for i in range(16)],"oracle_metadata_use":"post-decision metrics only"}
def _clean(issue,decoder,key,sources):
    started=time.perf_counter();result=decoder.decode(issue["clean_logits"].reshape(16,20),key,sources);elapsed=(time.perf_counter()-started)*1000;accepted=result["status"]=="authenticated" and result["token"]==issue["metadata"].token and result["source_id"]==issue["metadata"].source_id
    return {**_field_metrics(issue["clean_logits"],issue["bits"]),"accepted":accepted,"status":result["status"],"runtime_ms":elapsed,"candidate_count":result["candidate_count"],"search_budget_exhausted":result["search_budget_exhausted"]}
def _force_indices(logits,indices):
    x=logits[indices].clone()
    for row,index in enumerate(indices):
        bits=torch.tensor([(index>>shift)&1 for shift in (3,2,1,0)],dtype=torch.bool);x[row,:4]=torch.where(bits,torch.tensor(20.),torch.tensor(-20.))
    return x
def _grid_controls(issues,decoder,key,sources,seed):
    result={}
    for surviving in (16,14,12,10,8,7):
        for pattern in ("random","contiguous"):
            rows=[]
            for sample,issue in enumerate(issues):
                if pattern=="contiguous":start=(sample*3)%(17-surviving);indices=list(range(start,start+surviving))
                else:indices=sorted(random.Random(seed+sample*97+surviving).sample(range(16),surviving))
                decoded=decoder.decode(_force_indices(issue["clean_logits"].reshape(16,20),indices),key,sources);rows.append(decoded["status"]=="authenticated" and decoded["token"]==issue["metadata"].token)
            result[f"{surviving}_{pattern}"]={"acceptance":mean(rows),"samples":len(rows),"known_grid_identities":True}
    return result
def _aggregate(cases):
    if not cases:return {}
    times=sorted(x["runtime_ms"] for x in cases)
    return {"cases":len(cases),"authenticated_acceptance":mean(x["accepted"] for x in cases),"false_rejection":mean(x["false_rejection"] for x in cases),"false_acceptance":mean(x["false_acceptance"] for x in cases),"regional_index_accuracy":mean(x["regional_index_accuracy"] for x in cases),"rs_symbol_accuracy":mean(x["rs_symbol_accuracy"] for x in cases),"authentication_share_accuracy":mean(x["authentication_share_accuracy"] for x in cases),"candidate_coverage":mean(x["candidate_coverage"] for x in cases),"token_reconstruction_rate":mean(x["token_reconstruction"] for x in cases),"authenticator_reconstruction_rate":mean(x["authenticator_reconstruction"] for x in cases),"missing_precision":mean(x["missing_precision"] for x in cases),"missing_recall":mean(x["missing_recall"] for x in cases),"missing_f1":mean(x["missing_f1"] for x in cases),"valid_rate":mean(x["valid_rate"] for x in cases),"uncertain_rate":mean(x["uncertain_rate"] for x in cases),"average_runtime_ms":mean(times),"p95_runtime_ms":times[min(len(times)-1,math.ceil(.95*len(times))-1)],"average_candidate_count":mean(x["candidate_count"] for x in cases),"worst_candidate_count":max(x["candidate_count"] for x in cases),"search_budget_exhaustion_rate":mean(x["search_budget_exhausted"] for x in cases)}
def _evaluate_crops(issues,model,decoder,key,sources,config,split_name):
    output=Path(config["output_directory"])/"shards"/split_name;output.mkdir(parents=True,exist_ok=True);limits=_limits(config);all_cases=[];manifest_hash=hashlib.sha256(json.dumps({"split":[x["item"] for x in issues],"search":config["search"],"areas":config["retained_areas"],"modes":config["crop_modes"],"candidate":config["selected_candidate_sha256"],"allocation_version":config["allocation_version"]},sort_keys=True).encode()).hexdigest()
    for sample,issue in enumerate(issues):
        path=output/f"sample_{sample:03d}.json"
        if path.exists():
            payload=json.loads(path.read_text())
            if payload["manifest_hash"]!=manifest_hash:raise RuntimeError("crop shard manifest-hash verification failed")
            all_cases.extend(payload["cases"]);continue
        cases=[];donor=issues[(sample+1)%len(issues)]["issued256"];areas=config["retained_areas"];modes=config["crop_modes"]
        assignments=[(float(areas[sample%len(areas)]),modes[sample%len(modes)])]
        if sample<2:assignments.append((float(areas[(sample+3)%len(areas)]),modes[8+sample]))
        for area,mode in assignments:
            box=_crop_box(area,mode,sample,int(config["seed"]));questioned=_represent(issue["issued256"],box,mode);cases.append(_crop_case(questioned,_survival(box),issue,model,decoder,key,sources,limits,"crop_only",area,mode))
        if sample<len(config["compound_conditions"]):
            kind=config["compound_conditions"][sample];area=float(config["compound_area"]);mode="random_resize";box=_crop_box(area,mode,sample,int(config["seed"])+91);base=_represent(issue["issued256"],box,mode);cases.append(_crop_case(_compound(base,kind,donor),_survival(box),issue,model,decoder,key,sources,limits,kind,area,mode))
        path.write_text(json.dumps({"manifest_hash":manifest_hash,"sample":sample,"identifier":issue["item"]["identifier"],"cases":cases,"contains_expected_payload":False,"contains_secret":False})+"\n");all_cases.extend(cases)
    return all_cases,{"manifest_hash":manifest_hash,"shards":len(issues),"resumable":True}
def _summaries(cases):
    crop=[x for x in cases if x["label"]=="crop_only"];area={str(v):_aggregate([x for x in crop if x["retained_area"]==v]) for v in sorted({x["retained_area"] for x in crop})};symbols={g:_aggregate([x for x in crop if x["eligibility"]==g]) for g in (">=12","8-11","<8")};compound={v:_aggregate([x for x in cases if x["label"]==v]) for v in sorted({x["label"] for x in cases if x["label"]!="crop_only"})}
    return {"crop_only_by_retained_area":area,"crop_only_by_decodable_symbols":symbols,"compound":compound,"overall_crop_only":_aggregate(crop)}

def _negative_controls(issues,model,decoder,key,sources):
    controls={k:[] for k in ("unwatermarked","random_logits","wrong_key","mixed_identities","duplicate_indices","insufficient")}
    generator=torch.Generator().manual_seed(991)
    for i,issue in enumerate(issues):
        original=blind_regional_logits(model,issue["original"][None])[0].reshape(16,20);mixed=torch.cat((issue["clean_logits"].reshape(16,20)[:8],issues[(i+1)%len(issues)]["clean_logits"].reshape(16,20)[8:]));duplicate=torch.cat((issue["clean_logits"].reshape(16,20)[:15],issue["clean_logits"].reshape(16,20)[:1]))
        values={"unwatermarked":decoder.decode(original,key,sources),"random_logits":decoder.decode(torch.randn(16,20,generator=generator),key,sources),"wrong_key":decoder.decode(issue["clean_logits"].reshape(16,20),bytes(32),sources),"mixed_identities":decoder.decode(mixed,key,sources),"duplicate_indices":decoder.decode(duplicate,key,sources),"insufficient":decoder.decode(issue["clean_logits"].reshape(16,20)[:7],key,sources)}
        for name,value in values.items():controls[name].append(value["status"]=="authenticated")
    return {name:{"acceptance":mean(values),"false_accepts":sum(values),"samples":len(values)} for name,values in controls.items()}
def _figures(report,directory):
    import matplotlib.pyplot as plt
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);area=report["locked_final"]["crop_summary"]["crop_only_by_retained_area"];xs=sorted(map(float,area))
    for title,key in (("acceptance","authenticated_acceptance"),("false_rejection","false_rejection"),("missing_f1","missing_f1")):
        plt.figure();plt.plot(xs,[area[str(x)][key] for x in xs],marker="o");plt.xlabel("retained image area");plt.ylabel(key);plt.ylim(-.02,1.02);plt.grid(True);plt.savefig(directory/f"{title}_versus_retained_area.png",dpi=160,bbox_inches="tight");plt.close()
    cases=report["locked_final"]["crop_cases"];plt.figure();plt.scatter([x["hypothesis_count"] for x in cases],[x["runtime_ms"] for x in cases],s=8);plt.xlabel("search hypotheses");plt.ylabel("runtime ms");plt.savefig(directory/"runtime_versus_search_size.png",dpi=160,bbox_inches="tight");plt.close()
    symbols=report["locked_final"]["crop_summary"]["crop_only_by_decodable_symbols"];names=[">=12","8-11","<8"];plt.figure();plt.bar(names,[symbols[x].get("authenticated_acceptance",0) for x in names]);plt.ylabel("authenticated acceptance");plt.savefig(directory/"acceptance_versus_surviving_symbols.png",dpi=160,bbox_inches="tight");plt.close()
    maps=[x for x in cases if x["label"]=="crop_only"][:4];coding={"missing":0,"uncertain":1,"valid":2,"manipulated":3};fig,axes=plt.subplots(1,len(maps),figsize=(3*len(maps),3))
    for ax,row in zip(np.atleast_1d(axes),maps):ax.imshow(np.asarray([coding[x] for x in row["regional_map"]]).reshape(4,4),vmin=0,vmax=3);ax.set_title(f"area={row['retained_area']}");ax.axis("off")
    plt.savefig(directory/"representative_region_state_maps.png",dpi=160,bbox_inches="tight");plt.close()
def _clean_summary(rows):
    times=sorted(x["runtime_ms"] for x in rows);return {"acceptance":mean(x["accepted"] for x in rows),"samples":len(rows),"regional_index_accuracy":mean(x["regional_index_accuracy"] for x in rows),"rs_symbol_accuracy":mean(x["rs_symbol_accuracy"] for x in rows),"authentication_share_accuracy":mean(x["authentication_share_accuracy"] for x in rows),"average_runtime_ms":mean(times),"p95_runtime_ms":times[-1]}
def run_experiment(config):
    output=Path(config["output_directory"]);output.mkdir(parents=True,exist_ok=True);verification,model,decoder,decoder_params=_load_model(config);dataset=CocoImageDataset(config["data_root"]);split,population=_population(dataset,config);key=deterministic_scientific_key(int(config["seed"]))
    materials={name:_materials(dataset,items,config,key,10000*i) for i,(name,items) in enumerate(split.items(),1)};issues={name:[_issue(model,x) for x in values] for name,values in materials.items()};sources={name:[x["metadata"].source_id for x in values] for name,values in issues.items()}
    development_clean=[_clean(x,decoder,key,sources["decoder_development"]) for x in issues["decoder_development"]];selection_clean=[_clean(x,decoder,key,sources["selection_validation"]) for x in issues["selection_validation"]]
    selection_cases,selection_shards=_evaluate_crops(issues["selection_validation"],model,decoder,key,sources["selection_validation"],config,"selection_validation");frozen={"search":config["search"],"decoder_parameters":decoder_params,"selection_clean":_clean_summary(selection_clean),"selection_crop":_summaries(selection_cases)};(output/"frozen_selection.json").write_text(json.dumps(frozen,indent=2)+"\n")
    marker=output/"locked_final_complete.json"
    if marker.exists():raise RuntimeError("locked-final population already evaluated; use preserved report and shards")
    (output/"locked_final_started.json").write_text(json.dumps({"selection_frozen":True,"locked_final_evaluations":1})+"\n")
    final=issues["locked_final"];final_sources=sources["locked_final"];clean=[_clean(x,decoder,key,final_sources) for x in final];grid=_grid_controls(final,decoder,key,final_sources,int(config["seed"]));cases,shards=_evaluate_crops(final,model,decoder,key,final_sources,config,"locked_final");summary=_summaries(cases);negative=_negative_controls(final,model,decoder,key,final_sources)
    shuffled=[]
    for i,issue in enumerate(final):
        order=torch.randperm(16,generator=torch.Generator().manual_seed(int(config["seed"])+i));x=decoder.decode(issue["clean_logits"].reshape(16,20)[order],key,final_sources);shuffled.append(x["status"]=="authenticated" and x["token"]==issue["metadata"].token)
    clean_summary=_clean_summary(clean);eligible=summary["crop_only_by_decodable_symbols"][">=12"];g=config["gates"];grid12=mean(grid[f"12_{p}"]["acceptance"] for p in ("random","contiguous"));false_accepts=sum(x["false_accepts"] for x in negative.values());insufficient_rejection=1-mean((negative["insufficient"]["acceptance"],negative["random_logits"]["acceptance"]))
    gates={"clean_authenticated_acceptance":clean_summary["acceptance"]>=g["clean_authenticated_acceptance"],"grid_aware_12_of_16_acceptance":grid12>=g["grid_aware_12_of_16_acceptance"],"eligible_blind_crop_acceptance":eligible.get("cases",0)>0 and eligible.get("authenticated_acceptance",0)>=g["eligible_blind_crop_acceptance"],"shuffled_matches":abs(mean(shuffled)-clean_summary["acceptance"])<=1/len(clean),"zero_false_accepts":false_accepts==0,"insufficient_evidence_rejection":insufficient_rejection>=g["insufficient_evidence_rejection"],"missing_region_f1":summary["overall_crop_only"]["missing_f1"]>=g["missing_region_f1"],"search_budget_exhaustion":summary["overall_crop_only"]["search_budget_exhaustion_rate"]<=g["search_budget_exhaustion"],"p95_runtime":summary["overall_crop_only"]["p95_runtime_ms"]<g["p95_runtime_ms"]}
    protocol_passed=gates["grid_aware_12_of_16_acceptance"] and gates["zero_false_accepts"] and gates["insufficient_evidence_rejection"];blind_passed=all(gates.values())
    report={"schema_version":"blind_crop_threshold_recovery_v1.0","scope":{"protocol":"16 indexed regional code symbols jointly authenticated through threshold reconstruction of a distributed keyed authenticator","not_individual_macs":True,"blind_input":"cropped questioned image only","manipulation_classification":"provisional; no selected regional digest"},"parent_verification":verification,"candidate":{"label":"selected_step450_non_promoted_diagnostic_candidate","sha256":config["selected_candidate_sha256"],"modified":False},"soft_decoder":{"version":"v2 frozen","parameters":decoder_params,"larger_beam_used":False},"population":{"manifest":split,"audit":population,"locked_final_evaluations":1},"predeclared_search":config["search"],"predeclared_gates":g,"development_clean":_clean_summary(development_clean),"selection_validation":{"frozen_parameters":frozen,"shards":selection_shards},"locked_final":{"clean":clean_summary,"shuffled_acceptance":mean(shuffled),"grid_aware_erasure_control":grid,"crop_summary":summary,"compound_separate":summary["compound"],"negative_controls":negative,"crop_cases":cases,"shards":shards},"gate_results":gates,"protocol_erasure_feasibility_passed":protocol_passed,"blind_crop_feasibility_passed":blind_passed,"neural_stage_passed":False,"stage_e_permitted":False,"scientific_status":"passed_blind_crop_threshold_recovery_feasibility" if blind_passed else "blocked_by_blind_crop_threshold_recovery","no_secret_expected_payload_token_or_crop_coordinates_serialized":True,"novelty_claimed":False}
    text=json.dumps(report,indent=2)+"\n";(output/"report.json").write_text(text);committed=Path(config["committed_report"]);committed.parent.mkdir(parents=True,exist_ok=True);committed.write_text(text);marker.write_text(json.dumps({"complete":True,"locked_final_evaluations":1,"scientific_status":report["scientific_status"]})+"\n");_figures(report,config["figure_directory"]);return report
