"""Real-COCO regional digest calibration, locked testing and empirical collision controls."""
from __future__ import annotations
import hashlib,io,json,math,random,time
from pathlib import Path
from statistics import mean
import numpy as np
import torch
from scipy.stats import rankdata
from PIL import Image,ImageFilter
from torchvision.transforms import functional as TF
from kfrag.data import CocoImageDataset
from kfrag.protocols.regional_perceptual_digest_v1 import *

BENIGN=("jpeg_q90","jpeg_q75","jpeg_q60","resize_075","resize_050","brightness_095","brightness_105","contrast_090","contrast_110","colour_mild","blur_05","blur_10","watermark_proxy_005","watermark_proxy_010")
MALICIOUS=("replacement_full","splice_025","splice_050","splice_075","overlay_025","overlay_050","overlay_075","same_image_relocation","mixed_source_collage")
def deterministic_three_split(identifiers,counts,seed):
    order=list(identifiers);random.Random(seed).shuffle(order);a,b,c=counts
    if a+b+c>len(order):raise ValueError("insufficient real images for three disjoint populations")
    return {"calibration":order[:a],"selection_validation":order[a:a+b],"locked_final_test":order[a+b:a+b+c]}
def _jpeg(image,quality):
    buffer=io.BytesIO();TF.to_pil_image(image).save(buffer,format="JPEG",quality=quality);buffer.seek(0);return TF.pil_to_tensor(Image.open(buffer).convert("RGB")).float()/255
def _restore_resize(image,factor):
    size=image.shape[-1];small=TF.resize(image,[round(size*factor)]*2,antialias=True);return TF.resize(small,[size,size],antialias=True)
def _watermark_proxy(image,amplitude):
    size=image.shape[-1];axis=torch.arange(size);pattern=torch.sin(axis[:,None]*.71+axis[None,:]*1.13);return (image+amplitude*pattern.unsqueeze(0)).clamp(0,1)
def benign_transform(image,name):
    if name.startswith("jpeg_q"):return _jpeg(image,int(name[6:]))
    if name=="resize_075":return _restore_resize(image,.75)
    if name=="resize_050":return _restore_resize(image,.5)
    if name.startswith("brightness_"):return TF.adjust_brightness(image,int(name[-3:])/100)
    if name.startswith("contrast_"):return TF.adjust_contrast(image,int(name[-3:])/100)
    if name=="colour_mild":return TF.adjust_saturation(TF.adjust_hue(image,.02),.95)
    if name.startswith("blur_"):return TF.pil_to_tensor(TF.to_pil_image(image).filter(ImageFilter.GaussianBlur(float(name[-2:])/10))).float()/255
    if name=="watermark_proxy_005":return _watermark_proxy(image,.005)
    if name=="watermark_proxy_010":return _watermark_proxy(image,.01)
    raise ValueError(name)
def _bounds(index,size=256):c=size//4;y=index//4*c;x=index%4*c;return y,x,c
def malicious_transform(image,donor,name,target=5):
    changed=image.clone();mask=np.zeros(16,dtype=bool);y,x,c=_bounds(target,image.shape[-1]);mask[target]=True
    if name=="replacement_full":changed[:,y:y+c,x:x+c]=donor[:,y:y+c,x:x+c]
    elif name.startswith("splice_") or name.startswith("overlay_"):
        fraction=int(name[-3:])/100;side=max(1,round(c*math.sqrt(fraction)));offset=(c-side)//2
        if name.startswith("splice_"):changed[:,y+offset:y+offset+side,x+offset:x+offset+side]=donor[:,y+offset:y+offset+side,x+offset:x+offset+side]
        else:changed[:,y+offset:y+offset+side,x+offset:x+offset+side]=(1-image[:,y:y+c,x:x+c].mean()).clamp(0,1)
    elif name=="same_image_relocation":
        sy,sx,_=_bounds((target+6)%16,image.shape[-1]);changed[:,y:y+c,x:x+c]=image[:,sy:sy+c,sx:sx+c]
    elif name=="mixed_source_collage":
        for index in (1,5,9,13):yy,xx,cc=_bounds(index,image.shape[-1]);changed[:,yy:yy+cc,xx:xx+cc]=donor[:,yy:yy+cc,xx:xx+cc];mask[index]=True
    else:raise ValueError(name)
    return changed,mask
def _distribution(values):
    array=np.asarray(values,float);return {"count":len(array),"mean":float(array.mean()),"std":float(array.std()),"minimum":float(array.min()),"q05":float(np.quantile(array,.05)),"median":float(np.quantile(array,.5)),"q95":float(np.quantile(array,.95)),"q99":float(np.quantile(array,.99)),"maximum":float(array.max())}
def _auc(negative,positive):
    combined=np.asarray(list(negative)+list(positive));labels=np.asarray([0]*len(negative)+[1]*len(positive));ranks=rankdata(combined,method="average");return float((ranks[labels==1].sum()-len(positive)*(len(positive)+1)/2)/(len(positive)*len(negative)))
def _metrics(states,truth):
    tp=sum(s=="manipulated" and t for s,t in zip(states,truth));fp=sum(s=="manipulated" and not t for s,t in zip(states,truth));fn=sum(s!="manipulated" and t for s,t in zip(states,truth));precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.;return {"manipulated_precision":precision,"manipulated_recall":recall,"manipulated_f1":2*precision*recall/(precision+recall) if precision+recall else 0.,"false_manipulation_rate":fp/max(1,sum(not x for x in truth)),"missed_manipulation_rate":fn/max(1,sum(truth)),"uncertain_rate":sum(x=="uncertain" for x in states)/len(states),"state_counts":{x:states.count(x) for x in ("valid","missing","manipulated","uncertain")}}
def _population_distances(images,digest,kind):
    rows={name:[] for name in (BENIGN if kind=="benign" else MALICIOUS)};truth={name:[] for name in rows};generation=[];verification=[]
    references=[]
    for image in images:
        start=time.perf_counter();references.append(digest.digest_image(image));generation.append((time.perf_counter()-start)*1000)
    for i,image in enumerate(images):
        donor=images[(i+1)%len(images)]
        for name in rows:
            transformed=benign_transform(image,name) if kind=="benign" else malicious_transform(image,donor,name,target=(i*7)%16)[0];altered=np.zeros(16,bool) if kind=="benign" else malicious_transform(image,donor,name,target=(i*7)%16)[1];start=time.perf_counter();observed=digest.digest_image(transformed);dist=[digest.distance(a,b) for a,b in zip(references[i],observed)];verification.append((time.perf_counter()-start)*1000);rows[name].extend(dist);truth[name].extend(altered.tolist())
    return rows,truth,references,{"digest_generation_mean_ms":mean(generation),"verification_mean_ms":mean(verification),"verification_p95_ms":float(np.quantile(verification,.95))}
def _select_threshold(calibration_benign,validation_benign,validation_malicious,validation_truth):
    cal=[x for values in calibration_benign.values() for x in values];benign=[x for values in validation_benign.values() for x in values];positive=[distance for name,values in validation_malicious.items() for distance,changed in zip(values,validation_truth[name]) if changed];best=None
    valid_max=float(np.quantile(cal,.95));candidates=sorted(set(float(np.quantile(cal,q)) for q in (.98,.99,.995,.999))|set(float(np.quantile(positive,q)) for q in (.05,.10,.20)))
    for threshold in candidates:
        threshold=max(threshold,valid_max+1e-9);fp=sum(x>=threshold for x in benign);tp=sum(x>=threshold for x in positive);precision=tp/max(1,tp+fp);recall=tp/max(1,len(positive));f1=2*precision*recall/max(1e-12,precision+recall);fmr=fp/len(benign);score=(fmr<=.05,f1,-fmr,-threshold)
        if best is None or score>best[0]:best=(score,{"valid_max":valid_max,"manipulated_min":threshold,"validation_false_manipulation_rate":fmr,"validation_manipulated_precision":precision,"validation_manipulated_recall":recall,"validation_manipulated_f1":f1})
    return best[1]
def _evaluate(rows,truth,threshold,benign_pool):
    result={}
    for name,values in rows.items():
        states=list(classify_distances(values,[True]*len(values),True,threshold["valid_max"],threshold["manipulated_min"]));metrics=_metrics(states,truth[name]);metrics.update({"distance_distribution":_distribution(values),"roc_auc_vs_pooled_benign":None if not any(truth[name]) else _auc(benign_pool,[v for v,t in zip(values,truth[name]) if t])});result[name]=metrics
    return result
def run_real_image_digest(config):
    root=Path(config["data_root"]);dataset=CocoImageDataset(root);ids=[dataset[i]["relative_path"] for i in range(len(dataset))];split=deterministic_three_split(ids,(int(config["calibration_images"]),int(config["selection_images"]),int(config["final_test_images"])),int(config["seed"]));lookup={dataset[i]["relative_path"]:dataset[i]["image"] for i in range(len(dataset))};pop={name:[lookup[x] for x in values] for name,values in split.items()};key=hashlib.sha256(f"regional-digest:{config['seed']}".encode()).digest();report={"schema_version":"real_image_regional_digest_v1.0","scope":"real COCO perceptual-digest feasibility; not an image-attack benchmark","region_correspondence_assumed":True,"crop_synchronization_validated":False,"global_identity_authentication":"explicit trusted prerequisite simulated independently of digest matching","population_manifest":{"identifiers":split,"sha256":{identifier:hashlib.sha256((root/identifier).read_bytes()).hexdigest() for identifier in ids},"disjoint":True,"locked_final_evaluations":1},"registry_definition":"HMAC-SHA256 binds image identifier, protocol version, region index, digest type, digest version and digest value; runtime key is not serialized","digests":{},"neural_stage_passed":False,"stage_e_permitted":False}
    for digest in digest_candidates():
        cal_b,_,_,_= _population_distances(pop["calibration"],digest,"benign");val_b,_,_,_= _population_distances(pop["selection_validation"],digest,"benign");val_m,val_truth,_,_= _population_distances(pop["selection_validation"],digest,"malicious");threshold=_select_threshold(cal_b,val_b,val_m,val_truth)
        final_b,final_b_truth,references,runtime=_population_distances(pop["locked_final_test"],digest,"benign");final_m,final_m_truth,_,_= _population_distances(pop["locked_final_test"],digest,"malicious");benign_pool=[x for values in final_b.values() for x in values];benign_eval=_evaluate(final_b,final_b_truth,threshold,benign_pool);malicious_eval=_evaluate(final_m,final_m_truth,threshold,benign_pool)
        clean_repeat=[digest.distance(value,value) for image in references for value in image];records,_=create_registry(split["locked_final_test"][0],pop["locked_final_test"][0],digest,key);registry_ok=authenticate_registry(records,key);aggregate_b=[m["false_manipulation_rate"] for m in benign_eval.values()];aggregate_m=[m["missed_manipulation_rate"] for m in malicious_eval.values()];all_states=[];all_truth=[]
        for name,values in final_m.items():all_states.extend(classify_distances(values,[True]*len(values),True,threshold["valid_max"],threshold["manipulated_min"]));all_truth.extend(final_m_truth[name])
        aggregate=_metrics(all_states,all_truth);qualifies=mean(aggregate_b)<=float(config["qualification"]["maximum_benign_false_manipulation"]) and mean(aggregate_m)<=float(config["qualification"]["maximum_malicious_missed_manipulation"]) and aggregate["manipulated_f1"]>=float(config["qualification"]["minimum_f1"])
        report["digests"][digest.name]={"threshold_selection":threshold,"locked_benign":benign_eval,"locked_content_changing":malicious_eval,"aggregate":{"mean_benign_false_manipulation_rate":mean(aggregate_b),"mean_content_change_missed_manipulation_rate":mean(aggregate_m),**aggregate},"controls":{"clean_repeatability":_distribution(clean_repeat),"unmodified_issued_image":_distribution(clean_repeat)},"registry":{"authenticated":registry_ok,"storage_bits_per_region":digest.storage_bits_per_region,"digest_storage_bytes_per_image":digest.storage_bits_per_region*16//8,"record_authentication_bytes_per_image":32*16,"total_registry_bytes_per_image":digest.storage_bits_per_region*2+32*16},"runtime":runtime,"qualifies":qualifies}
    # Empirical collision controls use locked images only and never affect selection.
    stats=LowFrequencyStatistics();stats_refs=[stats.digest_image(x) for x in pop["locked_final_test"]]
    for digest in digest_candidates():
        refs=[digest.digest_image(x) for x in pop["locked_final_test"]];unrelated=[];similar=[]
        for i in range(len(refs)):
            for region in range(16):
                candidates=[(digest.distance(refs[i][region],refs[j][r]),j,r) for j in range(len(refs)) if j!=i for r in range(16)];unrelated.append(min(x[0] for x in candidates));j,r=min(((stats.distance(stats_refs[i][region],stats_refs[j][rr]),j,rr) for j in range(len(refs)) if j!=i for rr in range(16)))[1:];similar.append(digest.distance(refs[i][region],refs[j][r]))
        same=[]
        for i in range(len(refs)):
            for region in range(16):same.append(digest.distance(refs[i][region],refs[i][(region+5)%16]))
        threshold=report["digests"][digest.name]["threshold_selection"]["manipulated_min"];report["digests"][digest.name]["collision_controls"]={"nearest_unrelated_region":_distribution(unrelated),"nearest_unrelated_below_manipulated_threshold":mean(x<threshold for x in unrelated),"visually_similar_unrelated_region":_distribution(similar),"visually_similar_below_manipulated_threshold":mean(x<threshold for x in similar),"same_source_region_replacement":_distribution(same),"same_source_below_manipulated_threshold":mean(x<threshold for x in same),"interpretation":"empirical controls only; not adversarial collision-security proof"}
    qualified=[name for name,value in report["digests"].items() if value["qualifies"]];report["recommendation"]={"digest":min(qualified,key=lambda n:report["digests"][n]["registry"]["total_registry_bytes_per_image"]) if qualified else None,"qualified_candidates":qualified,"criteria":config["qualification"],"status":"conditional_real_image_digest_candidate" if qualified else "no_digest_achieved_locked_tradeoff"};report["scientific_status"]="passed_real_image_digest_feasibility" if qualified else "blocked_by_real_image_digest_tradeoff";(Path(config["output_directory"])/"report.json").parent.mkdir(parents=True,exist_ok=True);(Path(config["output_directory"])/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
