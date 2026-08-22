"""Frozen-threshold, resumable large-scale real-COCO digest reproduction."""
from __future__ import annotations
import hashlib,io,json,math,random,time,tracemalloc
from pathlib import Path
from statistics import mean
import numpy as np
import torch
from PIL import Image,ImageFilter,features as pil_features
from scipy.stats import rankdata
from torchvision.transforms import functional as TF
from kfrag.data import CocoImageDataset
from kfrag.protocols.regional_perceptual_digest_v1 import DCTPerceptualHash,CombinedDigest,classify_distances,regions,create_registry,authenticate_registry

SOURCE_COMMIT="1448a9a4e23f00b0855fa8061b9bec5d75e5d7ae"
IMPLEMENTATION_PATH=Path("kfrag/protocols/regional_perceptual_digest_v1.py")
STANDARD_BENIGN=("clean_repeat","jpeg_q95","jpeg_q85","jpeg_q75","jpeg_q60","webp_q95","webp_q80","webp_q60","resize_075","resize_050","blur_05","blur_10","brightness_095","brightness_105","contrast_090","contrast_110","colour_mild")
EXTREME_BENIGN=("jpeg_q40","resize_025")
MALICIOUS=("replacement_full","splice_010","splice_025","splice_050","splice_075","overlay_010","overlay_025","overlay_050","overlay_075","occlusion_010","occlusion_025","occlusion_050","occlusion_075","same_image_relocation","mixed_source_collage","visually_similar_replacement","same_source_replacement")

MINIMUM_REPRODUCTION_IMAGES = 1000


class DataPopulationError(RuntimeError):
    """Raised before scientific calculations when unseen COCO data are insufficient."""

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()
def _json_default(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
def load_frozen_thresholds(path,prior_report):
    frozen=json.loads(Path(path).read_text());prior=json.loads(Path(prior_report).read_text());checks={"source_commit":frozen["source_commit"]==SOURCE_COMMIT,"implementation_hash":_sha(IMPLEMENTATION_PATH)==frozen["implementation_sha256"].upper()}
    for name in ("dct_phash","combined_digest"):
        checks[name]=frozen[name]["valid_max"]==prior["digests"][name]["threshold_selection"]["valid_max"] and frozen[name]["manipulated_min"]==prior["digests"][name]["threshold_selection"]["manipulated_min"]
    if not all(checks.values()):raise RuntimeError(f"frozen digest prerequisite failed: {checks}")
    return frozen,checks
def select_unseen_population(dataset,prior_report,count,seed):
    if count < MINIMUM_REPRODUCTION_IMAGES:raise DataPopulationError(f"configured reproduction population {count} is below the mandatory minimum of {MINIMUM_REPRODUCTION_IMAGES} unseen COCO images")
    prior=json.loads(Path(prior_report).read_text())["population_manifest"];prior_ids={Path(x).name for values in prior["identifiers"].values() for x in values};prior_hashes={x.lower() for x in prior["sha256"].values()};order=list(range(len(dataset)));random.Random(seed).shuffle(order);unseen=[]
    for index in order:
        path=dataset.image_paths[index];identifier=path.relative_to(dataset.image_directory).as_posix();digest=_sha(path).lower()
        if path.name not in prior_ids and digest not in prior_hashes:unseen.append({"dataset_index":index,"identifier":identifier,"sha256":digest})
    if len(unseen)<count:raise DataPopulationError(f"only {len(unseen)} unseen COCO images are available after excluding {len(prior_ids)} prior identifiers and {len(prior_hashes)} prior SHA-256 hashes; {count} are required")
    selected=unseen[:count];selected_names={Path(x["identifier"]).name for x in selected};selected_hashes={x["sha256"] for x in selected}
    preflight={"available_unseen_coco_image_count":len(unseen),"excluded_prior_image_count":len(prior_ids),"excluded_prior_identifier_count":len(prior_ids),"excluded_prior_sha256_count":len(prior_hashes),"selected_reproduction_count":len(selected),"identifier_overlap":len(selected_names&prior_ids),"sha256_overlap":len(selected_hashes&prior_hashes)}
    preflight["selected_identifiers_and_sha256_disjoint"]=preflight["identifier_overlap"]==0 and preflight["sha256_overlap"]==0
    if not preflight["selected_identifiers_and_sha256_disjoint"]:raise DataPopulationError(f"selected reproduction population overlaps prior evidence: {preflight}")
    return selected,preflight
def _codec(image,kind,quality):
    buffer=io.BytesIO();TF.to_pil_image(image).save(buffer,format=kind,quality=quality);buffer.seek(0);return TF.pil_to_tensor(Image.open(buffer).convert("RGB")).float()/255
def _resize(image,factor):
    size=image.shape[-1];return TF.resize(TF.resize(image,[round(size*factor)]*2,antialias=True),[size,size],antialias=True)
def benign_transform(image,name):
    if name=="clean_repeat":return image.clone()
    if name.startswith("jpeg_q"):return _codec(image,"JPEG",int(name[6:]))
    if name.startswith("webp_q"):return _codec(image,"WEBP",int(name[6:]))
    if name.startswith("resize_"):return _resize(image,int(name[-3:])/100)
    if name=="blur_05" or name=="blur_10":return TF.pil_to_tensor(TF.to_pil_image(image).filter(ImageFilter.GaussianBlur(.5 if name.endswith("05") else 1.))).float()/255
    if name.startswith("brightness_"):return TF.adjust_brightness(image,int(name[-3:])/100)
    if name.startswith("contrast_"):return TF.adjust_contrast(image,int(name[-3:])/100)
    if name=="colour_mild":return TF.adjust_saturation(TF.adjust_hue(image,.02),.95)
    raise ValueError(name)
def _bounds(index,size=256):c=size//4;return index//4*c,index%4*c,c
def _paste_fraction(changed,source,target,fraction,mode):
    y,x,c=_bounds(target,changed.shape[-1]);side=max(1,round(c*math.sqrt(fraction)));o=(c-side)//2
    if mode=="splice":changed[:,y+o:y+o+side,x+o:x+o+side]=source[:,y+o:y+o+side,x+o:x+o+side]
    elif mode=="overlay":changed[:,y+o:y+o+side,x+o:x+o+side]=(.6*source[:,y+o:y+o+side,x+o:x+o+side]+.4*changed[:,y+o:y+o+side,x+o:x+o+side]).clamp(0,1)
    else:changed[:,y+o:y+o+side,x+o:x+o+side]=(1-changed[:,y:y+c,x:x+c].mean()).clamp(0,1)
def malicious_transform(image,donor,name,target,similar_region=None):
    changed=image.clone();truth=np.zeros(16,bool);y,x,c=_bounds(target,image.shape[-1]);truth[target]=True
    if name=="replacement_full":changed[:,y:y+c,x:x+c]=donor[:,y:y+c,x:x+c]
    elif name.startswith(("splice_","overlay_","occlusion_")):_paste_fraction(changed,donor,target,int(name[-3:])/100,name.split("_")[0])
    elif name in ("same_image_relocation","same_source_replacement"):
        source=(target+5)%16;sy,sx,_=_bounds(source,image.shape[-1]);changed[:,y:y+c,x:x+c]=image[:,sy:sy+c,sx:sx+c]
    elif name=="mixed_source_collage":
        truth[:]=False
        for index in (1,5,9,13):yy,xx,cc=_bounds(index,image.shape[-1]);changed[:,yy:yy+cc,xx:xx+cc]=donor[:,yy:yy+cc,xx:xx+cc];truth[index]=True
    elif name=="visually_similar_replacement":
        if similar_region is None:raise ValueError("similar donor region is required")
        donor_image,donor_region=similar_region;sy,sx,_=_bounds(donor_region,donor_image.shape[-1]);changed[:,y:y+c,x:x+c]=donor_image[:,sy:sy+c,sx:sx+c]
    else:raise ValueError(name)
    return changed,truth
def _colour_vectors(images):return np.stack([r.mean((1,2)).numpy() for image in images for r in regions(image)])
def _similar_map(images):
    from scipy.spatial import cKDTree
    vectors=_colour_vectors(images);tree=cKDTree(vectors);mapping={}
    for flat,vector in enumerate(vectors):
        image=flat//16;_,indices=tree.query(vector,k=min(len(vectors),40));choice=next(int(x) for x in np.atleast_1d(indices) if int(x)//16!=image);mapping[(image,flat%16)]=(choice//16,choice%16)
    return mapping
def _digest_values(digest,image):return digest.digest_image(image)
def _row_for_image(index,identifier,image,donor,similar,thresholds,webp,key):
    digests={"dct_phash":DCTPerceptualHash(),"combined_digest":CombinedDigest()};references={};generation={};rows={name:{} for name in digests}
    registry_authenticated={}
    for name,digest in digests.items():
        start=time.perf_counter();references[name]=_digest_values(digest,image);generation[name]=(time.perf_counter()-start)*1000;records,_=create_registry(identifier,image,digest,key);registry_authenticated[name]=authenticate_registry(records,key)
    benign=list(STANDARD_BENIGN)+list(EXTREME_BENIGN)
    if not webp:benign=[x for x in benign if not x.startswith("webp_")]
    transformed=[(name,benign_transform(image,name),np.zeros(16,bool)) for name in benign]
    target=(index*7)%16
    for name in MALICIOUS:
        similar_arg=(similar[0],similar[1]) if name=="visually_similar_replacement" else None;changed,truth=malicious_transform(image,donor,name,target,similar_arg);transformed.append((name,changed,truth))
    for transform,observed,truth in transformed:
        for name,digest in digests.items():
            start=time.perf_counter();values=_digest_values(digest,observed);runtime=(time.perf_counter()-start)*1000;distances=[digest.distance(a,b) for a,b in zip(references[name],values)];states=classify_distances(distances,[True]*16,True,thresholds[name]["valid_max"],thresholds[name]["manipulated_min"]);rows[name][transform]={"distances":distances,"truth":truth.tolist(),"states":states,"verification_ms":runtime}
    return {"identifier":identifier,"generation_ms":generation,"registry_authenticated":registry_authenticated,"digests":rows}
def write_population_preflight(config,preflight):
    output=Path(config["output_directory"]);output.mkdir(parents=True,exist_ok=True)
    path=output/"population_preflight.json";path.write_text(json.dumps(preflight,indent=2)+"\n")
    print("Stage-Digest reproduction population preflight:",json.dumps(preflight,sort_keys=True))
    return path
def _fingerprint(config,selected,frozen):return hashlib.sha256(json.dumps({"config":config,"selected":selected,"frozen":frozen},sort_keys=True).encode()).hexdigest()
def run_shards(config,dataset,selected,frozen):
    output=Path(config["output_directory"]);shards=output/"shards";shards.mkdir(parents=True,exist_ok=True);fingerprint=_fingerprint(config,selected,frozen);size=int(config["shard_size"]);webp=bool(pil_features.check("webp"));rows=[];resumed=0;peak_rss=0;key=hashlib.sha256(f"large-scale-registry:{config['seed']}".encode()).digest()
    try:
        import psutil;process=psutil.Process()
    except ImportError:process=None
    for start in range(0,len(selected),size):
        stop=min(len(selected),start+size);path=shards/f"shard_{start:05d}_{stop:05d}.json"
        if path.is_file():
            payload=json.loads(path.read_text())
            if payload.get("fingerprint")!=fingerprint:raise RuntimeError("existing shard manifest-hash verification failed")
            rows.extend(payload["rows"]);resumed+=1;continue
        images=[dataset[item["dataset_index"]]["image"] for item in selected[start:stop]];donors=[dataset[selected[(start+i+1)%len(selected)]["dataset_index"]]["image"] for i in range(len(images))]
        # Similar search is shard-local for transform generation; global exact collision search is separate.
        mapping=_similar_map(images+donors);shard_rows=[]
        for local,(item,image,donor) in enumerate(zip(selected[start:stop],images,donors)):
            donor_index,donor_region=mapping[(local,( (start+local)*7)%16)];similar_image=(images+donors)[donor_index];shard_rows.append(_row_for_image(start+local,item["identifier"],image,donor,(similar_image,donor_region),frozen,webp,key));peak_rss=max(peak_rss,process.memory_info().rss if process else 0)
        payload={"schema_version":"regional_digest_shard_v1","fingerprint":fingerprint,"start":start,"stop":stop,"rows":shard_rows};path.write_text(json.dumps(payload)+"\n");rows.extend(shard_rows)
    return rows,{"shard_count":math.ceil(len(selected)/size),"resumed_shards":resumed,"webp_available":webp,"peak_process_rss_bytes":peak_rss,"fingerprint":fingerprint}
def _counts(record):
    truth=np.asarray(record["truth"],bool);states=record["states"];tp=sum(s=="manipulated" and t for s,t in zip(states,truth));fp=sum(s=="manipulated" and not t for s,t in zip(states,truth));fn=sum(s!="manipulated" and t for s,t in zip(states,truth));return tp,fp,fn,sum(s=="valid" for s in states),sum(s=="manipulated" for s in states),sum(s=="uncertain" for s in states),len(states)
def _auc(negative,positive):
    if not positive:return None
    values=np.asarray(negative+positive);labels=np.asarray([0]*len(negative)+[1]*len(positive));ranks=rankdata(values,method="average");return float((ranks[labels==1].sum()-len(positive)*(len(positive)+1)/2)/(len(negative)*len(positive)))
def _ci(values,rng,iterations):
    values=np.asarray(values,float);samples=np.mean(values[rng.integers(0,len(values),(iterations,len(values)))],axis=1);return [float(np.quantile(samples,.025)),float(np.quantile(samples,.975))]
def summarize(rows,digest_name,bootstrap_iterations,seed):
    transforms=list(rows[0]["digests"][digest_name]);benign=[x for x in transforms if x in STANDARD_BENIGN or x in EXTREME_BENIGN];benign_by_image=[[distance for name in benign for distance in row["digests"][digest_name][name]["distances"]] for row in rows];result={};rng=np.random.default_rng(seed)
    for transform in transforms:
        records=[row["digests"][digest_name][transform] for row in rows];per=[]
        for record in records:
            tp,fp,fn,valid,manip,uncertain,total=_counts(record);truth=sum(record["truth"]);precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/truth if truth else 0.;per.append({"false_manipulation_rate":fp/max(1,total-truth),"missed_manipulation_rate":fn/max(1,truth),"valid_rate":valid/total,"manipulated_rate":manip/total,"uncertain_rate":uncertain/total,"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0.})
        macro={key:mean(x[key] for x in per) for key in per[0]};cis={key:_ci([x[key] for x in per],rng,bootstrap_iterations) for key in per[0]};all_dist=[x for record in records for x in record["distances"]];positives=[d for record in records for d,t in zip(record["distances"],record["truth"]) if t];negative=[x for values in benign_by_image for x in values];image_aucs=[_auc(benign_by_image[i],[d for d,t in zip(records[i]["distances"],records[i]["truth"]) if t]) for i in range(len(rows))];image_aucs=[x for x in image_aucs if x is not None]
        result[transform]={"macro_by_image":macro,"image_clustered_bootstrap_95_ci":cis,"region_level":{"false_manipulation_rate":sum(_counts(x)[1] for x in records)/max(1,sum(_counts(x)[6]-sum(x["truth"]) for x in records)),"missed_manipulation_rate":sum(_counts(x)[2] for x in records)/max(1,sum(sum(x["truth"]) for x in records)),"valid_rate":sum(_counts(x)[3] for x in records)/(16*len(records)),"manipulated_rate":sum(_counts(x)[4] for x in records)/(16*len(records)),"uncertain_rate":sum(_counts(x)[5] for x in records)/(16*len(records)),"distance":{"mean":mean(all_dist),"q05":float(np.quantile(all_dist,.05)),"median":float(np.quantile(all_dist,.5)),"q95":float(np.quantile(all_dist,.95)),"maximum":max(all_dist)},"roc_auc_vs_pooled_benign":_auc(negative,positives)},"roc_auc_macro_by_image":mean(image_aucs) if image_aucs else None,"roc_auc_image_clustered_95_ci":_ci(image_aucs,rng,bootstrap_iterations) if image_aucs else None}
    return result
def _exact_collision_controls(dataset,selected,digest_name):
    digest=DCTPerceptualHash() if digest_name=="dct_phash" else CombinedDigest();values=[];image_ids=[]
    for i,item in enumerate(selected):
        current=digest.digest_image(dataset[item["dataset_index"]]["image"]);values.extend(current);image_ids.extend([i]*16)
    image_ids=np.asarray(image_ids);nearest=[]
    if digest_name=="dct_phash":dct=np.asarray(values,bool)
    else:dct=np.asarray([x[0] for x in values],bool);diff=np.asarray([x[1] for x in values],bool);stats=np.asarray([x[2] for x in values],np.float32)
    for start in range(0,len(values),32):
        stop=min(len(values),start+32);distance=np.mean(dct[start:stop,None,:]!=dct[None,:,:],axis=2)
        if digest_name=="combined_digest":distance=.4*distance+.3*np.mean(diff[start:stop,None,:]!=diff[None,:,:],axis=2)+.3*np.minimum(1.,np.mean(np.abs(stats[start:stop,None,:]-stats[None,:,:]),axis=2)*4)
        distance[image_ids[None,:]==image_ids[start:stop,None]]=np.inf;nearest.extend(np.min(distance,axis=1).tolist())
    return {"nearest_digest_unrelated_region":{"count":len(nearest),"mean":mean(nearest),"q05":float(np.quantile(nearest,.05)),"median":float(np.quantile(nearest,.5)),"q95":float(np.quantile(nearest,.95)),"minimum":min(nearest)},"interpretation":"exact empirical nearest search over reproduction regions; not adversarial collision proof"}
def _paired_difference_ci(rows,a,b,transforms,iterations,seed):
    differences=[]
    for row in rows:
        recalls=[]
        for digest in (a,b):
            vals=[]
            for transform in transforms:
                record=row["digests"][digest][transform];tp,_,fn,*_=_counts(record);vals.append(tp/max(1,tp+fn))
            recalls.append(mean(vals))
        differences.append(recalls[1]-recalls[0])
    return {"mean":mean(differences),"image_clustered_bootstrap_95_ci":_ci(differences,np.random.default_rng(seed),iterations)}
def run_large_scale_reproduction(config):
    started=time.perf_counter();tracemalloc.start();frozen,threshold_checks=load_frozen_thresholds(config["frozen_thresholds"],config["prior_report"]);dataset=CocoImageDataset(config["data_root"]);selected,exclusion=select_unseen_population(dataset,config["prior_report"],int(config["population_size"]),int(config["seed"]));write_population_preflight(config,exclusion);rows,shards=run_shards(config,dataset,selected,frozen);summaries={name:summarize(rows,name,int(config["bootstrap_iterations"]),int(config["seed"])+i) for i,name in enumerate(("dct_phash","combined_digest"))};gates={};extreme=set(config["extreme_stress_conditions"])
    for name,summary in summaries.items():
        standard=[summary[x]["macro_by_image"]["false_manipulation_rate"] for x in STANDARD_BENIGN if x in summary and x not in extreme];malicious=[summary[x]["macro_by_image"]["recall"] for x in MALICIOUS];runtime=mean(row["digests"][name][transform]["verification_ms"] for row in rows for transform in row["digests"][name]);gate={"mean_benign_fmr":mean(standard)<=config["gates"]["mean_benign_false_manipulation"],"worst_standard_benign_fmr":max(standard)<=config["gates"]["worst_standard_benign_false_manipulation"],"aggregate_malicious_recall":mean(malicious)>=config["gates"]["aggregate_malicious_recall"],"splice_025":summary["splice_025"]["macro_by_image"]["recall"]>=config["gates"]["splice_025_recall"],"overlay_025":summary["overlay_025"]["macro_by_image"]["recall"]>=config["gates"]["overlay_025_recall"],"clean_repeatability":summary["clean_repeat"]["region_level"]["false_manipulation_rate"]==config["gates"]["clean_repeatability_failures"],"runtime":runtime<config["gates"]["verification_runtime_ms"],"registry_authentication":all(row["registry_authenticated"][name] for row in rows)};gates[name]={"results":gate,"passed":all(gate.values()),"mean_standard_benign_fmr":mean(standard),"worst_standard_benign_fmr":max(standard),"aggregate_malicious_recall":mean(malicious),"verification_runtime_ms":runtime}
    improvement=_paired_difference_ci(rows,"dct_phash","combined_digest",("splice_025","overlay_025"),int(config["bootstrap_iterations"]),int(config["seed"])+99);selected_candidate=None
    if gates["combined_digest"]["passed"] and improvement["image_clustered_bootstrap_95_ci"][0]>0:selected_candidate="combined_digest"
    elif gates["dct_phash"]["passed"] and (not gates["combined_digest"]["passed"] or improvement["image_clustered_bootstrap_95_ci"][0]<=0):selected_candidate="dct_phash"
    status="passed_large_scale_regional_digest_reproduction" if selected_candidate else "blocked_by_large_scale_regional_digest_gates";current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();elapsed=time.perf_counter()-started
    report={"schema_version":"large_scale_regional_digest_reproduction_v1.0","source_commit":SOURCE_COMMIT,"frozen_thresholds":{name:frozen[name] for name in ("dct_phash","combined_digest")},"threshold_checks":threshold_checks,"dataset":{"path":str(Path(config["data_root"])),"seed":config["seed"],"population_size":len(selected),"images":selected,"exclusion":exclusion,"preprocessing":config["preprocessing"]},"statistical_unit":"source image; region results are descriptive","shards":shards,"digests":summaries,"gates":gates,"small_area_combined_minus_dct_recall":improvement,"collision_controls":{name:_exact_collision_controls(dataset,selected,name) for name in ("dct_phash","combined_digest")},"resources":{"wall_time_seconds":elapsed,"throughput_images_per_second":len(selected)/elapsed,"peak_python_traced_bytes":peak,"peak_process_rss_bytes":shards["peak_process_rss_bytes"],"registry_bytes_per_image":{"dct_phash":640,"combined_digest":960},"digest_generation_mean_ms":{name:mean(row["generation_ms"][name] for row in rows) for name in ("dct_phash","combined_digest")},"verification_mean_ms":{name:gates[name]["verification_runtime_ms"] for name in ("dct_phash","combined_digest")}},"selection":{"candidate":selected_candidate,"rule":"prefer combined only when it passes all frozen gates and paired image-clustered small-area recall improvement CI is above zero; otherwise DCT only if it passes"},"region_correspondence_assumed":True,"crop_synchronization_validated":False,"global_identity_authentication":"trusted prerequisite simulated separately; digest matching is not source authentication","scientific_status":status,"neural_stage_passed":False,"stage_e_permitted":False,"novelty_claimed":False}
    output=Path(config["output_directory"]);output.mkdir(parents=True,exist_ok=True);text=json.dumps(report,indent=2,default=_json_default)+"\n";(output/"report.json").write_text(text);committed=Path(config["committed_report"]);committed.parent.mkdir(parents=True,exist_ok=True);committed.write_text(text);return report
