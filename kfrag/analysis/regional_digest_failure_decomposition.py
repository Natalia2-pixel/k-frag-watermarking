"""Read-only decomposition of frozen large-scale regional-digest evidence."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from statistics import mean
import numpy as np
from kfrag.data import CocoImageDataset
from kfrag.diagnostics.large_scale_regional_digest import MALICIOUS,_similar_map,regions

DIGESTS=("dct_phash","combined_digest")
FAMILIES={
    "unrelated_replacement":("replacement_full",),
    "splice":("splice_010","splice_025","splice_050","splice_075"),
    "overlay":("overlay_010","overlay_025","overlay_050","overlay_075"),
    "occlusion":("occlusion_010","occlusion_025","occlusion_050","occlusion_075"),
    "relocation":("same_image_relocation",),
    "collage":("mixed_source_collage",),
    "visually_similar_replacement":("visually_similar_replacement",),
    "same_source_replacement":("same_source_replacement",),
}
AREAS={family:{area:f"{family}_{area}" for area in ("010","025","050","075")} for family in ("splice","overlay","occlusion")}
PRESERVED={"scientific_status":"blocked_by_large_scale_regional_digest_gates","selected_candidate":None,"neural_stage_passed":False,"stage_e_permitted":False}

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _ci(values,seed,iterations):
    values=np.asarray(values,float);rng=np.random.default_rng(seed)
    samples=values[rng.integers(0,len(values),(iterations,len(values)))].mean(1)
    return [float(np.quantile(samples,.025)),float(np.quantile(samples,.975))]

def load_frozen_evidence(report_path,shard_directory,expected_shards,expected_images,expected_manifest_hash):
    report_path=Path(report_path);shard_paths=sorted(Path(shard_directory).glob("shard_*.json"))
    before={"report_sha256":_sha(report_path),"shard_sha256":{p.name:_sha(p) for p in shard_paths}}
    report=json.loads(report_path.read_text());payloads=[json.loads(p.read_text()) for p in shard_paths]
    if len(payloads)!=expected_shards:raise ValueError(f"expected {expected_shards} shards, found {len(payloads)}")
    if any(x.get("fingerprint")!=expected_manifest_hash for x in payloads):raise ValueError("manifest-hash verification failed")
    rows=[row for payload in payloads for row in payload["rows"]]
    if len(rows)!=expected_images:raise ValueError(f"expected {expected_images} rows, found {len(rows)}")
    if [x["identifier"] for x in rows]!=[x["identifier"] for x in report["dataset"]["images"]]:raise ValueError("shard/report population order mismatch")
    if report["scientific_status"]!=PRESERVED["scientific_status"] or report["selection"]["candidate"] is not None or report["neural_stage_passed"] or report["stage_e_permitted"]:raise ValueError("frozen scientific conclusion mismatch")
    after={"report_sha256":_sha(report_path),"shard_sha256":{p.name:_sha(p) for p in shard_paths}}
    if before!=after:raise RuntimeError("frozen evidence changed while loading")
    return report,rows,{"shard_count":len(payloads),"image_count":len(rows),"manifest_hash":expected_manifest_hash,"hashes_before":before,"hashes_after":after,"evidence_unchanged":True}

def _per_image(row,digest,transform):
    record=row["digests"][digest][transform];truth=np.asarray(record["truth"],bool);states=np.asarray(record["states"])
    n=max(1,int(truth.sum()));manip=float(((states=="manipulated")&truth).sum()/n);unc=float(((states=="uncertain")&truth).sum()/n)
    return {"recall":manip,"missed":1-manip,"uncertain":unc,"valid_miss":float(((states=="valid")&truth).sum()/n)}

def _summary(values,seed,iterations):
    return {"image_macro":mean(values),"image_clustered_95_ci":_ci(values,seed,iterations)}

def decompose(rows,report,seed,iterations):
    result={}
    for di,digest in enumerate(DIGESTS):
        per_transform={};all_missed=[];all_uncertain=[]
        for ti,transform in enumerate(MALICIOUS):
            vals=[_per_image(row,digest,transform) for row in rows];miss=[x["missed"] for x in vals];unc=[x["uncertain"] for x in vals]
            all_missed.append(mean(miss));all_uncertain.append(mean(unc))
            recall=1-mean(miss);official=report["digests"][digest][transform]
            records=[row["digests"][digest][transform] for row in rows]
            actual_missed=sum(sum(t and state!="manipulated" for t,state in zip(record["truth"],record["states"])) for record in records)
            actual_uncertain=sum(sum(t and state=="uncertain" for t,state in zip(record["truth"],record["states"])) for record in records)
            actual_altered=sum(sum(record["truth"]) for record in records)
            per_transform[transform]={
                "attack_family":next(k for k,v in FAMILIES.items() if transform in v),
                "alteration_area":int(transform[-3:])/100 if transform.startswith(("splice_","overlay_","occlusion_")) else (4/16 if transform=="mixed_source_collage" else 1.0),
                "source_relation":"same_source" if transform in ("same_image_relocation","same_source_replacement") else ("mixed_audit_required" if transform=="visually_similar_replacement" else "different_source"),
                "recall":recall,
                "recall_image_clustered_95_ci":official["image_clustered_bootstrap_95_ci"]["recall"],
                "missed_manipulation_rate":mean(miss),
                "missed_image_clustered_95_ci":official["image_clustered_bootstrap_95_ci"]["missed_manipulation_rate"],
                "uncertain_rate_on_altered_regions":mean(unc),
                "actual_altered_region_observations":actual_altered,
                "actual_missed_altered_region_observations":actual_missed,
                "actual_uncertain_altered_region_observations":actual_uncertain,
                "uncertain_contribution_to_all_attack_macro_uncertainty":0.0,
                "contribution_to_aggregate_missed_rate":mean(miss)/len(MALICIOUS),
                "counterfactual_aggregate_recall_if_perfect":report["gates"][digest]["aggregate_malicious_recall"]+mean(miss)/len(MALICIOUS),
            }
        missed_total=sum(all_missed);uncertain_total=sum(all_uncertain);actual_missed_total=sum(x["actual_missed_altered_region_observations"] for x in per_transform.values());actual_uncertain_total=sum(x["actual_uncertain_altered_region_observations"] for x in per_transform.values());aggregate=report["gates"][digest]["aggregate_malicious_recall"];gate_deficit=max(0,.90-aggregate)
        for transform,item in per_transform.items():
            item["share_of_aggregate_missed_manipulations"]=item["missed_manipulation_rate"]/missed_total if missed_total else 0
            item["share_of_aggregate_gate_deficit"]=item["contribution_to_aggregate_missed_rate"]/gate_deficit if gate_deficit else 0
            item["uncertain_contribution_to_all_attack_macro_uncertainty"]=item["uncertain_rate_on_altered_regions"]/uncertain_total if uncertain_total else 0
            item["share_of_actual_missed_altered_regions"]=item["actual_missed_altered_region_observations"]/actual_missed_total if actual_missed_total else 0
            item["share_of_actual_uncertain_altered_regions"]=item["actual_uncertain_altered_region_observations"]/actual_uncertain_total if actual_uncertain_total else 0
        family={}
        for fi,(name,transforms) in enumerate(FAMILIES.items()):
            vals=[]
            for row in rows:
                sample=[_per_image(row,digest,t) for t in transforms]
                vals.append({key:mean(x[key] for x in sample) for key in sample[0]})
            family[name]={
                "transforms":list(transforms),
                "recall":_summary([x["recall"] for x in vals],seed+di*100+fi,iterations),
                "missed":_summary([x["missed"] for x in vals],seed+di*100+fi+20,iterations),
                "uncertain_on_altered_regions":_summary([x["uncertain"] for x in vals],seed+di*100+fi+40,iterations),
                "contribution_to_aggregate_missed_rate":sum(per_transform[t]["contribution_to_aggregate_missed_rate"] for t in transforms),
                "share_of_aggregate_missed_manipulations":sum(per_transform[t]["share_of_aggregate_missed_manipulations"] for t in transforms),
                "share_of_aggregate_gate_deficit":sum(per_transform[t]["share_of_aggregate_gate_deficit"] for t in transforms),
                "actual_missed_altered_region_observations":sum(per_transform[t]["actual_missed_altered_region_observations"] for t in transforms),
                "share_of_actual_missed_altered_regions":sum(per_transform[t]["share_of_actual_missed_altered_regions"] for t in transforms),
                "actual_uncertain_altered_region_observations":sum(per_transform[t]["actual_uncertain_altered_region_observations"] for t in transforms),
                "share_of_actual_uncertain_altered_regions":sum(per_transform[t]["share_of_actual_uncertain_altered_regions"] for t in transforms),
            }
        result[digest]={
            "official_aggregate_recall":aggregate,
            "aggregate_recall_gate":.90,
            "aggregate_gate_deficit":gate_deficit,
            "official_aggregate_missed_rate":1-aggregate,
            "actual_missed_altered_region_observations":actual_missed_total,
            "actual_uncertain_altered_region_observations":actual_uncertain_total,
            "transforms":per_transform,
            "families":family,
            "area_decomposition":{f:{a:per_transform[t] for a,t in mapping.items()} for f,mapping in AREAS.items()},
            "overlay_025_gate":{"required_recall":.85,"observed_recall":per_transform["overlay_025"]["recall"],"deficit":max(0,.85-per_transform["overlay_025"]["recall"]),"missed_rate":per_transform["overlay_025"]["missed_manipulation_rate"],"uncertain_rate_on_altered_regions":per_transform["overlay_025"]["uncertain_rate_on_altered_regions"]},
        }
    return result

def audit_visually_similar_selection(data_root,manifest,rows):
    dataset=CocoImageDataset(data_root);records=[]
    for start in range(0,len(manifest),50):
        items=manifest[start:start+50];images=[dataset[x["dataset_index"]]["image"] for x in items]
        donor_items=[manifest[(start+i+1)%len(manifest)] for i in range(len(images))]
        donors=[dataset[x["dataset_index"]]["image"] for x in donor_items];pool=images+donors;pool_items=items+donor_items;mapping=_similar_map(pool)
        for local,(item,image) in enumerate(zip(items,images)):
            global_index=start+local;target=(global_index*7)%16;donor_index,donor_region=mapping[(local,target)]
            donor_item=pool_items[donor_index];source_region=regions(image)[target];donor_region_pixels=regions(pool[donor_index])[donor_region]
            rgb_a=source_region.mean((1,2)).numpy();rgb_b=donor_region_pixels.mean((1,2)).numpy()
            records.append({"source_identifier":item["identifier"],"source_region":target,"selected_identifier":donor_item["identifier"],"selected_region":donor_region,"same_identifier":item["identifier"]==donor_item["identifier"],"different_sha256":item["sha256"]!=donor_item["sha256"],"mean_rgb_euclidean":float(np.linalg.norm(rgb_a-rgb_b)),"pixel_mae":float((source_region-donor_region_pixels).abs().mean()),"pixel_exact_equal":bool(np.array_equal(source_region.numpy(),donor_region_pixels.numpy()))})
    def dist(key,subset):
        v=np.asarray([x[key] for x in subset],float);return {"count":len(v),"mean":float(v.mean()),"minimum":float(v.min()),"q05":float(np.quantile(v,.05)),"median":float(np.quantile(v,.5)),"q95":float(np.quantile(v,.95)),"maximum":float(v.max())}
    distinct=[x for x in records if not x["same_identifier"]];same=[x for x in records if x["same_identifier"]]
    strata={}
    for digest in DIGESTS:
        strata[digest]={}
        for label,indices in (("duplicate_source",[i for i,x in enumerate(records) if x["same_identifier"]]),("genuinely_different",[i for i,x in enumerate(records) if not x["same_identifier"]])):
            vals=[_per_image(rows[i],digest,"visually_similar_replacement") for i in indices]
            strata[digest][label]={"count":len(vals),"recall":_summary([x["recall"] for x in vals],3100+len(indices)+(0 if digest=="dct_phash" else 1),1000),"missed_rate":mean(x["missed"] for x in vals),"uncertain_rate":mean(x["uncertain"] for x in vals)}
    return {"selection_definition":"Within each 50-image shard, nearest Euclidean mean-RGB regional vector among the 50 source images plus a one-image-shifted duplicate donor list; only the same pool position was excluded, not duplicate underlying identifiers.","records":len(records),"same_identifier_count":sum(x["same_identifier"] for x in records),"different_identifier_count":sum(not x["same_identifier"] for x in records),"different_sha256_count":sum(x["different_sha256"] for x in records),"pixel_exact_equal_count":sum(x["pixel_exact_equal"] for x in records),"genuinely_different_image_and_content_count":sum((not x["same_identifier"]) and not x["pixel_exact_equal"] for x in records),"mean_rgb_similarity_distribution":{"all":dist("mean_rgb_euclidean",records),"duplicate_source":dist("mean_rgb_euclidean",same),"genuinely_different":dist("mean_rgb_euclidean",distinct)},"pixel_mae_distribution":{"all":dist("pixel_mae",records),"duplicate_source":dist("pixel_mae",same),"genuinely_different":dist("pixel_mae",distinct)},"digest_outcomes_by_source_relation":strata,"representative_numeric_examples":same[:3]+distinct[:3],"conclusion":"The generator did not reliably create visually similar cross-image replacements; duplicate source images in the shifted donor pool made most nominal replacements pixel-identical. This is a generator-validity defect in that attack condition, not digest evidence of semantic robustness."}

OBJECTIVES={
 "exact_content_integrity":{"dct_phash":"unsupported: lossy digest and benign-tolerance thresholds intentionally ignore many pixel changes","combined_digest":"unsupported: lossy digest and benign-tolerance thresholds intentionally ignore many pixel changes"},
 "perceptual_integrity":{"dct_phash":"partially supported: benign gates pass, but small overlays and the invalid visually-similar condition prevent the frozen reproduction gates","combined_digest":"partially supported and stronger on 25% splice/overlay, but still fails aggregate and overlay gates"},
 "semantic_integrity":{"dct_phash":"unsupported: no object, text, identity, or layout representation","combined_digest":"unsupported: low-frequency hash/statistics do not establish semantic change sensitivity"},
}
MECHANISMS={
 "robust_plus_fragile_dual_digest":{"benign_robustness":"robust branch tolerates declared benign processing; fragile branch deliberately does not","similar_replacement_sensitivity":"potentially high for pixel changes, but fragile alarms conflate benign processing","registry_storage":"two digests per region; likely tens of bytes/region","runtime":"low to moderate","external_requirement":"authenticated registry digest pairs; no original pixels at verification","adversarial_weaknesses":"fragile false alarms, robust collisions, digest transplantation","fixed_20bit_packet":"compatible only through registry lookup; not by embedding both digests","crop_sync":"required for fixed-region comparison"},
 "canonicalized_cryptographic_hashing":{"benign_robustness":"only to transformations perfectly removed by canonicalization","similar_replacement_sensitivity":"cryptographically strong after stable canonicalization","registry_storage":"at least 16-32 bytes/region","runtime":"low","external_requirement":"authenticated canonical hashes; deterministic canonicalizer","adversarial_weaknesses":"canonicalization instability or equivalence-class attacks","fixed_20bit_packet":"registry-bound, not carried in 8-bit share","crop_sync":"required"},
 "local_deep_feature_similarity":{"benign_robustness":"potentially strong; must be calibrated independently","similar_replacement_sensitivity":"potentially better for semantic differences, uncertain for lookalikes","registry_storage":"compressed embeddings commonly tens to hundreds of bytes/region","runtime":"moderate to high","external_requirement":"registry embeddings/model version; no original pixels","adversarial_weaknesses":"feature collisions, adversarial examples, model drift","fixed_20bit_packet":"registry-only compatibility","crop_sync":"required unless feature mechanism adds localization"},
 "semi_fragile_learned_signatures":{"benign_robustness":"learnable but requires real transformation validation","similar_replacement_sensitivity":"potentially tunable","registry_storage":"signature-dependent","runtime":"moderate","external_requirement":"model/version and authenticated expected signature","adversarial_weaknesses":"training-distribution gaps and adaptive attacks","fixed_20bit_packet":"unknown; extra neural capacity likely required","crop_sync":"required unless jointly trained for synchronization"},
 "object_text_layout_semantic_consistency":{"benign_robustness":"potentially strong for low-level changes","similar_replacement_sensitivity":"best aligned with semantic objective","registry_storage":"structured detections/text/layout may be hundreds to thousands of bytes/image","runtime":"high","external_requirement":"authenticated registry features and model versions","adversarial_weaknesses":"detector/OCR failures, semantic collisions, adversarial content","fixed_20bit_packet":"registry-only compatibility","crop_sync":"region correspondence or independent registration required"},
 "registry_multiscale_regional_features":{"benign_robustness":"can combine robust coarse and sensitive fine scales","similar_replacement_sensitivity":"testable and likely better than current compact digest","registry_storage":"explicit cost; likely hundreds of bytes/image or more","runtime":"moderate","external_requirement":"authenticated multiscale registry features","adversarial_weaknesses":"feature collisions and scale-dependent evasion","fixed_20bit_packet":"compatible via authenticated identity lookup","crop_sync":"required in this fixed-grid pilot"},
}

def run_analysis(config):
    report,rows,provenance=load_frozen_evidence(config["reproduction_report"],config["shard_directory"],int(config["expected_shards"]),int(config["expected_images"]),config["expected_manifest_hash"])
    decomposition=decompose(rows,report,int(config["seed"]),int(config["bootstrap_iterations"]))
    similarity=audit_visually_similar_selection(config["data_root"],report["dataset"]["images"],rows)
    result={"schema_version":"regional_digest_failure_decomposition_v1.0","analysis_type":"read-only frozen-evidence decomposition","evidence":provenance,"population_preflight":report["population_preflight"],"decomposition":decomposition,"visually_similar_generator_audit":similarity,"integrity_objectives":OBJECTIVES,"mechanism_comparison":MECHANISMS,"smallest_next_experiment":{"recommendation":"Replace only the invalid visually-similar generator with a cross-image, SHA-distinct, pixel-different selector; predeclare semantic/perceptual similarity strata using registry-stored multiscale regional features, then evaluate current frozen digests plus one local deep-feature baseline on a new disjoint locked population.","why_smallest":"It directly repairs the measured attack-condition validity defect and tests the current digest-space blind spot without threshold tuning, watermark training, or packet changes.","prohibitions":"Do not reuse the locked reproduction population for selection, do not tune frozen thresholds, and do not interpret feature similarity as cryptographic authentication."},"scientific_status":PRESERVED["scientific_status"],"selected_candidate":None,"neural_stage_passed":False,"stage_e_permitted":False,"novelty_claimed":False}
    output=Path(config["output_report"]);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+"\n")
    after={"report_sha256":_sha(config["reproduction_report"]),"shard_sha256":{p.name:_sha(p) for p in sorted(Path(config["shard_directory"]).glob("shard_*.json"))}}
    if after!=provenance["hashes_before"]:raise RuntimeError("analysis modified frozen evidence")
    return result
