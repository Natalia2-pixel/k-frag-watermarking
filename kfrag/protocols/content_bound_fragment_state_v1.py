"""Protocol-only content-bound fragment-state strategies and simulation."""
from __future__ import annotations
import hashlib,hmac,json,math,random
from dataclasses import dataclass
from typing import Iterable
import numpy as np

REGIONS=16
STATES=("valid","missing","manipulated","uncertain")

@dataclass(frozen=True)
class RegistryRecord:
    source_id:bytes;protocol_version:int;strategy:str;commitment:bytes;mac:bytes
@dataclass(frozen=True)
class VerificationOutput:
    authenticated_source_identity:bool;fragment_states:tuple[str,...]
    @property
    def valid_fragment_evidence(self):return tuple(i for i,x in enumerate(self.fragment_states) if x=="valid")
    @property
    def missing_fragment_evidence(self):return tuple(i for i,x in enumerate(self.fragment_states) if x=="missing")
    @property
    def manipulated_local_evidence(self):return tuple(i for i,x in enumerate(self.fragment_states) if x=="manipulated")
    @property
    def uncertain_local_evidence(self):return tuple(i for i,x in enumerate(self.fragment_states) if x=="uncertain")

def _message(source_id,version,strategy,commitment):return b"KFRAG-CONTENT-STATE"+bytes([version])+len(source_id).to_bytes(2,"big")+source_id+strategy.encode()+b"\0"+commitment
def _record(source_id,version,strategy,commitment,key):return RegistryRecord(source_id,version,strategy,commitment,hmac.new(key,_message(source_id,version,strategy,commitment),hashlib.sha256).digest())
def _record_valid(record,key):return hmac.compare_digest(record.mac,hmac.new(key,_message(record.source_id,record.protocol_version,record.strategy,record.commitment),hashlib.sha256).digest())
def _pack_bits(bits):return np.packbits(np.asarray(bits,dtype=np.uint8)).tobytes()

class ContentBindingStrategy:
    name="base";embedded_bits_per_region=20;additional_neural_bits=0;registry_requirement="authenticated content reference"
    def enroll(self,features,source_id,key):
        reference=self.reference(np.asarray(features,dtype=np.float64));commitment=self.commitment(reference);return _record(source_id,2,self.name,commitment,key),reference
    def calibrate(self,clean_population,seed=1):
        rng=np.random.default_rng(seed);scores=[]
        for features in clean_population:
            reference=self.reference(features)
            for observed in (features+rng.normal(0,.025,features.shape),np.round(features*12)/12,(features*.98)+.015):scores.extend(self.scores(reference,self.reference(observed),np.ones(REGIONS,dtype=bool)))
        valid=float(np.quantile(scores,.99));manipulated=float(np.quantile(scores,.999));step=self.minimum_gap();return {"valid_max":valid,"manipulated_min":max(manipulated,valid+step)}
    def verify(self,record,reference,observed_features,present,key,identity_evidence,thresholds):
        authenticated=bool(identity_evidence and _record_valid(record,key));present=np.asarray(present,dtype=bool);observed=self.reference(np.asarray(observed_features,dtype=np.float64));scores=self.scores(reference,observed,present);states=[]
        for i in range(REGIONS):
            if not present[i]:states.append("missing")
            elif not authenticated:states.append("uncertain")
            elif scores[i]<=thresholds["valid_max"]:states.append("valid")
            elif scores[i]>=thresholds["manipulated_min"]:states.append("manipulated")
            else:states.append("uncertain")
        return VerificationOutput(authenticated,tuple(states))
    def minimum_gap(self):return 1e-6

class RobustPerceptualDigest(ContentBindingStrategy):
    name="robust_perceptual_digest";registry_requirement="authenticated 64-bit perceptual digest per region; no original pixels"
    def __init__(self,dimension=32,digest_bits=64,seed=101):self.digest_bits=digest_bits;self.projection=np.random.default_rng(seed).normal(size=(dimension,digest_bits))
    def reference(self,features):
        normalized=features/(np.linalg.norm(features,axis=1,keepdims=True)+1e-9);return normalized@self.projection>=0
    def commitment(self,reference):return _pack_bits(reference)
    def scores(self,expected,observed,present):return np.mean(expected!=observed,axis=1)
    def minimum_gap(self):return 1/self.digest_bits

class SemiFragileLearnedSignature(ContentBindingStrategy):
    name="semi_fragile_learned_signature";registry_requirement="authenticated 16-dimensional learned signature per region and fixed public extractor"
    def __init__(self,dimension=32,signature_dimensions=16,seed=202):self.signature_dimensions=signature_dimensions;self.weights=np.random.default_rng(seed).normal(size=(dimension,signature_dimensions))/math.sqrt(dimension)
    def reference(self,features):return np.tanh(features@self.weights)
    def commitment(self,reference):return np.round((reference+1)*127.5).clip(0,255).astype(np.uint8).tobytes()
    def scores(self,expected,observed,present):return np.mean(np.abs(expected-observed),axis=1)

class CrossRegionParitySyndrome(ContentBindingStrategy):
    name="cross_region_parity_syndrome";registry_requirement="authenticated expected parity syndrome and public parity graph"
    def __init__(self,dimension=32,seed=303):self.vector=np.random.default_rng(seed).normal(size=dimension);self.edges=tuple((i,(i+1)%REGIONS) for i in range(REGIONS))+tuple((i,(i+4)%REGIONS) for i in range(REGIONS))
    def reference(self,features):
        bits=(features@self.vector)>=0;syndrome=np.asarray([bits[a]^bits[b] for a,b in self.edges],dtype=bool);return bits,syndrome
    def commitment(self,reference):return _pack_bits(reference[1])
    def scores(self,expected,observed,present):
        _,expected_syndrome=expected;_,observed_syndrome=observed;violated=expected_syndrome!=observed_syndrome;scores=np.zeros(REGIONS);counts=np.zeros(REGIONS)
        for edge,(a,b) in enumerate(self.edges):
            if present[a] and present[b]:scores[a]+=violated[edge];scores[b]+=violated[edge];counts[a]+=1;counts[b]+=1
        return scores/np.maximum(counts,1)
    def minimum_gap(self):return .25

def strategy_comparison():
    return [
      {"strategy":"robust_perceptual_digest","embedded_bits_per_region":20,"authentication_share_bits":8,"total_content_reference_bits":1024,"external_registry":"64-bit digest for every region","blind_without_original":True,"benign_tolerance":"projection digest calibrated for JPEG/resize/colour proxies","splice_overlay_replacement":"local digest distance; uncertain band before manipulated","replay_collage_weakness":"same-source replay and perceptually colliding collage remain possible","security_assumptions":"HMAC registry integrity, digest collision resistance is empirical not cryptographic","additional_neural_capacity_bits":0,"self_contained_cost":"at least 64 additional bits/region"},
      {"strategy":"semi_fragile_learned_signature","embedded_bits_per_region":20,"authentication_share_bits":8,"total_content_reference_bits":2048,"external_registry":"quantized 16-dimensional signature per region plus extractor version","blind_without_original":True,"benign_tolerance":"must be learned/calibrated for JPEG, resize and colour transformations","splice_overlay_replacement":"signature distance may localize distributional change","replay_collage_weakness":"extractor collisions, transfer attacks and same-signature collage","security_assumptions":"HMAC registry integrity and robust generalization of learned extractor","additional_neural_capacity_bits":0,"self_contained_cost":"approximately 16-128 additional bits/region depending quantization"},
      {"strategy":"cross_region_parity_syndrome","embedded_bits_per_region":20,"authentication_share_bits":8,"total_content_reference_bits":32,"external_registry":"expected authenticated syndrome and parity graph","blind_without_original":True,"benign_tolerance":"quantization must remain stable; parity can amplify benign bit changes","splice_overlay_replacement":"inconsistent incident checks infer suspect regions, not independent authentication","replay_collage_weakness":"coordinated replacements preserving parity and even-number cancellations","security_assumptions":"HMAC registry integrity and sufficient parity-graph distance","additional_neural_capacity_bits":0,"self_contained_cost":"approximately 4-8 additional parity bits/region"},
    ]

def _scenario(clean,name,rng):
    observed=clean.copy();present=np.ones(REGIONS,dtype=bool);truth=np.asarray(["valid"]*REGIONS,dtype=object);identity=True
    if name=="benign_noise":observed+=rng.normal(0,.03,observed.shape)
    elif name=="jpeg_proxy":observed=np.round(observed*10)/10
    elif name=="resize_proxy":observed=.96*observed+.04*np.roll(observed,1,axis=0)
    elif name=="colour_proxy":observed=observed*.97+.02
    elif name=="erasure":present[rng.choice(REGIONS,4,replace=False)]=False;truth[~present]="missing"
    elif name=="replacement":
        indices=rng.choice(REGIONS,2,replace=False);observed[indices]=rng.normal(size=(2,clean.shape[1]));truth[indices]="manipulated"
    elif name=="splice":
        indices=rng.choice(REGIONS,2,replace=False);observed[indices]=.25*clean[indices]+.75*rng.normal(size=(2,clean.shape[1]));truth[indices]="manipulated"
    elif name=="mixed_source":observed[8:]=rng.normal(size=observed[8:].shape);truth[8:]="manipulated";identity=False
    else:raise ValueError(name)
    return observed,present,truth,identity

def simulate_content_binding(trials=1000,calibration_trials=256,dimension=32,seed=404):
    rng=np.random.default_rng(seed);key=hashlib.sha256(f"content-state:{seed}".encode()).digest();calibration=rng.normal(size=(calibration_trials,REGIONS,dimension));strategies=(RobustPerceptualDigest(dimension),SemiFragileLearnedSignature(dimension),CrossRegionParitySyndrome(dimension));scenarios=("benign_noise","jpeg_proxy","resize_proxy","colour_proxy","erasure","replacement","splice","mixed_source");report={"trials_per_strategy":trials,"calibration_trials":calibration_trials,"strategies":{},"comparison":strategy_comparison(),"neural_stage_passed":False,"stage_e_permitted":False}
    for strategy in strategies:
        thresholds=strategy.calibrate(calibration,seed+1);counts={name:{"false_manipulated":0,"missed_manipulated":0,"true_manipulated":0,"true_valid":0,"uncertain":0,"identity_authenticated":0,"state_counts":{x:0 for x in STATES}} for name in scenarios}
        for trial in range(trials):
            clean=rng.normal(size=(REGIONS,dimension));source=hashlib.sha256(f"{strategy.name}:{trial}".encode()).digest()[:8];record,reference=strategy.enroll(clean,source,key);name=scenarios[trial%len(scenarios)];observed,present,truth,identity=_scenario(clean,name,rng);output=strategy.verify(record,reference,observed,present,key,identity,thresholds);row=counts[name];row["identity_authenticated"]+=output.authenticated_source_identity
            for expected,predicted in zip(truth,output.fragment_states):
                row["state_counts"][predicted]+=1;row["uncertain"]+=predicted=="uncertain"
                if expected=="valid":row["true_valid"]+=1;row["false_manipulated"]+=predicted=="manipulated"
                elif expected=="manipulated":row["true_manipulated"]+=1;row["missed_manipulated"]+=predicted!="manipulated"
        for row in counts.values():row["false_manipulation_rate"]=row["false_manipulated"]/max(1,row["true_valid"]);row["missed_manipulation_rate"]=row["missed_manipulated"]/max(1,row["true_manipulated"]);row["uncertain_rate"]=row["uncertain"]/(trials//len(scenarios)*REGIONS)
        report["strategies"][strategy.name]={"thresholds":thresholds,"scenarios":counts}
    report["recommendation"]={"strategy":"registry-assisted robust perceptual digest","conditional":True,"reason":"only candidate with no added neural packet bits, direct local evidence, and measured benign calibration; requires an authenticated registry and empirical robustness validation","embedded_bits_per_region":20,"warning":"the 8-bit field remains a distributed global-authentication share, not an independent local MAC"};return report

