"""Non-learned 4x4 regional perceptual digests and authenticated registry records."""
from __future__ import annotations
import hashlib,hmac,math,time
from dataclasses import dataclass
import numpy as np
import torch
from scipy.fft import dctn
from torch.nn import functional as F

GRID=4
REGIONS=16

def regions(image:torch.Tensor):
    if image.ndim!=3 or image.shape[0]!=3 or image.shape[1]!=image.shape[2] or image.shape[1]%4:raise ValueError("issued image must be RGB square and divisible by four")
    c=image.shape[-1]//4;return image.reshape(3,4,c,4,c).permute(1,3,0,2,4).reshape(16,3,c,c)
def _gray(region):return (.299*region[0]+.587*region[1]+.114*region[2]).unsqueeze(0).unsqueeze(0)
def _resize_gray(region,size):return F.interpolate(_gray(region),size=size,mode="bilinear",align_corners=False,antialias=True)[0,0].numpy()

class RegionalDigest:
    name="base";version=1;storage_bits_per_region=0
    def digest_region(self,region):raise NotImplementedError
    def digest_image(self,image):return tuple(self.digest_region(r) for r in regions(image))
    def distance(self,a,b):raise NotImplementedError
class DCTPerceptualHash(RegionalDigest):
    name="dct_phash";storage_bits_per_region=64
    def digest_region(self,region):
        coefficients=dctn(_resize_gray(region,(32,32)),norm="ortho")[:8,:8];median=np.median(coefficients.flatten()[1:]);return (coefficients>=median).flatten()
    def distance(self,a,b):return float(np.mean(np.asarray(a)!=np.asarray(b)))
class DifferenceHash(RegionalDigest):
    name="difference_hash";storage_bits_per_region=64
    def digest_region(self,region):
        sample=_resize_gray(region,(8,9));return (sample[:,1:]>=sample[:,:-1]).flatten()
    def distance(self,a,b):return float(np.mean(np.asarray(a)!=np.asarray(b)))
class LowFrequencyStatistics(RegionalDigest):
    name="low_frequency_statistics";storage_bits_per_region=96
    def digest_region(self,region):
        small=F.interpolate(region.unsqueeze(0),size=(2,2),mode="area")[0].permute(1,2,0).reshape(-1);return small.numpy().astype(np.float32)
    def distance(self,a,b):return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))
class CombinedDigest(RegionalDigest):
    name="combined_digest";storage_bits_per_region=224
    def __init__(self):self.dct=DCTPerceptualHash();self.diff=DifferenceHash();self.stats=LowFrequencyStatistics()
    def digest_region(self,region):return self.dct.digest_region(region),self.diff.digest_region(region),self.stats.digest_region(region)
    def distance(self,a,b):return .4*self.dct.distance(a[0],b[0])+.3*self.diff.distance(a[1],b[1])+.3*min(1.,self.stats.distance(a[2],b[2])*4)

@dataclass(frozen=True)
class RegionalDigestRecord:
    image_identifier:str;protocol_version:int;region_index:int;digest_type:str;digest_version:int;digest_value:bytes;authentication_tag:bytes
def serialize_digest(value):
    if isinstance(value,tuple):return b"".join(len(serialize_digest(x)).to_bytes(2,"big")+serialize_digest(x) for x in value)
    array=np.asarray(value);return (np.packbits(array.astype(np.uint8)).tobytes() if array.dtype==bool else np.round(array*255).clip(0,255).astype(np.uint8).tobytes())
def registry_message(image_identifier,protocol_version,region_index,digest_type,digest_version,digest_value):
    identifier=image_identifier.encode();kind=digest_type.encode();return b"KFRAG-REGIONAL-DIGEST"+len(identifier).to_bytes(2,"big")+identifier+bytes([protocol_version,region_index,len(kind)])+kind+bytes([digest_version])+len(digest_value).to_bytes(2,"big")+digest_value
def create_registry(image_identifier,image,digest,key,protocol_version=2):
    values=digest.digest_image(image);records=[]
    for index,value in enumerate(values):
        packed=serialize_digest(value);message=registry_message(image_identifier,protocol_version,index,digest.name,digest.version,packed);tag=hmac.new(key,message,hashlib.sha256).digest();records.append(RegionalDigestRecord(image_identifier,protocol_version,index,digest.name,digest.version,packed,tag))
    return tuple(records),values
def authenticate_registry(records,key):
    return all(hmac.compare_digest(record.authentication_tag,hmac.new(key,registry_message(record.image_identifier,record.protocol_version,record.region_index,record.digest_type,record.digest_version,record.digest_value),hashlib.sha256).digest()) for record in records)
def classify_distances(distances,present,identity_authenticated,valid_max,manipulated_min):
    states=[]
    for distance,exists in zip(distances,present):
        if not exists:states.append("missing")
        elif not identity_authenticated:states.append("uncertain")
        elif distance<=valid_max:states.append("valid")
        elif distance>=manipulated_min:states.append("manipulated")
        else:states.append("uncertain")
    return tuple(states)
def digest_candidates():return (DCTPerceptualHash(),DifferenceHash(),LowFrequencyStatistics(),CombinedDigest())

