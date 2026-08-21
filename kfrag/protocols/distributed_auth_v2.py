"""Protocol-only simulation of distributed authentication for 16 fragments."""
from __future__ import annotations
import hashlib,hmac,math,random
from dataclasses import dataclass
from collections import Counter
from reedsolo import RSCodec,ReedSolomonError
from kfrag.crypto.reed_solomon import encode as encode_token,reconstruct as reconstruct_token
from kfrag.crypto.token import ProvenanceToken

REGIONS=16
@dataclass(frozen=True)
class AuthFragment:
    index:int;symbol:int;share:int;share_bits:int=8
    def __post_init__(self):
        if not 0<=self.index<16 or not 0<=self.symbol<256 or not 0<=self.share<(1<<self.share_bits):raise ValueError("invalid fragment field")

def _mac(key,message,bits):
    value=int.from_bytes(hmac.new(key,message,hashlib.sha256).digest(),"big")>>(256-bits);return value
def canonical_fragment_message(source_id,token,index,symbol,version=2):return b"KFRAG-IF"+bytes([version])+len(source_id).to_bytes(2,"big")+source_id+token.pack()+bytes([index,symbol])
def canonical_global_message(source_id,token,symbols,version=2):return b"KFRAG-GLOBAL"+bytes([version])+len(source_id).to_bytes(2,"big")+source_id+token.pack()+bytes(range(16))+bytes(symbols)
def _unique(fragments):
    grouped={};duplicates=set()
    for fragment in fragments:
        if fragment.index in grouped:duplicates.add(fragment.index)
        else:grouped[fragment.index]=fragment
    return grouped,duplicates

class IndependentMAC:
    def __init__(self,bits):
        if bits not in (8,12,16):raise ValueError("independent MAC width must be 8, 12, or 16")
        self.bits=bits
    def issue(self,token,source_id,key):
        symbols=encode_token(token.pack());return [AuthFragment(i,s,_mac(key,canonical_fragment_message(source_id,token,i,s),self.bits),self.bits) for i,s in enumerate(symbols)]
    def verify(self,fragments,token,source_id,key):
        unique,duplicates=_unique(fragments);states={i:"missing" for i in range(16)}
        for i,f in unique.items():states[i]="valid" if i not in duplicates and hmac.compare_digest(f.share.to_bytes((self.bits+7)//8,"big"),_mac(key,canonical_fragment_message(source_id,token,f.index,f.symbol),self.bits).to_bytes((self.bits+7)//8,"big")) else "manipulated"
        return states
    def one_forge_probability(self):return 2**-self.bits
    def all_forges_probability(self,count):return 2**(-self.bits*count)
    def any_forge_probability(self,count):return 1-(1-2**-self.bits)**count

class DistributedGlobalMAC:
    """A global tag split raw (128-bit) or RS(16,8)-coded (64-bit)."""
    def __init__(self,tag_bits=64):
        if tag_bits not in (64,128):raise ValueError("global tag must be 64 or 128 bits")
        self.tag_bits=tag_bits;self.minimum_shares=8 if tag_bits==64 else 16
    def _shares(self,token,source_id,key):
        symbols=encode_token(token.pack());tag=_mac(key,canonical_global_message(source_id,token,symbols),self.tag_bits).to_bytes(self.tag_bits//8,"big")
        return bytes(RSCodec(8).encode(tag)) if self.tag_bits==64 else tag
    def issue(self,token,source_id,key):
        symbols=encode_token(token.pack());shares=self._shares(token,source_id,key);return [AuthFragment(i,s,shares[i]) for i,s in enumerate(symbols)]
    def verify(self,fragments,token,source_id,key):return classify_against_expected(fragments,self._shares(token,source_id,key),encode_token(token.pack()))
    def reconstruct_authenticator(self,fragments):
        unique,duplicates=_unique(fragments);usable=[f for i,f in unique.items() if i not in duplicates]
        if len(usable)<self.minimum_shares:raise ValueError("insufficient authentication shares")
        if self.tag_bits==128:
            if len(usable)!=16:raise ValueError("128-bit raw split requires all 16 shares")
            return bytes(unique[i].share for i in range(16))
        codeword=bytearray(16);present=set()
        for f in usable:codeword[f.index]=f.share;present.add(f.index)
        try:return bytes(RSCodec(8).decode(bytes(codeword),erase_pos=[i for i in range(16) if i not in present])[0])
        except ReedSolomonError as exc:raise ValueError("authentication shares are inconsistent") from exc
    def aggregate_forge_probability(self):return 2**-self.tag_bits

class JointFragmentCode(DistributedGlobalMAC):
    """64-bit global MAC bound to source, token, version, indices and all symbols."""
    def __init__(self):super().__init__(64)
    def recover_and_verify(self,fragments,source_id,key):
        unique,duplicates=_unique(fragments);usable=[(i,f.symbol) for i,f in unique.items() if i not in duplicates]
        if len(usable)<12:return {"status":"insufficient","token":None,"states":{i:("manipulated" if i in duplicates else "missing" if i not in unique else "unverified") for i in range(16)}}
        try:token=ProvenanceToken.unpack(reconstruct_token(usable))
        except ValueError:return {"status":"manipulated","token":None,"states":{i:("missing" if i not in unique else "manipulated") for i in range(16)}}
        states=self.verify(fragments,token,source_id,key);valid=sum(x=="valid" for x in states.values());status="valid" if valid>=12 and not any(x=="manipulated" for x in states.values()) else "manipulated"
        return {"status":status,"token":token,"states":states}

def classify_against_expected(fragments,expected_shares,expected_symbols=None):
    unique,duplicates=_unique(fragments);states={i:"missing" for i in range(16)}
    for i,f in unique.items():states[i]="valid" if i not in duplicates and f.share==expected_shares[i] and (expected_symbols is None or f.symbol==expected_symbols[i]) else "manipulated"
    return states

def monte_carlo(construction,trials=1000,seed=1):
    rng=random.Random(seed);key=hashlib.sha256(f"key:{seed}".encode()).digest();accepted=Counter()
    for trial in range(trials):
        token=ProvenanceToken(trial%(1<<24),rng.getrandbits(64),2);source=hashlib.sha256(f"source:{trial}".encode()).digest()[:8];fragments=construction.issue(token,source,key);case=trial%6
        if case==0: # random bit error
            j=rng.randrange(16);f=fragments[j];fragments[j]=AuthFragment(f.index,f.symbol^(1<<rng.randrange(8)),f.share,f.share_bits)
        elif case==1:fragments.pop(rng.randrange(16))
        elif case==2:rng.shuffle(fragments)
        elif case==3:fragments.append(fragments[rng.randrange(16)])
        elif case==4:
            other=construction.issue(ProvenanceToken((trial+1)%(1<<24),rng.getrandbits(64),2),source,key);fragments[8:]=other[8:]
        else:
            j=rng.randrange(16);f=fragments[j];fragments[j]=AuthFragment(f.index,f.symbol,rng.randrange(1<<f.share_bits),f.share_bits)
        if isinstance(construction,JointFragmentCode):result=construction.recover_and_verify(fragments,source,key);accepted[(case,result["status"])]+=1
        else:
            states=construction.verify(fragments,token,source,key);accepted[(case,"accepted" if all(x in ("valid","missing") for x in states.values()) else "rejected")]+=1
    return {f"case_{case}_{state}":count for (case,state),count in sorted(accepted.items())}

def construction_comparison():
    rows=[]
    for bits in (8,12,16):
        c=IndependentMAC(bits);rows.append({"construction":f"independent_mac_{bits}","bits_per_region":12+bits,"total_authentication_bits":16*bits,"minimum_auth_fragments":1,"token_recovery_fragments":12,"one_forged_fragment_acceptance":c.one_forge_probability(),"forged_identity_acceptance_at_12":c.all_forges_probability(12),"localization":"independently authenticated after candidate identity lookup","complexity":"O(n)"})
    rows.extend([{"construction":"global_hmac128_raw_split","bits_per_region":20,"total_authentication_bits":128,"minimum_auth_fragments":16,"token_recovery_fragments":12,"one_forged_fragment_acceptance":2**-8,"forged_identity_acceptance_at_12":None,"forged_identity_acceptance_at_16":2**-128,"localization":"inferred by comparison with recomputed global codeword","complexity":"O(n)"},{"construction":"global_hmac64_rs16_8","bits_per_region":20,"total_authentication_bits":128,"minimum_auth_fragments":8,"token_recovery_fragments":12,"one_forged_fragment_acceptance":2**-8,"forged_identity_acceptance_at_12":2**-64,"localization":"inferred by global codeword consistency","complexity":"O(n)+RS decoding"},{"construction":"joint_fragment_code64_rs16_8","bits_per_region":20,"total_authentication_bits":128,"minimum_auth_fragments":8,"token_recovery_fragments":12,"one_forged_fragment_acceptance":2**-8,"forged_identity_acceptance_at_12":2**-64,"localization":"jointly authenticated after identity reconstruction","complexity":"O(n)+two RS decodes"}]);return rows
