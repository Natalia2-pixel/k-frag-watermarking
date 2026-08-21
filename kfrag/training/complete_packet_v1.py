"""Protocol bridge and field-aware Stage-D training utilities."""
from __future__ import annotations
import hashlib,secrets
from pathlib import Path
import torch
from torch.nn import functional as F
from kfrag.crypto.token import ProvenanceToken
from kfrag.crypto.packets import RegionalPacket,create_packets
from kfrag.crypto.authentication import verify_tag

CAPACITIES=(12,20,28,36,44)

def ephemeral_key():return secrets.token_bytes(32)
def packet_bits(packet:RegionalPacket):
    values=[(packet.region_index>>s)&1 for s in range(3,-1,-1)]+[(packet.coded_symbol>>s)&1 for s in range(7,-1,-1)]
    values += [(byte>>s)&1 for byte in packet.authentication_tag for s in range(7,-1,-1)];return values
def bits_packet(bits):
    v=[int(x) for x in bits];index=sum(v[i]<<(3-i) for i in range(4));symbol=sum(v[4+i]<<(7-i) for i in range(8));tag=bytes(sum(v[12+i+j]<<(7-j) for j in range(8)) for i in range(0,32,8));return RegionalPacket(index,symbol,tag)
def fresh_packet_batch(count,key,generator):
    grids=[];tokens=[]
    for _ in range(count):
        issuer=int(torch.randint(0,1<<24,(1,),generator=generator));hi=int(torch.randint(0,1<<32,(1,),generator=generator));lo=int(torch.randint(0,1<<32,(1,),generator=generator));token=ProvenanceToken(issuer,(hi<<32)|lo,1);packets=create_packets(token,key);grids.append(torch.tensor([packet_bits(p) for p in packets]).reshape(4,4,44));tokens.append(token)
    return torch.stack(grids).float(),tokens
def field_losses(logits,bits,active_bits):
    fields={"index":(0,4),"symbol":(4,12),"tag":(12,active_bits)};losses={k:F.binary_cross_entropy_with_logits(logits[...,a:z],bits[...,a:z]) for k,(a,z) in fields.items() if z>a};losses["complete"]=F.binary_cross_entropy_with_logits(logits[...,:active_bits],bits[...,:active_bits]);return losses
def verify_decoded_grid(bits,key,token):
    packets=[bits_packet(bits.reshape(16,44)[i].tolist()) for i in range(16)];token_bytes=token.pack();return [verify_tag(key,token_bytes,p.region_index,p.coded_symbol,p.authentication_tag) for p in packets]
def stage_d_gates(m):
    return {"overall":m["overall_bit_accuracy"]>=.95,"index":m["index_bit_accuracy"]>=.98,"symbol":m["symbol_bit_accuracy"]>=.95,"tag":m["tag_bit_accuracy"]>=.95,
      "regions":min(m["per_region_accuracy"])>=.90,"bits":min(m["per_bit_accuracy"])>=.90,"shuffled_margin":m["correct_minus_shuffled_margin"]>=.40,"spatial_margin":m["correct_minus_spatial_margin"]>=.40,
      "original":.45<=m["original_image_accuracy"]<=.55,"leakage":m["cross_region_leakage"]<=.10,"wrong_key":m["wrong_key_acceptance_rate"]<=.01,"mutation":m["one_bit_mutation_acceptance_rate"]==0,
      "psnr":m["psnr"]>=35,"ssim":m["ssim"]>=.95,"saturation":m["residual_saturation_fraction"]<=.001,"analytical_zero":m["analytical_contribution"]==0,"disjoint":m["disjoint_images"],"blind":m["blind_decoder"],"secure_serialization":m["no_secret_or_expected_payload_serialized"]}
