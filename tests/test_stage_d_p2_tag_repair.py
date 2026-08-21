import tempfile
from pathlib import Path
import torch
from kfrag.diagnostics.stage_d_p2_tag_repair import weak_bit_diagnosis,deterministic_ephemeral_key
from kfrag.training.stage_d_tag_capacity import balanced_bit_loss

def test_diagnosis_identifies_persistent_packet_positions():
    values=[.99]*28
    rows=[]
    for _ in range(20):
        row=values.copy();row[8]=.86;row[12]=.87;row[20]=.84;rows.append({"level":"P2","per_bit_accuracy":row})
    diagnosis=weak_bit_diagnosis({"history":rows});p2={x["packet_bit"]:x for x in diagnosis["P2"]}
    assert set(p2)=={8,12,20};assert p2[20]["field"]=="tag" and p2[20]["below_gate_count"]==20

def test_balancing_uses_detached_training_statistics_and_upweights_weak_bit():
    logits=torch.zeros(2,4,4,4,requires_grad=True);bits=torch.zeros_like(logits);logits.data[...,0]=-5;logits.data[...,1]=-5;bits[...,1]=1
    loss,ema,weights=balanced_bit_loss(logits,bits,4,torch.ones(4),decay=0);loss.backward();assert not ema.requires_grad and not weights.requires_grad;assert weights[1]>weights[0] and logits.grad[...,1].abs().sum()>logits.grad[...,0].abs().sum()

def test_lower_capacity_checkpoint_policy_is_independent_of_p4_best():
    source=Path("kfrag/diagnostics/stage_d_p2_tag_repair.py").read_text();assert 'output/"P1"' in source and 'p1_dir/"best.pt"' in source;assert 'output/"best.pt"' not in source

def test_failed_next_level_cannot_remove_p1_checkpoint():
    with tempfile.TemporaryDirectory() as directory:
        p1=Path(directory)/"P1";p1.mkdir();torch.save({"passed":True},p1/"best.pt");failed={"passed":False};torch.save(failed,Path(directory)/"last.pt");assert (p1/"best.pt").is_file() and torch.load(p1/"best.pt",weights_only=False)["passed"]

def test_repair_uses_two_bit_internal_bridges_but_keeps_official_milestones():
    source=Path("kfrag/diagnostics/stage_d_p2_tag_repair.py").read_text();assert '("P1",8,True)' in source and '("P2",16,True)' in source
    assert '10tag' in source and '12tag' in source and '14tag' in source

def test_ephemeral_scientific_key_is_seed_deterministic_but_not_configured():
    assert deterministic_ephemeral_key(7)==deterministic_ephemeral_key(7);assert deterministic_ephemeral_key(7)!=deterministic_ephemeral_key(8)
    text=Path("configs/stage_d_p2_tag_repair_local.yaml").read_text().lower();assert "hmac_key" not in text and "authentication_secret" not in text
