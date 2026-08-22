# Blind crop threshold recovery: incomplete repaired run

Scientific status: **incomplete; no repaired crop result is claimed**.

The repaired output root is:

`outputs/blind_crop_threshold_recovery_v1/local_coco_repaired`

At the usage checkpoint, the experiment process had exited before writing any
atomic selection-validation or locked-final shard:

- selection-validation shards: 0 of 8;
- locked-final shards: 0 of 8;
- repaired locked-final marker: absent;
- repaired final report: absent.

Consequently there is no repaired locked-final population to analyse. The
earlier directory
`outputs/blind_crop_threshold_recovery_v1/invalid_mixed_allocation_attempt`
remains quarantined with `INVALID_SCIENTIFIC_EVIDENCE.txt`; its observations
must not be used as scientific evidence.

The repaired configuration declares
`allocation_version: stratified_population_v1`. The allocation version is part
of the shard manifest hash, so a resumed run rejects shards made under a
different allocation strategy. No valid repaired shard existed at this
checkpoint, so there is no partial observation to promote or discard.

Resume from the repository root with exactly:

```powershell
python scripts/run_blind_crop_threshold_recovery.py --config configs/blind_crop_threshold_recovery_v1.yaml
```

The run must complete all eight selection-validation shards, all eight
locked-final shards, and write the locked-final marker before any final crop
result is claimed.

The fixed scientific boundaries remain:

- `neural_stage_passed=false`;
- `stage_e_permitted=false`;
- no Stage-E authorization;
- no promotion of the selected step-450 diagnostic candidate.
