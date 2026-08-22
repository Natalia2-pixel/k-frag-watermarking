# Regional digest failure decomposition v1 - concise report

This read-only analysis consumed the committed report and 20 manifest-hash-verified shards covering 1,000 unseen COCO images. Source report and shard SHA-256 values were identical before and after analysis. The image, not its 16 regions, is the bootstrap unit. The complete transform, area, family, source-relation, confidence-interval, miss-contribution, uncertainty-contribution, mechanism, and representative-example tables are in regional_digest_failure_decomposition_v1_report.json.

## Frozen conclusion

- scientific_status=blocked_by_large_scale_regional_digest_gates
- selected_candidate=none
- neural_stage_passed=false
- stage_e_permitted=false

| Digest | Aggregate recall | Gate deficit | Missed altered regions | Uncertain altered regions |
|---|---:|---:|---:|---:|
| DCT pHash | 0.809824 | 0.090176 | 3,233 | 2,070 |
| Combined | 0.849529 | 0.050471 | 2,558 | 1,481 |

Overlay accounts for 33.13% of DCT misses and 46.23% of its uncertainty, and 32.64% of Combined misses and 51.92% of its uncertainty. At 25% area, DCT overlay recall is 0.686 (image-clustered 95% CI [0.651, 0.716]), 0.164 below gate; Combined is 0.834 [0.810, 0.856], 0.016 below gate. This independent failure remains real.

## Visually-similar generator audit

The selector used nearest Euclidean regional mean-RGB distance within each 50-image shard's 50 sources plus a shifted duplicate donor list. It excluded the same pool position but not duplicate underlying identifiers.

- 980/1,000 selections were the same identifier, SHA-256, region, and pixels; RGB distance and pixel MAE were zero.
- Only 20 were genuinely cross-image and pixel-different.
- Cross-image RGB distance: mean 0.028883, median 0.020144, range [0.007908, 0.136697].
- Cross-image pixel MAE: mean 0.199800, median 0.205843, range [0.040948, 0.362544].
- Both digests detected 20/20 cross-image cases, but this selector-biased n=20 cannot validate general similar-replacement sensitivity.
- The nominal condition contributes 0.057647 absolute aggregate missed recall. Correcting only it would put DCT at 0.867471 (still failing) and Combined at 0.907176 (aggregate passing), but Combined would still fail the unchanged 25% overlay gate.

This is a generator-validity defect, not evidence that either digest misses 98% of genuine similar cross-image replacements. Completed gates are not redefined.

## Integrity objectives

Neither digest supports exact-content integrity because both are lossy and tolerate benign pixel changes. Both partially support perceptual integrity: benign gates pass, and Combined is stronger on 25% splice/overlay, but frozen gates still fail. Neither supports semantic integrity because neither represents objects, identity, text, layout, or meaning.

## Decision

The smallest next experiment is a new disjoint locked population with a corrected cross-image selector enforcing different identifier, different SHA-256, and nonzero pixel difference. Predeclare perceptual/semantic similarity strata using authenticated registry-stored multiscale features, and compare the two frozen digests with one local deep-feature baseline. Do not retune thresholds.

Mechanisms considered, without implementation: robust-plus-fragile dual digest, canonicalized cryptographic hashing, local deep-feature similarity, semi-fragile learned signatures, object/text/layout consistency, and registry multiscale features. Their robustness, storage, runtime, external evidence, weaknesses, 20-bit compatibility, and crop-synchronization requirements are recorded in the JSON report.

This is a design recommendation, not a novelty claim. Fixed 4x4 correspondence remains assumed; crop synchronization remains unvalidated.
