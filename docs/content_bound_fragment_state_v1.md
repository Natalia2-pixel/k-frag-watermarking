# Content-bound fragment state v1

## Scientific boundary

This is a protocol design and controlled feature-vector simulation, not an image-attack benchmark. It preserves the locked local soft-decoder identity recovery result of **0.96875**, while keeping `neural_stage_passed=false` and `stage_e_permitted=false`. Globally authenticated identity recovery and local manipulation evidence are separate claims.

The verifier returns four distinct outputs: authenticated source identity; valid fragment evidence; missing fragment evidence; and manipulated versus uncertain local evidence. Missing means no observation was assigned. Manipulated requires successful source authentication plus strong content-bound contradiction. Everything else is uncertain. A packet disagreement alone is insufficient: channel decoding errors, benign distortion and malicious content changes can produce the same bit disagreement, so packet error has no reliable causal label.

## Strategies and capacity

### A. Registry-assisted robust local perceptual digest

A 64-bit projection digest is enrolled per region, included in a globally HMAC-authenticated registry record, and compared with the digest extracted from the questioned image. The neural packet remains 20 bits: index 4, RS symbol 8, distributed global-authentication share 8. Verification is blind to original pixels but requires the public source identifier, runtime HMAC key and registry digest. JPEG, resize and colour tolerance must be calibrated. Splice, overlay and replacement should increase digest distance, but perceptual collisions, same-source replay and collages remain weaknesses. Self-contained carriage would cost at least another 64 bits per region.

### B. Semi-fragile learned local-content signature

A fixed public learned extractor produces a 16-dimensional signature whose enrolled value is globally authenticated in the registry. It can remain within the 20-bit packet only in registry-assisted form; self-contained quantization costs roughly 16–128 extra bits per region. It may learn tolerance to JPEG, resize and mild colour changes while responding to overlays and replacements. Its security depends on extractor generalization and resistance to collisions or transfer attacks; it is not a cryptographic primitive.

### C. Authenticated cross-region parity syndromes

Quantized regional content bits form authenticated checks over a ring/grid parity graph. Violated incident checks identify suspect sets by global consistency, not independent authentication. Registry-assisted operation adds no neural bits; self-contained operation needs approximately 4–8 additional parity bits per region and cannot take those bits from the 8-bit authentication share without reducing the 64-bit aggregate authenticator. Benign quantization changes can propagate syndromes; coordinated replacements, even-number cancellations, replay and parity-preserving collages are weaknesses.

All three remain blind to the original image only because they use authenticated registry references. Crop synchronization is unresolved. An 8-bit regional value is never treated as strong independent authentication.

## Threat interpretation

The key and registry integrity are trusted. The adversary may erase, replace, splice, overlay, replay or collage fragments, but cannot forge the registry HMAC. The simulation uses Gaussian feature vectors and proxy perturbations only. Its rates are evidence about the decision rules under that synthetic model, not claims about JPEG, resizing, colour transforms or real image manipulation.

## Conditional recommendation

The registry-assisted robust perceptual digest is the only current candidate that offers direct local evidence while retaining the approximately 20-bit neural packet. This recommendation is conditional on real-image robustness, collision, replay and collage studies. Without a registry, none of the compared approaches fits the current capacity honestly.

## Falsifiable novelty hypothesis

**Hypothesis:** combining a threshold-recovered distributed global authenticator with registry-authenticated robust regional digests and an explicit four-state decision rule yields lower false-manipulation rates than packet-disagreement localization at the same 20-bit neural packet budget.

This is not an established novelty claim. Before any such claim, literature must be compared in: semi-fragile watermarking; content-dependent authentication; self-synchronizing watermarking; distributed authentication; tamper localization; and fragment and erasure decoding. The hypothesis is falsified if prior work already provides the combination, if real benign transforms exceed the calibrated false-manipulation target, or if splice/replacement manipulations remain mostly uncertain.
