# Real-image regional digest feasibility v1

This experiment evaluates non-learned regional perceptual digests on 100 real COCO images. It does not execute or retrain a watermark model. Images are deterministically divided into 40 calibration, 30 threshold-selection validation and 30 locked final-test images. The final population is evaluated once after all digest definitions and thresholds are fixed.

The issued representation is RGB, float `[0,1]`, resized to 256×256 by the existing COCO loader. Its fixed 4×4 layout yields sixteen 64×64 regions. Region correspondence is assumed; crop synchronization is not solved.

For each issued image and region, the registry binds the image identifier, protocol version, region index, digest type, digest version and digest bytes with HMAC-SHA256. The runtime key is generated deterministically in memory for scientific reproducibility and is never serialized. Global identity authentication is an explicit trusted prerequisite and is not inferred from digest agreement.

Compared digests are a 64-bit DCT perceptual hash, 64-bit difference hash, 96-bit low-frequency RGB statistic, and 224-bit combined digest. These are perceptual descriptors, not cryptographically collision-resistant hashes.

Benign processing includes JPEG qualities 90/75/60, downscale-and-restore factors 0.75/0.5, mild brightness/contrast/colour changes, Gaussian blur, and bounded sinusoidal watermark-strength proxies. Content-changing populations include full replacement, 25/50/75% splice, 25/50/75% overlay/occlusion, same-image relocation and four-region mixed-source collage. These categories remain separate.

The four-state rule is unchanged: valid requires authenticated identity and digest agreement; missing requires no assigned observation; manipulated requires authenticated identity and strong digest contradiction; uncertain covers the calibrated gap and unauthenticated evidence. Collision-oriented nearest-unrelated, visually similar unrelated and same-source controls are empirical only, not adversarial security proof.

A digest is conditionally recommended only when the locked test has mean benign false-manipulation at most 0.05, mean content-change miss rate at most 0.25 and aggregate manipulated F1 at least 0.75. This is a feasibility threshold, not a publication or security claim. `neural_stage_passed=false` and `stage_e_permitted=false` remain fixed.
