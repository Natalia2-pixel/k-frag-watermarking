# Large-scale regional digest reproduction v1

This independent real-COCO reproduction compares only the compact 64-bit-per-region DCT pHash and the 224-bit-per-region combined digest. Their implementations are frozen by the SHA-256 of the source file at commit `1448a9a4e23f00b0855fa8061b9bec5d75e5d7ae`. Thresholds are loaded at full precision from `regional_digest_frozen_thresholds_v1.json`, asserted against the preceding report, and never estimated from reproduction images.

The official COCO val2017 population is deterministically filtered by both identifier and SHA-256 to exclude all 100 prior calibration, validation and locked-test images. Exactly 1,000 previously unseen images are selected. The source image is the statistical unit. Reports contain macro image metrics and image-clustered bootstrap 95% confidence intervals; pooled region statistics are explicitly descriptive.

The issued representation remains RGB float `[0,1]`, resized bilinearly with antialiasing to 256×256. A fixed 4×4 grid gives sixteen 64×64 regions. Correspondence is assumed: this experiment does not validate synchronization or arbitrary crops.

Benign processing comprises clean repeatability; JPEG 95/85/75/60/40; WebP 95/80/60 when supported; resize-and-restore 0.75/0.5/0.25; blur; brightness, contrast and colour changes. JPEG 40 and resize 0.25 are predeclared extreme stress conditions and excluded only from the worst-standard-benign gate, while still being reported. Content changes comprise full replacement; 10/25/50/75% splice, overlay and occlusion; relocation; mixed collage; visually similar replacement; and same-source replacement.

The four-state rule is unchanged. Valid requires authenticated identity plus digest agreement; missing requires no assigned observation; manipulated requires authenticated identity plus strong contradiction; uncertain covers intermediate evidence. Global authentication is a separately simulated trusted prerequisite. Digest agreement never establishes identity.

The completed evidence consists of 20 manifest-hash-verified shards covering 1,000 unseen COCO images. Each 50-image shard stores an experiment manifest hash over configuration, population and frozen thresholds. Manifest-hash verification permits matching shards to resume and makes mismatched shards abort. Runtime, throughput, Python traced memory and process RSS are reported. The exact nearest unrelated digest search covers all reproduction regions and remains an empirical collision control, not adversarial collision proof.

Predeclared gates are mean standard benign FMR ≤0.02, worst standard benign FMR ≤0.05, aggregate malicious recall ≥0.90, 25% splice and overlay recall ≥0.85, zero clean failures and verification below 100 ms/image. Combined is selected only if all gates pass and its paired image-clustered small-area recall improvement CI is above zero. Otherwise DCT may be selected only if it passes. No failing gate is retuned.

No watermark model, neural checkpoint or soft decoder is executed or modified. `neural_stage_passed=false` and `stage_e_permitted=false`; no novelty is claimed.


## Mandatory population preflight

Before any digest, transformation, collision, or statistical calculation, the executable hashes the complete available COCO population and excludes every prior calibration, validation, and locked-test image by both identifier and SHA-256. It aborts with DataPopulationError if the configured reproduction count is below 1,000, fewer than the requested unseen images remain, or either overlap is nonzero. It writes and prints population_preflight.json containing the available unseen count, excluded prior count, selected count, both overlap counts, and the combined disjointness assertion.
