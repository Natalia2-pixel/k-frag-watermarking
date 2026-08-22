# Regional digest failure decomposition v1

This is a read-only analysis of the frozen 1,000-image reproduction. It does not rerun transformations, select thresholds, train neural components, or modify shard evidence. The source image remains the independent statistical unit; all confidence intervals use image-clustered bootstrap resampling.

The analysis preserves scientific_status=blocked_by_large_scale_regional_digest_gates, selects no candidate, keeps neural_stage_passed=false, and keeps stage_e_permitted=false. Fixed 4x4 correspondence remains assumed and crop synchronization remains unvalidated.

The visually-similar generator is audited by deterministically reconstructing only its recorded selection procedure from the existing image manifest. No transformation result is recalculated. The smallest recommended next experiment must use genuinely cross-image, SHA-distinct, pixel-different replacements on a new locked population and compare current frozen digests with one registry-backed multiscale/deep-feature baseline. This is a design recommendation, not a novelty claim.
