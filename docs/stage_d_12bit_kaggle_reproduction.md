# Independent real-COCO Stage-D 12-bit reproduction

This run stops after R4. It cannot train tag bits and always records `stage_e_permitted=false`.

Required Kaggle inputs are the repository checkout, the Stage-C final evidence dataset containing `kaggle_fidelity_repair_v1/best.pt` and `report.json`, the Stage-B V2 evidence needed to instantiate the compatible parent architecture, and COCO 2017 `val2017` images. Every path can be overridden on the runner CLI.

The setup cell copies the read-only repository into `/kaggle/working/kfrag-stage-d` and installs it editable. The verification cell checks the configured inputs, exact Stage-C hash and report before optimization. The execution uses 128/64 disjoint COCO images and 256 final samples.

The rollback-v2 repair evaluates the complete applicable gate set at every checkpoint, retains the best passing candidate, requires two successful evaluations, and stops after ten stale evaluations. It uses a fresh optimizer and cosine schedule per R-level, an index loss weight of 2.0, and a 3e-4 learning rate. The output directory is separate from the failed COCO attempt.
