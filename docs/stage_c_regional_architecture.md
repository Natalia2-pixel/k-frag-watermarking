# Stage-C regional-symbol channel

Stage C validates only sixteen simultaneous eight-bit regional symbols on a fixed 4×4 grid. It does not validate authentication tags, 44-bit packets, crop synchronization, geometric attacks, Reed–Solomon reconstruction, localization, or end-to-end provenance.

The mandatory Stage-B V2 learned-only encoder and decoder are loaded only after hash, metadata, tensor, preprocessing, mapping, finite-value, and round-trip checks. A shared regional router places each signed eight-bit carrier into one cell using smooth overlap-aware windows. Routed features enter the validated content-conditioned encoder at all existing payload fusion levels. Hidden activations remain SiLU and the final residual remains amplitude-times-mask-times-tanh.

The blind decoder receives questioned RGB pixels only. It partitions spatial features on the fixed grid and applies shared RGB and fixed-high-pass feature extraction and a shared regional head. It outputs raw `[B,4,4,8]` logits trained with `BCEWithLogitsLoss`; no global image pooling destroys region layout.

The gated curriculum verifies the parent checkpoint, cycles one active region, advances through 2/4/8/16 active regions with rollback, and evaluates all sixteen regions with no analytical contribution. `last.pt` is unconditional and `best.pt` is created only when every Stage-C gate passes. Stage D is never enabled automatically.
