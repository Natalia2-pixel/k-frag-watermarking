# Stage-B V2 natural-image channel

Stage-B V2 is isolated from the immutable Stage-A, Stage-B V1, and COCO-pilot baselines. Its scope is exactly eight independently represented regional-symbol bits. Region-index bits, authentication tags, full packets, Reed–Solomon reconstruction, localization, and provenance verification are excluded.

The encoder maps canonical `[B,8]` bits from `{0,1}` to `{-1,+1}` and multiplies them by eight learned spatial bases. RGB and payload stems fuse at full and half resolution in a compact residual U-Net. A differentiable bounded strength mask modulates `amplitude * mask * tanh(raw_residual)`; the image is clamped to `[0,1]`.

The blind decoder accepts only questioned RGB pixels. Separate RGB and fixed high-pass branches (Laplacian, horizontal/vertical differences, and image-minus-average-blur) fuse after initial extraction. Multi-scale residual blocks, global average pooling, and one linear layer produce exactly eight raw logits. Hidden blocks use SiLU by default (GELU is a configured ablation); the residual head uses tanh; decoder output has no activation; communication uses `BCEWithLogitsLoss`.

The curriculum is analytical decoder warm-up, analytical-to-learned transition, then learned-only joint training. Passing is impossible unless the final analytical weight is zero. A 32/16 seeded, disjoint population is used with fresh independent payloads. Checkpoints contain hashed relative identifiers and never evaluation payloads, pixels, or authentication material. `last.pt` is unconditional; `best.pt` is gate-only. Stage C remains blocked unless every repair-pilot gate passes.

Stage-A carrier/decoder names, shapes, dtypes, and preprocessing are audited. V2 is architecturally and semantically different, so incompatible optimizer or model state is not loaded; Stage A remains controlled prerequisite evidence.
