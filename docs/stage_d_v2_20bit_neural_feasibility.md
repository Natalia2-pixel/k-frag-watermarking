# Stage-D v2 20-bit neural feasibility

This local-only stage tests one transition: the immutable validated 12-bit parent to a 20-bit regional packet. It does not implement the former 28/36/44-bit progression, attacks, Kaggle execution, Stage E, or a novelty claim.

The regional layout is index bits 0–3, frozen-parent RS bits 4–11, and bits 12–19 containing one byte of the RS(16,8) codeword of the global 64-bit HMAC defined by `distributed_auth_v2`. Training targets are issued by `JointFragmentCode`; they are not independent random tag bytes. Fresh tokens and source identifiers are generated on every occurrence. The deterministic scientific key is derived in memory from the run seed and is excluded from checkpoints and reports.

P0 requires bit-identical residual, watermarked image, and parent logits. The parent is frozen and audited for zero updates. P1 trains only the existing shared tag carrier/head at amplitude 0.014. Selection uses a disjoint validation population; a third disjoint final-test population is untouched by training, balancing, early stopping, and checkpoint selection.

Protocol evaluation converts blind-decoder logits to hard packet bits before constructing fragments. Recovery is never credited from issuance bits. It separately measures token reconstruction, 64-bit authenticator reconstruction, authenticated acceptance, fragment states, duplicates, corruption, mixed identities, erasures, and insufficient evidence. `best.pt` exists only when every neural gate passes; `last.pt` is unconditional. Stage E remains disabled.
