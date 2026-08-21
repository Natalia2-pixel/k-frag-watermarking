# Stage-D complete regional-packet pilot

Stage D retains the validated Stage-C 4×4 spatial channel and its 0.014 bounded-residual operating point, but expands each shared regional route and blind regional output from 8 to 44 bits. The packet layout is index bits 0–3, RS symbol bits 4–11, and 32 truncated-HMAC bits 12–43. No global pooling or independent per-region network is used.

The protocol is unchanged. A 96-bit `ProvenanceToken` is RS(16,12) encoded. Each tag authenticates `token.pack() || region_index || coded_symbol` with HMAC-SHA256 truncated to four bytes. The key is ephemeral process memory and is excluded from configuration, reports, logs, and checkpoints.

The gated capacity sequence is 12, 20, 28, 36, then 44 bits. Future bits are masked out of both carrier routing and loss. Final evaluation is learned-only. Stage E is always disabled.

The exact-packet pilot gate is 0.10: the mandatory 0.95 bit-accuracy floor implies an independent-error exact yield of `0.95^44 = 0.1047`. Both exact packet recovery and HMAC-valid packet fraction must meet this floor; their full values are always reported.
