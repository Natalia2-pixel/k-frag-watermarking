# Stage-D 12-bit transition repair

This repair preserves the blocked complete-packet pilot as a baseline. Positions 4–11 directly use the validated Stage-C router, encoder, and blind decoder. R0 is an exact parent call. Four new index carriers and a blind index head are introduced one bit at a time in R1–R4 while the Stage-C pathway remains frozen and its logits are distilled.

Shuffled margin is calculated on randomized RS bits only, so deterministic grid indices cannot inflate it. Original-image chance accuracy is likewise measured against randomized RS targets, with index behavior reported separately. A successful R4 stops and creates a separately reviewable D1 checkpoint; it never enters tag-capacity training or enables Stage E.
