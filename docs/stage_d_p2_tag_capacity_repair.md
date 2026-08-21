# Stage-D P2 tag-capacity repair

The blocked baseline is preserved at commit `c5654750702b1adf19c826ace84dd4b2e9697dbc`. Its histories show persistent weak packet positions 8 (retained RS), 12 (first P1 tag bit), and 20 (first newly introduced P2 tag bit); positions 12 and 20 reuse carrier channel zero in successive tag banks. This is stable per-bit imbalance and carrier competition, not merely terminal-step degradation.

The repair retains official P1=20 and P2=28 milestones but inserts 22-, 24-, and 26-bit bridges, adding two tag bits at a time. It computes balancing weights exclusively from an EMA of training BCE, distills every previously selected pathway, freezes the real-COCO 12-bit parent, and stops at P2. A passing P1 is saved independently under `P1/best.pt`; root `best.pt` remains withheld because P3/P4 are outside this repair.
