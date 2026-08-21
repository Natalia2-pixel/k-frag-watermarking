# Experimental isolated tag subspace

This local-only branch freezes the validated real-COCO 12-bit parent and constructs 32 tag feature carriers by projection and Gram–Schmidt removal of the complete 12-carrier parent subspace. The resulting declared maximum parent/tag carrier cosine is `1e-5`.

Tags use a separate blind decoder and a separate RGB residual projection. The tag residual consumes only a configured fraction of the remaining positive or negative residual room after the unchanged parent residual, so the combined residual remains bounded by 0.014 without rescaling the parent. P0 delegates exactly to the parent.

The curriculum is P0, P1, 22/24/26-bit internal bridges, P2, P3, and P4, with complete gates at every level and immediate failure stopping. Official lower-capacity checkpoints are preserved independently. No attacks, novelty mechanism, Kaggle support, or Stage E are included.
