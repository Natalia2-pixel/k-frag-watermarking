# Observation-space tag isolation feasibility

The observation operator is the exact blind matched-filter path used here: regional RGB residual, fixed Laplacian/horizontal/vertical/image-minus-average responses, flattening, L2 normalization, and normalized inner-product pooling. Parent RGB patterns are mapped through that operator; SVD constructs a numerically independent parent observation basis and tag RGB preimages are projected from it. A second projection controls float32 Gram–Schmidt drift.

The validated 12-bit parent remains frozen and P0 delegates to it exactly. Tags use a separate blind decoder. Their residual consumes a fixed fraction of the remaining signed 0.014 budget. Training constrains parent-logit drift and penalizes decoder-gradient cosine above 0.20. This local feasibility run executes only P0 and P1; no attacks, Kaggle support, novelty claim, later capacity, or Stage E are included.
