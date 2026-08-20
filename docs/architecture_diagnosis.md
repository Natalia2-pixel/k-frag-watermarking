# Learned-channel architectural diagnosis

V1 and V2 are retained as failed baselines. Their positive payload-sensitivity
checks only establish that changing a payload changes the encoder residual and
therefore some decoder logits. They do not establish recoverable communication.
The decoder can remain dominated by image content while still being sensitive.

The V2 fidelity change reduced residual energy and raised PSNR to 44.85 dB, but
it did not create spatially separable bit carriers or an adequate signal-to-
interference ratio. Consequently an eight-bit symbol is still effectively a
uniform guess: exact accuracy is approximately 1/256 even when some individual
bit metrics fluctuate above 0.5. Normalization and clamping can further attenuate
the weak residual, and full-packet training spreads scarce capacity over 44 bits.

The replacement path therefore starts with only the eight RS-symbol bits. It
uses distinct local carriers, an image-conditioned residual refinement path,
scheduled residual amplitude, a blind receptive-field decoder, raw-logit BCE,
gradient clipping, and explicit fidelity and saturation penalties. Progression
is gated by synthetic communication, fresh-payload overfit, then disjoint 32/16
image evaluation. These diagnostics validate implementation prerequisites only;
they establish neither natural-image robustness nor novelty.
