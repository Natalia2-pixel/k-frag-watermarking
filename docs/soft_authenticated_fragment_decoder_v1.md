# Blind soft authenticated fragment decoder v1

This decoder-only local experiment preserves the failed hard-threshold feasibility result and freezes the selected step-450 neural candidate. It changes no encoder, carrier, parent decoder, residual, amplitude, or neural gate.

Candidate generation consumes only each region's 20 raw logits. Bounded bitwise beams produce index, RS-byte, and authentication-share candidates with log likelihoods and confidence gaps. A deterministic likelihood assignment resolves unordered and duplicate index observations. RS(16,12) token decoding combines hard decoding, confidence erasures, and bounded symbol candidates. RS(16,8) reconstructs candidate authenticators. Only then does exact HMAC comparison occur for caller-supplied public registry/candidate-source identifiers. No expected token or packet enters the API.

The decoder fails closed on no match, ambiguity, insufficient evidence, or budget exhaustion. Region states remain distinct: valid, missing, manipulated, and uncertain. The hard decoder remains the baseline.

The proposition that blind soft list decoding plus global authentication can recover identities that hard decoding rejects is a **falsifiable novelty hypothesis requiring literature comparison**, not an established novelty claim.
