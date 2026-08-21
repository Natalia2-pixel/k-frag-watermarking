# Soft authenticated fragment decoder v2

V2 is a decoder-only local feasibility experiment. Neural weights, carriers, residual amplitude, neural gates, and the hard baseline are immutable. Development, selection-validation, and locked-final logit populations are disjoint. The locked final set is constructed and evaluated once only after parameter selection.

The decoder uses calibrated log likelihoods, maximum-likelihood index assignment, bounded systematic-token and RS(16,12) lists, adaptive confidence erasures, bounded RS(16,8) authenticator reconstruction, and exact HMAC verification against caller-supplied public candidate-source identifiers. Candidate generation cannot receive oracle truth. Ground truth appears only in post-decision coverage/failure diagnosis.

Identity recovery, fragment validity, and manipulation localization are reported separately. Uncertain evidence is never merged into manipulated evidence. Search exhaustion, ambiguity, insufficient evidence, and absence of an exact HMAC match fail closed.

The method remains a **falsifiable novelty hypothesis requiring literature comparison**, not an established novelty claim. Stage E remains disabled.
