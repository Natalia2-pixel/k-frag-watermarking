# Distributed fragment authentication v2 — protocol design report

## Scope and prerequisites

This is a protocol simulation, not a neural or image-security result. A decoded fragment is `(region_index, RS_symbol, authentication_share)`. The intended neural width is 20 bits: 4 index, 8 symbol, and 8 authentication-share bits. Authentication uses an externally held key. The verifier does **not** know the expected payload from the decoder.

For blind verification it first needs at least 12 indexed symbols to reconstruct the 96-bit token under RS(16,12), and then needs a public `source_id` supplied by the application or a bounded registry/candidate-source lookup. The token, source identifier, protocol version, all indices, and all sixteen coded symbols are input to the recommended global MAC. With `C` registry candidates, verification costs `O(Cn)` MAC/codeword comparisons.

## Information bound

Sixteen 8-bit shares carry at most 128 bits. Therefore a 128-bit global authenticator using exactly eight bits per region has no information-theoretic room for erasure redundancy: all 16 shares are required. A k-of-16 encoding of a 128-bit tag with `k<16` requires more than eight bits per region or a smaller authenticator. This is a guaranteed counting result, not a simulation observation.

## A — independent truncated MACs

Each region authenticates `domain || version || source_id || token || index || symbol` independently. Exact probabilities under the PRF assumption are:

| Tag | Bits/region total | One forged region | All 4 forged | All 8 forged | All 12 forged | All 16 forged |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 20 | 2^-8 | 2^-32 | 2^-64 | 2^-96 | 2^-128 |
| 12 | 24 | 2^-12 | 2^-48 | 2^-96 | 2^-144 | 2^-192 |
| 16 | 28 | 2^-16 | 2^-64 | 2^-128 | 2^-192 | 2^-256 |

The probability that at least one of `m` independently forged fragments is accepted is `1-(1-2^-b)^m`. Thus an ordinary 8-bit MAC has only 1/256 single-fragment resistance and is not strong independent authentication, regardless of the much smaller probability of forging many fragments simultaneously.

Advantages: immediate per-fragment states after candidate lookup, unordered arrival, and independent localization. Missing fragments are explicit. Disadvantages: weak single-fragment security at 8 bits; 12/16-bit variants exceed the near-20-bit target. Replay remains valid without an external freshness registry.

## B — distributed global HMAC

Two variants are simulated:

1. A 128-bit HMAC split into sixteen raw bytes. It provides 2^-128 aggregate forgery resistance but requires all 16 shares. One missing share prevents authentication. Localization is inferred by comparing against a recomputed candidate codeword; an isolated random share still matches with probability 2^-8.
2. A 64-bit HMAC encoded as RS(16,8), producing sixteen one-byte shares. The authentication code tolerates eight erasures, four errors, or combinations `2e+s<=8`. Aggregate forgery probability is 2^-64. Token recovery remains the bottleneck at RS(16,12): four erasures, two errors, or `2e+s<=4` before candidate-assisted filtering.

Both bind a global message, but the second deliberately provides 64—not 128—bits of aggregate authentication.

## C — jointly authenticated fragment code

The recommended construction is the 64-bit joint code:

```text
M = "KFRAG-GLOBAL"
    || protocol_version
    || source_id_length || source_id
    || 96-bit ProvenanceToken
    || canonical indices 0..15
    || all 16 RS(16,12) symbols

T = Trunc64(HMAC-SHA256(K, M))
shares[0..15] = RS(16,8).encode(T)
fragment[i] = i || symbol[i] || shares[i]
```

It uses 20 bits per region and 128 transmitted authentication-share bits. At least eight shares reconstruct `T`, but at least twelve valid symbol fragments are needed to reconstruct the token. A forged reconstructed identity is accepted with probability at most 2^-64 under the PRF assumption and correct candidate binding. A random forged share matches its expected byte with probability 2^-8, but a single matching byte is never treated as identity authentication.

After token reconstruction and source lookup, the verifier recomputes the entire expected symbol/share codeword. It reports every position as `valid`, `missing`, or `manipulated`. Localization is jointly authenticated/inferred from global consistency, not independently authenticated by each byte. Duplicate indices are manipulated. Shuffling is harmless. Mixed-image fragments, wrong sources, symbol changes, and share changes are inconsistent.

If ordinary RS token decoding fails because too many symbols are corrupt, exact localization is unavailable without candidate-token lookup. With a candidate registry, expected shares/symbols can be compared first, after which any 12 valid symbols recover the token. Without a registry, the initial RS(16,12) bound applies.

## Comparison

| Construction | Region bits | Auth bits/image | Auth threshold | Token threshold | Auth error/erasure bound | Forged identity | Localization |
|---|---:|---:|---:|---:|---|---:|---|
| Independent MAC-8 | 20 | 128 | 1 per local check | 12 | Per-fragment | 2^-96 for 12 simultaneous forgeries; only 2^-8 for one | Independent after candidate lookup |
| Independent MAC-12 | 24 | 192 | 1 | 12 | Per-fragment | 2^-144 at 12 | Independent |
| Independent MAC-16 | 28 | 256 | 1 | 12 | Per-fragment | 2^-192 at 12 | Independent |
| Global HMAC-128 raw | 20 | 128 | 16 | 12 | No auth erasures | 2^-128 | Global inference |
| Global HMAC-64 RS(16,8) | 20 | 128 | 8 | 12 | `2e+s<=8` auth; `2e+s<=4` token | 2^-64 | Global inference |
| Joint code-64 RS(16,8) | 20 | 128 | 8 | 12 | Same bounds | 2^-64 | Joint consistency |

Subset search is unnecessary for the implemented bounded-error path: two RS decodes plus linear codeword comparison suffice. Candidate-registry mode is `O(Cn)`. Exponential subset search would only be needed for an unbounded adversarial mixture without candidate identities, and is not recommended.

## Simulation results and guarantees

Mathematical guarantees, assuming ideal HMAC/PRF behavior and correct key separation:

- exact forgery probabilities above;
- RS error/erasure bounds;
- the 128-bit/no-redundancy information bound;
- deterministic canonical-message binding.

The 10,000-trial-per-construction simulation exercises random symbol errors, erasures, shuffling, duplicate indices, mixed valid identities, and random forged shares. Exhaustive tests enumerate all 256 possible 8-bit shares, all erasure subsets and permutations of a four-fragment case, and every two-source split in that case. These validate the implementation paths; they do not prove HMAC security.

## Assumptions and unresolved weaknesses

- HMAC-SHA256 is a secure PRF and keys remain secret and domain-separated.
- The registry maps a reconstructed token or supplied source identifier to the correct candidate namespace.
- Neural decoding errors fit the stated RS bounds.
- Replay is **not detected** without an external freshness/issuance registry, epoch, revocation state, or application nonce.
- The 64-bit joint code meets the requested minimum but not preferred 128-bit aggregate security.
- A single share has only eight bits of coincidence resistance and must never be accepted independently.
- If more than two unknown token symbols are corrupted and no candidate lookup is available, token reconstruction/localization may fail.
- Content-bound anti-splicing, image manipulation localization, geometric synchronization, and neural communication remain outside this protocol simulation.

## Recommendation

The joint 64-bit RS(16,8) fragment code is the only evaluated construction recommended for a future neural feasibility test at 20 bits per region. The recommendation is conditional: applications must accept 64-bit aggregate security, provide a source/candidate registry and replay policy, require at least 12 valid fragments for identity recovery, and never interpret one 8-bit share as independent authentication. If 128-bit aggregate security with missing-fragment tolerance is mandatory, the 20-bit regional budget is insufficient and no construction here should proceed to neural implementation.

## Proposed formal threat model

The adversary may observe, erase, reorder, duplicate, replay, modify, forge, or splice any subset of decoded fragments and knows all public protocol details. The adversary does not know the HMAC key, cannot alter the verifier registry, and cannot break HMAC or RS algebra. Acceptance means reconstructing a token from at least 12 valid symbols and validating the joint code under the claimed source/version. Security targets are at most 2^-64 acceptance of a forged identity, explicit non-valid state under insufficient evidence, and identification of inconsistent positions whenever a candidate identity can be established. Replay resistance is explicitly delegated to registry state.

## Falsifiable novelty hypothesis

**Hypothesis, not an established novelty claim:** a globally MAC-bound RS share code jointly covering the provenance token and complete indexed RS symbol vector can provide more useful 20-bit regional authentication/recovery behavior than independent 8-bit tags, because security is attached to a reconstructed multi-fragment identity while per-region bytes serve only as consistency evidence. It is falsified if formal comparison finds an equivalent established construction with the same threat model and allocation, or if neural errors prevent the required 12 valid fragments and 64-bit aggregate verification under the unchanged fidelity budget.
