# K-FRAG

K-FRAG is an experimental implementation of blind, fragment-resilient image
provenance watermarking. It includes 96-bit registered identities, RS(16,12)
erasure recovery, independently authenticated 44-bit regional packets, a
content-adaptive residual encoder, questioned-image-only decoding and
synchronization, attacks, evidence maps, controlled curriculum gates, dataset
manifests, evaluation, and immutable run artifacts.

The current COCO development result does **not** pass the predefined natural
image communication gate. Later phases are implemented but scientifically
unvalidated and must not be described as successful.

## Install

Python 3.9 or newer is required. Create and activate a virtual environment,
then install the package and test dependencies:

```bash
python -m venv .venv
python -m pip install -e ".[test]"
```

## Test

```bash
python -m pytest
```

Secret keys are supplied by callers at runtime. Never print them or commit them
to source control or configuration files.

For inference set `KFRAG_HMAC_KEY` in the process environment. Authentication
uses domain-separated HMAC-SHA-256. The development packet truncates its tag to
32 bits, which provides limited forgery resistance; `tag_bits` is configurable
for capacity/security ablations. The public asset namespace must come from
registered verifier metadata and is never inferred from the expected identity.

## Development run

```bash
python scripts/train_kfrag.py --config configs/kfrag_clean_natural_dev_v1.yaml
python scripts/validate_run.py outputs/kfrag/clean_natural_dev_v1/<run-id>
```
