# K-FRAG

K-FRAG is a research repository for blind, fragment-resilient image provenance
watermarking. This initial phase implements only the non-neural provenance
protocol: a fixed 96-bit token, RS(16,12) erasure recovery, and authenticated
regional packets.

No watermark neural network, dataset loader, application, or training code is
included yet.

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
