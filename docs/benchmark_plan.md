# K-FRAG benchmark plan

This document is a plan only. No baseline has been implemented or run, and no
baseline result is claimed. Missing implementations and measurements must be
recorded as **unavailable**, never estimated or inferred.

## Baseline registry

The six closest papers have already been selected in the literature review, but
their bibliographic data must be copied from the verified literature-extraction
spreadsheet. Titles are intentionally not guessed here.

| ID | Exact title | Venue/year | Official code/checkpoint | Dataset/resolution | Payload capacity | Embedding/decoding conditions | Supported attacks | Reported metrics | Reproduction status |
|---|---|---|---|---|---|---|---|---|---|
| baseline_1 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| baseline_2 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| baseline_3 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| baseline_4 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| baseline_5 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| baseline_6 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

Each row must eventually contain the exact title; venue and year; official code
and checkpoint availability; dataset and image resolution; payload capacity;
embedding and decoding conditions; supported attacks; reported metrics; and
reproduction status.

## Result provenance and equivalence

Every results table will have separate columns for **original-paper reported
results** and **results reproduced under the K-FRAG experimental protocol**.
Unreported measurements and unavailable implementations remain “unavailable.”
No value may migrate between these namespaces.

Comparisons will use equivalent datasets, image resolutions, payload sizes,
attack parameters, and sample counts wherever technically possible. Any
unavoidable mismatch will be disclosed beside the affected result, rather than
hidden in an aggregate ranking.

K-FRAG reporting preserves PSNR, SSIM, LPIPS, overall and non-index bit
accuracy, coded-symbol accuracy, authentication-tag accuracy, exact regional-
packet accuracy, exact full-payload accuracy, provenance-token recovery rate,
false acceptance and false rejection rates, tamper-localization IoU and F1,
minimum surviving image area, number of surviving authenticated fragments,
inference time, and parameter count.

## Semantic distinctions

The comparison must explicitly distinguish registered source identity from
watermark-presence detection; arbitrary payload recovery from fixed
classification; global recovery from independent regional recovery;
authenticated packets from unauthenticated bit predictions; and k-of-n
threshold reconstruction from full-payload decoding.

The threat-model table must separately state whether decoding is blind to the
original image, expected payload, and crop coordinates. Evidence outcomes must
be represented as valid, missing, corrupted, or manipulated rather than being
collapsed into a single failure state.

## Future ablations

Planned K-FRAG ablations are: without RS coding; without HMAC authentication;
without content binding; without synchronization; a global payload instead of
regional packets; different k-of-n thresholds; and different HMAC-tag lengths.
These are not results and must remain marked unavailable until run.
