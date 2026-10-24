# Deepfake detection — research repository

The **new, preliminary v4 implementation** is isolated at
[**v4/README.md**](v4/README.md).

The root Python modules and notebooks are the **historical exploratory code**,
including CWT, Transformer and diffusion experiments. They have **not**
been migrated, fixed or revalidated by the v4 branch.

## Proposed course direction

**Research question:** Does pixel-wise temporal-frequency information
improve deepfake video detection on unseen manipulation methods compared
with RGB-only models?

The preliminary v4 includes a disjoint-source manifest audit,
deterministic clip extraction, an RGB CNN+GRU baseline, a temporal pixel-wise
FFT CNN, their late fusion, an experimental CWT alternative, evaluation
at video level, and automated unit tests.

See [v4/README.md](v4/README.md) for explicit limitations, reproducible
commands, hypotheses, and missing work. Results from older runs are not
comparable until source-group splits and protocol are verified.

**Academic status:** Not yet evaluated with controlled FF++ / cross-domain
experiments. Do not present synthetic unit-test results as classification AUC.
