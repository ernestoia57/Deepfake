# v4.1 — Compression-aware manifest and stratified crop audit

## Motivation (local audit, 2026-10-23)

Observed inventory (NOT general FF++ dataset counts):

| Method | Count | Quality |
| --- | ---: | --- |
| Original | 1363 | RAW |
| Deepfakes | 1000 | RAW |
| Face2Face | 1000 | C23 |
| DeepFakeDetection (previously unknown_fake) | 1029 | RAW |

The existing group-preserving split put 864 DeepFakeDetection videos in
validation but only 64 in train. Mixing these domains may reward
compression shortcuts instead of facial-forgery artifacts.

The preliminary 300-clip audit (FIRST 100 rows of each split, not random)
reported 14 face-detection failures and 2 short videos. This is not an
unbiased estimate of the global failure rate.

## What is implemented

- Reclassifies DeepFakeDetection via an explicit folder component.
- Adds a `compression` column to a **new** manifest.
- Emits per-split method x compression counts and alerts.
- Creates `raw_deepfakes` subset (only REAL RAW + Deepfakes RAW) with
  original source_group and split preserved. Not an official FF++ split.
- Uses reproducible, stratified *random* crop/decode probing, rather than
  checking only sorted first N videos.
- Rejects unknown sources, ambiguous compressions and label/path mismatch.
- Does NOT auto-delete, resplit, modify or transcode any video.

## Commands (Anaconda Prompt / CMD in repository root)

```bat
conda activate deepfake_v4
cd /d C:\Deepfake\Deepfake_v4

python -m pytest -q v4\tests

python -m v4.protocol report ^
  --manifest manifests\ffpp_v4.csv ^
  --out runs\v4_1_domains.json

python -m v4.protocol prepare ^
  --manifest manifests\ffpp_v4.csv ^
  --profile annotated_all ^
  --out manifests\ffpp_v4_domains.csv

python -m v4.protocol prepare ^
  --manifest manifests\ffpp_v4.csv ^
  --profile raw_deepfakes ^
  --out manifests\ffpp_v4_raw_deepfakes.csv

python -m v4.protocol probe ^
  --manifest manifests\ffpp_v4_raw_deepfakes.csv ^
  --samples-per-stratum 20 ^
  --frames 16 ^
  --size 64 ^
  --face haar ^
  --seed 42 ^
  --out runs\v4_1_crop_probe.json
```

The probe inspects up to 20 samples per (split, class, method, compression).
If any face/decode failures remain, training will fail fast; do not
substitute black images or claim a valid model. Increase
`--samples-per-stratum` to examine more clips. `--face center` is
strictly debug-only and not a scientifically justified face crop.

## Single-epoch integration smoke (only after a clean probe)

```bat
python -m v4.pipeline train ^
  --manifest manifests\ffpp_v4_raw_deepfakes.csv ^
  --mode rgb ^
  --frames 16 --size 64 --dim 64 ^
  --batch 2 --epochs 1 --workers 0 --seed 42 ^
  --out runs\v4_1_rgb_smoke
```

**Warning:** `--epochs 1` still loads the full train subset and can
fail on an unprobed video. For a guaranteed smoke test, first create a
separate small independent-group manifest containing both labels in each
split, or implement offline clip-validity indexing. Do not test on
a manifest filtered using test labels to tune model or preprocessing.

Do not claim generalization to Face2Face or DeepFakeDetection from this
baseline. Leave both datasets for a separate, provenance-checked,
pre-registered evaluation. Face2Face C23 cannot isolate synthesis method
from compression until matched-quality originals and other manipulations
are available.
