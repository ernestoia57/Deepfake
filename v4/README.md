# Deepfake v4 — preliminary, controlled research prototype

**Status:** engineering preview; NO new model accuracy claims. This is an isolated
implementation; the original repository root and diffusion experiments remain unchanged.

Research question: *Does pixel-wise temporal frequency improve out-of-manipulation
generalization of a video deepfake detector beyond a comparable RGB baseline?*

Based on: Rössler et al., FaceForensics++ (ICCV 2019); Lewis et al.,
"Deepfake Video Detection Based on Spatial, Spectral, and Temporal
Inconsistencies" (AIPR 2020); Kim et al., "Beyond Spatial Frequency:
Pixel-wise Temporal Frequency-based Deepfake Video Detection" (ICCV 2025).
Temporal pixel-wise FFT is **not a novel claim**.

## NEW: v4.3 prevalidated windows (required before real training)

See [DATA_PROTOCOL_v4_3.md](DATA_PROTOCOL_v4_3.md). Run
`python -m v4.window_index` on your matched RAW YouTube+Deepfakes manifest
to find several deterministic, Haar-validated 16-frame windows per video
and save each stable face square. Pass `--window-index <JSON>` to all
`v4.pipeline train` commands; the saved checkpoint retains the index
reference for validation/test. The program **refuses incomplete indexes**
instead of skipping videos or silently generating synthetic labeled frames.
Training multiple architectures with the same index/seed uses the same frames
and crops. This is an engineering gate, not a validated face tracker.

## NEW: v4.1 dataset-domain audit

**Start with [DATA_PROTOCOL_v4_1.md](DATA_PROTOCOL_v4_1.md).**
Observed local folders contain REAL RAW, Deepfakes RAW, Face2Face C23 and
DeepFakeDetection RAW. The original v4 manifest mislabels DeepFakeDetection
as `unknown_fake` and mixes method-dependent quality. The new
`python -m v4.protocol` commands **do not change the original source
manifest, any videos or split assignments**. They explicitly classify
domains, generate a comparable RAW original+Deepfakes baseline and run
random stratified face/decode probes.

These data corrections are prerequisites to interpreting any accuracy.

## Install

Python 3.10–3.12, PyTorch, OpenCV, NumPy, SciPy, scikit-learn, pytest.
Install from the repo root (prefer a fresh environment):

```bash
python -m pip install -r v4/requirements.txt
python -m pytest -q v4/tests
```

## Data contract

1. Obtain FF++ videos yourself under the original access terms.
2. The manifest generator supports conventional FF++ directories containing
   `original_sequences` / `manipulated_sequences`, or exploratory
   `real` / `fake` folders. A name must include numeric source IDs, e.g.
   `000.mp4`, `000_003.mp4`, or `01_02__scene.mp4`.
3. Related source IDs are placed in the **same source-connected component**
   before splitting. This is conservative: a large connected component can
   make train/val/test impossible, in which case the run fails.
4. This exploratory generated split does **not** claim to reproduce the official
   FF++ protocol. For comparable published scores, supply official split
   membership with the same source-group and class-presence audits.
5. Generated manifest has real paths local to YOUR machine and must not be
   committed. Do not publish face crops, subjects' identities or raw videos.
6. Missing faces are errors, not auto-labeled black images; if many failures
   occur, audit quality first. Face detector is preliminary Haar, not a
   validated multi-person tracker. `--face center` is **debug-only**.

## Run

```bash
python -m v4.pipeline manifest --root /data/FaceForensics++ --out manifests/ffpp_v4.csv
python -m v4.pipeline audit --manifest manifests/ffpp_v4.csv --inspect-clips 5
python -m v4.pipeline teardown --mode fusion --out runs/v4_teardown.json
python -m v4.pipeline train --manifest manifests/ffpp_v4.csv --mode rgb --out runs/v4_rgb_seed42 --seed 42
python -m v4.pipeline train --manifest manifests/ffpp_v4.csv --mode freq --out runs/v4_fft_seed42 --seed 42
python -m v4.pipeline train --manifest manifests/ffpp_v4.csv --mode fusion --out runs/v4_fusion_seed42 --seed 42
python -m v4.pipeline evaluate --checkpoint runs/v4_fusion_seed42/best.pt --split test --out runs/v4_fusion_test
```

These example baseline commands assume an already-audited, matched-quality
manifest; with the discovered 4,392-video mix use
`manifests/ffpp_v4_raw_deepfakes.csv` prepared as described in v4.1 instead.

Run also seeds 43 and 44; for statistical rigor report matched-seed paired
differences and uncertainty. **Do not tune on test.** Model selection is by
validation video-AUC; test is evaluated once after finalizing choices.

**Compute constraint:** ConvEncoder applies a CNN per frame for RGB.
Start with `--frames 16 --size 64 --dim 64` for an integration smoke test,
then use identical final settings across RGB/frequency/fusion runs.

**CWT exploration:** `--mode freq --frequency cwt --cwt-fmin 6 --cwt-fmax 10`.
This simple valid-center Morlet statistic is *not* equivalent to a full
time-resolved CWT classifier. It rejects under-resolved bands rather than
inventing boundary values. At 32 frames and 30 fps the default band selection
can still be tight; tests use explicit supported bands. Frequency bands should
be selected before using a held-out test set.

## What this prototype does and does not measure

* RGB CNN + GRU versus time-order-invariant RGB mean pooling.
* Pixel-wise rFFT (time-axis only) of per-frame median-filtered residuals.
* Learnable late concatenation fusion (not fixed arbitrary weights).
* Video-level probabilities aggregated over deterministic validation/test clips.
* Model parameters, approximate MACs, GPU latency and peak allocated GPU
  memory; **full FLOPs including preprocessing are not provided**.
* Bootstrap video-AUC interval. Related sources create dependencies, so
  video-bootstrap intervals may be overconfident: use source-group bootstrap
  and paired seeds for publication-grade inference.
* Does NOT claim pre-trained SOTA reproduction, validated face-tracking, or
  domain-shift generalization until independent experiments are run.

## Known limitations to resolve before E3

1. Official FF++ partitions and provenance review for derived/duplicate videos.
2. Video frame decoding depends on OpenCV/FFmpeg; verify FPS/timestamps,
   VFR footage and decode errors with ffprobe on target machine.
3. Strict failure-fast facial crops; audit discard rate by label/manipulation
   before final experiment. Replace Haar with RetinaFace + robust tracking.
4. Architecture teardown lacks visual feature maps, receptive field and exact
   operator FLOP profiling; add instrumented model inspections.
5. Evaluate leave-one-manipulation-out and then DFDC/Celeb-DF untouched.
6. Dataset license, consent, distribution shift and group-level demographic
   bias must be documented in the final report.

## Research design log

| Decision | Rationale | Status |
|---|---|---|
| Keep old code unchanged | Preserve negative experiments/reproducibility | Implemented |
| Disable diffusion from new core | ELBO is not verified as face validity/authenticity | Implemented |
| Fix temporal-axis representation | Existing frequency pipelines were confounded | Implemented |
| Source-connected split | Avoid shared source appearing train/test | Implemented, inspect source rules |
| Reject missing face/decoded frame | Prevent silent shortcut learning | Implemented |
| Compression-aware domain profiling | Avoid raw-vs-c23 spurious cues | Added in v4.1 |
| Stratified random face/decode probe | Detect biased failure rates | Added in v4.1 |
| RGB/FFT/fusion ablations | Clear counterfactual research hypotheses | Code ready; measurements pending |
| Model GRU temporal order | Mean pooling cannot model order | Code ready; accuracy pending |
| AI-assisted code generation | Assistance used to draft v4 and unit tests; all code needs human review | Documented |

**Do not treat v4 as a validated detector or forensic proof of authenticity.**
