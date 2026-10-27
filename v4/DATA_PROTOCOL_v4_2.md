# v4.2 matched YouTube RAW experiment

The user data audit found per split exactly 704/148/148 originals from
`original_sequences/youtube/raw` and 704/148/148 manipulated videos in
`manipulated_sequences/Deepfakes/raw`. There are also 363 RAW actors
originals, heavily concentrated in validation. They are deliberately excluded
from this *initial* within-domain baseline to remove a known provenance
imbalance, not discarded from the dataset.

**Numbers alone do not prove pairing.** The new profile verifies each fake
video's source IDs against original filenames *within its original split*
and requires the same `source_group`. If your sources are organized
differently or a parent is missing, the command fails closed rather than
inventing associations.

From Windows CMD / Anaconda Prompt in the repository root:

```bat
python -m pytest -q v4\tests

python -m v4.protocol prepare ^
  --manifest manifests\ffpp_v4.csv ^
  --profile raw_youtube_deepfakes ^
  --out manifests\ffpp_v4_youtube_pairs.csv

python -m v4.protocol probe ^
  --manifest manifests\ffpp_v4_youtube_pairs.csv ^
  --samples-per-stratum 25 ^
  --frames 16 --size 64 --face haar --seed 42 ^
  --out runs\v4_2_pair_crop_probe.json
```

The profile does not modify the source manifest, video files, or split
assignments. Expected counts are 1,408 training / 296 validation / 296
test; actual counts and source matching MUST be verified by the command.

If provenance fails, preserve the error and inspect real/fake filename
relationships before trying to train. Do not bypass provenance checks just
to get an experiment to run.

Current data pipeline may fail upon unprobed corrupted/face-missing videos;
an exhaustive offline clip index and a robust face tracker are still needed
for fully validated experiments. No model AUC is claimed yet.
