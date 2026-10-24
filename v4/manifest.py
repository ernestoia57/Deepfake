"""FF++ manifest creation and disjoint-source split integrity checks."""
import csv
import json
import random
from collections import Counter
from pathlib import Path

REQUIRED = {"path", "video_id", "label", "manipulation", "source_group", "split"}
SPLITS = ("train", "val", "test")


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, a):
        self.parent.setdefault(a, a)
        if self.parent[a] != a:
            self.parent[a] = self.find(self.parent[a])
        return self.parent[a]

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def source_ids(path):
    """Derive known FF++ source IDs (e.g. 001_002, 01_02__scene).
    Do NOT use arbitrary DFDC filenames as provenance; they need metadata.
    """
    prefix = path.stem.split("__")[0]
    parts = prefix.split("_")
    if not (1 <= len(parts) <= 2 and all(p.isdigit() for p in parts)):
        raise ValueError(f"Unrecognized FF++ source ID: {path.name}. Supply verified provenance.")
    return parts


def discover_ffpp(root):
    root = Path(root).resolve()
    rows = []
    for p in sorted(root.rglob("*.mp4")):
        rel = p.relative_to(root)
        tokens = [s.lower() for s in rel.parts]
        if "real" in tokens or "original_sequences" in tokens:
            label = 0
            manip = "original"
        elif "fake" in tokens or "manipulated_sequences" in tokens:
            label = 1
            known = ("Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures")
            manip = next((x for x in known if x.lower() in tokens), "unknown_fake")
        else:
            continue
        ids = source_ids(p)
        rows.append(dict(path=str(p), video_id=str(rel), label=label,
                         manipulation=manip, source_ids=ids))
    if not rows:
        raise ValueError("No FF++ videos found; check --root structure")
    return rows


def create_manifest(root, output, seed=42, val_fraction=0.15, test_fraction=0.15):
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and
            val_fraction + test_fraction < 1):
        raise ValueError("Invalid split fractions")
    rows = discover_ffpp(root)
    uf = UnionFind()
    for r in rows:
        ids = r["source_ids"]
        for source in ids[1:]:
            uf.union(ids[0], source)
    groups = {}
    for r in rows:
        r["source_group"] = uf.find(r["source_ids"][0])
        del r["source_ids"]
        groups.setdefault(r["source_group"], []).append(r)
    group_keys = sorted(groups)
    if len(group_keys) < 3:
        raise ValueError(
            f"Only {len(group_keys)} independent source groups. Cannot safely form "
            "train/val/test without source leakage; obtain more source videos."
        )
    rng = random.Random(seed)
    rng.shuffle(group_keys)
    n = len(group_keys)
    n_test = max(1, round(n*test_fraction))
    n_val = max(1, round(n*val_fraction))
    if n_test + n_val >= n:
        raise ValueError("Not enough independent groups for three splits")
    assignment = {}
    for idx, group in enumerate(group_keys):
        assignment[group] = ("test" if idx < n_test else
                             "val" if idx < n_test+n_val else "train")
    for r in rows:
        r["split"] = assignment[r["source_group"]]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(REQUIRED))
        writer.writeheader()
        writer.writerows(rows)
    return audit_manifest(output)


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest missing columns: {REQUIRED-set(reader.fieldnames or [])}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    for row in rows:
        if row["split"] not in SPLITS or row["label"] not in ("0", "1"):
            raise ValueError("Invalid split or label in manifest")
    return rows


def audit_manifest(path, verify_files=True):
    rows = read_manifest(path)
    owners = {}
    seen_ids = set()
    for r in rows:
        key = r["source_group"]
        if key in owners and owners[key] != r["split"]:
            raise ValueError(f"LEAKAGE: source_group={key} in both {owners[key]} and {r['split']}")
        owners[key] = r["split"]
        if r["video_id"] in seen_ids:
            raise ValueError(f"Duplicate video_id: {r['video_id']}")
        seen_ids.add(r["video_id"])
        if verify_files and not Path(r["path"]).is_file():
            raise FileNotFoundError(r["path"])
    counts = {s: dict(Counter(r["label"] for r in rows if r["split"] == s)) for s in SPLITS}
    for s in SPLITS:
        if not all(counts[s].get(label, 0) for label in ("0", "1")):
            raise ValueError(f"Split {s} missing a real or fake class: {counts[s]}")
    return {"videos": len(rows), "independent_source_groups": len(owners),
            "counts_by_split_label": counts}


def write_audit(path, output):
    results = audit_manifest(path)
    Path(output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
