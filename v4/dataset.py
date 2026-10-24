"""Deterministic video-level clip sampling with explicit decode / face failures.

No replacement of failed reads by black labeled tensors. No silent whole-frame
fallback when face detection fails. Validation/test sample fixed clips.
"""
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .manifest import read_manifest, audit_manifest
from .frequency import temporal_fft, temporal_cwt


class SampleError(RuntimeError):
    pass


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    area = max(0, ix1-ix0) * max(0, iy1-iy0)
    den = aw*ah + bw*bh - area
    return area / den if den > 0 else 0.0


class StableFaceCrop:
    """Preliminary Haar detector. Index stores the selected square crop."""

    def __init__(self, enlargement=1.35, min_face=32):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise SampleError("OpenCV Haar detector could not be loaded")
        self.enlargement = enlargement
        self.min_face = min_face

    def locate(self, frames):
        """Return one integer [x0,y0,side] stable square for the whole clip."""
        T = len(frames)
        prior, boxes = None, []
        for i in sorted(set([0, T//2, T-1])):
            gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            found = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4)
            found = [tuple(map(int, r)) for r in found
                     if r[2] >= self.min_face and r[3] >= self.min_face]
            if not found:
                continue
            if prior is None:
                selected = max(found, key=lambda b: b[2]*b[3])
            else:
                selected = max(found, key=lambda b: _iou(b, prior))
                if _iou(selected, prior) < 0.15:
                    continue
            prior = selected
            boxes.append(selected)
        if len(boxes) < 2:
            raise SampleError("Face not reliably found in at least two anchor frames")
        H, W = frames[0].shape[:2]
        mx, my, mw, mh = np.median(np.asarray(boxes), axis=0)
        side = int(round(max(mw, mh) * self.enlargement))
        side = max(1, min(side, H, W))
        x0 = int(round(mx + mw/2 - side/2))
        y0 = int(round(my + mh/2 - side/2))
        x0 = max(0, min(x0, W-side))
        y0 = max(0, min(y0, H-side))
        return [x0, y0, side]

    def __call__(self, frames, size, bbox=None):
        """Reuse prevalidated coordinates when provided; never silently redetect."""
        x0, y0, side = self.locate(frames) if bbox is None else bbox
        H, W = frames[0].shape[:2]
        if not all(isinstance(v, (int, np.integer)) for v in (x0, y0, side)):
            raise SampleError("Invalid noninteger indexed face box")
        if side <= 0 or x0 < 0 or y0 < 0 or x0+side > W or y0+side > H:
            raise SampleError("Indexed face box outside video frame")
        return np.stack([cv2.resize(
            f[y0:y0+side, x0:x0+side], (size, size),
            interpolation=cv2.INTER_AREA) for f in frames])


def video_metadata(path):
    """Fetch valid frame count and FPS; fail rather than accept broken metadata."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SampleError(f"Cannot open video: {path}")
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if not (np.isfinite(fps) and fps > 0) or count <= 0:
        raise SampleError(f"Invalid FPS/frame count: {path}, frames={count}, fps={fps}")
    return count, fps


def read_video_window(path, nframes, start):
    """Decode exactly one clip at a fixed zero-based frame index."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SampleError(f"Cannot open video: {path}")
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if count < nframes or not (np.isfinite(fps) and fps > 0):
            raise SampleError(f"Short video or invalid FPS: {path}, frames={count}, fps={fps}")
        if not (isinstance(start, int) and 0 <= start <= count-nframes):
            raise SampleError(f"Invalid window start={start} for {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(nframes):
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                raise SampleError(f"Frame decode failed for {path} at clip start={start}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return np.stack(frames), fps


class VideoClipDataset(Dataset):
    def __init__(self, manifest, split, frames=32, size=128,
                 frequency="fft", filter_mode="median", face="haar",
                 clips_per_video=1, seed=42, cwt_bands=8, cwt_fmin=6.0,
                 cwt_fmax=10.0, window_index=None):
        audit_manifest(manifest)
        self.rows = [r for r in read_manifest(manifest) if r["split"] == split]
        if not self.rows:
            raise ValueError("No videos in requested split")
        if frames < 8 or frames % 2 or size < 32:
            raise ValueError("Require even frames>=8 and size>=32")
        if face not in ("haar", "center"):
            raise ValueError("face mode must be haar or center")
        if frequency not in ("fft", "cwt"):
            raise ValueError("frequency must be fft or cwt")
        self.split, self.frames, self.size = split, frames, size
        self.frequency, self.filter_mode = frequency, filter_mode
        self.face_mode, self.seed = face, seed
        self.clips_per_video, self.epoch = clips_per_video, 0
        self.cwt_bands, self.cwt_fmin, self.cwt_fmax = cwt_bands, cwt_fmin, cwt_fmax
        self.cropper = StableFaceCrop() if face == "haar" else None
        self.indexed_windows = None
        if window_index:
            from .window_index import load_index
            self.indexed_windows = load_index(window_index, manifest,
                                               frames=frames, size=size, face=face)
            for row in self.rows:
                windows = self.indexed_windows[row["video_id"]]
                required = 1 if split == "train" else clips_per_video
                if len(windows) < required:
                    raise ValueError(f"Indexed video {row['video_id']} has "
                                     f"{len(windows)} valid windows; need {required}")


    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows) * self.clips_per_video

    def __getitem__(self, index):
        video_pos, clip_id = divmod(index, self.clips_per_video)
        rec = self.rows[video_pos]
        path = rec["path"]
        indexed_box = None
        if self.indexed_windows is not None:
            windows = self.indexed_windows[rec["video_id"]]
            if self.split == "train":
                key = f"{self.seed}:{self.epoch}:{rec['video_id']}:{clip_id}"
                number = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],
                                        "little")
                chosen = windows[number % len(windows)]
            else:
                chosen = windows[clip_id]
            start = int(chosen["start"])
            indexed_box = chosen.get("bbox")
        else:
            count, _ = video_metadata(path)
            if count < self.frames:
                raise SampleError(f"Short video: {path}, frames={count}")
            max_start = count - self.frames
            if self.split == "train":
                key = f"{self.seed}:{self.epoch}:{rec['video_id']}:{clip_id}"
                seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],
                                      "little")
                start = int(np.random.default_rng(seed).integers(0, max_start+1))
            elif self.clips_per_video <= 1:
                start = max_start // 2
            else:
                start = int(round(max_start * clip_id / (self.clips_per_video-1)))
        frames, fps = read_video_window(path, self.frames, start)
        if self.cropper is not None:
            frames = self.cropper(frames, self.size, bbox=indexed_box)
        else:
            # DEBUG-ONLY control: deliberately not treated as a face crop
            h, w = frames[0].shape[:2]
            side = min(h, w)
            x0, y0 = (w-side)//2, (h-side)//2
            frames = np.stack([
                cv2.resize(f[y0:y0+side, x0:x0+side], (self.size, self.size),
                           interpolation=cv2.INTER_AREA) for f in frames])
        if self.frequency == "fft":
            freq = temporal_fft(frames, filter_mode=self.filter_mode)
        else:
            freq = temporal_cwt(frames, fps=fps, bands=self.cwt_bands,
                                fmin=self.cwt_fmin, fmax=self.cwt_fmax,
                                filter_mode=self.filter_mode)
        rgb = frames.astype(np.float32).transpose(0, 3, 1, 2) / 255.
        return {"rgb": torch.from_numpy(rgb.copy()),
                "freq": torch.from_numpy(freq.copy()),
                "label": torch.tensor(float(rec["label"]), dtype=torch.float32),
                "video_id": rec["video_id"], "source_group": rec["source_group"],
                "fps": fps, "start": start}
