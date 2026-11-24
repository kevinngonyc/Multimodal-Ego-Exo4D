"""Shared paths, take metadata and video helpers for the self-supervised adaptation (run everything from idea_1)."""
import json
import math
from pathlib import Path

import av
import numpy as np

DATA_ROOT = next(p / "data" for p in (Path.cwd(), *Path.cwd().parents) if (p / "data" / "annotations").is_dir())
ANNOTATION_ROOT = DATA_ROOT / "annotations"
BANK_DIR = Path("embeddings") / "ssl_bank"
RUN_DIR = Path("models") / "ssl_adapt"
SCENARIOS = ("Basketball", "Cooking", "Dance", "Music", "Rock Climbing", "Soccer")
FPS, CLIP_FRAMES, SIZE = 8, 8, 224
LEVELS = {"Novice": 1, "Early Expert": 2, "Intermediate Expert": 3, "Late Expert": 4}

TAKES = {t["take_name"]: t for t in json.loads((DATA_ROOT / "takes.json").read_text(encoding="utf-8"))}

def downscaled(rel):
    parts = rel.split("/")
    parts[-1:-1] = ["downscaled", "448"]
    return DATA_ROOT.joinpath(*parts)

OFFICIAL = {}  # take -> dict(split, scenario, score, participant, ego, exo1)
for split, fname in (("train", "proficiency_demonstrator_train.json"), ("val", "proficiency_demonstrator_val.json")):
    for m in json.loads((ANNOTATION_ROOT / fname).read_text(encoding="utf-8"))["annotations"]:
        take = m["video_paths"]["ego"].split("/")[1]
        OFFICIAL[take] = dict(split=split, scenario=m["scenario_name"], score=LEVELS[m["proficiency_score"]],
                              participant=m["origin_participant_id"], ego=downscaled(m["video_paths"]["ego"]),
                              exo1=downscaled(m["video_paths"]["exo1"]))

def video_dir(take):
    return DATA_ROOT / "takes" / take / "frame_aligned_videos" / "downscaled" / "448"

def ego_path(take):
    if take in OFFICIAL:
        return OFFICIAL[take]["ego"]
    c = sorted(video_dir(take).glob("aria*_214-1.mp4"))
    return c[0] if c else None

def exo_camera(take):
    """Same rule as the main caches: best_exo if its video exists, else exo1 (official) / first static camera."""
    best = TAKES.get(take, {}).get("best_exo")
    if best and (video_dir(take) / f"{best}.mp4").exists():
        return best, video_dir(take) / f"{best}.mp4"
    if take in OFFICIAL:
        return "exo1 (no best_exo)", OFFICIAL[take]["exo1"]
    c = sorted(p for p in video_dir(take).glob("*.mp4") if p.stem.startswith(("cam", "gp")))
    return (c[0].stem, c[0]) if c else (None, None)

def frame_count(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return int(s.frames), float(s.average_rate)

def clip_frame_indices(k0, rate):
    """frame indices of the 8 FPS samples k0..k0+7 (same rounding as the main extraction)."""
    return [int(math.floor(k * rate / FPS + 0.5)) for k in range(k0, k0 + CLIP_FRAMES)]

def grab(path, wanted, seek_gap=90):
    """Decode only the frames in `wanted` (sorted frame indices) -> {idx: 224x224x3 uint8}, squashed with the same
    bilinear resize as the main caches. Seeks (to the previous keyframe) when the next wanted frame is far ahead."""
    out = {}
    wanted = sorted(set(wanted))
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        rate, tb = float(s.average_rate), float(s.time_base)
        start = s.start_time or 0
        pos, i = -10**9, 0
        while i < len(wanted):
            target = wanted[i]
            if target - pos > seek_gap:
                c.seek(int(start + target / rate / tb), stream=s, backward=True, any_frame=False)
            for fr in c.decode(s):
                idx = int(round((fr.pts - start) * tb * rate))
                pos = idx
                while i < len(wanted) and wanted[i] < idx:
                    i += 1  # frame missing in the stream: skip it (filled by the caller)
                if i < len(wanted) and wanted[i] == idx:
                    out[idx] = fr.to_ndarray(format="rgb24", width=SIZE, height=SIZE, interpolation="BILINEAR")
                    i += 1
                    # only leave to seek right after capturing a frame (never right after landing on a keyframe,
                    # which would seek back to the same keyframe forever)
                    if i >= len(wanted) or wanted[i] - idx > seek_gap:
                        break
                elif i >= len(wanted):
                    break
            else:
                break  # end of stream
    return out
