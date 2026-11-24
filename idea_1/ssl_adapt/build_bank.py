"""Pre-decode synchronized ego + exo 1 s clips (8 frames @ 8 FPS, squashed to 224, JPEG q95) into a flat byte bank.

usage (from idea_1):  python ssl_adapt/build_bank.py train|probe [LIMIT]
  train: every take of the 6 proficiency scenarios with ego + exo video, EXCEPT official val takes (kept clean for
         the official-split evaluation). ~1 clip per 6 s at random 8 FPS offsets, 8..96 clips per take.
  probe: the 2,111 official takes (train + val), 16 clips per take on the extraction grid (clip c = 8 FPS samples
         8c..8c+7, as in the main caches), used only to track a level probe during training (no level labels are
         ever used by the training loss).
-> embeddings/ssl_bank/{name}.bin (concatenated JPEGs) + {name}_index.pt
   index = {"takes": [names], "take_meta": {take: dict(participant, scenario, university, setting, camera, official)},
            "item_take": int32[N], "item_k0": int32[N] (first 8 FPS sample), "off": int64[N,16], "len": int32[N,16]}
   columns 0-7 = ego frames, 8-15 = exo frames.
"""
import hashlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (BANK_DIR, CLIP_FRAMES, FPS, OFFICIAL, SCENARIOS, TAKES, clip_frame_indices, ego_path,
                    exo_camera, frame_count, grab)

NAME = sys.argv[1]
assert NAME in ("train", "probe")
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
CLIPS_PER_SEC, MIN_CLIPS, MAX_CLIPS, PROBE_CLIPS = 1 / 6, 8, 96, 16

def take_list():
    if NAME == "probe":
        return sorted(t for t in OFFICIAL if ego_path(t) is not None and ego_path(t).exists() and exo_camera(t)[1] is not None
                      and Path(exo_camera(t)[1]).exists())
    out = []
    for t, m in TAKES.items():
        if m["parent_task_name"] not in SCENARIOS or m.get("is_dropped"):
            continue
        if t in OFFICIAL and OFFICIAL[t]["split"] == "val":
            continue
        e, (cam, x) = ego_path(t), exo_camera(t)
        if e is not None and e.exists() and x is not None and Path(x).exists():
            out.append(t)
    return sorted(out)

def participant_of(t):
    # takes.json participant_uid for every take; the official annotation id is kept separately for the probe
    return int(TAKES[t]["participant_uid"]) if TAKES.get(t, {}).get("participant_uid") is not None else -1

def work(take):
    ep, (cam, xp) = ego_path(take), exo_camera(take)
    ne, re_ = frame_count(ep)
    nx, rx = frame_count(xp)
    rate = rx
    n = min(ne, nx)
    n_dense = int((n - 1) * FPS / rate) + 1
    if NAME == "probe":
        n_clips = max(1, n_dense // CLIP_FRAMES)
        cs = np.unique(np.linspace(0, n_clips - 1, min(PROBE_CLIPS, n_clips)).round().astype(int))
        k0s = [int(c) * CLIP_FRAMES for c in cs]
    else:
        rng = np.random.default_rng(int(hashlib.md5(take.encode()).hexdigest()[:8], 16))
        last = n_dense - CLIP_FRAMES
        if last < 0:
            return take, cam, []
        k = int(np.clip(round(n / rate * CLIPS_PER_SEC), MIN_CLIPS, MAX_CLIPS))
        k0s = sorted(set(rng.integers(0, last + 1, size=k).tolist()))
    idx = {k0: clip_frame_indices(k0, rate) for k0 in k0s}
    wanted = sorted({i for v in idx.values() for i in v if i < n})
    ego, exo = grab(ep, wanted), grab(xp, wanted)
    enc = [cv2.IMWRITE_JPEG_QUALITY, 95]
    clips = []
    for k0, ii in idx.items():
        row = []
        for src in (ego, exo):
            frames = [src.get(i) for i in ii]
            if sum(f is None for f in frames) > 2:
                row = None
                break
            for j in range(len(frames)):  # fill isolated gaps with the nearest decoded frame
                if frames[j] is None:
                    frames[j] = next(frames[q] for q in sorted(range(len(frames)), key=lambda q: abs(q - j)) if frames[q] is not None)
            row += [cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), enc)[1].tobytes() for f in frames]
        if row is not None:
            clips.append((k0, row))
    return take, cam, clips

def main():
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    takes = take_list()[:LIMIT]
    suffix = "" if LIMIT is None else "_smoke"
    binp, idxp = BANK_DIR / f"{NAME}{suffix}.bin", BANK_DIR / f"{NAME}{suffix}_index.pt"
    print(f"{NAME}: {len(takes)} takes -> {binp}", flush=True)
    take_id = {t: i for i, t in enumerate(takes)}
    meta = {}
    item_take, item_k0, offs, lens = [], [], [], []
    pos, t0, failed = 0, time.time(), []
    with open(binp, "wb") as fh, ProcessPoolExecutor(10) as pool:
        futs = {pool.submit(work, t): t for t in takes}
        for done, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            try:
                _, cam, clips = fut.result()
            except Exception as e:
                failed.append((t, repr(e)))
                continue
            m = TAKES.get(t, {})
            meta[t] = dict(participant=participant_of(t), scenario=m.get("parent_task_name", OFFICIAL.get(t, {}).get("scenario")),
                           university=m.get("university_name"), setting=m.get("physical_setting_uid"), camera=cam,
                           official=OFFICIAL.get(t, {}).get("split"))
            for k0, row in clips:
                o, l = [], []
                for b in row:
                    fh.write(b); o.append(pos); l.append(len(b)); pos += len(b)
                item_take.append(take_id[t]); item_k0.append(k0); offs.append(o); lens.append(l)
            if done % 100 == 0 or done == len(takes):
                el = time.time() - t0
                print(f"{done}/{len(takes)} takes, {len(item_take)} clips, {pos / 1e9:.1f} GB, {el / 60:.1f} min, "
                      f"eta {el / done * (len(takes) - done) / 60:.0f} min", flush=True)
    torch.save({"takes": takes, "take_meta": meta, "item_take": np.array(item_take, np.int32),
                "item_k0": np.array(item_k0, np.int32), "off": np.array(offs, np.int64), "len": np.array(lens, np.int32)}, idxp)
    print(f"done: {len(item_take)} clips from {len(meta)} takes, {pos / 1e9:.1f} GB; failed {len(failed)} {failed[:3]}")

if __name__ == "__main__":
    main()
