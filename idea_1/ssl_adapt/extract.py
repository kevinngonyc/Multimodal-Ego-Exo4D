"""Full-take features with an adapted checkpoint, in exactly the format of the main K600 caches.

usage (from idea_1):  python ssl_adapt/extract.py --run NAME --ckpt step20000 --view exo|ego [--limit N]
Every 1 s clip (8 frames @ 8 FPS, squashed to 224, sequential decode) of the 2,111 official takes; LoRA merged into
the weights; stats = (L6, L12) x (CLS, patch mean, patch std), fp16.
-> embeddings/timesformer_ssl_{run}_{ckpt}_{view}8fps_clips.pt = {"camera": {take: cam}, "clips": {"squash": {take: (n, 2, 3, 768)}}}
"""
import argparse
import math
import queue
import sys
import threading
import time
from pathlib import Path

import av
import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import CLIP_FRAMES, FPS, OFFICIAL, RUN_DIR, SIZE, ego_path, exo_camera
from train import MEAN, STD, build_model

BATCH = 48

def decode(take, view):
    camera, path = ("ego", ego_path(take)) if view == "ego" else exo_camera(take)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate, total = float(stream.average_rate), stream.frames
        n_dense = int((total - 1) * FPS / rate) + 1
        n_clips = max(1, n_dense // CLIP_FRAMES)
        wanted = {int(math.floor(k * rate / FPS + 0.5)): k for k in range(n_clips * CLIP_FRAMES)}
        frames = np.zeros((n_clips * CLIP_FRAMES, SIZE, SIZE, 3), np.uint8)
        got = np.zeros(n_clips * CLIP_FRAMES, bool)
        for i, frame in enumerate(container.decode(video=0)):
            k = wanted.get(i)
            if k is None:
                continue
            frames[k] = frame.to_ndarray(format="rgb24", width=SIZE, height=SIZE, interpolation="BILINEAR")
            got[k] = True
    for k in np.flatnonzero(~got):
        frames[k] = frames[max(k - 1, 0)]
    return camera, frames.reshape(n_clips, CLIP_FRAMES, SIZE, SIZE, 3)

def load_adapted(run, ckpt):
    ck = torch.load(RUN_DIR / run / f"{ckpt}.pt", weights_only=False)
    model = build_model(ck["args"]["lora_r"])
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected and not [k for k in missing if "lora_" in k], (unexpected[:3], missing[:3])
    model = model.merge_and_unload()
    model.gradient_checkpointing_disable()
    return model.half().cuda().eval()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="step20000")
    ap.add_argument("--view", choices=("exo", "ego"), required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    out = Path("embeddings") / f"timesformer_ssl_{a.run}_{a.ckpt}_{a.view}8fps_clips{'_smoke' if a.limit else ''}.pt"
    takes = sorted(t for t in OFFICIAL if (ego_path(t) if a.view == "ego" else exo_camera(t)[1]) is not None
                   and Path(ego_path(t) if a.view == "ego" else exo_camera(t)[1]).exists())[:a.limit]
    cache = torch.load(out, weights_only=False) if out.exists() else {"camera": {}, "clips": {"squash": {}}}
    todo = [t for t in takes if t not in cache["clips"]["squash"]]
    print(f"{len(takes)} takes, {len(todo)} to go -> {out}", flush=True)
    model = load_adapted(a.run, a.ckpt)
    mean = torch.tensor(MEAN, device="cuda", dtype=torch.float16)
    std = torch.tensor(STD, device="cuda", dtype=torch.float16)

    @torch.inference_mode()
    def embed(clips):
        res = []
        for i in range(0, len(clips), BATCH):
            x = torch.from_numpy(clips[i:i + BATCH]).cuda().half().div_(255).sub_(mean).div_(std)
            x = x.permute(0, 1, 4, 2, 3).contiguous()
            r = model(pixel_values=x, output_hidden_states=True)
            layers = []
            for hidden in (r.hidden_states[6], r.last_hidden_state):
                p = hidden[:, 1:].float()
                layers.append(torch.stack([hidden[:, 0].float(), p.mean(1), p.std(1)], 1))
            res.append(torch.stack(layers, 1).half().cpu())
        return torch.cat(res)

    decoded = queue.Queue(maxsize=6)
    def producer(chunk):
        for take in chunk:
            try:
                decoded.put((take, *decode(take, a.view)))
            except Exception as error:
                decoded.put((take, None, error))
    for i in range(6):
        threading.Thread(target=producer, args=(todo[i::6],), daemon=True).start()
    t0, failed = time.time(), []
    for k in tqdm(range(1, len(todo) + 1), desc="takes", mininterval=30):
        take, camera, frames = decoded.get()
        if camera is None:
            failed.append((take, repr(frames)))
        else:
            cache["camera"][take] = camera
            cache["clips"]["squash"][take] = embed(frames)
        if k % 300 == 0 or k == len(todo):
            torch.save(cache, out)
    print(f"done in {(time.time() - t0) / 60:.1f} min; {len(failed)} failed: {failed[:5]}")

if __name__ == "__main__":
    main()
