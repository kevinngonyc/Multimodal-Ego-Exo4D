"""Dense frozen TimeSformer-K600 clip features: 8 consecutive frames at the source 30 FPS (0.27 s per clip).

usage (from idea_1):  python dense/extract_dense.py [--scenario Dance] [--limit N]
Same model, frame squash (whole frame bilinear to 224x224), normalisation and camera choice (ego = Aria; exo =
best_exo, else exo1) as the main 8 FPS caches (ssl_adapt/extract.py); only the frame sampling differs: every source
frame, grouped into non-overlapping 8-frame clips. Stored: layer-12 patch-token std (768) per clip, fp16.
-> embeddings/dense30_{view}_{scenario}.pt = {take: (n, 768)}
   embeddings/cls_lang_clipfeats_dense30_{scenario}.pt = {take: (n, 1536)} ego | exo, truncated to the shorter view
   (the format train.load_takes reads)
"""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import av
import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ssl_adapt"))
from common import OFFICIAL, SIZE, ego_path, exo_camera  # noqa: E402

CLIP, BATCH, MEAN, STD = 8, 48, 0.45, 0.225


def decode(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        total = stream.frames
        n_clips = max(1, total // CLIP)
        frames = np.zeros((n_clips * CLIP, SIZE, SIZE, 3), np.uint8)
        got = 0
        for i, frame in enumerate(container.decode(video=0)):
            if i >= n_clips * CLIP:
                break
            frames[i] = frame.to_ndarray(format="rgb24", width=SIZE, height=SIZE, interpolation="BILINEAR")
            got = i + 1
    frames[got:] = frames[max(got - 1, 0)]
    return frames.reshape(n_clips, CLIP, SIZE, SIZE, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="Dance")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    from transformers import TimesformerModel
    model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k600").half().cuda().eval()
    mean = torch.tensor(MEAN, device="cuda", dtype=torch.float16)
    std = torch.tensor(STD, device="cuda", dtype=torch.float16)

    @torch.inference_mode()
    def embed(clips):
        res = []
        for i in range(0, len(clips), BATCH):
            x = torch.from_numpy(clips[i:i + BATCH]).cuda().half().div_(255).sub_(mean).div_(std)
            h = model(pixel_values=x.permute(0, 1, 4, 2, 3).contiguous()).last_hidden_state[:, 1:].float()
            res.append(h.std(1).half().cpu())
        return torch.cat(res)

    takes = sorted(t for t, m in OFFICIAL.items() if m["scenario"] == a.scenario)[:a.limit]
    tag = f"{a.scenario.replace(' ', '')}{'_smoke' if a.limit else ''}"
    caches = {}
    for view in ("ego", "exo"):
        out = Path("embeddings") / f"dense30_{view}_{tag}.pt"
        cache = torch.load(out, weights_only=False) if out.exists() else {}
        paths = {t: (ego_path(t) if view == "ego" else exo_camera(t)[1]) for t in takes}
        todo = [t for t in takes if t not in cache and paths[t] is not None and Path(paths[t]).exists()]
        print(f"{view}: {len(takes)} takes, {len(todo)} to go -> {out}", flush=True)
        q = queue.Queue(maxsize=3)

        def producer(chunk):
            for t in chunk:
                try:
                    q.put((t, decode(paths[t])))
                except Exception as e:
                    q.put((t, e))
        for i in range(4):
            threading.Thread(target=producer, args=(todo[i::4],), daemon=True).start()
        t0, failed, nclips = time.time(), [], 0
        for k in tqdm(range(1, len(todo) + 1), desc=view, mininterval=60):
            t, frames = q.get()
            if isinstance(frames, Exception):
                failed.append((t, repr(frames)))
            else:
                cache[t] = embed(frames); nclips += len(frames)
            if k % 100 == 0 or k == len(todo):
                torch.save(cache, out)
        print(f"{view} done in {(time.time() - t0) / 60:.1f} min, {nclips} clips; {len(failed)} failed {failed[:3]}", flush=True)
        caches[view] = cache
    both = {}
    for t in takes:
        if t in caches["ego"] and t in caches["exo"]:
            n = min(len(caches["ego"][t]), len(caches["exo"][t]))
            both[t] = torch.cat([caches["ego"][t][:n], caches["exo"][t][:n]], 1)
    out = Path("embeddings") / f"cls_lang_clipfeats_dense30_{tag}.pt"
    torch.save(both, out)
    L = np.array([len(v) for v in both.values()])
    print(f"saved {out}: {len(both)} takes, clips per take median {np.median(L):.0f}, max {L.max()}")


if __name__ == "__main__":
    main()
