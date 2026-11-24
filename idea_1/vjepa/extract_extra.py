"""Frozen V-JEPA 2 ViT-L features for the official proficiency takes NOT in the main V-JEPA caches (run from idea_1).

usage:  python vjepa/extract_extra.py {exo|ego} [N]      then   python vjepa/extract_extra.py build
Same model, framing, layers and stats as the main caches (embeddings/vjepa2_vitl_{view}8fps_clips.pt, made for the
1,338 commentary takes): back-to-back 2 s clips of 16 frames at 8 FPS, whole frame squashed to 256x256, ImageNet
mean/std; per clip, layer 12 and layer 24 (last), mean and std over the 2048 tokens -> (2 layers, 2 stats, 1024) fp16.
Cameras as for the TimeSformer extra caches: ego, and best_exo (exo1 if missing).
-> embeddings/vjepa2_vitl_{view}8fps_clips_extra.pt = {"camera": {take: cam}, "clips": {take: tensor}}
build: embeddings/cls_lang_clipfeats_vjepa.pt = {take: (n 2-s clips, 2048) fp16} = [ego L12 token std, exo L12 token std]
       for every take of the TimeSformer clip-feature cache (cls_lang_clipfeats.pt), n = min over the two views.
"""
import json
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

DATA_ROOT = next(p / "data" for p in (Path.cwd(), *Path.cwd().parents) if (p / "data" / "annotations").is_dir())
ANNOTATION_ROOT = DATA_ROOT / "annotations"
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
FPS, CLIP_FRAMES, SIZE, BATCH, LAYERS = 8, 16, 256, 4, (12, 24)
EMB = Path("embeddings")


def downscaled(rel):
    parts = rel.split("/")
    parts[-1:-1] = ["downscaled", "448"]
    return DATA_ROOT.joinpath(*parts)


def build():
    ts = torch.load(EMB / "cls_lang_clipfeats.pt", weights_only=False)
    V = {}
    for view in ("ego", "exo"):
        c = dict(torch.load(EMB / f"vjepa2_vitl_{view}8fps_clips.pt", weights_only=False)["clips"])
        c.update(torch.load(EMB / f"vjepa2_vitl_{view}8fps_clips_extra.pt", weights_only=False)["clips"])
        V[view] = c
    X, missing = {}, []
    for t in ts:
        if t not in V["ego"] or t not in V["exo"]:
            missing.append(t)
            continue
        n = min(len(V["ego"][t]), len(V["exo"][t]))
        X[t] = torch.cat([V["ego"][t][:n, 0, 1], V["exo"][t][:n, 0, 1]], 1).half()
    torch.save(X, EMB / "cls_lang_clipfeats_vjepa.pt")
    r = np.median([len(ts[t]) / len(X[t]) for t in X])
    print(f"{len(X)} of {len(ts)} takes ({len(missing)} missing: {missing[:5]}); D={next(iter(X.values())).shape[1]}; "
          f"median TimeSformer clips / V-JEPA clips = {r:.2f}")


def main(view, limit):
    from transformers import VJEPA2Model
    main_cache = EMB / f"vjepa2_vitl_{view}8fps_clips.pt"
    cache_path = EMB / (f"vjepa2_vitl_{view}8fps_clips_extra.pt" if limit is None else f"vjepa2_{view}_extra_smoke.pt")
    takes_meta = {t["take_name"]: t for t in json.loads((DATA_ROOT / "takes.json").read_text(encoding="utf-8"))}
    ego_path, exo1_path = {}, {}
    for fn in ("proficiency_demonstrator_train.json", "proficiency_demonstrator_val.json"):
        for m in json.loads((ANNOTATION_ROOT / fn).read_text(encoding="utf-8"))["annotations"]:
            take = m["video_paths"]["ego"].split("/")[1]
            ego_path[take] = downscaled(m["video_paths"]["ego"])
            exo1_path[take] = downscaled(m["video_paths"]["exo1"])

    def camera_of(take):
        if view == "ego":
            return "ego", ego_path[take]
        best = takes_meta.get(take, {}).get("best_exo")
        path = DATA_ROOT / "takes" / take / "frame_aligned_videos" / "downscaled" / "448" / f"{best}.mp4"
        if best and path.exists():
            return best, path
        return "exo1 (no best_exo)", exo1_path[take]

    def decode(take):
        camera, path = camera_of(take)
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
                if k is not None:
                    frames[k] = frame.to_ndarray(format="rgb24", width=SIZE, height=SIZE, interpolation="BILINEAR")
                    got[k] = True
        for k in np.flatnonzero(~got):
            frames[k] = frames[max(k - 1, 0)]
        return camera, frames.reshape(n_clips, CLIP_FRAMES, SIZE, SIZE, 3)

    have = set(torch.load(main_cache, mmap=True, weights_only=False)["clips"])
    ts_takes = set(torch.load(EMB / "cls_lang_clipfeats.pt", mmap=True, weights_only=False))
    takes = sorted(t for t in ts_takes if t not in have)
    print(f"{len(ts_takes)} takes in the TimeSformer clip cache; {len(have)} in the main V-JEPA {view} cache; "
          f"{len(takes)} to extract", flush=True)
    takes = takes[:limit]
    cache = torch.load(cache_path, weights_only=False) if cache_path.exists() else {"camera": {}, "clips": {}}
    todo = [t for t in takes if t not in cache["clips"]]
    print(f"{len(takes) - len(todo)} cached, {len(todo)} to go -> {cache_path}", flush=True)
    if not todo:
        return
    model = VJEPA2Model.from_pretrained(MODEL_ID, torch_dtype=torch.float16).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 1, 3, 1, 1).half()
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 1, 3, 1, 1).half()

    @torch.inference_mode()
    def embed(clips):
        out = []
        for i in range(0, len(clips), BATCH):
            x = torch.from_numpy(clips[i:i + BATCH]).cuda().permute(0, 1, 4, 2, 3).half().div_(255)
            x = ((x - mean) / std).contiguous()
            res = model(pixel_values_videos=x, output_hidden_states=True, skip_predictor=True)
            layers = []
            for layer in LAYERS:
                h = (res.last_hidden_state if layer == LAYERS[-1] else res.hidden_states[layer]).float()
                layers.append(torch.stack([h.mean(1), h.std(1)], 1))
            out.append(torch.stack(layers, 1).half().cpu())
        return torch.cat(out)

    decoded = queue.Queue(maxsize=3)

    def producer(chunk):
        for take in chunk:
            try:
                decoded.put((take, *decode(take)))
            except Exception as error:
                decoded.put((take, None, error))

    for i in range(4):
        threading.Thread(target=producer, args=(todo[i::4],), daemon=True).start()
    t0, failed = time.time(), []
    for k in tqdm(range(1, len(todo) + 1), desc="takes", mininterval=60):
        take, camera, frames = decoded.get()
        if camera is None:
            failed.append((take, repr(frames)))
        else:
            cache["camera"][take] = camera
            cache["clips"][take] = embed(frames)
        if k % 100 == 0 or k == len(todo):
            torch.save(cache, cache_path)
    print(f"done in {(time.time() - t0) / 60:.1f} min; {len(failed)} failed: {failed[:5]}", flush=True)
    t = next(iter(cache["clips"]))
    x = cache["clips"][t].float()
    print(f"{len(cache['clips'])} takes; {t} {cache['camera'][t]} {tuple(x.shape)} finite {bool(torch.isfinite(x).all())}")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        assert sys.argv[1] in ("ego", "exo")
        main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
