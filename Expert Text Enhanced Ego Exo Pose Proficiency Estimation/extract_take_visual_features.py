"""Extract frozen ResNet-50 Ego and Exo features for visual pretraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import ResNet50_Weights, resnet50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--frames-per-take", type=int, default=8)
    return parser.parse_args()


def video_feature(path: Path, encoder, transform, device: str, frames_per_take: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or count < 1:
        capture.release()
        return None
    frames = []
    for frame_index in np.linspace(0, count - 1, frames_per_take, dtype=int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(torch.from_numpy(rgb).permute(2, 0, 1)))
    capture.release()
    if not frames:
        return None
    with torch.no_grad():
        return encoder(torch.stack(frames).to(device)).mean(dim=0).cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    benchmark = {
        item["take_uid"]: item
        for item in json.loads(
            (args.data_root / "annotations" / "proficiency_demonstrator_train.json").read_text(
                encoding="utf-8"
            )
        )["annotations"]
    }
    weights = ResNet50_Weights.DEFAULT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = resnet50(weights=None)
    encoder.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    encoder.fc = torch.nn.Identity()
    encoder.to(device).eval()
    transform = weights.transforms()
    usable, ego_vectors, exo_vectors, skipped = [], [], [], []
    for index, row in enumerate(rows, start=1):
        item = benchmark.get(row["take_uid"])
        if not item:
            skipped.append({"take_uid": row["take_uid"], "reason": "missing_label"})
            continue
        folder = args.data_root / "takes" / row["take_name"] / "frame_aligned_videos" / "downscaled" / "448"
        ego_path = folder / Path(item["video_paths"]["ego"]).name
        exo_path = folder / Path(item["video_paths"]["exo1"]).name
        if not ego_path.exists() or not exo_path.exists():
            skipped.append({"take_uid": row["take_uid"], "reason": "missing_video"})
            continue
        print(f"[{index}/{len(rows)}] {row['take_name']}", flush=True)
        ego = video_feature(ego_path, encoder, transform, device, args.frames_per_take)
        exo = video_feature(exo_path, encoder, transform, device, args.frames_per_take)
        if ego is None or exo is None:
            skipped.append({"take_uid": row["take_uid"], "reason": "unreadable_video"})
            continue
        usable.append(row)
        ego_vectors.append(ego)
        exo_vectors.append(exo)
    if not usable:
        raise RuntimeError("No complete Ego/Exo pairs were found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ego=np.vstack(ego_vectors), exo=np.vstack(exo_vectors))
    args.metadata_output.write_text(json.dumps(usable, indent=2), encoding="utf-8")
    args.metadata_output.with_name(args.metadata_output.stem + "_skipped.json").write_text(
        json.dumps(skipped, indent=2), encoding="utf-8"
    )
    print(json.dumps({"takes": len(usable), "skipped": len(skipped)}, indent=2))


if __name__ == "__main__":
    main()
