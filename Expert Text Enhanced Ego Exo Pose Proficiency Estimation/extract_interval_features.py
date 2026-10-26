"""Create aligned Ego, Exo, pose, and masked-text features for each comment.

Each row represents one expert-commentary timestamp.  Ego and Exo RGB frames,
3D pose, and the masked commentary segment are extracted from a fixed window
centred on that timestamp.  The output is a cached feature file for the
timestamp-level fusion experiments.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torchvision.models import ResNet50_Weights, resnet50


LABELS = ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]
JOINTS = [
    "nose", "left-eye", "right-eye", "left-ear", "right-ear",
    "left-shoulder", "right-shoulder", "left-elbow", "right-elbow",
    "left-wrist", "right-wrist", "left-hip", "right-hip",
    "left-knee", "right-knee", "left-ankle", "right-ankle",
]
RATING_TERMS = re.compile(
    r"\b(?:novice|early\s+expert|intermediate\s+expert|late\s+expert|"
    r"proficien\w*|rating|score|good\s+execution|tip\s+for\s+improvement|beginner)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--frames-per-window", type=int, default=2)
    parser.add_argument(
        "--max-segments-per-take",
        type=int,
        default=0,
        help="Use this many evenly spaced comment intervals per take; 0 uses all intervals.",
    )
    parser.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    return parser.parse_args()


def mask_rating_terms(text: str) -> str:
    return RATING_TERMS.sub("[MASK]", text).strip()


def commentary_segments(commentary_root: Path, take_name: str) -> list[tuple[float, str]]:
    """Return (video timestamp, masked text) segments for one take."""
    results: list[tuple[float, str]] = []
    take_root = commentary_root / take_name
    for data_path in take_root.rglob("data.json"):
        transcription_path = data_path.parent / "transcriptions.json"
        if not transcription_path.exists():
            continue
        data = json.loads(data_path.read_text(encoding="utf-8"))
        transcriptions = json.loads(transcription_path.read_text(encoding="utf-8"))
        for annotation in data.get("annotations", []):
            text = transcriptions.get(annotation.get("recording_path"), {}).get("text", "")
            text = mask_rating_terms(text)
            timestamp = annotation.get("video_time")
            if text and isinstance(timestamp, (int, float)):
                results.append((float(timestamp), text))
    return results


def normalized_pose_window(path: Path, centre: float, window_seconds: float) -> np.ndarray | None:
    """Summarize hip-centred, shoulder-normalized 3D pose around one timestamp."""
    annotations = json.loads(path.read_text(encoding="utf-8"))
    frames: list[tuple[float, np.ndarray]] = []
    for timestamp, observations in annotations.items():
        try:
            time = float(timestamp)
        except ValueError:
            continue
        for observation in observations:
            points = observation.get("annotation3D", {})
            required = {"left-hip", "right-hip", "left-shoulder", "right-shoulder"}
            if not required.issubset(points):
                continue
            hips = np.array(
                [[points["left-hip"][axis], points["right-hip"][axis]] for axis in "xyz"],
                dtype=np.float32,
            )
            hip_centre = hips.mean(axis=1)
            shoulder_width = np.linalg.norm(
                np.array([points["left-shoulder"][axis] for axis in "xyz"], dtype=np.float32)
                - np.array([points["right-shoulder"][axis] for axis in "xyz"], dtype=np.float32)
            )
            if not np.isfinite(shoulder_width) or shoulder_width < 1e-5:
                continue
            vector = []
            for joint in JOINTS:
                if joint in points:
                    vector.extend(
                        (np.array([points[joint][axis] for axis in "xyz"], dtype=np.float32) - hip_centre)
                        / shoulder_width
                    )
                else:
                    vector.extend([np.nan, np.nan, np.nan])
            frames.append((time, np.asarray(vector, dtype=np.float32)))
    if not frames:
        return None
    half = window_seconds / 2
    selected = [vector for time, vector in frames if abs(time - centre) <= half]
    if not selected:
        selected = [
            vector
            for _, vector in sorted(frames, key=lambda item: abs(item[0] - centre))[: min(3, len(frames))]
        ]
    sequence = np.nan_to_num(np.asarray(selected), nan=0.0, posinf=0.0, neginf=0.0)
    positions = np.concatenate(
        [
            sequence.mean(axis=0),
            sequence.std(axis=0),
            np.percentile(sequence, 10, axis=0),
            np.percentile(sequence, 90, axis=0),
        ]
    )
    velocity = np.diff(sequence, axis=0)
    motion = (
        np.concatenate([np.abs(velocity).mean(axis=0), velocity.std(axis=0)])
        if len(velocity)
        else np.zeros(sequence.shape[1] * 2, dtype=np.float32)
    )
    return np.concatenate([positions, motion]).astype(np.float32)


def read_window_feature(
    path: Path,
    centre: float,
    window_seconds: float,
    frames_per_window: int,
    encoder: torch.nn.Module,
    transform,
    device: str,
) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    duration = frame_count / fps
    offsets = np.linspace(-window_seconds / 2, window_seconds / 2, frames_per_window)
    tensors = []
    for offset in offsets:
        time = min(max(0.0, centre + float(offset)), max(0.0, duration - 1.0 / fps))
        capture.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
        ok, frame = capture.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensors.append(transform(torch.from_numpy(rgb).permute(2, 0, 1)))
    capture.release()
    if not tensors:
        return None
    with torch.no_grad():
        return encoder(torch.stack(tensors).to(device)).mean(dim=0).cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    annotations = args.data_root / "annotations"
    pose_paths = {
        path.stem: path
        for path in (annotations / "ego_pose" / "train" / "body" / "annotation").glob("*.json")
    }
    benchmark = {
        item["take_uid"]: item
        for item in json.loads(
            (annotations / "proficiency_demonstrator_train.json").read_text(encoding="utf-8")
        )["annotations"]
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ResNet50_Weights.DEFAULT
    encoder = resnet50(weights=None)
    encoder.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    encoder.fc = torch.nn.Identity()
    encoder.to(device).eval()
    text_encoder = SentenceTransformer(args.text_model, local_files_only=True, device=device)
    transform = weights.transforms()

    metadata, ego_features, exo_features, pose_features, text_features = [], [], [], [], []
    skipped = []
    for take_index, row in enumerate(rows, start=1):
        uid, take_name = row["take_uid"], row["take_name"]
        if uid not in pose_paths or uid not in benchmark:
            skipped.append({"take_uid": uid, "reason": "missing_pose_or_label"})
            continue
        video_paths = benchmark[uid]["video_paths"]
        video_dir = args.data_root / "takes" / take_name / "frame_aligned_videos" / "downscaled" / "448"
        ego_path = video_dir / Path(video_paths["ego"]).name
        exo_path = video_dir / Path(video_paths["exo1"]).name
        if not ego_path.exists() or not exo_path.exists():
            skipped.append({"take_uid": uid, "reason": "missing_ego_or_exo_video"})
            continue
        segments = commentary_segments(annotations / "expert_commentary", take_name)
        if args.max_segments_per_take and len(segments) > args.max_segments_per_take:
            sample_indices = np.linspace(
                0, len(segments) - 1, args.max_segments_per_take, dtype=int
            )
            segments = [segments[index] for index in sample_indices]
        print(f"[{take_index}/{len(rows)}] {take_name}: {len(segments)} commentary segments", flush=True)
        for segment_index, (timestamp, text) in enumerate(segments):
            ego = read_window_feature(
                ego_path, timestamp, args.window_seconds, args.frames_per_window, encoder, transform, device
            )
            exo = read_window_feature(
                exo_path, timestamp, args.window_seconds, args.frames_per_window, encoder, transform, device
            )
            pose = normalized_pose_window(pose_paths[uid], timestamp, args.window_seconds)
            if ego is None or exo is None or pose is None:
                skipped.append(
                    {"take_uid": uid, "segment_index": segment_index, "reason": "unreadable_interval"}
                )
                continue
            text_feature = text_encoder.encode(
                text, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)
            metadata.append(
                {
                    "take_uid": uid,
                    "take_name": take_name,
                    "participant_uid": row["participant_uid"],
                    "label": row["label"],
                    "activity": row["activity"],
                    "timestamp_seconds": timestamp,
                    "segment_index": segment_index,
                    "masked_text": text,
                }
            )
            ego_features.append(ego)
            exo_features.append(exo)
            pose_features.append(pose)
            text_features.append(text_feature)

    if not metadata:
        raise RuntimeError("No complete aligned intervals were extracted.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ego=np.vstack(ego_features),
        exo=np.vstack(exo_features),
        pose=np.vstack(pose_features),
        text=np.vstack(text_features),
    )
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    args.metadata_output.with_name(args.metadata_output.stem + "_skipped.json").write_text(
        json.dumps(skipped, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "intervals": len(metadata),
                "takes": len({row["take_uid"] for row in metadata}),
                "skipped": len(skipped),
                "feature_file": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
