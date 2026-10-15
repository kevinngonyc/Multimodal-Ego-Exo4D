"""Participant-disjoint frozen-feature baselines for proficiency estimation.

Model inputs are ego RGB, exo RGB, and 3D body pose only. Expert text is not
used. The script evaluates four increasingly multimodal conditions with the
same folds: ego, exo, ego+exo, and ego+exo+pose.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet50_Weights, resnet50

SEED = 42
N_SPLITS = 3
FRAMES_PER_TAKE = 6
LABELS = ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]
POSE_JOINTS = [
    "nose", "left-eye", "right-eye", "left-ear", "right-ear",
    "left-shoulder", "right-shoulder", "left-elbow", "right-elbow",
    "left-wrist", "right-wrist", "left-hip", "right-hip",
    "left-knee", "right-knee", "left-ankle", "right-ankle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True, help="Local ImageNet ResNet-50 checkpoint.")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def video_feature(path: Path, model: torch.nn.Module, transform, device: str) -> np.ndarray | None:
    """Uniformly sample RGB frames and average frozen ResNet-50 features."""
    capture = cv2.VideoCapture(str(path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        capture.release()
        return None
    tensors: list[torch.Tensor] = []
    for frame_index in np.linspace(0, total_frames - 1, FRAMES_PER_TAKE, dtype=int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if success:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensors.append(transform(torch.from_numpy(rgb).permute(2, 0, 1)))
    capture.release()
    if not tensors:
        return None
    with torch.no_grad():
        return model(torch.stack(tensors).to(device)).mean(dim=0).cpu().numpy()


def pose_feature(path: Path) -> np.ndarray:
    """Compute normalized 3D-pose position and velocity summary statistics.

    Every frame is centered at the midpoint of the hips and scaled by shoulder
    width. Statistics are mean, standard deviation, 10th percentile, and 90th
    percentile for positions and mean/std for frame-to-frame velocity.
    """
    annotations = json.loads(path.read_text(encoding="utf-8"))
    frames: list[np.ndarray] = []
    for timestamp in sorted(annotations, key=float):
        for observation in annotations[timestamp]:
            joints = observation.get("annotation3D", {})
            if not {"left-hip", "right-hip", "left-shoulder", "right-shoulder"}.issubset(joints):
                continue
            hip_center = np.mean([[joints["left-hip"][axis], joints["right-hip"][axis]] for axis in "xyz"], axis=1)
            shoulder_width = np.linalg.norm(
                np.array([joints["left-shoulder"][axis] for axis in "xyz"])
                - np.array([joints["right-shoulder"][axis] for axis in "xyz"])
            )
            if not np.isfinite(shoulder_width) or shoulder_width < 1e-5:
                continue
            values = []
            for joint in POSE_JOINTS:
                if joint in joints:
                    values.extend((np.array([joints[joint][axis] for axis in "xyz"]) - hip_center) / shoulder_width)
                else:
                    values.extend([np.nan, np.nan, np.nan])
            frames.append(np.asarray(values, dtype=np.float32))
    if not frames:
        raise ValueError(f"No usable 3D pose frames in {path}")
    # Missing joints are represented as zero after centering and scaling. This
    # gives every take the same feature dimension without emitting NaN-based
    # summary warnings.
    sequence = np.nan_to_num(np.asarray(frames), nan=0.0, posinf=0.0, neginf=0.0)
    position = np.concatenate([
        np.mean(sequence, axis=0), np.std(sequence, axis=0),
        np.percentile(sequence, 10, axis=0), np.percentile(sequence, 90, axis=0),
    ])
    velocity = np.diff(sequence, axis=0)
    motion = np.concatenate([np.mean(np.abs(velocity), axis=0), np.std(velocity, axis=0)]) if len(velocity) else np.zeros(sequence.shape[1] * 2)
    return np.concatenate([position, motion])


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, ax=axis)
    axis.set(xlabel="Predicted proficiency", ylabel="True proficiency", title="Ego + Exo + Pose: participant-disjoint CV")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    print("Loading matched ego/exo/pose subset...", flush=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = output_root / "feature_cache"
    cache_root.mkdir(exist_ok=True)
    annotations_root = args.data_root / "annotations"

    takes = pd.read_csv(args.data_root / "selected_takes_128.csv").rename(
        columns={"uid": "take_uid", "take": "take_name", "label": "proficiency", "scenario": "activity"}
    )
    pose_paths = {path.stem: path for path in (annotations_root / "ego_pose" / "train" / "body" / "annotation").glob("*.json")}
    takes = takes[takes.take_uid.isin(pose_paths)].copy().reset_index(drop=True)
    print(f"Found {len(takes)} pose-covered takes.", flush=True)
    metadata = {item["take_uid"]: item for item in json.loads((args.data_root / "takes.json").read_text(encoding="utf-8"))}
    benchmark = {item["take_uid"]: item for item in json.loads((annotations_root / "proficiency_demonstrator_train.json").read_text(encoding="utf-8"))["annotations"]}
    takes["participant_uid"] = takes.take_uid.map(lambda uid: metadata[uid]["participant_uid"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ResNet50_Weights.DEFAULT
    encoder = resnet50(weights=None)
    encoder.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    encoder.fc = torch.nn.Identity()
    encoder.to(device).eval()
    print(f"Extracting frozen video features on {device}...", flush=True)

    rows, ego_vectors, exo_vectors, pose_vectors = [], [], [], []
    for take in takes.itertuples(index=False):
        print(f"Processing {take.take_name}", flush=True)
        cache_path = cache_root / f"{take.take_uid}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            ego, exo = cached["ego"], cached["exo"]
            pose = pose_feature(pose_paths[take.take_uid])
            print("  loaded cached features", flush=True)
        else:
            paths = benchmark[take.take_uid]["video_paths"]
            video_dir = args.data_root / "takes" / take.take_name / "frame_aligned_videos" / "downscaled" / "448"
            ego = video_feature(video_dir / Path(paths["ego"]).name, encoder, weights.transforms(), device)
            exo = video_feature(video_dir / Path(paths["exo1"]).name, encoder, weights.transforms(), device)
            pose = pose_feature(pose_paths[take.take_uid])
            if ego is not None and exo is not None:
                np.savez_compressed(cache_path, ego=ego, exo=exo)
        if ego is None or exo is None:
            print(f"Skipping unreadable video pair: {take.take_name}")
            continue
        rows.append(take._asdict())
        ego_vectors.append(ego)
        exo_vectors.append(exo)
        pose_vectors.append(pose)

    table = pd.DataFrame(rows)
    print(f"Training/evaluating on {len(table)} complete takes...", flush=True)
    ego_array, exo_array, pose_array = np.vstack(ego_vectors), np.vstack(exo_vectors), np.vstack(pose_vectors)
    np.save(output_root / "ego_features.npy", ego_array)
    np.save(output_root / "exo_features.npy", exo_array)
    np.save(output_root / "pose_features.npy", pose_array)

    inputs = {
        "ego": ego_array,
        "exo": exo_array,
        "ego_exo": np.hstack([ego_array, exo_array]),
        "ego_exo_pose": np.hstack([ego_array, exo_array, pose_array]),
    }
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    y = table.proficiency.to_numpy()
    groups = table.participant_uid.to_numpy()
    all_predictions: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "evaluation": "3-fold stratified participant-disjoint cross-validation",
        "seed": SEED,
        "takes": int(len(table)),
        "participants": int(table.participant_uid.nunique()),
        "frames_per_take": FRAMES_PER_TAKE,
        "classes": LABELS,
        "class_counts": table.proficiency.value_counts().to_dict(),
        "models": {},
    }
    for name, feature_matrix in inputs.items():
        predictions = np.empty(len(table), dtype=object)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(feature_matrix, y, groups), start=1):
            classifier = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                LogisticRegression(max_iter=4000, class_weight="balanced", C=0.1, random_state=SEED),
            )
            classifier.fit(feature_matrix[train_idx], y[train_idx])
            predictions[test_idx] = classifier.predict(feature_matrix[test_idx])
            for index in test_idx:
                all_predictions.append({"model": name, "fold": fold, "take_uid": table.take_uid.iloc[index],
                                        "participant_uid": table.participant_uid.iloc[index], "true_label": y[index],
                                        "predicted_label": predictions[index]})
        summary["models"][name] = {
            "accuracy": float(accuracy_score(y, predictions)),
            "macro_f1": float(f1_score(y, predictions, labels=LABELS, average="macro", zero_division=0)),
        }
        if name == "ego_exo_pose":
            plot_confusion(y, predictions, output_root / "ego_exo_pose_confusion_matrix.png")

    table.to_csv(output_root / "matched_take_manifest.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(output_root / "cross_validation_predictions.csv", index=False)
    (output_root / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
