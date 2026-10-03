"""Extract reproducible ResNet-50 ego/exo features and visualize them with t-SNE."""
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
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torchvision.models import ResNet50_Weights, resnet50

SEED = 42
FRAMES_PER_TAKE = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True, help="Local ResNet-50 ImageNet checkpoint.")
    return parser.parse_args()


def read_features(video_path: Path, model: torch.nn.Module, transform, device: str) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 1:
        capture.release()
        return None
    frames = []
    for frame_index in np.linspace(0, frame_count - 1, FRAMES_PER_TAKE, dtype=int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if success:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(torch.from_numpy(rgb).permute(2, 0, 1)))
    capture.release()
    if not frames:
        return None
    with torch.no_grad():
        return model(torch.stack(frames).to(device)).mean(dim=0).cpu().numpy()


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    data_root, output_root = args.data_root, args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    annotations_dir = data_root / "annotations"

    takes = pd.read_csv(data_root / "selected_takes_128.csv").rename(
        columns={"uid": "take_uid", "take": "take_name", "label": "proficiency", "scenario": "activity"}
    )
    labels = json.loads((annotations_dir / "proficiency_demonstrator_train.json").read_text(encoding="utf-8"))["annotations"]
    eligible = takes[takes.take_uid.isin({item["take_uid"] for item in labels})].copy()
    sample = pd.concat([group.sample(n=min(8, len(group)), random_state=SEED) for _, group in eligible.groupby("proficiency")])
    annotation_by_uid = {item["take_uid"]: item for item in labels}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Identity()
    model.to(device).eval()

    records, ego_features, exo_features = [], [], []
    for take in sample.itertuples(index=False):
        paths = annotation_by_uid[take.take_uid]["video_paths"]
        video_dir = data_root / "takes" / take.take_name / "frame_aligned_videos" / "downscaled" / "448"
        ego_path = video_dir / Path(paths["ego"]).name
        exo_path = video_dir / Path(paths["exo1"]).name
        ego = read_features(ego_path, model, weights.transforms(), device)
        exo = read_features(exo_path, model, weights.transforms(), device)
        if ego is None or exo is None:
            print(f"Skipping unreadable pair: {take.take_name}")
            continue
        records.append({"take_uid": take.take_uid, "take_name": take.take_name, "activity": take.activity, "proficiency": take.proficiency})
        ego_features.append(ego)
        exo_features.append(exo)

    table = pd.DataFrame(records)
    ego_array, exo_array = np.vstack(ego_features), np.vstack(exo_features)
    np.save(output_root / "ego_resnet_features.npy", ego_array)
    np.save(output_root / "exo_resnet_features.npy", exo_array)

    metrics = {"sampled_takes": len(table), "frames_per_take": FRAMES_PER_TAKE, "seed": SEED, "device": device}
    sns.set_theme(style="whitegrid")
    for name, features in [("ego", ego_array), ("exo", exo_array)]:
        metrics[name] = {key: float(silhouette_score(features, table[key])) for key in ("proficiency", "activity")}
        perplexity = min(20, max(5, (len(features) - 1) // 3))
        coordinates = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=SEED).fit_transform(features)
        table[f"{name}_tsne_x"], table[f"{name}_tsne_y"] = coordinates[:, 0], coordinates[:, 1]
        figure, axis = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=table, x=f"{name}_tsne_x", y=f"{name}_tsne_y", hue="proficiency", s=80, alpha=.85, ax=axis)
        axis.set(title=f"Pretrained {name.title()}-Video Features by Proficiency", xlabel="t-SNE dimension 1", ylabel="t-SNE dimension 2")
        figure.tight_layout()
        figure.savefig(output_root / f"{name}_features.png", dpi=300)
        plt.close(figure)
    table.to_csv(output_root / "visual_embeddings_tsne.csv", index=False)
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
