"""Evaluate timestamp-level multimodal proficiency models.

The script trains a compact gated-fusion model using participant-disjoint
cross-validation.  Segment logits are mean-pooled within each take before the
four-level proficiency loss and take-level metrics are computed.  The
shuffled-text control trains on aligned data, then permutes text segments
within each held-out take at inference time.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


LABELS = ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    parser.add_argument(
        "--visual-pretrain", type=Path,
        help="Optional Ego--Exo visual projector checkpoint trained on the larger cohort.",
    )
    parser.add_argument(
        "--freeze-visual-projectors", action="store_true",
        help="Keep transferred Ego and Exo projectors fixed during multimodal training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GatedFusion(torch.nn.Module):
    def __init__(self, input_dims: dict[str, int], modalities: tuple[str, ...], hidden_dim: int) -> None:
        super().__init__()
        self.modalities = modalities
        self.projectors = torch.nn.ModuleDict(
            {
                name: torch.nn.Sequential(
                    torch.nn.LayerNorm(input_dims[name]),
                    torch.nn.Linear(input_dims[name], hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.20),
                )
                for name in modalities
            }
        )
        self.gates = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * len(modalities), hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, len(modalities)),
            torch.nn.Sigmoid(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * len(modalities), hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.30),
            torch.nn.Linear(hidden_dim, len(LABELS)),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = [self.projectors[name](inputs[name]) for name in self.modalities]
        joined = torch.cat(embeddings, dim=1)
        gates = self.gates(joined)
        weighted = torch.cat(
            [embedding * gates[:, index : index + 1] for index, embedding in enumerate(embeddings)],
            dim=1,
        )
        return self.classifier(weighted)


def take_groups(rows: list[dict], indices: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for local_index, global_index in enumerate(indices):
        grouped[rows[int(global_index)]["take_uid"]].append(local_index)
    groups = [np.asarray(indices_for_take, dtype=np.int64) for indices_for_take in grouped.values()]
    labels = np.asarray([LABELS.index(rows[int(indices[group[0]])]["label"]) for group in groups], dtype=np.int64)
    take_ids = list(grouped)
    return groups, labels, take_ids


def pooled_loss(logits: torch.Tensor, groups: list[np.ndarray], targets: torch.Tensor, criterion) -> torch.Tensor:
    pooled = torch.stack([logits[torch.as_tensor(group, device=logits.device)].mean(dim=0) for group in groups])
    return criterion(pooled, targets)


def pooled_predictions(logits: torch.Tensor, groups: list[np.ndarray]) -> np.ndarray:
    pooled = torch.stack([logits[torch.as_tensor(group, device=logits.device)].mean(dim=0) for group in groups])
    return pooled.argmax(dim=1).detach().cpu().numpy()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.arange(len(LABELS)), average="macro", zero_division=0)),
    }


def standardize(
    features: dict[str, np.ndarray], train_index: np.ndarray, test_index: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train, test = {}, {}
    for name, matrix in features.items():
        scaler = StandardScaler()
        train[name] = scaler.fit_transform(matrix[train_index]).astype(np.float32)
        test[name] = scaler.transform(matrix[test_index]).astype(np.float32)
    return train, test


def as_tensors(features: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    return {name: torch.from_numpy(matrix).to(device) for name, matrix in features.items()}


def permute_text_within_take(
    text: np.ndarray, rows: list[dict], test_index: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle text only within a held-out take, keeping activity/person context fixed."""
    shuffled = text.copy()
    by_take: dict[str, list[int]] = defaultdict(list)
    for local_index, global_index in enumerate(test_index):
        by_take[rows[int(global_index)]["take_uid"]].append(local_index)
    for local_indices in by_take.values():
        if len(local_indices) > 1:
            shuffled[local_indices] = shuffled[rng.permutation(local_indices)]
    return shuffled


def train_one(
    train_features: dict[str, np.ndarray],
    test_features: dict[str, np.ndarray],
    train_groups: list[np.ndarray],
    test_groups: list[np.ndarray],
    train_targets: np.ndarray,
    modalities: tuple[str, ...],
    args: argparse.Namespace,
    seed: int,
    device: str,
) -> tuple[GatedFusion, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    set_seed(seed)
    input_dims = {name: matrix.shape[1] for name, matrix in train_features.items()}
    model = GatedFusion(input_dims, modalities, args.hidden_dim).to(device)
    if args.visual_pretrain:
        checkpoint = torch.load(args.visual_pretrain, map_location="cpu", weights_only=False)
        if checkpoint.get("hidden_dim") != args.hidden_dim:
            raise ValueError("Visual-pretraining hidden dimension does not match the fusion model.")
        source = checkpoint["state_dict"]
        destination = model.state_dict()
        transferred = 0
        for key, value in source.items():
            if key.startswith("projectors.ego.") or key.startswith("projectors.exo."):
                if key in destination and destination[key].shape == value.shape:
                    destination[key] = value
                    transferred += 1
        has_visual_projector = "ego" in model.projectors or "exo" in model.projectors
        if has_visual_projector and not transferred:
            raise ValueError("The visual-pretraining checkpoint has no compatible Ego/Exo projectors.")
        model.load_state_dict(destination)
        if args.freeze_visual_projectors:
            for name in ("ego", "exo"):
                if name in model.projectors:
                    for parameter in model.projectors[name].parameters():
                        parameter.requires_grad = False
    class_counts = np.bincount(train_targets, minlength=len(LABELS))
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_tensors, test_tensors = as_tensors(train_features, device), as_tensors(test_features, device)
    target_tensor = torch.from_numpy(train_targets).to(device)
    model.train()
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = pooled_loss(model(train_tensors), train_groups, target_tensor, criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model.eval(), train_tensors, test_tensors


def evaluate_configuration(
    name: str,
    modalities: tuple[str, ...],
    features: dict[str, np.ndarray],
    rows: list[dict],
    splitter: StratifiedGroupKFold,
    args: argparse.Namespace,
    device: str,
) -> tuple[dict, list[dict], list[dict]]:
    take_rows = {}
    for row in rows:
        take_rows.setdefault(row["take_uid"], row)
    representative = list(take_rows.values())
    # Map the take-level splitter back to interval rows.
    take_labels = np.asarray([LABELS.index(row["label"]) for row in representative])
    take_groups_ids = np.asarray([row["participant_uid"] for row in representative])
    take_ids = np.asarray([row["take_uid"] for row in representative])
    take_to_interval = defaultdict(list)
    for index, row in enumerate(rows):
        take_to_interval[row["take_uid"]].append(index)

    all_true, all_pred = [], []
    prediction_rows, shuffled_fold_results = [], []
    for fold, (take_train, take_test) in enumerate(
        splitter.split(take_ids, take_labels, take_groups_ids), start=1
    ):
        train_index = np.asarray(
            [index for take_index in take_train for index in take_to_interval[take_ids[take_index]]], dtype=np.int64
        )
        test_index = np.asarray(
            [index for take_index in take_test for index in take_to_interval[take_ids[take_index]]], dtype=np.int64
        )
        train_features, test_features = standardize(features, train_index, test_index)
        train_groups, train_targets, _ = take_groups(rows, train_index)
        test_groups, test_targets, test_take_ids = take_groups(rows, test_index)
        model, _, test_tensors = train_one(
            train_features,
            test_features,
            train_groups,
            test_groups,
            train_targets,
            modalities,
            args,
            SEED + fold,
            device,
        )
        with torch.no_grad():
            prediction = pooled_predictions(model(test_tensors), test_groups)
        all_true.extend(test_targets.tolist())
        all_pred.extend(prediction.tolist())
        for take_uid, truth, predicted in zip(test_take_ids, test_targets, prediction):
            prediction_rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "take_uid": take_uid,
                    "true_label": LABELS[int(truth)],
                    "predicted_label": LABELS[int(predicted)],
                }
            )

        if name == "ego_exo_pose_aligned_text":
            shuffled_results = []
            for repeat in range(args.shuffle_repeats):
                rng = np.random.default_rng(SEED + fold * 100 + repeat)
                altered = {key: value.clone() for key, value in test_tensors.items()}
                shuffled_text = permute_text_within_take(test_features["text"], rows, test_index, rng)
                altered["text"] = torch.from_numpy(shuffled_text).to(device)
                with torch.no_grad():
                    shuffled_prediction = pooled_predictions(model(altered), test_groups)
                shuffled_results.append(metrics(test_targets, shuffled_prediction))
            shuffled_fold_results.extend(shuffled_results)

    result = metrics(np.asarray(all_true), np.asarray(all_pred))
    result["takes"] = len(all_true)
    if shuffled_fold_results:
        result["shuffled_text_control"] = {
            metric: float(np.mean([repeat[metric] for repeat in shuffled_fold_results]))
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        }
        result["shuffled_text_repeats"] = len(shuffled_fold_results)
    return result, prediction_rows, shuffled_fold_results


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = np.load(args.features)
    features = {name: archive[name].astype(np.float32) for name in ("ego", "exo", "pose", "text")}
    rows = json.loads(args.metadata.read_text(encoding="utf-8"))
    if len(rows) != len(features["ego"]):
        raise ValueError("Feature rows and metadata rows do not match.")
    if len({row["take_uid"] for row in rows}) < args.folds:
        raise ValueError("Not enough takes for requested cross-validation folds.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}: {len(rows)} intervals from {len({r['take_uid'] for r in rows})} takes.")
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    configurations = {
        "ego_exo": ("ego", "exo"),
        "ego_exo_pose": ("ego", "exo", "pose"),
        "ego_exo_pose_aligned_text": ("ego", "exo", "pose", "text"),
        "masked_text_only": ("text",),
    }
    summary = {
        "evaluation": "participant-disjoint stratified cross-validation, take-level mean pooling of timestamp logits",
        "device": device,
        "takes": len({row["take_uid"] for row in rows}),
        "intervals": len(rows),
        "classes": LABELS,
        "class_counts": {
            label: sum(row["label"] == label for row in {r["take_uid"]: r for r in rows}.values())
            for label in LABELS
        },
        "models": {},
    }
    predictions, shuffled = [], []
    for name, modalities in configurations.items():
        print(f"Training {name}...", flush=True)
        result, rows_out, shuffled_out = evaluate_configuration(
            name, modalities, features, rows, splitter, args, device
        )
        summary["models"][name] = result
        predictions.extend(rows_out)
        shuffled.extend(shuffled_out)
    (args.output_dir / "interval_fusion_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "interval_fusion_predictions.json").write_text(
        json.dumps(predictions, indent=2), encoding="utf-8"
    )
    (args.output_dir / "shuffled_text_repeats.json").write_text(json.dumps(shuffled, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
