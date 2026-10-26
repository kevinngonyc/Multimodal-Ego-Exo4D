"""Train transferable Ego--Exo projection layers on the visual cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_interval_fusion import GatedFusion, LABELS, SEED, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    archive = np.load(args.features)
    features = {
        "ego": archive["ego"].astype(np.float32),
        "exo": archive["exo"].astype(np.float32),
    }
    rows = json.loads(args.metadata.read_text(encoding="utf-8"))
    if len(rows) != len(features["ego"]):
        raise ValueError("Feature and metadata row counts do not match.")
    targets = np.asarray([LABELS.index(row["label"]) for row in rows], dtype=np.int64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GatedFusion(
        {name: matrix.shape[1] for name, matrix in features.items()},
        ("ego", "exo"),
        args.hidden_dim,
    ).to(device)
    tensors = {name: torch.from_numpy(matrix).to(device) for name, matrix in features.items()}
    target_tensor = torch.from_numpy(targets).to(device)
    counts = np.bincount(targets, minlength=len(LABELS))
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(tensors), target_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                train_accuracy = (model(tensors).argmax(dim=1) == target_tensor).float().mean().item()
            print(f"epoch={epoch} loss={loss.item():.4f} train_accuracy={train_accuracy:.3f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dim": args.hidden_dim,
            "modalities": ["ego", "exo"],
            "classes": LABELS,
            "takes": len(rows),
            "class_counts": {label: int((targets == index).sum()) for index, label in enumerate(LABELS)},
        },
        args.output,
    )
    print(f"Saved transferable visual projectors to {args.output}")


if __name__ == "__main__":
    main()
