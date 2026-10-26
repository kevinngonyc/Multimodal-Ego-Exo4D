"""Build the complete 295-take visual pretraining manifest.

It joins newly selected takes with already-downloaded visual takes, while
excluding the 52 aligned four-modality takes reserved for the final
pose-and-text experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--reserved-uids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = {
        item["take_uid"]: item
        for item in json.loads((args.data_root / "takes.json").read_text(encoding="utf-8"))
    }
    labels = json.loads(
        (args.data_root / "annotations" / "proficiency_demonstrator_train.json").read_text(
            encoding="utf-8"
        )
    )["annotations"]
    reserved = set(args.reserved_uids.read_text(encoding="utf-8").split())
    new_rows = json.loads(args.new_manifest.read_text(encoding="utf-8"))
    rows = {row["take_uid"]: row for row in new_rows}
    on_disk = {
        path.name
        for path in (args.data_root / "takes").iterdir()
        if path.is_dir()
    }
    for label in labels:
        uid = label["take_uid"]
        take = metadata.get(uid)
        if not take or uid in reserved or take["take_name"] not in on_disk or uid in rows:
            continue
        rows[uid] = {
            "take_uid": uid,
            "take_name": take["take_name"],
            "participant_uid": take.get("participant_uid") or label.get("origin_participant_id", "unknown"),
            "label": label["proficiency_score"],
            "activity": label.get("scenario_name", "unknown"),
        }
    complete = sorted(rows.values(), key=lambda row: (row["label"], row["take_uid"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(complete, indent=2), encoding="utf-8")
    counts = {}
    for row in complete:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(json.dumps({"takes": len(complete), "class_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
