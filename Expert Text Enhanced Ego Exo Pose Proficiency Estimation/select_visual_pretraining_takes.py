"""Select a balanced, participant-capped Ego--Exo proficiency pretraining cohort.

The cohort intentionally excludes the four-modality takes reserved for the
expert-text--pose experiment.  It uses only official training-split labels and
downloads synchronized 448p Ego and Exo video after this script writes UID
lists and a manifest.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=60)
    parser.add_argument("--max-takes-per-participant", type=int, default=2)
    parser.add_argument(
        "--exclude-uids-file", type=Path,
        help="Optional UID list reserved for a separate fine-tuning/evaluation cohort.",
    )
    return parser.parse_args()


def existing_take_names(data_root: Path) -> set[str]:
    takes_dir = data_root / "takes"
    if not takes_dir.exists():
        return set()
    return {path.name for path in takes_dir.iterdir() if path.is_dir()}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        row["take_uid"]: row
        for row in json.loads((args.data_root / "takes.json").read_text(encoding="utf-8"))
    }
    labels = json.loads(
        (args.data_root / "annotations" / "proficiency_demonstrator_train.json").read_text(
            encoding="utf-8"
        )
    )["annotations"]
    excluded = set()
    if args.exclude_uids_file and args.exclude_uids_file.exists():
        excluded = {
            line.strip() for line in args.exclude_uids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    on_disk = existing_take_names(args.data_root)

    pools: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    skipped = Counter()
    for item in labels:
        uid = item["take_uid"]
        take = metadata.get(uid)
        if take is None:
            skipped["missing_metadata"] += 1
            continue
        if uid in excluded:
            skipped["reserved_four_modality"] += 1
            continue
        if take["take_name"] in on_disk:
            skipped["already_downloaded"] += 1
            continue
        label = item["proficiency_score"]
        if label not in LABELS:
            skipped["unknown_label"] += 1
            continue
        pools[label][item.get("scenario_name", "unknown")].append(
            {
                "take_uid": uid,
                "take_name": take["take_name"],
                "participant_uid": take.get("participant_uid") or item.get("origin_participant_id", "unknown"),
                "label": label,
                "activity": item.get("scenario_name", "unknown"),
            }
        )

    selected: list[dict] = []
    participant_counts: Counter[str] = Counter()
    for label in LABELS:
        by_activity = pools[label]
        activities = sorted(by_activity)
        for activity in activities:
            by_activity[activity].sort(key=lambda row: (row["participant_uid"], row["take_uid"]))
        chosen = 0
        while activities and chosen < args.per_label:
            next_activities = []
            made_progress = False
            for activity in activities:
                candidates = by_activity[activity]
                while candidates and participant_counts[candidates[0]["participant_uid"]] >= args.max_takes_per_participant:
                    candidates.pop(0)
                if candidates and chosen < args.per_label:
                    row = candidates.pop(0)
                    selected.append(row)
                    participant_counts[row["participant_uid"]] += 1
                    chosen += 1
                    made_progress = True
                if candidates:
                    next_activities.append(activity)
            if not made_progress:
                break
            activities = next_activities

    selected.sort(key=lambda row: (row["label"], row["activity"], row["participant_uid"], row["take_uid"]))
    (args.output_dir / "visual_pretraining_train_uids.txt").write_text(
        "\n".join(row["take_uid"] for row in selected) + "\n", encoding="utf-8"
    )
    (args.output_dir / "visual_pretraining_train_manifest.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    summary = {
        "selected_takes": len(selected),
        "selected_participants": len(participant_counts),
        "selected_by_label": dict(Counter(row["label"] for row in selected)),
        "selected_by_activity": dict(Counter(row["activity"] for row in selected)),
        "excluded": dict(skipped),
        "settings": {
            "per_label": args.per_label,
            "max_takes_per_participant": args.max_takes_per_participant,
            "reserved_uids": len(excluded),
            "takes_already_on_disk": len(on_disk),
        },
    }
    (args.output_dir / "visual_pretraining_selection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
