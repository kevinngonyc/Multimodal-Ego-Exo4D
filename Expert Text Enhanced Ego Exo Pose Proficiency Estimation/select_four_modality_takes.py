"""Select pose-and-commentary eligible Ego-Exo4D proficiency takes.

Run after downloading the proficiency, EgoPose, and expert-commentary
annotations. The script writes train/validation UID lists that can be given to
the Ego-Exo4D downloader to retrieve only synchronized 448p ego/exo videos.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


LABELS = ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-per-label", type=int, default=75)
    parser.add_argument("--max-val-per-label", type=int, default=30)
    parser.add_argument("--max-takes-per-participant", type=int, default=2)
    return parser.parse_args()


def has_expert_text(commentary_root: Path, take_name: str) -> bool:
    """Return True when an expert-commentary transcription exists for a take."""
    take_root = commentary_root / take_name
    for data_path in take_root.rglob("data.json"):
        transcript_path = data_path.parent / "transcriptions.json"
        if not transcript_path.exists():
            continue
        annotations = json.loads(data_path.read_text(encoding="utf-8"))
        transcripts = json.loads(transcript_path.read_text(encoding="utf-8"))
        for annotation in annotations.get("annotations", []):
            text = transcripts.get(annotation.get("recording_path"), {}).get("text", "").strip()
            if text:
                return True
    return False


def select_split(
    data_root: Path,
    split: str,
    metadata: dict[str, dict],
    limit_per_label: int,
    max_takes_per_participant: int,
) -> tuple[list[dict], dict[str, int]]:
    annotations_root = data_root / "annotations"
    annotation_path = annotations_root / f"proficiency_demonstrator_{split}.json"
    pose_root = annotations_root / "ego_pose" / split / "body" / "annotation"
    commentary_root = annotations_root / "expert_commentary"
    if not annotation_path.exists() or not pose_root.exists() or not commentary_root.exists():
        return [], {"missing_required_annotation_directory": 1}

    annotated = json.loads(annotation_path.read_text(encoding="utf-8"))["annotations"]
    eligible = []
    for item in annotated:
        uid = item["take_uid"]
        take = metadata.get(uid)
        if take is None or not (pose_root / f"{uid}.json").exists():
            continue
        if not has_expert_text(commentary_root, take["take_name"]):
            continue
        eligible.append({
            "take_uid": uid,
            "take_name": take["take_name"],
            "participant_uid": take.get("participant_uid", "unknown"),
            "label": item["proficiency_score"],
            "activity": item.get("scenario_name", "unknown"),
        })

    # Round-robin over activity within each class, with a participant cap.
    chosen = []
    counts_by_label: dict[str, int] = defaultdict(int)
    counts_by_participant: dict[str, int] = defaultdict(int)
    by_label_activity: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in sorted(eligible, key=lambda x: (x["label"], x["activity"], x["participant_uid"], x["take_uid"])):
        by_label_activity[row["label"]][row["activity"]].append(row)

    for label in LABELS:
        pools = by_label_activity[label]
        activities = sorted(pools)
        while activities and counts_by_label[label] < limit_per_label:
            next_activities = []
            selected_this_round = False
            for activity in activities:
                while pools[activity] and counts_by_participant[pools[activity][0]["participant_uid"]] >= max_takes_per_participant:
                    pools[activity].pop(0)
                if pools[activity] and counts_by_label[label] < limit_per_label:
                    row = pools[activity].pop(0)
                    chosen.append(row)
                    counts_by_label[label] += 1
                    counts_by_participant[row["participant_uid"]] += 1
                    selected_this_round = True
                if pools[activity]:
                    next_activities.append(activity)
            if not selected_this_round:
                break
            activities = next_activities

    availability = {label: sum(row["label"] == label for row in eligible) for label in LABELS}
    return chosen, availability


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        take["take_uid"]: take
        for take in json.loads((args.data_root / "takes.json").read_text(encoding="utf-8"))
    }
    summary = {}
    for split, limit in (("train", args.max_train_per_label), ("val", args.max_val_per_label)):
        selected, available = select_split(
            args.data_root, split, metadata, limit, args.max_takes_per_participant
        )
        (args.output_dir / f"four_modality_{split}_uids.txt").write_text(
            "\n".join(row["take_uid"] for row in selected) + "\n", encoding="utf-8"
        )
        (args.output_dir / f"four_modality_{split}_manifest.json").write_text(
            json.dumps(selected, indent=2), encoding="utf-8"
        )
        summary[split] = {
            "eligible_by_label": available,
            "selected_by_label": {label: sum(row["label"] == label for row in selected) for label in LABELS},
            "selected_takes": len(selected),
            "selected_participants": len({row["participant_uid"] for row in selected}),
        }
    (args.output_dir / "four_modality_selection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
