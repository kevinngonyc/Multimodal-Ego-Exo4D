"""Audit availability of ego/exo video, body pose, and aligned expert text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def has_ego_and_exo(video_dir: Path) -> tuple[bool, bool]:
    names = [path.name.lower() for path in video_dir.glob("*.mp4")]
    return any(name.startswith("aria") for name in names), any(
        name.startswith(("cam", "gp")) for name in names
    )


def has_aligned_text(commentary_dir: Path) -> bool:
    """Require both an annotation record and its transcription file."""
    for data_path in commentary_dir.rglob("data.json"):
        if not (data_path.parent / "transcriptions.json").exists():
            continue
        data = json.loads(data_path.read_text(encoding="utf-8"))
        if any(annotation.get("recording_path") for annotation in data.get("annotations", [])):
            return True
    return False


def main() -> None:
    args = parse_args()
    data_root, output_root = args.data_root, args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    annotations = data_root / "annotations"

    takes = pd.read_csv(data_root / "selected_takes_128.csv").rename(
        columns={"uid": "take_uid", "take": "take_name", "label": "proficiency", "scenario": "activity"}
    )
    body_pose_uids = {
        path.stem for path in (annotations / "ego_pose").rglob("*.json") if "body" in path.parts
    }

    records = []
    for take in takes.itertuples(index=False):
        video_dir = data_root / "takes" / take.take_name / "frame_aligned_videos" / "downscaled" / "448"
        has_ego, has_exo = has_ego_and_exo(video_dir)
        text_dir = annotations / "expert_commentary" / take.take_name
        records.append(
            {
                "take_uid": take.take_uid,
                "take_name": take.take_name,
                "activity": take.activity,
                "proficiency": take.proficiency,
                "ego_video": has_ego,
                "exo_video": has_exo,
                "body_pose": take.take_uid in body_pose_uids,
                "aligned_expert_text": has_aligned_text(text_dir) if text_dir.exists() else False,
            }
        )

    table = pd.DataFrame(records)
    table["complete_model_input"] = table.ego_video & table.exo_video & table.body_pose & table.aligned_expert_text
    table.to_csv(output_root / "take_level_coverage.csv", index=False)

    names = ["Ego + exo video", "Body pose", "Pose + aligned expert text"]
    values = [
        int((table.ego_video & table.exo_video).sum()),
        int(table.body_pose.sum()),
        int(table.complete_model_input.sum()),
    ]
    figure, axis = plt.subplots(figsize=(6.2, 3.35))
    axis.barh(names, values, color="#4C78A8")
    axis.invert_yaxis()
    axis.set_xlim(0, 140)
    axis.set_xlabel("Number of selected takes")
    axis.set_title("Coverage for Expert Text--Video--Pose Analysis")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    for index, value in enumerate(values):
        axis.text(value + 2, index, str(value), va="center")
    figure.tight_layout()
    figure.savefig(output_root / "text_video_pose_coverage.png", dpi=300)
    plt.close(figure)

    summary = {
        "selected_takes": int(len(table)),
        "ego_exo_video": values[0],
        "body_pose": values[1],
        "takes_with_any_aligned_expert_text": int(table.aligned_expert_text.sum()),
        "complete_model_input": int(table.complete_model_input.sum()),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
