"""Create timestamp-aligned expert-text records and analyze text embeddings."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

RATING_PATTERN = re.compile(
    r"\b(?:novice|early\s+expert|intermediate\s+expert|late\s+expert|"
    r"proficien\w*|rating|score|good\s+execution|tip\s+for\s+improvement|beginner)\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root, output_root = args.data_root, args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    annotations = data_root / "annotations"
    takes = pd.read_csv(data_root / "selected_takes_128.csv").rename(
        columns={"uid": "take_uid", "take": "take_name", "label": "proficiency", "scenario": "activity"}
    )
    body_uids = {path.stem for path in (annotations / "ego_pose").rglob("*.json") if "body" in path.parts}
    takes = takes[takes.take_uid.isin(body_uids)]

    segments = []
    for take in takes.itertuples(index=False):
        for data_path in (annotations / "expert_commentary" / take.take_name).rglob("data.json"):
            transcript_path = data_path.parent / "transcriptions.json"
            if not transcript_path.exists():
                continue
            data = json.loads(data_path.read_text(encoding="utf-8"))
            transcripts = json.loads(transcript_path.read_text(encoding="utf-8"))
            for annotation in data.get("annotations", []):
                text = transcripts.get(annotation.get("recording_path"), {}).get("text", "").strip()
                if text:
                    segments.append({
                        "take_uid": take.take_uid, "take_name": take.take_name,
                        "activity": take.activity, "proficiency": take.proficiency,
                        "timestamp_s": annotation.get("video_time"),
                        "duration_s": annotation.get("duration_approx"), "text": text,
                        "word_count": len(re.findall(r"\b\w+[\w'-]*\b", text)),
                        "direct_rating_term": bool(RATING_PATTERN.search(text)),
                    })
    segment_table = pd.DataFrame(segments)
    if segment_table.empty:
        raise RuntimeError("No aligned expert-text segments found.")
    segment_table.to_csv(output_root / "aligned_text_segments.csv", index=False)

    # Use every selected training take with commentary for the language-modality
    # analysis. The 52-take pose subset remains the correct subset for temporal
    # alignment and later multimodal model comparisons.
    all_takes = pd.read_csv(data_root / "selected_takes_128.csv").rename(
        columns={"uid": "take_uid", "take": "take_name", "label": "proficiency", "scenario": "activity"}
    )
    documents_rows = []
    for take in all_takes.itertuples(index=False):
        texts = []
        for transcript_path in (annotations / "expert_commentary" / take.take_name).rglob("transcriptions.json"):
            transcripts = json.loads(transcript_path.read_text(encoding="utf-8"))
            texts.extend(item.get("text", "").strip() for item in transcripts.values() if item.get("text", "").strip())
        if texts:
            documents_rows.append({
                "take_uid": take.take_uid, "take_name": take.take_name,
                "activity": take.activity, "proficiency": take.proficiency,
                "text": " ".join(texts),
            })
    documents = pd.DataFrame(documents_rows)
    documents["aligned_segments"] = documents.take_uid.map(
        segment_table.groupby("take_uid").size()
    ).fillna(0).astype(int)
    documents.to_csv(output_root / "commentary_documents.csv", index=False)

    # The model was downloaded before analysis. Requiring a local copy keeps
    # reruns deterministic and prevents an unexpected network dependency.
    model = SentenceTransformer(args.model, local_files_only=True)
    features = model.encode(documents.text.tolist(), normalize_embeddings=True, show_progress_bar=True)
    scores = {key: float(silhouette_score(features, documents[key])) for key in ("activity", "proficiency")}
    perplexity = min(30, max(5, (len(documents) - 1) // 3))
    coordinates = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(features)
    documents["tsne_x"], documents["tsne_y"] = coordinates[:, 0], coordinates[:, 1]
    documents.to_csv(output_root / "commentary_neural_embeddings_tsne.csv", index=False)

    sns.set_theme(style="whitegrid")
    for column, filename, title in [
        ("proficiency", "commentary_neural_by_proficiency.png", "Neural Expert-Text Embeddings by Proficiency"),
        ("activity", "commentary_neural_by_activity.png", "Neural Expert-Text Embeddings by Activity"),
    ]:
        figure, axis = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=documents, x="tsne_x", y="tsne_y", hue=column, s=72, alpha=.85, ax=axis)
        axis.set(title=title, xlabel="t-SNE dimension 1", ylabel="t-SNE dimension 2")
        figure.tight_layout()
        figure.savefig(output_root / filename, dpi=300)
        plt.close(figure)

    metrics = {
        "aligned_segments": int(len(segment_table)),
        "commentary_documents": int(len(documents)),
        "pose_covered_takes_with_text": int(segment_table.take_uid.nunique()),
        "median_segments_per_pose_covered_take": float(segment_table.groupby("take_uid").size().median()),
        "direct_rating_segments": int(segment_table.direct_rating_term.sum()),
        "model": args.model,
        "silhouette": scores,
    }
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
