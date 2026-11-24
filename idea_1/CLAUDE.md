# idea_1: proficiency estimation on Ego-Exo4D

Start with `final_report/README.md`. It explains the project, the final model, the evaluation protocol, every result
(as CSV in `final_report/tables/`), the experiment history and where each number comes from. The full write-up is
`final_report/report_doc.pdf` (Markdown copy: `final_report/report_doc.md`).

- On a fresh clone, `data/`, `embeddings/`, `*.pt`, `*.log` and `*.out` are absent (gitignored). Experiments cannot be
  rerun or rescored there; use the tracked tables, logs (`final_report/logs/*.txt`) and figures.
- Final model: `python cls_lang/joint.py --lam 0.5` (tag `joint_comm`), commentary weight 0.5, not 4.
- Scope set by the user: main target = 1–10 expert rating, official level second; video-only inference; no audio,
  no pose; commentary, ratings and levels are training signals only.
- When running experiments: one GPU job at a time (12 GB card); lay out the full plan (what runs, what changes,
  control, metric, decision rule, time) and wait for the user's go-ahead; change one thing at a time.
