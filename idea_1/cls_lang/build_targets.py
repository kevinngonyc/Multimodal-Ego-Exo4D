"""Sentence-level commentary targets for the CLS + language model (run from idea_1).

Every transcribed timestamped comment (expert_commentary/<take>/<expert>/data.json + transcriptions.json) on the
official demonstrator takes -> sentences, each with
  emb      all-MiniLM-L6-v2 sentence embedding (384, unit norm)            -> L_text target
  p_good   P(praise) vs tip, logistic regression trained on the official good-execution / tip-for-improvement lists
           (proficiency_demonstration_*.json, 40k sentences, 5 scenarios; transfers to Rock Climbing by language)
  aspect   P(aspect) over aspects_for(scenario) (same vocabulary as goodtip_aspects.ipynb): per-scenario logistic
           regression on the Qwen3.5-4B tags of the list sentences; Rock Climbing (no lists) tagged by Qwen directly.
           The last aspect, "other" (describes what happens / not about the performance), marks neutral sentences.
Checks: CV accuracy of both classifiers on the list sentences; polarity AUC on raw comments that match a list entry
(same take, expert and pause time).
-> embeddings/cls_lang_sentences.pt
"""
import glob
import json
import re
import string
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

S = Path(__file__).parent
sys.path.insert(0, str(S))
from vocab import ACTIVITY, aspects_for  # noqa: E402

DATA = next(p / "data" for p in (Path.cwd(), *Path.cwd().parents) if (p / "data" / "annotations").is_dir())
A = DATA / "annotations"
OUT = Path("embeddings") / "cls_lang_sentences.pt"
RC_CACHE = Path("embeddings") / "qwen35_rc_raw_aspects.pt"
LETTERS = string.ascii_uppercase


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(re.sub(r"[^A-Za-z]", "", s)) >= 3]


def raw_sentences(official):
    rows = []
    for f in glob.glob(str(A / "expert_commentary" / "*" / "*" / "data.json")):
        take = Path(f).parent.parent.name
        if take not in official:
            continue
        tp = Path(f).with_name("transcriptions.json")
        if not tp.exists():
            continue
        d, tr = json.load(open(f, encoding="utf-8")), json.load(open(tp, encoding="utf-8"))
        for ci, a in enumerate(d.get("annotations", [])):
            text = tr.get(a.get("recording_path"), {}).get("text", "").strip()
            for s in split_sentences(text):
                rows.append(dict(take=take, scenario=official[take], expert=d.get("user_id"), comment=ci,
                                 time=float(a["video_time"]), sentence=s, context=text))
    return rows


def rc_prompt(sentence, context):
    options = "\n".join(f"{LETTERS[i]}. {d}" for i, (_, d) in enumerate(aspects_for("Rock Climbing")))
    ctx = f"The full comment was: \"{context}\"\n\n" if context != sentence else ""
    return (f"An expert watching a person's {ACTIVITY['Rock Climbing']} performance paused the video and made a "
            f"comment.\n\n{ctx}Which aspect of the performance is this sentence about?\n\"{sentence}\"\n\n"
            f"{options}\n\nChoose the most specific option that fits. Reply with exactly one letter.")


def tag_rock_climbing(keys):
    cache = torch.load(RC_CACHE) if RC_CACHE.exists() else {}
    todo = sorted(set(keys) - set(cache), key=lambda k: len(rc_prompt(*k)))
    print(f"Rock Climbing: {len(todo)} sentences to tag with Qwen", flush=True)
    if todo:
        from tqdm.auto import tqdm
        from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-4B", quantization_config=bnb,
                                                         device_map="auto").eval()
        tok = AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B").tokenizer
        tok.padding_side = "right"
        n = len(aspects_for("Rock Climbing"))
        ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS[:n]]

        @torch.inference_mode()
        def probs(batch_keys):
            texts = [tok.apply_chat_template([{"role": "user", "content": rc_prompt(*k)}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) for k in batch_keys]
            b = tok(texts, return_tensors="pt", padding=True).to(model.device)
            last = b["attention_mask"].sum(1) - 1
            keep = torch.unique(last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(**b, logits_to_keep=keep).logits
            r = lg[torch.arange(len(texts)), torch.searchsorted(keep, last)].float()
            return r[:, ids].softmax(-1).cpu()

        for start in tqdm(range(0, len(todo), 2000), desc="RC chunks", mininterval=60):
            chunk = todo[start:start + 2000]
            for b in range(0, len(chunk), 16):
                for k, p in zip(chunk[b:b + 16], probs(chunk[b:b + 16])):
                    cache[k] = p.half()
            torch.save(cache, RC_CACHE)
        del model
        torch.cuda.empty_cache()
    return cache


def main():
    LEVELS = {"Novice": 1, "Early Expert": 2, "Intermediate Expert": 3, "Late Expert": 4}
    official = {}
    for fn in ("proficiency_demonstrator_train.json", "proficiency_demonstrator_val.json"):
        for m in json.load(open(A / fn, encoding="utf-8"))["annotations"]:
            official[m["video_paths"]["ego"].split("/")[1]] = m["scenario_name"]
    rows = raw_sentences(official)
    print(f"{len(rows)} raw sentences, {len({(r['take'], r['expert'], r['comment']) for r in rows})} comments, "
          f"{len({r['take'] for r in rows})} takes", flush=True)

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    embed = lambda xs: st.encode(xs, batch_size=512, normalize_embeddings=True, show_progress_bar=False,
                                 convert_to_numpy=True)
    E = embed([r["sentence"] for r in rows])

    lists = torch.load("embeddings/qwen35_goodtip_aspects.pt", weights_only=False)["scores"]
    lk = list(lists)
    LE = embed([k[2] for k in lk])
    lscen = np.array([k[0] for k in lk])
    lpol = np.array([k[1] for k in lk])

    # polarity: one classifier for all scenarios (+ scenario one-hot)
    scen_names = sorted(set(official.values()))
    oh = lambda sc: (np.asarray(sc)[:, None] == np.array(scen_names)[None]).astype(np.float32)
    Xl = np.hstack([LE, oh(lscen)])
    pol = LogisticRegression(C=1.0, max_iter=3000)
    print(f"polarity CV accuracy on list sentences: {cross_val_score(pol, Xl, lpol, cv=5).mean():.3f}", flush=True)
    pol.fit(Xl, lpol)
    rscen = np.array([r["scenario"] for r in rows])
    p_good = pol.predict_proba(np.hstack([E, oh(rscen)]))[:, 1]

    # polarity check on raw comments that match a list entry (same take, expert, pause time)
    dem = {}
    for fn in ("proficiency_demonstration_train.json", "proficiency_demonstration_val.json"):
        for m in json.load(open(A / fn, encoding="utf-8"))["annotations"]:
            dem[m["video_paths"]["ego"].split("/")[1]] = m
    labels = {}
    for take, m in dem.items():
        for key, v in (("good_executions", 1), ("tips_for_improvement", 0)):
            for e in m.get(key) or []:
                labels.setdefault((take, e["expert_uid"], round(float(e["video_time"]), 1)), set()).add(v)
    cid = [(r["take"], r["expert"], round(r["time"], 1)) for r in rows]
    agg = {}
    for k, p in zip(cid, p_good):
        agg.setdefault(k, []).append(p)
    pure = [(np.mean(agg[k]), next(iter(v))) for k, v in labels.items() if k in agg and len(v) == 1]
    if pure:
        y = np.array([b for _, b in pure]); s = np.array([a for a, _ in pure])
        print(f"raw comments matched to a single-polarity list entry: {len(pure)}; AUC of mean p_good = "
              f"{roc_auc_score(y, s):.3f}", flush=True)

    # aspects
    n_max = max(len(aspects_for(s)) for s in scen_names)
    aspect = np.zeros((len(rows), n_max), np.float32)
    for sc in scen_names:
        m = rscen == sc
        if sc == "Rock Climbing":
            continue
        lm = lscen == sc
        y = np.array([int(lists[k].float().argmax()) for k, keep in zip(lk, lm) if keep])
        clf = LogisticRegression(C=1.0, max_iter=3000)
        acc = cross_val_score(clf, LE[lm], y, cv=5).mean() if len(np.unique(y)) > 1 and lm.sum() > 50 else float("nan")
        clf.fit(LE[lm], y)
        P = np.zeros((m.sum(), len(aspects_for(sc))), np.float32)
        P[:, clf.classes_] = clf.predict_proba(E[m])
        aspect[np.flatnonzero(m), :P.shape[1]] = P
        print(f"aspect classifier {sc}: {lm.sum()} list sentences, CV accuracy {acc:.3f}", flush=True)
    rc = np.flatnonzero(rscen == "Rock Climbing")
    keys = [(rows[i]["sentence"], rows[i]["context"]) for i in rc]
    cache = tag_rock_climbing(keys)
    for i, k in zip(rc, keys):
        p = cache[k].float().numpy()
        aspect[i, :len(p)] = p

    out = {k: np.array([r[k] for r in rows]) for k in ("take", "scenario", "expert", "comment", "time", "sentence")}
    out.update(emb=E.astype(np.float16), p_good=p_good.astype(np.float32), aspect=aspect,
               aspect_names={s: [a for a, _ in aspects_for(s)] for s in scen_names})
    torch.save(out, OUT)
    print(f"saved {OUT}: {len(rows)} sentences")


if __name__ == "__main__":
    main()
