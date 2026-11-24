"""Commentary-head AUC experiments for the hard-expert model (run from idea_1).

usage:  python cls_lang/comm_auc.py --pres_src hard|all [--window -2,0] [--repeats 1] [--seeds 2] [--aspect] [--smoke]

Commentary-only training (pretrain.train_comm: presence BCE + polarity SE + text 1-cos [+ aspect]) with
  --pres_src hard: every target from the best-ranked expert on the take (hard selection, as before)
  --pres_src all:  presence + text (+ aspect) from ALL experts' comments (incl. Soccer); polarity still from the hard
                   expert only (polarity is where agreement with level matters)
  --window lo,hi:  a comment paused at time T labels clips floor(T)+lo .. floor(T)+hi (default -2,0 as before)
Held-out diagnostics (outer test takes, targets built with the fold's training-only expert ranking):
  presence AUC vs own targets, vs hard-expert targets and vs all-expert targets (same window), position baseline;
  polarity (hard expert) AUC overall, WITHIN-take (clip AUC inside each take with both classes, weighted by clips),
  and the take-mean baseline (every covered clip gets its take's mean predicted polarity: pure between-take signal);
  text cosine vs own targets and vs the training-mean embedding.
-> cls_lang/comm_auc_{tag}.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from train import DEV, expert_rho, load_ratings, load_takes, weight_fn  # noqa: E402
from pretrain import SentencesA, encode, train_comm  # noqa: E402

ALL = lambda e: 1.0


def make_targets(sents, j, wh, pres_src):
    th = sents.targets(j, wh)
    if pres_src == "hard":
        return th
    ta = sents.targets(j, ALL)
    if ta is None:
        return None
    if th is None:
        ta["pol"] = np.zeros_like(ta["pol"]); ta["pol_w"] = np.zeros_like(ta["pol_w"])
    else:
        ta["pol"], ta["pol_w"] = th["pol"], th["pol_w"]
    return ta


def pres_auc(clip, te, T):
    ys, ps = [], []
    for j in te:
        tg = T.get(int(j))
        if tg is not None:
            ys.append(tg["pres"] > 0); ps.append(clip[int(j)][0])
    y = np.concatenate(ys)
    return float(roc_auc_score(y, np.concatenate(ps))) if 0 < y.mean() < 1 else float("nan")


def diagnostics(clip, te, Town, Th, Ta, mean_txt):
    out = dict(pres_auc_own=pres_auc(clip, te, Town), pres_auc_hard=pres_auc(clip, te, Th), pres_auc_all=pres_auc(clip, te, Ta))
    ys, pos = [], []
    for j in te:
        tg = Town.get(int(j))
        if tg is not None:
            n = len(tg["pres"]); ys.append(tg["pres"] > 0); pos.append(np.arange(n) / max(n - 1, 1))
    out["pres_auc_position"] = float(roc_auc_score(np.concatenate(ys), np.concatenate(pos)))
    py, pp, pm, within, wn = [], [], [], [], []
    for j in te:
        tg = Th.get(int(j))
        if tg is None:
            continue
        w = tg["pol_w"]; cov = w.sum(1) > 1e-3
        if not cov.any():
            continue
        q = clip[int(j)][1]
        yy = (tg["pol"][cov] * w[cov]).sum(1) / w[cov].sum(1) > 0
        pr = (q[cov] * w[cov]).sum(1) / w[cov].sum(1)
        py.append(yy); pp.append(pr); pm.append(np.full(len(pr), pr.mean()))
        if 0 < yy.mean() < 1:
            within.append(roc_auc_score(yy, pr)); wn.append(len(yy))
    py, pp, pm = map(np.concatenate, (py, pp, pm))
    out.update(pol_auc=float(roc_auc_score(py, pp)), pol_auc_takemean=float(roc_auc_score(py, pm)),
               pol_auc_within=float(np.average(within, weights=wn)), pol_frac_good=float(py.mean()))
    cm, cb = [], []
    for j in te:
        tg = Town.get(int(j))
        if tg is None:
            continue
        c = tg["txt_w"] > 0
        if c.any():
            tt = tg["txt"][c] / (np.linalg.norm(tg["txt"][c], axis=1, keepdims=True) + 1e-8)
            cm.append((clip[int(j)][2][c] * tt).sum(1)); cb.append(tt @ mean_txt)
    out.update(txt_cos=float(np.concatenate(cm).mean()), txt_cos_meanbaseline=float(np.concatenate(cb).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pres_src", choices=("hard", "all"), default="hard")
    ap.add_argument("--window", default="-2,0")
    ap.add_argument("--aspect", action="store_true")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    lo, hi = map(int, a.window.split(","))
    tag = f"{a.pres_src}_w{lo}_{hi}{'_aspect' if a.aspect else ''}{'_smoke' if a.smoke else ''}"
    torch.backends.cuda.matmul.allow_tf32 = True
    names, Xraw, s, y, p = load_takes()
    n = len(names)
    take_ix = {t: j for j, t in enumerate(names)}
    sents = SentencesA(take_ix, [len(x) for x in Xraw])
    base = sents.last.copy()                      # floor(pause time)
    sents.first, sents.last = np.maximum(base + lo, 0), np.maximum(base + hi, 0)
    valid = torch.as_tensor(sents.valid, device=DEV)
    R = load_ratings(take_ix)
    Xmean = np.stack([x.float().mean(0).numpy() for x in Xraw])
    print(f"{n} takes; {tag}", flush=True)
    log = []
    t0 = time.time()
    for r in range(a.repeats):
        for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(np.zeros(n), y, p)):
            fit_rel, iv_rel = next(StratifiedGroupKFold(7, shuffle=True, random_state=100 + r).split(np.zeros(len(tr)), y[tr], p[tr]))
            fit_i, iv_i = tr[fit_rel], tr[iv_rel]
            mu = np.stack([Xmean[tr][s[tr] == k].mean(0) for k in range(6)])
            sd = (Xmean[tr] - mu[s[tr]]).std(0) + 1e-6
            mu_t, sd_t = torch.as_tensor(mu, dtype=torch.float32), torch.as_tensor(sd, dtype=torch.float32)
            Xs = [(x.float() - mu_t[s[j]]) / sd_t for j, x in enumerate(Xraw)]
            wf = weight_fn("hard", expert_rho(R, s, y, tr), sents, s)
            build = lambda idx, src: {int(j): tg for j in idx if (tg := make_targets(sents, int(j), wf(int(j)), src)) is not None}
            T = build(tr, a.pres_src)
            Town, Th, Ta = build(te, a.pres_src), build(te, "hard"), build(te, "all")
            fit_c = np.array([j for j in fit_i if int(j) in T]); iv_c = np.array([j for j in iv_i if int(j) in T])
            mean_txt = np.mean([tg["txt"][tg["txt_w"] > 0].mean(0) for tg in T.values()], 0)
            mean_txt /= np.linalg.norm(mean_txt)
            for seed in range(a.seeds):
                net, ep, best = train_comm(Xs, s, fit_c, iv_c, T, valid, a.aspect, seed + 10 * r, a.epochs, a.patience, a.smoke)
                _, _, clip = encode(net, Xs, s)
                d = diagnostics(clip, te, Town, Th, Ta, mean_txt)
                log.append(dict(repeat=r, fold=f, seed=seed, best_epoch=ep, n_train_takes=len(T), **d))
                del net; torch.cuda.empty_cache()
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in log[-1].items()}),
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
                if a.smoke:
                    break
            if a.smoke:
                break
        if a.smoke:
            break
    keys = [k for k in log[0] if isinstance(log[0][k], float)]
    summary = {k: round(float(np.mean([l[k] for l in log])), 3) for k in keys}
    summary["best_epoch_median"] = float(np.median([l["best_epoch"] for l in log]))
    print("MEAN", json.dumps(summary), flush=True)
    torch.save(dict(log=log, summary=summary, args=vars(a)), Path("cls_lang") / f"comm_auc_{tag}.pt")


if __name__ == "__main__":
    main()
