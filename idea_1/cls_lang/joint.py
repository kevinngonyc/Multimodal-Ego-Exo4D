"""One network for both targets: expert rating and official level, optionally + all-expert commentary (run from idea_1).

usage:  python cls_lang/joint.py [--lam 0.5] [--rank 1] [--official] [--view ego|exo] [--feat_cache F --clip_sec 2] [--tag joint] [--repeats 4] [--seeds 2] [--smoke]
Same encoder as train.Net (frozen ego+exo clip tokens -> 128, 2-layer transformer) but TWO task tokens in front, each
+ the scenario embedding: [RATING] -> per-scenario rating head (z-scored expert rating, as rating.py --head
per_scenario); [LEVEL] -> level score (+ scenario bias) and 4 class logits (+ scenario bias).
Loss = MSE(rating) on rated takes + MSE(level) + CE(level) on all takes
       [+ lam * (presence BCE + polarity SE + text 1-cos) on the clip tokens, from ALL experts' timestamped comments,
        polarity centred per expert with training-fold means, exactly as rating.py --arm rating_comm].
All training takes are used. Early stopping per task on the inner participant split (1/7): the rating checkpoint = best
within-scenario Spearman with the z-rating, the level checkpoint = best within-scenario Spearman with level; training
stops once neither has improved for 8 epochs. Commentary outputs are read from the rating checkpoint.
Test takes get their scenario from a video classifier fit on the training takes.
--feat_cache / --clip_sec: another clip-feature cache (dict take -> (n clips, D)) whose clips last clip_sec seconds
(V-JEPA 2: 2 s); comment windows and the training crop (384 s) are converted to that clip length.
Held-out commentary diagnostics (lam > 0; rating checkpoint, outer test takes): presence AUC (all-expert targets) and
position baseline; polarity AUC on raw (uncentred) all-expert polarity, overall / within-take / take-mean baseline;
text cosine vs the training-mean embedding.
Also per fold, for the rating report: the frozen probes of rating.py (shared and per-scenario).
--official: one split instead of the 4 x 5 folds: train on the official train takes, test on official val; also fits
the frozen PCA level probe on that split (fz_prob / fz_reg) and saves per-seed test predictions (seed_out) and, with
lam > 0, per-clip presence logits, per-aspect polarity and text embeddings of the val takes (clip_out); report with
cls_lang/official_report.py [TAG].
--view ego|exo: one view only (768 dims); the frozen ridge rating probes are skipped (NaN), the level probe uses that view.
-> cls_lang/rating_{tag}.pt  (rating.py --report --report_tags ... {tag}); level in lvl_reg / lvl_prob
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import train as _train  # noqa: E402
from train import DEV, Net, Sentences, batchify, load_ratings, load_takes  # noqa: E402
from rating import level_readout, probe, probe_ps, rho_w, z_rating  # noqa: E402


class JointNet(Net):
    def __init__(self, d_in=1536):
        super().__init__(d_in=d_in)
        nn.init.normal_(self.cls, std=0.02)
        self.cls2 = nn.Parameter(torch.randn(1, 1, self.d) * 0.02)
        self.rreg = nn.Linear(self.d, 6)
        self.rreg_b = nn.Parameter(torch.zeros(6))

    def forward(self, x, tpos, mask, s):
        h = self.inp(x) + self.time_enc(tpos)
        e = self.scen(s)[:, None]
        h = torch.cat([self.cls + e, self.cls2 + e, h], 1)
        m = torch.cat([torch.zeros(len(x), 2, dtype=torch.bool, device=x.device), ~mask], 1)
        h = self.norm(self.enc(h, src_key_padding_mask=m))
        zr, zl, tok = h[:, 0], h[:, 1], h[:, 2:]
        rating = self.rreg(zr).gather(1, s[:, None]).squeeze(1) + self.rreg_b[s]
        return (rating, self.reg(zl).squeeze(-1) + self.reg_b[s], self.cl(zl) + self.cl_b[s],
                self.pres(tok).squeeze(-1), torch.tanh(self.pol(tok)), self.txt(tok))


@torch.no_grad()
def predict(net, idx, Xs, s, bs=32, clips=False):
    net.eval()
    order = sorted(idx, key=lambda j: len(Xs[j]))
    out, clip = {}, {}
    for i in range(0, len(order), bs):
        ch = order[i:i + bs]
        b = batchify(ch, Xs, s)
        r, l, lg, pres, pol, txt = net(b["x"], b["t"], b["mask"], b["s"])
        txt = F.normalize(txt, dim=-1)
        for k, (j, a, c, d) in enumerate(zip(ch, r.cpu().numpy(), l.cpu().numpy(), F.softmax(lg, -1).cpu().numpy())):
            out[j] = (a, c, d)
            if clips:
                n = b["lens"][k]
                clip[j] = (pres[k, :n].cpu().numpy(), pol[k, :n].cpu().numpy(), txt[k, :n].cpu().numpy())
    net.train()
    res = (np.array([out[j][0] for j in idx]), np.array([out[j][1] for j in idx]), np.stack([out[j][2] for j in idx]))
    return (*res, clip) if clips else res


def comm_diagnostics(clip, te, T, mean_txt):
    """held-out commentary metrics; T = raw all-expert targets of the test takes."""
    ys, ps, pos = [], [], []
    py, pp, pm, within, wn = [], [], [], [], []
    cm, cb = [], []
    for j in te:
        tg = T.get(int(j))
        if tg is None:
            continue
        pr_, po_, tx_ = clip[int(j)]
        n = len(tg["pres"])
        ys.append(tg["pres"] > 0); ps.append(pr_); pos.append(np.arange(n) / max(n - 1, 1))
        w = tg["pol_w"]; cov = w.sum(1) > 1e-3
        if cov.any():
            yy = (tg["pol"][cov] * w[cov]).sum(1) / w[cov].sum(1) > 0
            q = (po_[cov] * w[cov]).sum(1) / w[cov].sum(1)
            py.append(yy); pp.append(q); pm.append(np.full(len(q), q.mean()))
            if 0 < yy.mean() < 1:
                within.append(roc_auc_score(yy, q)); wn.append(len(yy))
        c = tg["txt_w"] > 0
        if c.any():
            tt = tg["txt"][c] / (np.linalg.norm(tg["txt"][c], axis=1, keepdims=True) + 1e-8)
            cm.append((tx_[c] * tt).sum(1)); cb.append(tt @ mean_txt)
    y = np.concatenate(ys)
    py, pp, pm = map(np.concatenate, (py, pp, pm))
    return dict(pres_auc=float(roc_auc_score(y, np.concatenate(ps))),
                pres_auc_position=float(roc_auc_score(y, np.concatenate(pos))),
                pol_auc=float(roc_auc_score(py, pp)), pol_auc_takemean=float(roc_auc_score(py, pm)),
                pol_auc_within=float(np.average(within, weights=wn)),
                txt_cos=float(np.concatenate(cm).mean()), txt_cos_meanbaseline=float(np.concatenate(cb).mean()))


def train_joint(Xs, s, y, t, fit_i, iv_i, seed, epochs, smoke, d_in, T=None, lam=0.0, crop=384, pairs=None, rank=0.0,
                n_pairs=8):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    prng = np.random.default_rng(seed + 1000)  # separate stream: the main batches are drawn exactly as without --rank
    if rank > 0 and pairs is not None and len(pairs):
        # scenario share of the sampled pairs = its share of rated fitting takes (as in the rating MSE), not of pairs
        ps_ = s[pairs[:, 0].astype(int)]
        rated_fit = fit_i[~np.isnan(t[fit_i])]
        w_sc = {k: (s[rated_fit] == k).sum() / max((ps_ == k).sum(), 1) for k in range(6)}
        pairs_p = np.array([w_sc[k] for k in ps_]); pairs_p /= pairs_p.sum()
    net = JointNet(d_in).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    yt = torch.as_tensor(y, dtype=torch.float32, device=DEV)
    tt = torch.as_tensor(np.nan_to_num(t), dtype=torch.float32, device=DEV)
    has = torch.as_tensor(~np.isnan(t), device=DEV)
    ivr = iv_i[~np.isnan(t[iv_i])]
    best = {"rating": (-2, None, 0), "level": (-2, None, 0)}
    since = {"rating": 0, "level": 0}
    for ep in range(1, epochs + 1):
        perm = rng.permutation(fit_i)
        for i in range(0, len(perm), 16):
            idx = perm[i:i + 16]
            b = batchify(idx, Xs, s, crop=crop, rng=rng, T=T if lam > 0 else None)
            r, l, lg, pres, pol, txt = net(b["x"], b["t"], b["mask"], b["s"])
            ix = torch.as_tensor(idx, device=DEV)
            loss = F.mse_loss(l, yt[ix]) + F.cross_entropy(lg, (yt[ix] - 1).long())
            hm = has[ix]
            if hm.any():
                loss = loss + F.mse_loss(r[hm], tt[ix][hm])
            if lam > 0 and b["has"].sum() > 0:
                tm = b["mask"] & (b["has"][:, None] > 0)
                l_pres = F.binary_cross_entropy_with_logits(pres[tm], b["pres"][tm])
                l_pol = ((pol - b["pol"]) ** 2 * b["polw"])[tm].sum() / tm.sum()
                cos = F.cosine_similarity(txt, b["txt"] + 1e-8, dim=-1)
                l_txt = ((1 - cos) * b["txtw"])[tm].sum() / tm.sum()
                loss = loss + lam * (l_pres + l_pol + l_txt)
            if rank > 0 and pairs is not None and len(pairs):
                pk = pairs[prng.choice(len(pairs), size=n_pairs, p=pairs_p)]
                pa, pb, sg = pk[:, 0].astype(int), pk[:, 1].astype(int), torch.as_tensor(pk[:, 2], dtype=torch.float32, device=DEV)
                bp = batchify(np.concatenate([pa, pb]), Xs, s, crop=crop, rng=prng)
                rp = net(bp["x"], bp["t"], bp["mask"], bp["s"])[0]
                loss = loss + rank * F.softplus(-(rp[:n_pairs] - rp[n_pairs:]) * sg).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        pr, pl, _ = predict(net, iv_i, Xs, s)
        pos = {j: k for k, j in enumerate(iv_i)}
        cur = {"rating": rho_w(pr[[pos[j] for j in ivr]], t[ivr], s[ivr]), "level": rho_w(pl, y[iv_i], s[iv_i])}
        for task in best:
            if cur[task] > best[task][0]:
                best[task] = (cur[task], {k: v.clone() for k, v in net.state_dict().items()}, ep); since[task] = 0
            else:
                since[task] += 1
        if min(since.values()) >= 8 or (smoke and ep >= 2):
            break
    nets = {}
    for task, (_, state, _) in best.items():
        nn_ = JointNet(d_in).to(DEV); nn_.load_state_dict(state); nets[task] = nn_
    return nets, {task: best[task][2] for task in best}, {task: best[task][0] for task in best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--feat_cache", default=None)
    ap.add_argument("--clip_sec", type=int, default=1)
    ap.add_argument("--rank", type=float, default=0.0, help="weight of the within-expert, same-task pairwise ranking loss")
    ap.add_argument("--official", action="store_true",
                    help="one split: train on the official train takes, test on official val (repeats forced to 1)")
    ap.add_argument("--view", default="both", choices=["both", "ego", "exo"],
                    help="input view: both (ego + best exo, 1,536 dims) or one view only (768 dims); frozen ridge probes skipped")
    ap.add_argument("--tag", default="joint")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    if a.feat_cache:
        _train.FEAT_CACHE = Path(a.feat_cache)
    names, Xraw, s, y, p = load_takes()
    VS = {"both": [slice(0, 768), slice(768, 1536)], "ego": [slice(0, 768)], "exo": [slice(768, 1536)]}[a.view]
    if a.view != "both":  # cache columns are [ego 768 | exo 768]
        Xraw = [x[:, VS[0]].contiguous() for x in Xraw]
        VS = [slice(0, 768)]
    y = y.astype(float)
    n = len(names)
    d_in = Xraw[0].shape[1]
    crop = 384 // a.clip_sec
    take_ix = {tk: j for j, tk in enumerate(names)}
    R = load_ratings(take_ix)
    rated = np.zeros(n, bool)
    for (j, _) in R:
        rated[j] = True
    task_name = {m["take_name"]: m.get("task_name", "?") for m in json.load(open(_train.DATA / "takes.json", encoding="utf-8"))}
    task_of = [task_name.get(tk, "?") for tk in names]
    sents = v_raw = None
    if a.lam > 0:
        sents = Sentences(take_ix, [len(x) for x in Xraw])
        if a.clip_sec > 1:  # comment at time T labels the clips covering floor(T) - 2 .. floor(T) seconds
            sents.last = sents.last // a.clip_sec
            sents.first = np.maximum(sents.last - int(np.ceil(2 / a.clip_sec)), 0)
        v_raw = sents.v.copy()
    Xmean = np.stack([x.float().mean(0).numpy() for x in Xraw])
    if a.official:
        a.repeats = 1
        val = {m["video_paths"]["ego"].split("/")[1]
               for m in json.load(open(_train.A / "proficiency_demonstrator_val.json", encoding="utf-8"))["annotations"]}
        is_val = np.array([tk in val for tk in names])
        off_split = [(np.flatnonzero(~is_val), np.flatnonzero(is_val))]
        print(f"official split: {len(off_split[0][0])} train, {len(off_split[0][1])} val takes", flush=True)
    fz_prob = np.full((a.repeats, n, 4), np.nan); fz_reg = np.full((a.repeats, n), np.nan)
    reg = np.full((a.repeats, n), np.nan); preg = np.full((a.repeats, n), np.nan); pps = np.full((a.repeats, n), np.nan)
    lvl_reg = np.full((a.repeats, n), np.nan); lvl_prob = np.full((a.repeats, n, 4), np.nan)
    fold_of = np.zeros((a.repeats, n), int); log = []; scen_err = []; comm = []
    seed_out = []; clip_out = {}  # --official only: per-seed test predictions and per-clip commentary outputs
    t0 = time.time()
    print(f"{n} takes, {rated.sum()} rated, D={d_in}, clip {a.clip_sec} s, crop {crop}; joint rating + level, lam {a.lam}",
          flush=True)
    for r in range(a.repeats):
        split = off_split if a.official else StratifiedGroupKFold(5, shuffle=True, random_state=r).split(np.zeros(n), y, p)
        for f, (tr, te) in enumerate(split):
            fold_of[r, te] = f
            s_in = s.copy()
            clf = make_pipeline(StandardScaler(), PCA(128, svd_solver="full"), LogisticRegression(C=1.0, max_iter=3000)).fit(Xmean[tr], s[tr])
            s_in[te] = clf.predict(Xmean[te])
            scen_err.append([names[j] for j in te if s_in[j] != s[j]])
            fit_rel, iv_rel = next(StratifiedGroupKFold(7, shuffle=True, random_state=100 + r).split(np.zeros(len(tr)), y[tr], p[tr]))
            fit_i, iv_i = tr[fit_rel], tr[iv_rel]
            trr = tr[rated[tr]]
            t = z_rating(R, n, trr)
            t[~rated] = np.nan
            mu = np.stack([Xmean[tr][s[tr] == k].mean(0) for k in range(6)])
            sd = (Xmean[tr] - mu[s[tr]]).std(0) + 1e-6
            mu_t, sd_t = torch.as_tensor(mu, dtype=torch.float32), torch.as_tensor(sd, dtype=torch.float32)
            Xs = [(x.float() - mu_t[s_in[j]]) / sd_t for j, x in enumerate(Xraw)]
            T = Tte = mean_txt = None
            if a.lam > 0:
                em = defaultdict(list)
                for j in tr:
                    for k in sents.by_take.get(int(j), []):
                        em[sents.expert[k]].append(v_raw[k])
                emean = {e: float(np.mean(v)) for e, v in em.items()}
                sents.v = np.clip(v_raw - np.array([emean.get(e, 0.0) for e in sents.expert]), -1, 1)
                T = {int(j): tg for j in tr if (tg := sents.targets(int(j), lambda e: 1.0)) is not None}
                sents.v = v_raw.copy()
                Tte = {int(j): tg for j in te if (tg := sents.targets(int(j), lambda e: 1.0)) is not None}
                ks = [k for j in tr for k in sents.by_take.get(int(j), [])]
                e_ = sents.emb[ks] / (np.linalg.norm(sents.emb[ks], axis=1, keepdims=True) + 1e-8)
                mean_txt = e_.mean(0); mean_txt /= np.linalg.norm(mean_txt)
            pairs = None
            if a.rank > 0:  # same expert, same task, different rating, both takes in the fitting set
                fs = set(int(j) for j in fit_i)
                by_e = defaultdict(list)
                for (j, e), rt in R.items():
                    if j in fs:
                        by_e[e].append((j, rt))
                pl = [(ja, jb, float(np.sign(ra - rb))) for v in by_e.values() for i_, (ja, ra) in enumerate(v)
                      for (jb, rb) in v[i_ + 1:] if ra != rb and task_of[ja] == task_of[jb]]
                pairs = np.array(pl)
                if r == 0 and f == 0:
                    cnt = defaultdict(int)
                    for ja, _, _ in pl:
                        cnt[_train.SCEN[s[int(ja)]]] += 1
                    print(f"rank pairs (repeat 0 fold 0): {len(pl)} {dict(cnt)}", flush=True)
            pr_, pl_, pp_, cl_ = [], [], [], []
            for seed in range(a.seeds):
                nets, eps, inner = train_joint(Xs, s, y, t, fit_i, iv_i, seed + 10 * r, a.epochs, a.smoke, d_in, T, a.lam, crop,
                                               pairs, a.rank)
                if a.lam > 0:
                    pr, _, _, cl = predict(nets["rating"], te, Xs, s_in, clips=True); cl_.append(cl)
                else:
                    pr = predict(nets["rating"], te, Xs, s_in)[0]
                pr_.append(pr)
                _, l_, p_ = predict(nets["level"], te, Xs, s_in)
                pl_.append(l_); pp_.append(p_)
                log.append(dict(repeat=r, fold=f, seed=seed, best_epoch=eps["rating"], best_epoch_level=eps["level"],
                                inner_rating=inner["rating"], inner_level=inner["level"]))
                del nets; torch.cuda.empty_cache()
            reg[r, te] = np.mean(pr_, 0); lvl_reg[r, te] = np.mean(pl_, 0); lvl_prob[r, te] = np.mean(pp_, 0)
            if a.official:
                seed_out = [dict(seed=k_, reg=pr_[k_], lvl_reg=pl_[k_], lvl_prob=pp_[k_]) for k_ in range(a.seeds)]
                if a.lam > 0:  # presence logit and per-aspect polarity per 1 s clip, averaged over seeds
                    clip_out = {names[j]: (np.mean([c[j][0] for c in cl_], 0).astype(np.float32),
                                           np.mean([c[j][1] for c in cl_], 0).astype(np.float32),
                                           np.mean([c[j][2] for c in cl_], 0).astype(np.float16)) for j in te}
            cd = {}
            if a.lam > 0:
                clip = {j: tuple(np.mean([c[j][q] for c in cl_], 0) for q in range(3)) for j in cl_[0]}
                cd = comm_diagnostics(clip, te, Tte, mean_txt)
                comm.append(dict(repeat=r, fold=f, **cd))
            if a.official:  # reference: frozen PCA-64 per view level probe (as rating.py --level_probe), same split
                fitall = np.zeros(n, bool); fitall[tr] = True
                P = []
                for v in VS:
                    Zv = Xmean[:, v].astype(np.float64).copy()
                    for k in range(6):
                        mk = s_in == k
                        if (fitall & mk).any():
                            Zv[mk] -= Zv[fitall & mk].mean(0)
                    P.append(PCA(64, svd_solver="full").fit(Zv[tr]).transform(Zv))
                fz_prob[r, te], fz_reg[r, te] = level_readout(np.hstack(P), s_in, y, tr, te)
            fitm = np.zeros(n, bool); fitm[trr] = True
            if a.view == "both":
                preg[r, te] = probe(Xmean, s_in, t, fitm, te)
                pps[r, te] = probe_ps(Xmean, s_in, t, fitm, te)
            tz = z_rating(R, n, np.flatnonzero(rated)); ter = te[rated[te]]
            print(json.dumps(dict(repeat=r, fold=f, epochs_rating=[l["best_epoch"] for l in log[-a.seeds:]],
                                  epochs_level=[l["best_epoch_level"] for l in log[-a.seeds:]],
                                  rating_rho=round(rho_w(reg[r, ter], tz[ter], s[ter]), 3),
                                  level_rho=round(rho_w(lvl_reg[r, te], y[te], s[te]), 3),
                                  level_acc=round(float((lvl_prob[r, te].argmax(1) + 1 == y[te]).mean()), 3),
                                  **{k: round(v, 3) for k, v in cd.items()})),
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if a.smoke:
                break
        if a.smoke:
            break
    if comm:
        keys = [k for k in comm[0] if k not in ("repeat", "fold")]
        print("commentary (mean over folds): " + ", ".join(f"{k} {np.mean([c[k] for c in comm]):.3f}" for k in keys))
    torch.save(dict(names=names, reg=reg, probe_reg=preg, probe_ps_reg=pps, lvl_reg=lvl_reg, lvl_prob=lvl_prob,
                    fold_of=fold_of, log=log, args=vars(a), scenario_errors=scen_err, comm=comm,
                    **(dict(fz_prob=fz_prob, fz_reg=fz_reg, test_idx=te, seed_out=seed_out, clip_out=clip_out,
                           te_names=[names[j] for j in te]) if a.official else {})),
               Path("cls_lang") / f"rating_{a.tag}{'_smoke' if a.smoke else ''}.pt")


if __name__ == "__main__":
    main()
