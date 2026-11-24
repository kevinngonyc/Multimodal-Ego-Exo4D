"""Expert rating (1-10) as the CLS target, with and without all-expert commentary (run from idea_1).

usage:  python cls_lang/rating.py --arm rating|rating_comm [--repeats 4] [--seeds 2] [--smoke]
        python cls_lang/rating.py --report

Target (training): per take, the mean over its experts of the per-expert z-scored rating, expert mean / sd computed on
the outer training takes only. Only takes with a rating are trained on; the model sees video only at inference.
Arms (train.Net on frozen ego+exo K600 L12 token-std clip tokens, CLS regression head, scenario bias; no class head):
  rating       MSE on the z-rating
  rating_comm  + lam * (presence BCE + polarity SE + text 1-cos) from ALL experts' timestamped comments (weight 1 each);
               polarity = 2 p_good - 1 minus the expert's own mean over training takes (per-expert bias removed)
Every fold also fits the frozen probe (scenario-centred PCA-64 per view, scenario one-hot, RidgeCV) on the same rating
target. Early stopping: inner participant split (1/7), within-scenario Spearman with the z-rating.
Evaluation (expert_rating.ipynb protocol): rows = (take, expert i) on takes with >= 2 rated experts; the take's
prediction vs the mean rating of the OTHER experts (raw and per-expert z, z from all ratings as in the notebook);
human = expert i vs the others on the same rows; within-scenario Spearman, predictions centred within
(repeat, fold, scenario) over rated takes and averaged over repeats; participant bootstrap. Also vs official level.
Protocol: the main 4 x 5 StratifiedGroupKFold by participant over the 2,111 official takes.
-> cls_lang/rating_{arm}.pt
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
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from train import DEV, SCEN, Net, Sentences, batchify, centred, load_ratings, load_takes  # noqa: E402


class NetPS(Net):
    """train.Net with one rating head per scenario on the CLS token (instead of one shared head + scenario bias)."""
    def __init__(self):
        super().__init__()
        self.reg = nn.Linear(self.d, 6)

    def forward(self, x, tpos, mask, s):
        h = self.inp(x) + self.time_enc(tpos)
        h = torch.cat([self.cls + self.scen(s)[:, None], h], 1)
        m = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool, device=x.device), ~mask], 1)
        h = self.norm(self.enc(h, src_key_padding_mask=m))
        z, tok = h[:, 0], h[:, 1:]
        r = self.reg(z).gather(1, s[:, None]).squeeze(1) + self.reg_b[s]
        self.last_cls = z
        return (r, self.cl(z) + self.cl_b[s], self.pres(tok).squeeze(-1), torch.tanh(self.pol(tok)), self.txt(tok))


@torch.no_grad()
def cls_embed(net, idx, Xs, s, bs=32):
    """frozen CLS output (after the final LayerNorm) for takes idx -> (len(idx), d)."""
    net.eval()
    order = sorted(idx, key=lambda j: len(Xs[j]))
    out = {}
    for i in range(0, len(order), bs):
        ch = order[i:i + bs]
        b = batchify(ch, Xs, s)
        net(b["x"], b["t"], b["mask"], b["s"])
        for j, v in zip(ch, net.last_cls.float().cpu().numpy()):
            out[j] = v
    net.train()
    return np.stack([out[j] for j in idx])


def level_readout(Z, s_in, y, tr, te):
    """4-level logistic (probabilities) + ridge (ordinal score) on scenario one-hot + standardised Z, fit on tr."""
    from sklearn.linear_model import LogisticRegressionCV
    F_ = np.hstack([(s_in[:, None] == np.arange(6)[None]).astype(float), Z])
    sc = StandardScaler().fit(F_[tr])
    F_ = sc.transform(F_)
    prob = LogisticRegressionCV(Cs=np.logspace(-4, 2, 7), cv=5, max_iter=5000, n_jobs=4).fit(F_[tr], y[tr]).predict_proba(F_[te])
    reg = RidgeCV(alphas=np.logspace(-2, 6, 17)).fit(F_[tr], y[tr]).predict(F_[te])
    return prob, reg


def probe_ps(Xmean, s, t, fit, held):
    """frozen probe fit separately per scenario: PCA (<= 64) per view + RidgeCV on that scenario's training takes."""
    out = np.zeros(len(held))
    for k in np.unique(s[held]):
        f = fit & (s == k); h = held[s[held] == k]
        if f.sum() < 5:
            continue
        parts = []
        for v in (slice(0, 768), slice(768, 1536)):
            Z = Xmean[:, v].astype(np.float64)
            pca = PCA(min(64, f.sum() - 1), svd_solver="full").fit(Z[f])
            parts.append(pca.transform(Z))
        Z = np.hstack(parts)
        sc = StandardScaler().fit(Z[f])
        r = RidgeCV(alphas=np.logspace(-2, 6, 17)).fit(sc.transform(Z[f]), t[f]).predict(sc.transform(Z[h]))
        out[np.searchsorted(held, h)] = r
    return out


def rho_w(pred, t, s):
    return spearmanr(centred(pred, s), centred(t, s))[0]


def z_rating(R, n, fit_takes):
    """per-take mean of per-expert z-scored ratings; expert stats from fit_takes only. NaN = no rating."""
    fit = set(int(j) for j in fit_takes)
    by = defaultdict(list)
    for (j, e), r in R.items():
        if j in fit:
            by[e].append(r)
    mu = {e: np.mean(v) for e, v in by.items()}
    sd = {e: (np.std(v) if np.std(v) > 0 else 1.0) for e, v in by.items()}
    acc = defaultdict(list)
    for (j, e), r in R.items():
        if e in mu:
            acc[j].append((r - mu[e]) / sd[e])
    t = np.full(n, np.nan)
    for j, v in acc.items():
        t[j] = np.mean(v)
    return t


@torch.no_grad()
def predict(net, idx, Xs, s, bs=32):
    net.eval()
    order = sorted(idx, key=lambda j: len(Xs[j]))
    out = {}
    for i in range(0, len(order), bs):
        ch = order[i:i + bs]
        b = batchify(ch, Xs, s)
        r = net(b["x"], b["t"], b["mask"], b["s"])[0]
        for j, v in zip(ch, r.cpu().numpy()):
            out[j] = v
    net.train()
    return np.array([out[j] for j in idx])


def train_one(Xs, s, t, fit_i, iv_i, T, lam, seed, epochs, smoke, crop=384, head="shared"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = (NetPS() if head == "per_scenario" else Net()).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    tt = torch.as_tensor(np.nan_to_num(t), dtype=torch.float32, device=DEV)
    best, best_state, bad, best_ep = -2, None, 0, 0
    for ep in range(1, epochs + 1):
        perm = rng.permutation(fit_i)
        for i in range(0, len(perm), 16):
            idx = perm[i:i + 16]
            b = batchify(idx, Xs, s, crop=crop, rng=rng, T=T if lam > 0 else None)
            reg, _, pres, pol, txt = net(b["x"], b["t"], b["mask"], b["s"])
            loss = F.mse_loss(reg, tt[torch.as_tensor(idx, device=DEV)])
            if lam > 0 and b["has"].sum() > 0:
                tm = b["mask"] & (b["has"][:, None] > 0)
                l_pres = F.binary_cross_entropy_with_logits(pres[tm], b["pres"][tm])
                l_pol = ((pol - b["pol"]) ** 2 * b["polw"])[tm].sum() / tm.sum()
                cos = F.cosine_similarity(txt, b["txt"] + 1e-8, dim=-1)
                l_txt = ((1 - cos) * b["txtw"])[tm].sum() / tm.sum()
                loss = loss + lam * (l_pres + l_pol + l_txt)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        cur = rho_w(predict(net, iv_i, Xs, s), t[iv_i], s[iv_i])
        if cur > best:
            best, best_state, bad, best_ep = cur, {k: v.clone() for k, v in net.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad >= 8:
                break
        if smoke and ep >= 2:
            break
    net.load_state_dict(best_state)
    return net, best_ep, best


def probe(Xmean, s, t, fit, held):
    parts = [(s[:, None] == np.arange(6)[None]).astype(float)]
    for v in (slice(0, 768), slice(768, 1536)):
        Z = Xmean[:, v].astype(np.float64).copy()
        for k in range(6):
            m = s == k
            if (fit & m).any():
                Z[m] -= Z[fit & m].mean(0)
        parts.append(PCA(64, svd_solver="full").fit(Z[fit]).transform(Z))
    Z = StandardScaler().fit(np.hstack(parts)[fit]).transform(np.hstack(parts))
    return RidgeCV(alphas=np.logspace(-2, 6, 17)).fit(Z[fit], t[fit]).predict(Z[held])


def run(a):
    torch.backends.cuda.matmul.allow_tf32 = True
    if a.feat_cache:
        import train as _train
        _train.FEAT_CACHE = Path(a.feat_cache)
    names, Xraw, s, y, p = load_takes()
    if a.scenario:
        keep = np.flatnonzero(s == SCEN.index(a.scenario))
        names, Xraw, s, y, p = [names[i] for i in keep], [Xraw[i] for i in keep], s[keep], y[keep], p[keep]
    n = len(names)
    take_ix = {t: j for j, t in enumerate(names)}
    R = load_ratings(take_ix)
    rated = np.zeros(n, bool)
    for (j, _) in R:
        rated[j] = True
    Xmean = np.stack([x.float().mean(0).numpy() for x in Xraw])
    lam = a.lam if a.arm == "rating_comm" else 0.0
    sents = Sentences(take_ix, [len(x) for x in Xraw]) if lam > 0 else None
    v_raw = sents.v.copy() if sents else None
    print(f"{n} takes, {rated.sum()} with a rating, {len(R)} ratings; arm {a.arm}, lam {lam}", flush=True)
    reg = np.full((a.repeats, n), np.nan); preg = np.full((a.repeats, n), np.nan); pps = np.full((a.repeats, n), np.nan)
    fold_of = np.zeros((a.repeats, n), int); log = []; scen_err = []
    lvl_prob = np.full((a.repeats, n, 4), np.nan); lvl_reg = np.full((a.repeats, n), np.nan)
    fz_prob = np.full((a.repeats, n, 4), np.nan); fz_reg = np.full((a.repeats, n), np.nan)
    t0 = time.time()
    for r in range(a.repeats):
        for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(np.zeros(n), y, p)):
            fold_of[r, te] = f
            s_in = s.copy()  # scenario as model INPUT: true for training takes, predicted from video for test takes
            if a.pred_scenario:
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline
                clf = make_pipeline(StandardScaler(), PCA(128, svd_solver="full"),
                                    LogisticRegression(C=1.0, max_iter=3000)).fit(Xmean[tr], s[tr])
                s_in[te] = clf.predict(Xmean[te])
                scen_err.append([names[j] for j in te if s_in[j] != s[j]])
            trr = tr[rated[tr]]
            fit_rel, iv_rel = next(StratifiedGroupKFold(7, shuffle=True, random_state=100 + r).split(np.zeros(len(trr)), y[trr], p[trr]))
            fit_i, iv_i = trr[fit_rel], trr[iv_rel]
            t = z_rating(R, n, trr)
            mu = np.stack([Xmean[tr][s[tr] == k].mean(0) for k in range(6)])
            sd = (Xmean[tr] - mu[s[tr]]).std(0) + 1e-6
            mu_t, sd_t = torch.as_tensor(mu, dtype=torch.float32), torch.as_tensor(sd, dtype=torch.float32)
            Xs = [(x.float() - mu_t[s_in[j]]) / sd_t for j, x in enumerate(Xraw)]
            T = None
            if lam > 0:
                fitset = set(int(j) for j in trr)
                em = defaultdict(list)
                for j in fitset:
                    for k in sents.by_take.get(j, []):
                        em[sents.expert[k]].append(v_raw[k])
                emean = {e: float(np.mean(v)) for e, v in em.items()}
                sents.v = np.clip(v_raw - np.array([emean.get(e, 0.0) for e in sents.expert]), -1, 1)
                T = {int(j): tg for j in fit_i if (tg := sents.targets(int(j), lambda e: 1.0)) is not None}
                T.update({int(j): tg for j in iv_i if (tg := sents.targets(int(j), lambda e: 1.0)) is not None})
            preds, eps, lp, lr_ = [], [], [], []
            for seed in range(a.seeds):
                net, ep, inner = train_one(Xs, s, t, fit_i, iv_i, T, lam, seed + 10 * r, a.epochs, a.smoke, a.crop, a.head)
                preds.append(predict(net, te, Xs, s_in)); eps.append(ep)
                if a.level_probe:
                    Z = cls_embed(net, np.arange(n), Xs, s_in)
                    pr_, rg_ = level_readout(Z, s_in, y, tr, te)
                    lp.append(pr_); lr_.append(rg_)
                log.append(dict(repeat=r, fold=f, seed=seed, best_epoch=ep, inner_rho=inner))
                del net; torch.cuda.empty_cache()
            reg[r, te] = np.mean(preds, 0)
            if a.level_probe:
                lvl_prob[r, te], lvl_reg[r, te] = np.mean(lp, 0), np.mean(lr_, 0)
                # reference: the same level readout on the frozen PCA features (as in train.probe), same folds
                fitall = np.zeros(n, bool); fitall[tr] = True
                P = []
                for v in (slice(0, 768), slice(768, 1536)):
                    Zv = Xmean[:, v].astype(np.float64).copy()
                    for k in range(6):
                        mk = s_in == k
                        if (fitall & mk).any():
                            Zv[mk] -= Zv[fitall & mk].mean(0)
                    P.append(PCA(64, svd_solver="full").fit(Zv[tr]).transform(Zv))
                fz_prob[r, te], fz_reg[r, te] = level_readout(np.hstack(P), s_in, y, tr, te)
            fitm = np.zeros(n, bool); fitm[trr] = True
            preg[r, te] = probe(Xmean, s_in, t, fitm, te)
            pps[r, te] = probe_ps(Xmean, s_in, t, fitm, te)
            tz = z_rating(R, n, np.flatnonzero(rated))  # evaluation-only check on this fold's rated test takes
            ter = te[rated[te]]
            print(json.dumps(dict(repeat=r, fold=f, n_train=len(trr), n_targets=len(T) if T else 0, epochs=eps,
                                  cls_vs_zrating=round(rho_w(reg[r, ter], tz[ter], s[ter]), 3),
                                  probe_vs_zrating=round(rho_w(preg[r, ter], tz[ter], s[ter]), 3))),
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if a.smoke:
                break
        if a.smoke:
            break
    if a.pred_scenario:
        errs = [e for fold in scen_err for e in fold]
        print(f"scenario predicted from video: {len(errs)} errors over {a.repeats} repeats: {sorted(set(errs))}", flush=True)
    torch.save(dict(names=names, reg=reg, probe_reg=preg, probe_ps_reg=pps, fold_of=fold_of, log=log, args=vars(a),
                    scenario_errors=scen_err, lvl_prob=lvl_prob, lvl_reg=lvl_reg, fz_prob=fz_prob, fz_reg=fz_reg),
               Path("cls_lang") / f"rating_{a.arm}{'_' + a.tag if a.tag else ''}{'_smoke' if a.smoke else ''}.pt")


def report(tags=("rating", "rating_comm")):
    res = {a_: torch.load(Path("cls_lang") / f"rating_{a_}.pt", weights_only=False) for a_ in tags
           if (Path("cls_lang") / f"rating_{a_}.pt").exists()}
    first = next(iter(res.values()))
    names = first["names"]
    from train import A, LEVELS
    meta = {}
    for fn in ("proficiency_demonstrator_train.json", "proficiency_demonstrator_val.json"):
        for m in json.load(open(A / fn, encoding="utf-8"))["annotations"]:
            meta[m["video_paths"]["ego"].split("/")[1]] = (SCEN.index(m["scenario_name"]), LEVELS[m["proficiency_score"]],
                                                           m["origin_participant_id"])
    s = np.array([meta[t][0] for t in names]); y = np.array([meta[t][1] for t in names], float)
    p = np.array([meta[t][2] for t in names])
    n = len(names)
    take_ix = {t: j for j, t in enumerate(names)}
    R = load_ratings(take_ix)
    rated = np.zeros(n, bool)
    for (j, _) in R:
        rated[j] = True
    by_e = defaultdict(list)
    for (j, e), r in R.items():
        by_e[e].append(r)
    mu = {e: np.mean(v) for e, v in by_e.items()}; sd = {e: (np.std(v) or 1.0) for e, v in by_e.items()}
    by_t = defaultdict(list)
    for (j, e), r in R.items():
        by_t[j].append((e, r, (r - mu[e]) / sd[e]))
    H = []
    for j, g in by_t.items():
        if len(g) < 2:
            continue
        for i, (e, r, z) in enumerate(g):
            o = [x for k, x in enumerate(g) if k != i]
            H.append((j, r, z, np.mean([x[1] for x in o]), np.mean([x[2] for x in o])))
    H = np.array(H)
    hi = H[:, 0].astype(int); hs = s[hi]
    cols = {k: centred(H[:, c], hs) for k, c in (("one", 1), ("one_z", 2), ("oth", 3), ("oth_z", 4))}
    lev_c = centred(y, s)

    def pooled(P0, fold_of):
        P = P0.copy()
        for r in range(len(P)):
            for f in range(5):
                for k in range(6):
                    g = (fold_of[r] == f) & (s == k) & rated
                    P[r, g] -= P[r, g].mean()
        return P.mean(0)

    M = {}
    for a_, r_ in res.items():
        M[f"probe: {a_}"] = pooled(r_["probe_reg"], r_["fold_of"])
        if "probe_ps_reg" in r_:
            M[f"probe-ps: {a_}"] = pooled(r_["probe_ps_reg"], r_["fold_of"])
        M[f"CLS: {a_}"] = pooled(r_["reg"], r_["fold_of"])
    U = np.unique(p); rows_t = {u: np.flatnonzero((p == u) & rated) for u in U}
    hp = p[hi]; rows_h = {u: np.flatnonzero(hp == u) for u in U}
    rng = np.random.default_rng(0)
    draws = [rng.choice(U, len(U)) for _ in range(1000)]
    Bt = [np.concatenate([rows_t[u] for u in d]) for d in draws]
    Bh = [np.concatenate([rows_h[u] for u in d]) for d in draws]
    sp = lambda a, b: spearmanr(a, b)[0]
    allh = np.arange(len(hi)); allt = np.flatnonzero(rated)

    def metrics(x, human=False):
        if human:
            return dict(raw=(cols["one"], allh), z=(cols["one_z"], allh))
        return dict(raw=(x[hi], allh), z=(x[hi], allh), level=(x, allt))
    print(f"{rated.sum()} rated takes, {len(hi)} (take, expert) rows on takes with >= 2 experts, {len(U)} participants")
    print(f"{'model':24s} {'vs experts raw':>16s} {'vs experts z':>14s} {'vs level':>10s}")
    print(f"{'human: one expert':24s} {sp(cols['one'], cols['oth']):16.3f} {sp(cols['one_z'], cols['oth_z']):14.3f} "
          f"{sp(cols['one'], lev_c[hi]):10.3f}")
    for m, x in M.items():
        print(f"{m:24s} {sp(x[hi], cols['oth']):16.3f} {sp(x[hi], cols['oth_z']):14.3f} {sp(x[allt], lev_c[allt]):10.3f}")

    def diff(a_, b_):
        out = []
        for key, tgt in (("raw", cols["oth"]), ("z", cols["oth_z"])):
            d = [sp(M[a_][hi][b], tgt[b]) - sp(M[b_][hi][b], tgt[b]) for b in Bh]
            out.append(f"{key} {sp(M[a_][hi], tgt) - sp(M[b_][hi], tgt):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
        d = [sp(M[a_][b], lev_c[b]) - sp(M[b_][b], lev_c[b]) for b in Bt]
        out.append(f"level {sp(M[a_][allt], lev_c[allt]) - sp(M[b_][allt], lev_c[allt]):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
        print(f"  {a_} - {b_}: " + "   ".join(out))
    print("paired differences (95% participant bootstrap)")
    for a_ in res:
        diff(f"CLS: {a_}", f"probe: {a_}")
        if f"probe-ps: {a_}" in M:
            diff(f"probe-ps: {a_}", f"probe: {a_}")
            diff(f"CLS: {a_}", f"probe-ps: {a_}")
    tl = list(res)
    for i in range(1, len(tl)):
        diff(f"CLS: {tl[i]}", f"CLS: {tl[0]}")
        diff(f"probe: {tl[i]}", f"probe: {tl[0]}")
    hdiff = [sp(cols["one_z"][b], cols["oth_z"][b]) for b in Bh]
    print(f"human z CI: [{np.percentile(hdiff, 2.5):.3f}, {np.percentile(hdiff, 97.5):.3f}]")
    print("per scenario, vs experts z (human first)")
    for k in range(6):
        m = hs == k
        if m.sum() < 10:
            continue
        print(f"  {SCEN[k]:14s} rows={m.sum():4d} takes={len(np.unique(hi[m])):4d} human {sp(cols['one_z'][m], cols['oth_z'][m]):.3f}  "
              + "  ".join(f"{mm} {sp(x[hi][m], cols['oth_z'][m]):.3f}" for mm, x in M.items()))
    for a_, r_ in res.items():
        ep = [l["best_epoch"] for l in r_["log"]]
        print(f"{a_}: best epoch median {np.median(ep):.0f} (range {min(ep)}-{max(ep)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("rating", "rating_comm"))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--feat_cache", help="clip-feature cache {take: (n, 1536)}; default embeddings/cls_lang_clipfeats.pt")
    ap.add_argument("--scenario", help="train and evaluate on this scenario only")
    ap.add_argument("--crop", type=int, default=384, help="training crop in tokens")
    ap.add_argument("--tag", default="")
    ap.add_argument("--head", choices=("shared", "per_scenario"), default="shared")
    ap.add_argument("--level_probe", action="store_true",
                    help="also fit a 4-level logistic / ridge on the frozen CLS token (+ the same on frozen PCA features)")
    ap.add_argument("--pred_scenario", action="store_true",
                    help="test takes get their scenario from a video classifier fit on the training takes (video-only inference)")
    ap.add_argument("--report_tags", nargs="+", default=["rating", "rating_comm"])
    a = ap.parse_args()
    report(a.report_tags) if a.report else run(a)
