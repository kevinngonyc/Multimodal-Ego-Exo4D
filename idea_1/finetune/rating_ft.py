"""Pilot: fine-tune the top of TimeSformer on rating + within-expert ranking + commentary + level, one scenario, one fold.

usage (from idea_1):  python finetune/rating_ft.py [--fold 0] [--scenario Dance] [--smoke]

Split = repeat 0 of the joint model (cls_lang/joint.py): StratifiedGroupKFold(5, shuffle, random_state=0) by participant
over all 2,111 takes; inner split StratifiedGroupKFold(7, shuffle, random_state=100) of the training takes. Only the
scenario's takes are used for training and testing, so the final joint model's repeat-0 predictions for the same test
takes (cls_lang/rating_joint_lam4.pt) are directly comparable.

Head = cls_lang/joint.JointNet ([RATING], [LEVEL], per-clip commentary heads) on ego | exo clip features (1 s clip, 8
frames at 8 FPS, whole frame squashed to 224, layer-12 patch-token std; standardised with the frozen training-fold
statistics, exactly as the joint model).
Training samples = PAIRS of training takes rated by the same expert (same task when possible) with different ratings;
from each take a random window of W consecutive clips. Loss per pair =
   MSE(rating head, mean per-expert z-rating) + rank * softplus(-(r_a - r_b) * sign(rating_a - rating_b))
 + MSE(level score) + CE(level)  + lam * (presence BCE + polarity SE + text 1 - cos)   [lam 4, as the final model]
Stage 1 (frozen, LP): the head alone on the cached frozen features, early stopping on the inner split (best
within-scenario Spearman with the z-rating; patience 8). This is the frozen control and the start of stage 2.
Stage 2 (FT): LoRA (rank 8) on the attention / temporal layers of the top transformer blocks (--layers), LoRA B = 0 so
epoch 0 == stage 1; features computed live from the decoded video; same loss and sampler; the head keeps training.
Checkpoint = best inner-split Spearman (epoch 0 included); patience 3.
Test (held-out takes of the fold, whole takes): frozen control (cached features), FT best, FT last epoch. Metrics:
agreement with held-out experts (per-expert z, the rating report's rows), Spearman with the mean z-rating, level
Spearman / accuracy, commentary presence AUC / text cosine; participant bootstrap FT - frozen and FT - final model.
-> models/rating_ft/{scenario}_fold{k}/{log.jsonl, ft_best.pt, preds.pt}
"""
import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cls_lang"))
sys.path.insert(0, str(ROOT / "ssl_adapt"))
from common import TAKES, clip_frame_indices, ego_path, exo_camera, frame_count, grab  # noqa: E402  (ssl_adapt)

MEAN, STD, SIZE, FPS, CF = 0.45, 0.225, 224, 8, 8
TARGETS = (r"attention\.attention\.qkv|temporal_attention\.attention\.qkv|attention\.output\.dense|"
           r"temporal_attention\.output\.dense|temporal_dense")


# ---------------------------------------------------------------- video (same frames as the cached features)
def view_path(take, view):
    return ego_path(take) if view == "ego" else exo_camera(take)[1]


def decode_full(take, view, n):
    """every 1 s clip, sequential decode (identical to ssl_adapt/extract.decode), first n clips."""
    with av.open(str(view_path(take, view))) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        rate, total = float(st.average_rate), st.frames
        n_clips = max(1, (int((total - 1) * FPS / rate) + 1) // CF)
        wanted = {int(math.floor(k * rate / FPS + 0.5)): k for k in range(n_clips * CF)}
        fr = np.zeros((n_clips * CF, SIZE, SIZE, 3), np.uint8)
        got = np.zeros(n_clips * CF, bool)
        for i, frame in enumerate(c.decode(video=0)):
            k = wanted.get(i)
            if k is not None:
                fr[k] = frame.to_ndarray(format="rgb24", width=SIZE, height=SIZE, interpolation="BILINEAR")
                got[k] = True
    for k in np.flatnonzero(~got):
        fr[k] = fr[max(k - 1, 0)]
    fr = fr.reshape(n_clips, CF, SIZE, SIZE, 3)[:n]
    if len(fr) < n:
        fr = np.concatenate([fr, np.repeat(fr[-1:], n - len(fr), 0)])
    return fr


def decode_window(take, view, o, w):
    """clips o .. o+w-1 by seeking (same frame indices as decode_full)."""
    path = view_path(take, view)
    _, rate = frame_count(path)
    flat = [i for k in range(o, o + w) for i in clip_frame_indices(CF * k, rate)]
    got = grab(path, flat)
    fr = np.zeros((w * CF, SIZE, SIZE, 3), np.uint8)
    last = next(iter(got.values())) if got else fr[0]
    for q, i in enumerate(flat):
        last = got.get(i, last)
        fr[q] = last
    return fr.reshape(w, CF, SIZE, SIZE, 3)


class Windows(Dataset):
    def __init__(self, names):
        self.names = names

    def __len__(self):
        return 10**9

    def __getitem__(self, spec):
        j, o, w = spec
        t = self.names[j]
        return j, o, torch.from_numpy(decode_window(t, "ego", o, w)), torch.from_numpy(decode_window(t, "exo", o, w))


class Full(Dataset):
    def __init__(self, names, idx, lens):
        self.names, self.idx, self.lens = names, list(idx), lens

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        j = int(self.idx[k])
        t, n = self.names[j], self.lens[j]
        return j, torch.from_numpy(decode_full(t, "ego", n)), torch.from_numpy(decode_full(t, "exo", n))


class SpecSampler:
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        return self.gen

    def __len__(self):
        return 10**9


def first(batch):
    return batch


# ---------------------------------------------------------------- backbone
def build_backbone(r, layers):
    from peft import LoraConfig, get_peft_model
    from transformers import TimesformerModel
    base = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k600")
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rx = rf".*encoder\.layer\.({'|'.join(str(l) for l in layers)})\.({TARGETS})$"
    bb = get_peft_model(base, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=rx, bias="none"))
    try:
        base.disable_input_require_grads()
    except Exception:
        pass
    return bb


def clip_std(bb, x, chunk=64):
    """uint8 (N, 8, 224, 224, 3) on GPU -> (N, 768) layer-12 patch-token std (float32)."""
    out = []
    for i in range(0, len(x), chunk):
        v = x[i:i + chunk].permute(0, 1, 4, 2, 3).float().div_(255).sub_(MEAN).div_(STD)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = bb(pixel_values=v).last_hidden_state
        out.append(h[:, 1:].float().std(1))
    return torch.cat(out)


# ---------------------------------------------------------------- metrics
def rho(a, b):
    return float(spearmanr(a, b)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--scenario", default="Dance")
    ap.add_argument("--W", type=int, default=32, help="clips per training window")
    ap.add_argument("--pairs", type=int, default=4, help="pairs per optimiser step (one pair per micro-batch)")
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--rank", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--layers", default="8,9,10,11")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--ft_epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import joint as J
    from rating import rho_w, z_rating
    from train import DEV, SCEN, Sentences, load_ratings, load_takes
    from sklearn.model_selection import StratifiedGroupKFold

    out = Path("models") / "rating_ft" / f"{a.scenario.replace(' ', '')}_fold{a.fold}{'_smoke' if a.smoke else ''}"
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.jsonl", "w")

    def note(rec):
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n"); logf.flush()

    torch.manual_seed(0); random.seed(0)
    rng = np.random.default_rng(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    t_start = time.time()
    mins = lambda: round((time.time() - t_start) / 60, 1)

    # ------------------------------------------------ data, split, targets (as joint.py repeat 0)
    names, Xraw, s, y, p = load_takes()
    n = len(names)
    lens = [len(x) for x in Xraw]
    take_ix = {t: j for j, t in enumerate(names)}
    R = load_ratings(take_ix)
    rated = np.zeros(n, bool)
    for (j, _) in R:
        rated[j] = True
    tr, te = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(n), y, p))[a.fold]
    fit_rel, iv_rel = next(StratifiedGroupKFold(7, shuffle=True, random_state=100).split(np.zeros(len(tr)), y[tr], p[tr]))
    fit_i, iv_i = tr[fit_rel], tr[iv_rel]
    sc = SCEN.index(a.scenario)
    D = s == sc
    fit_d, iv_d, te_d = fit_i[D[fit_i]], iv_i[D[iv_i]], te[D[te]]
    t = z_rating(R, n, tr[rated[tr]]); t[~rated] = np.nan
    Xmean = np.stack([x.float().mean(0).numpy() for x in Xraw])
    mu = np.stack([Xmean[tr][s[tr] == k].mean(0) for k in range(6)])
    sd = (Xmean[tr] - mu[s[tr]]).std(0) + 1e-6
    mu_t, sd_t = torch.as_tensor(mu, dtype=torch.float32), torch.as_tensor(sd, dtype=torch.float32)
    Xs = [(x.float() - mu_t[s[j]]) / sd_t for j, x in enumerate(Xraw)]
    mu_g, sd_g = mu_t.to(DEV), sd_t.to(DEV)

    sents = Sentences(take_ix, lens)
    v_raw = sents.v.copy()
    em = defaultdict(list)
    for j in tr:
        for k in sents.by_take.get(int(j), []):
            em[sents.expert[k]].append(v_raw[k])
    emean = {e: float(np.mean(v)) for e, v in em.items()}
    sents.v = np.clip(v_raw - np.array([emean.get(e, 0.0) for e in sents.expert]), -1, 1)
    T = {int(j): tg for j in tr if (tg := sents.targets(int(j), lambda e: 1.0)) is not None}
    sents.v = v_raw.copy()
    Tte = {int(j): tg for j in te_d if (tg := sents.targets(int(j), lambda e: 1.0)) is not None}
    ks = [k for j in tr for k in sents.by_take.get(int(j), [])]
    e_ = sents.emb[ks] / (np.linalg.norm(sents.emb[ks], axis=1, keepdims=True) + 1e-8)
    mean_txt = e_.mean(0); mean_txt /= np.linalg.norm(mean_txt)

    # within-expert pairs among the scenario's fitting takes (same task when the expert has such pairs)
    fitset = set(int(j) for j in fit_d)
    by_e = defaultdict(list)
    for (j, e), r in R.items():
        if j in fitset:
            by_e[e].append((j, r))
    task = {j: TAKES.get(names[j], {}).get("task_name", "?") for j in fitset}
    pairs_same, pairs_any = [], []
    for e, v in by_e.items():
        for i in range(len(v)):
            for k in range(i + 1, len(v)):
                (ja, ra), (jb, rb) = v[i], v[k]
                if ra != rb:
                    (pairs_same if task[ja] == task[jb] else pairs_any).append((ja, jb, float(np.sign(ra - rb))))
    pairs = pairs_same if len(pairs_same) >= 200 else pairs_same + pairs_any
    note(dict(scenario=a.scenario, fold=a.fold, n_fit=len(fit_d), n_innerval=len(iv_d), n_test=len(te_d),
              rated_fit=int(rated[fit_d].sum()), rated_innerval=int(rated[iv_d].sum()), rated_test=int(rated[te_d].sum()),
              experts=len(by_e), pairs_same_task=len(pairs_same), pairs_other=len(pairs_any), pairs_used=len(pairs),
              args=vars(a)))

    y_t = torch.as_tensor(y, dtype=torch.float32, device=DEV)
    t_t = torch.as_tensor(np.nan_to_num(t), dtype=torch.float32, device=DEV)
    s_t = torch.as_tensor(s, device=DEV)

    def windows():
        """endless micro-batches: one pair = [(j, o, w), (j, o, w)]"""
        while True:
            ja, jb, sg = pairs[rng.integers(len(pairs))]
            spec = []
            for j in (ja, jb):
                w = min(a.W, lens[j])
                spec.append((j, int(rng.integers(0, lens[j] - w + 1)), w))
            yield spec, sg

    def head_loss(net, feats, spec, sg):
        """feats: list of (w, 1536) standardised features for the pair's windows."""
        x = nn.utils.rnn.pad_sequence(feats, batch_first=True)
        L = x.shape[1]
        mask = torch.zeros(2, L, dtype=torch.bool, device=DEV)
        tpos = torch.zeros(2, L, device=DEV)
        for b, (j, o, w) in enumerate(spec):
            mask[b, :w] = True
            tpos[b, :w] = torch.arange(o, o + w, device=DEV)
        ix = torch.as_tensor([j for j, _, _ in spec], device=DEV)
        r, l, lg, pres, pol, txt = net(x, tpos, mask, s_t[ix])
        loss = F.mse_loss(r, t_t[ix]) + a.rank * F.softplus(-(r[0] - r[1]) * sg)
        loss = loss + F.mse_loss(l, y_t[ix]) + F.cross_entropy(lg, (y_t[ix] - 1).long())
        if a.lam > 0:
            P = torch.zeros(2, L, device=DEV); PO = torch.zeros(2, L, pol.shape[-1], device=DEV)
            PW = torch.zeros_like(PO); TX = torch.zeros(2, L, txt.shape[-1], device=DEV); TW = torch.zeros(2, L, device=DEV)
            has = torch.zeros(2, dtype=torch.bool, device=DEV)
            for b, (j, o, w) in enumerate(spec):
                tg = T.get(j)
                if tg is None:
                    continue
                has[b] = True
                P[b, :w] = torch.from_numpy(tg["pres"][o:o + w]); PO[b, :w] = torch.from_numpy(tg["pol"][o:o + w])
                PW[b, :w] = torch.from_numpy(tg["pol_w"][o:o + w]); TX[b, :w] = torch.from_numpy(tg["txt"][o:o + w])
                TW[b, :w] = torch.from_numpy(tg["txt_w"][o:o + w])
            tm = mask & has[:, None]
            if tm.any():
                l_pres = F.binary_cross_entropy_with_logits(pres[tm], P[tm])
                l_pol = ((pol - PO) ** 2 * PW)[tm].sum() / tm.sum()
                l_txt = ((1 - F.cosine_similarity(txt, TX + 1e-8, dim=-1)) * TW)[tm].sum() / tm.sum()
                loss = loss + a.lam * (l_pres + l_pol + l_txt)
        return loss

    def inner_rho(net, feats_list):
        ivr = iv_d[rated[iv_d]]
        pr = J.predict(net, ivr, feats_list, s)[0]
        return rho_w(pr, t[ivr], s[ivr])

    spe = 4 if a.smoke else max(1, len(fit_d) // (2 * a.pairs))  # each fitting take ~once per epoch

    # ------------------------------------------------ stage 1: frozen control (head on cached features)
    torch.manual_seed(0)
    net = J.JointNet(1536).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    gen = windows()
    best, best_state, best_ep, bad = -2.0, None, 0, 0
    for ep in range(1, (2 if a.smoke else 60) + 1):
        net.train()
        for _ in range(spe):
            opt.zero_grad(set_to_none=True)
            for _ in range(a.pairs):
                spec, sg = next(gen)
                feats = [Xs[j][o:o + w].to(DEV) for j, o, w in spec]
                (head_loss(net, feats, spec, sg) / a.pairs).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        cur = inner_rho(net, Xs)
        if cur > best:
            best, best_state, best_ep, bad = cur, {k: v.clone() for k, v in net.state_dict().items()}, ep, 0
        else:
            bad += 1
            if bad >= 8:
                break
    net.load_state_dict(best_state)
    note(dict(stage="frozen", best_epoch=best_ep, inner_rho=round(best, 4), min=mins()))
    frozen_state = {k: v.clone() for k, v in best_state.items()}

    # ------------------------------------------------ stage 2: LoRA fine-tuning
    bb = build_backbone(a.lora_r, [int(l) for l in a.layers.split(",")]).to(DEV)
    lora = [q for q in bb.parameters() if q.requires_grad]
    note(dict(lora_params=sum(q.numel() for q in lora)))

    def live(idx, max_takes=None):
        """standardised live features for whole takes -> list indexed like Xs (None elsewhere)"""
        idx = list(idx)[:max_takes]
        feats = [None] * n
        bb.eval()
        dl = DataLoader(Full(names, idx, lens), batch_size=None, num_workers=min(4, a.workers), prefetch_factor=1)
        with torch.no_grad():
            for j, ego, exo in dl:
                fe = clip_std(bb, ego.to(DEV, non_blocking=True)); fx = clip_std(bb, exo.to(DEV, non_blocking=True))
                feats[j] = ((torch.cat([fe, fx], 1) - mu_g[s[j]]) / sd_g).cpu()
        return feats, idx

    # sanity: live frozen features == cached features (whole takes and windows)
    chk = list(iv_d[:2])
    F0, _ = live(chk)
    cos_full = [float(F.cosine_similarity(F0[j], Xs[j], dim=1).mean()) for j in chk]
    w0 = Windows(names)[(int(chk[0]), 3, 4)]
    with torch.no_grad():
        fw = torch.cat([clip_std(bb, w0[2].to(DEV)), clip_std(bb, w0[3].to(DEV))], 1)
        fw = ((fw - mu_g[s[chk[0]]]) / sd_g).cpu()
    cos_win = float(F.cosine_similarity(fw, Xs[chk[0]][3:7], dim=1).mean())
    note(dict(sanity_cos_full=[round(c, 4) for c in cos_full], sanity_cos_window=round(cos_win, 4), min=mins()))

    iv_eval = iv_d[rated[iv_d]][: (4 if a.smoke else None)]
    IV, _ = live(iv_eval)
    r0 = rho_w(J.predict(net, iv_eval, IV, s)[0], t[iv_eval], s[iv_eval])
    note(dict(epoch=0, inner_rho_live=round(r0, 4), min=mins()))

    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr, "weight_decay": 0.01},
                             {"params": list(net.parameters()), "lr": a.head_lr, "weight_decay": 0.05}])
    E = 1 if a.smoke else a.ft_epochs
    total, warm = spe * E, max(1, min(20, spe * E // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, k / total))))

    def snap():
        return ({k: v.detach().cpu().clone() for k, v in bb.state_dict().items() if "lora_" in k},
                {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})

    best, best_snap, best_ep, bad = r0, snap(), 0, 0
    signs = deque()  # the sampler runs in this process and the loader keeps order -> FIFO of pair signs

    def spec_stream():
        for spec, sg in windows():
            signs.append(sg)
            yield spec
    dl = DataLoader(Windows(names), batch_sampler=SpecSampler(spec_stream()), num_workers=a.workers,
                    collate_fn=first, persistent_workers=True, prefetch_factor=2, pin_memory=True)
    it = iter(dl)
    for ep in range(1, E + 1):
        net.train(); bb.train()  # TimeSformer has no dropout; train mode turns on gradient checkpointing
        t1, hist, wait = time.time(), [], 0.0
        for step in range(spe):
            opt.zero_grad(set_to_none=True)
            lsum = 0.0
            for _ in range(a.pairs):
                t2 = time.time()
                batch = next(it)
                wait += time.time() - t2
                spec = [(j, o, ego.shape[0]) for j, o, ego, exo in batch]
                sg = signs.popleft()
                feats = []
                for j, o, ego, exo in batch:
                    f = torch.cat([clip_std(bb, ego.to(DEV, non_blocking=True)),
                                   clip_std(bb, exo.to(DEV, non_blocking=True))], 1)
                    feats.append((f - mu_g[s[j]]) / sd_g)
                loss = head_loss(net, feats, spec, sg) / a.pairs
                loss.backward()
                lsum += loss.item()
            nn.utils.clip_grad_norm_(lora + list(net.parameters()), 1.0)
            opt.step(); sched.step()
            hist.append(lsum)
            if a.smoke or (ep == 1 and step == 4):
                note(dict(epoch=ep, step=step, loss=round(lsum, 4), sec_per_step=round((time.time() - t1) / (step + 1), 1),
                          data_wait=round(wait, 1), mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)))
        IV, _ = live(iv_eval)
        cur = rho_w(J.predict(net, iv_eval, IV, s)[0], t[iv_eval], s[iv_eval])
        note(dict(epoch=ep, loss=round(float(np.mean(hist)), 4), inner_rho_live=round(cur, 4),
                  train_min=round((time.time() - t1) / 60, 1), data_wait_min=round(wait / 60, 1), min=mins()))
        if cur > best:
            best, best_snap, best_ep, bad = cur, snap(), ep, 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    last_snap, last_ep = snap(), ep
    torch.save(dict(lora=best_snap[0], head=best_snap[1], epoch=best_ep, args=vars(a)), out / "ft_best.pt")

    # ------------------------------------------------ test
    tst = te_d[: (6 if a.smoke else None)]
    preds = {}
    net.load_state_dict(frozen_state)
    preds["frozen"] = J.predict(net, tst, Xs, s, clips=True)
    variants = [("ft_best", best_snap)] + ([("ft_last", last_snap)] if last_ep != best_ep else [])
    for name, (lw, hw) in variants:
        bb.load_state_dict(lw, strict=False); net.load_state_dict(hw)
        TF, _ = live(tst)
        preds[name] = J.predict(net, tst, TF, s, clips=True)
    jf = torch.load(Path("cls_lang") / "rating_joint_lam4.pt", weights_only=False)
    assert list(jf["names"]) == list(names) and (jf["fold_of"][0][tst] == a.fold).all()
    ref = dict(final=(jf["reg"][0][tst], jf["lvl_reg"][0][tst], jf["lvl_prob"][0][tst]))

    # held-out-expert rows (as rating.py --report): expert stats over all takes; takes with >= 2 experts
    by_e = defaultdict(list)
    for (j, e), r in R.items():
        by_e[e].append(r)
    emu = {e: np.mean(v) for e, v in by_e.items()}; esd = {e: (np.std(v) or 1.0) for e, v in by_e.items()}
    by_t = defaultdict(list)
    for (j, e), r in R.items():
        by_t[j].append((r - emu[e]) / esd[e])
    pos = {int(j): k for k, j in enumerate(tst)}
    rows = [(j, z) for j in pos if len(by_t.get(j, [])) >= 2 for z in by_t[j]]
    hj = np.array([j for j, _ in rows]); hz = np.array([z for _, z in rows])
    tz = z_rating(R, n, np.flatnonzero(rated))
    tr_ = np.array([j for j in tst if rated[j]])
    U = np.unique(p[tst]); rng_b = np.random.default_rng(0)
    draws = [rng_b.choice(U, len(U)) for _ in range(1000)]
    rowp = p[hj]
    Bh = [np.concatenate([np.flatnonzero(rowp == u) for u in d]) for d in draws]

    def score(rv, lv, lp):
        rr = np.array([rv[pos[j]] for j in hj])
        return dict(experts_z=rho(rr, hz), mean_z=rho(rv[[pos[j] for j in tr_]], tz[tr_]),
                    level_rho=rho(lv, y[tst]), level_acc=float((lp.argmax(1) + 1 == y[tst]).mean())), rr

    res, rowpred = {}, {}
    for k, v in preds.items():
        res[k], rowpred[k] = score(v[0], v[1], v[2])
        res[k].update({q: round(x, 3) for q, x in J.comm_diagnostics(v[3], tst, Tte, mean_txt).items()
                       if q in ("pres_auc", "txt_cos", "txt_cos_meanbaseline")})
    res["final"], rowpred["final"] = score(*ref["final"])
    diffs = {}
    for k in [q for q in preds if q.startswith("ft")]:
        for base in ("frozen", "final"):
            d = [rho(rowpred[k][b], hz[b]) - rho(rowpred[base][b], hz[b]) for b in Bh]
            diffs[f"{k} - {base}"] = [round(rho(rowpred[k], hz) - rho(rowpred[base], hz), 3),
                                      round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]
    note(dict(test={k: {q: round(x, 3) for q, x in v.items()} for k, v in res.items()}, experts_z_diff_ci=diffs,
              best_epoch=best_ep, last_epoch=last_ep, rows=len(hz), takes=len(tst), min=mins()))
    torch.save(dict(takes=[names[j] for j in tst], preds={k: v[:3] for k, v in preds.items()}, ref=ref, res=res,
                    diffs=diffs), out / "preds.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
