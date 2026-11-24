"""Supervised LoRA fine-tuning of TimeSformer K600 on official proficiency level (LP-FT, video only).

usage (from idea_1):  python level_ft/finetune.py --fold 0|1|2|3|4|official [--run NAME] [--smoke]

Folds: StratifiedGroupKFold(5, shuffle, random_state=0) by participant over the 2,111 official takes = the folds of
repeat 0 of the frozen level probe (ssl_adapt/train.py level_probe). --fold official: train = official train,
test = official val. The test takes are only touched once, at the end, with the inner-val-selected checkpoint.

Model: shared TimeSformer K600 backbone (LoRA on blocks --layers of the 12, so the lower blocks run forward-only) for
ego and exo; per take, L12 patch-token std averaged over C clips per view -> concat (1536) -> scenario-centred,
standardised with frozen-feature statistics of the training takes -> linear regression head (+ scenario bias) and a
4-class head (+ scenario bias). LP-FT: both heads start at the frozen-feature solution (ridge / logistic, penalties by
participant-grouped CV), LoRA B = 0, so epoch 0 == the frozen linear probe.
Loss = MSE(level) + lam_ce * CE(level). Takes sampled with weight 1/sqrt(#takes of the participant x scenario)
(level is a participant attribute; stops a few heavy participants dominating). Clips per take come from the probe
bank (16) + the self-supervised train bank (random offsets; never contains official val takes).
Inner val = ~1/7 of the training participants; early stopping on its within-scenario Spearman (8 clips per take).
-> models/level_ft/{run}/fold{k}_{log.jsonl, best.pt, preds.pt}
"""
import argparse
import copy
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).parent.parent / "ssl_adapt"))
from common import BANK_DIR, OFFICIAL  # noqa: E402
from train import Bank, normalize  # noqa: E402

OUT = Path("models") / "level_ft"
SCEN = ("Basketball", "Cooking", "Dance", "Music", "Rock Climbing", "Soccer")
TARGETS = (r"attention\.attention\.qkv|temporal_attention\.attention\.qkv|attention\.output\.dense|"
           r"temporal_attention\.output\.dense|temporal_dense")


# ---------------------------------------------------------------- splits / metrics
def official_table(takes):
    y = np.array([OFFICIAL[t]["score"] for t in takes])
    s = np.array([SCEN.index(OFFICIAL[t]["scenario"]) for t in takes])
    p = np.array([OFFICIAL[t]["participant"] for t in takes])
    split = np.array([OFFICIAL[t]["split"] for t in takes])
    return y, s, p, split


def split_fold(fold, y, p, split):
    from sklearn.model_selection import StratifiedGroupKFold
    n = len(y)
    if fold == "official":
        tr, te = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    else:
        tr, te = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(n), y, p))[int(fold)]
    fit_i, iv_i = next(StratifiedGroupKFold(7, shuffle=True, random_state=1).split(np.zeros(len(tr)), y[tr], p[tr]))
    return tr[fit_i], tr[iv_i], te


def centred(v, s):
    v = v.astype(float).copy()
    for k in np.unique(s):
        v[s == k] -= v[s == k].mean()
    return v


def metrics(reg, logit, y, s):
    out = {"rho": float(spearmanr(centred(reg, s), centred(y, s))[0])}
    if logit is not None:
        out["acc"] = float((logit.argmax(1) + 1 == y).mean())
    return out


# ---------------------------------------------------------------- data
class TakeClips(Dataset):
    """local take j -> C random clips (ego, exo) from the take's pool of (bank, item)."""
    def __init__(self, banks, pools, C):
        self.banks, self.pools, self.C = banks, pools, C

    def __len__(self):
        return len(self.pools)

    def __getitem__(self, j):
        rng = random.Random()
        pool = self.pools[j]
        pick = rng.sample(pool, self.C) if len(pool) >= self.C else [rng.choice(pool) for _ in range(self.C)]
        ego, exo = [], []
        for b, i in pick:
            r2 = random.Random(rng.random())
            ego.append(self.banks[b]._clip(i, slice(0, 8), r2))
            exo.append(self.banks[b]._clip(i, slice(8, 16), random.Random(rng.random())))
        return torch.stack(ego), torch.stack(exo), j


class WeightedTakes(Sampler):
    def __init__(self, weights, B, seed=0):
        self.w, self.B, self.g = torch.as_tensor(weights, dtype=torch.float), B, torch.Generator().manual_seed(seed)

    def __iter__(self):
        while True:
            yield torch.multinomial(self.w, self.B, replacement=False, generator=self.g).tolist()

    def __len__(self):
        return 10**9


class EvalItems(Dataset):
    def __init__(self, bank, items):
        self.bank, self.items = bank, items  # [(probe item, local take j)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, j = self.items[k]
        return self.bank._clip(i, slice(0, 8), None), self.bank._clip(i, slice(8, 16), None), j


# ---------------------------------------------------------------- model
class LevelModel(nn.Module):
    def __init__(self, r, layers, mu, sd):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import TimesformerModel
        base = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k600")
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        rx = rf".*encoder\.layer\.({'|'.join(str(l) for l in layers)})\.({TARGETS})$"
        self.backbone = get_peft_model(base, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.1, target_modules=rx,
                                                        bias="none"))
        try:
            base.disable_input_require_grads()
        except Exception:
            pass
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float))  # (6, 1536) scenario means
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float))  # (1536,)
        self.reg = nn.Linear(1536, 1, bias=False)
        self.reg_b = nn.Parameter(torch.zeros(6))
        self.cls = nn.Linear(1536, 4)
        self.cls_b = nn.Parameter(torch.zeros(6, 4))

    def clip_feats(self, x):  # uint8 (N, 8, 3, 224, 224) -> (N, 768)
        h = self.backbone(pixel_values=normalize(x)).last_hidden_state
        return h[:, 1:].float().std(1)

    def head(self, f, s):  # raw take features (B, 1536), scenario idx (B,)
        z = (f.float() - self.mu[s]) / self.sd
        return self.reg(z).squeeze(-1) + self.reg_b[s], self.cls(z) + self.cls_b[s]

    def head_params(self):
        return [self.reg.weight, self.reg_b, self.cls.weight, self.cls.bias, self.cls_b]


def lp_init(model, Xf, y, s, p):
    """ridge / logistic on frozen standardised features; penalties by participant-grouped 5-fold CV."""
    from sklearn.linear_model import LogisticRegressionCV, Ridge
    from sklearn.model_selection import GridSearchCV, GroupKFold
    Z = (Xf - model.mu.cpu().numpy()[s]) / model.sd.cpu().numpy()
    yc = y - np.array([y[s == k].mean() for k in range(6)])[s]
    cv = list(GroupKFold(5).split(Z, y, p))
    gs = GridSearchCV(Ridge(fit_intercept=False), {"alpha": np.logspace(1, 6, 11)}, cv=cv,
                      scoring="neg_mean_squared_error").fit(Z, yc)
    oh = (s[:, None] == np.arange(6)[None]).astype(float)
    lr = LogisticRegressionCV(Cs=np.logspace(-4, -1, 7), cv=cv, max_iter=3000, n_jobs=7).fit(np.hstack([Z, oh]), y)
    assert list(lr.classes_) == [1, 2, 3, 4]
    with torch.no_grad():
        model.reg.weight.copy_(torch.as_tensor(gs.best_estimator_.coef_[None], dtype=torch.float))
        model.reg_b.copy_(torch.as_tensor([y[s == k].mean() for k in range(6)], dtype=torch.float))
        model.cls.weight.copy_(torch.as_tensor(lr.coef_[:, :1536], dtype=torch.float))
        model.cls.bias.copy_(torch.as_tensor(lr.intercept_, dtype=torch.float))
        model.cls_b.copy_(torch.as_tensor(lr.coef_[:, 1536:].T, dtype=torch.float))
    return {"ridge_alpha": float(gs.best_params_["alpha"]), "logistic_C": float(lr.C_[0])}


@torch.no_grad()
def take_features(model, loader, n):
    was = model.training
    model.eval()
    acc, cnt = torch.zeros(n, 1536), torch.zeros(n)
    for ego, exo, j in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fe = model.clip_feats(ego.cuda(non_blocking=True)).cpu()
            fx = model.clip_feats(exo.cuda(non_blocking=True)).cpu()
        acc.index_add_(0, j, torch.cat([fe, fx], 1)); cnt.index_add_(0, j, torch.ones(len(j)))
    model.train(was)
    return acc / cnt.clamp(min=1)[:, None]


@torch.no_grad()
def predict(model, f, s):
    reg, logit = model.head(f.cuda(), torch.as_tensor(s).cuda())
    return reg.cpu().numpy(), logit.float().cpu().numpy()


# ---------------------------------------------------------------- frozen PCA-ridge reference (same as level_probe)
def frozen_reference(Xe, Xx, y, s, fit, test):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.preprocessing import StandardScaler
    parts = [(s[:, None] == np.arange(6)[None]).astype(float)]
    for X in (Xe, Xx):
        X = X.copy()
        for k in range(6):
            m = s == k
            X[m] -= X[fit & m].mean(0)
        parts.append(PCA(64, svd_solver="full").fit(X[fit]).transform(X))
    X = np.hstack(parts)
    X = StandardScaler().fit(X[fit]).transform(X)
    reg = RidgeCV(alphas=np.logspace(-2, 6, 17)).fit(X[fit], y[fit]).predict(X[test])
    logit = LogisticRegression(C=0.01, max_iter=5000).fit(X[fit], y[fit]).predict_proba(X[test])
    return reg, logit


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--run", default="lpft_r8_top6")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--layers", default="6,7,8,9,10,11")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--lam-ce", type=float, default=0.5)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--C", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--eval-clips", type=int, default=8)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = OUT / a.run
    out.mkdir(parents=True, exist_ok=True)
    tag = f"fold{a.fold}" + ("_smoke" if a.smoke else "")
    log = open(out / f"{tag}_log.jsonl", "w")

    def note(rec):
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()

    torch.manual_seed(0); random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    probe_tr, probe_ev = Bank("probe", augment=True), Bank("probe", augment=False)
    train_bank = Bank("train", augment=True)
    takes = probe_ev.ix["takes"]
    y, s, p, split = official_table(takes)
    fit_i, iv_i, te_i = split_fold(a.fold, y, p, split)
    if a.smoke:
        rs = np.random.default_rng(0)
        fit_i, iv_i, te_i = (np.sort(rs.choice(v, n, replace=False)) for v, n in ((fit_i, 300), (iv_i, 48), (te_i, 48)))
    note({"fold": a.fold, "n_fit": len(fit_i), "n_innerval": len(iv_i), "n_test": len(te_i), "args": vars(a)})

    # clip pools (probe bank = bank 0, train bank = bank 1)
    pools = defaultdict(list)
    for i, t in enumerate(probe_ev.ix["item_take"]):
        pools[int(t)].append((0, i))
    tb_id = {t: k for k, t in enumerate(takes)}
    for i, t in enumerate(train_bank.ix["item_take"]):
        name = train_bank.ix["takes"][int(t)]
        if name in tb_id:
            pools[tb_id[name]].append((1, i))
    probe_items = defaultdict(list)
    for i, (t, k0) in enumerate(zip(probe_ev.ix["item_take"], probe_ev.ix["item_k0"])):
        probe_items[int(t)].append((int(k0), i))

    def eval_loader(idx, K):
        items = []
        for j, t in enumerate(idx):
            v = [i for _, i in sorted(probe_items[int(t)])]
            pick = np.unique(np.linspace(0, len(v) - 1, min(K, len(v))).round().astype(int)) if K else range(len(v))
            items += [(v[q], j) for q in pick]
        return DataLoader(EvalItems(probe_ev, items), batch_size=48, shuffle=False, num_workers=6,
                          persistent_workers=K is not None and K < 16)

    # frozen features (16-clip means, cached by ssl_adapt/mix_probe.py) -> standardisation + LP init
    frozen = torch.load(BANK_DIR / "probe_feats.pt", weights_only=False)["frozen"]
    Xf = np.hstack([frozen["ego"], frozen["exo"]])
    mu = np.stack([Xf[fit_i][s[fit_i] == k].mean(0) for k in range(6)])
    sd = (Xf[fit_i] - mu[s[fit_i]]).std(0) + 1e-6
    model = LevelModel(a.lora_r, [int(l) for l in a.layers.split(",")], mu, sd).cuda()
    model.backbone.print_trainable_parameters()
    t0 = time.time()
    note({"lp_init": lp_init(model, Xf[fit_i], y[fit_i], s[fit_i], p[fit_i]), "min": (time.time() - t0) / 60})

    Xf_t = torch.as_tensor(Xf, dtype=torch.float)
    ref = {}
    for name, idx in (("innerval", iv_i), ("test", te_i)):
        r, l = predict(model, Xf_t[idx], s[idx])
        ref[f"lp_frozen16_{name}"] = metrics(r, l, y[idx], s[idx])
    fitmask, temask = np.zeros(len(y), bool), np.zeros(len(y), bool)
    fitmask[np.concatenate([fit_i, iv_i])] = True; temask[te_i] = True
    r, l = frozen_reference(frozen["ego"], frozen["exo"], y, s, fitmask, temask)
    ref["pca_ridge_frozen16_test"] = metrics(r, l, y[te_i], s[te_i])
    note({"reference": ref})

    # training data
    grp = defaultdict(int)
    for t in fit_i:
        grp[(p[t], s[t])] += 1
    weights = [1 / math.sqrt(grp[(p[t], s[t])]) for t in fit_i]
    ds = TakeClips({0: probe_tr, 1: train_bank}, [pools[int(t)] for t in fit_i], a.C)
    loader = DataLoader(ds, batch_sampler=WeightedTakes(weights, a.B), num_workers=a.workers, pin_memory=True,
                        persistent_workers=True, prefetch_factor=4)
    yt = torch.as_tensor(y[fit_i], dtype=torch.float)
    st = torch.as_tensor(s[fit_i])
    iv_loader = eval_loader(iv_i, 2 if a.smoke else a.eval_clips)

    lora = [q for q in model.backbone.parameters() if q.requires_grad]
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr, "weight_decay": 0.01},
                             {"params": model.head_params(), "lr": a.head_lr, "weight_decay": 0.0}])
    spe = 5 if a.smoke else len(fit_i) // a.B
    total = spe * (2 if a.smoke else a.epochs)
    warm = min(100, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, k / total))))

    def evaluate(epoch):
        t1 = time.time()
        f = take_features(model, iv_loader, len(iv_i))
        r, l = predict(model, f, s[iv_i])
        return {"epoch": epoch, "innerval": metrics(r, l, y[iv_i], s[iv_i]), "eval_min": (time.time() - t1) / 60}

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k or "backbone" not in k}

    res = evaluate(0)
    note(res)
    best, best_state, bad = res["innerval"]["rho"], snapshot(), 0
    best_epoch = 0
    it = iter(loader)
    model.train()
    for epoch in range(1, (2 if a.smoke else a.epochs) + 1):
        hist, t1 = defaultdict(list), time.time()
        for _ in range(spe):
            ego, exo, j = next(it)
            B, C = ego.shape[:2]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fe = model.clip_feats(ego.flatten(0, 1).cuda(non_blocking=True)).view(B, C, -1).mean(1)
                fx = model.clip_feats(exo.flatten(0, 1).cuda(non_blocking=True)).view(B, C, -1).mean(1)
            yb, sb = yt[j].cuda(), st[j].cuda()
            reg, logit = model.head(torch.cat([fe, fx], 1), sb)
            l_mse = F.mse_loss(reg, yb)
            l_ce = F.cross_entropy(logit.float(), (yb - 1).long())
            loss = l_mse + a.lam_ce * l_ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora + model.head_params(), 1.0)
            opt.step(); sched.step()
            hist["mse"].append(float(l_mse)); hist["ce"].append(float(l_ce))
        rec = evaluate(epoch)
        rec.update({k: float(np.mean(v)) for k, v in hist.items()})
        rec.update({"train_min": (time.time() - t1) / 60, "lr": sched.get_last_lr()[0],
                    "mem_gb": torch.cuda.max_memory_allocated() / 1e9})
        note(rec)
        if rec["innerval"]["rho"] > best:
            best, best_state, bad, best_epoch = rec["innerval"]["rho"], snapshot(), 0, epoch
        else:
            bad += 1
            if bad >= a.patience:
                break

    # final: best (inner-val) checkpoint on the held-out takes, all 16 probe clips
    model.load_state_dict(best_state, strict=False)
    torch.save({"state": best_state, "epoch": best_epoch, "args": vars(a)}, out / f"{tag}_best.pt")
    te_loader = eval_loader(te_i, None)
    f = take_features(model, te_loader, len(te_i))
    r, l = predict(model, f, s[te_i])
    final = {"best_epoch": best_epoch, "best_innerval_rho": best, "test": metrics(r, l, y[te_i], s[te_i]), **ref}
    note({"final": final})
    torch.save({"takes": [takes[t] for t in te_i], "reg": r, "logit": l, "feats": f.numpy(), "final": final},
               out / f"{tag}_preds.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
