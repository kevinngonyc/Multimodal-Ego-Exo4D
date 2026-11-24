"""Self-supervised LoRA adaptation of TimeSformer K600 to Ego-Exo4D (no level labels in the loss).

usage (from idea_1):  python ssl_adapt/train.py --run NAME [--steps N] [--probe-every N] [--smoke] [--resume]

Losses on the downstream feature itself (L12 patch-token std of last_hidden_state, as in the main caches):
  sync  : symmetric InfoNCE between ego and exo clips of the same moment. Every take contributes C clips >= 2 s apart,
          so the same-take / other-time clips are hard negatives (the match must come from what happens, not from
          the scene).
  part  : SupCon over participants, positives = same participant in a DIFFERENT take (same-take pairs are masked
          out entirely), negatives = other participants from the same site and scenario (batches are drawn within
          a (university, scenario) group).
Batch = P participants x T takes x C clips (default 4 x 2 x 2 = 16 ego/exo pairs = 32 clips).
Every --probe-every steps: features for the probe bank (16 clips x 2,111 official takes, no augmentation) -> level
probe (5-fold StratifiedGroupKFold by participant x 2 repeats, scenario-centred PCA-64 per view, ridge,
within-scenario Spearman; plus official train->val accuracy), compared with the frozen model at step 0.
"""
import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).parent))
from common import BANK_DIR, CLIP_FRAMES, OFFICIAL, RUN_DIR, SIZE

MEAN, STD = 0.45, 0.225
LORA_TARGETS = r".*\.(attention\.attention\.qkv|temporal_attention\.attention\.qkv|attention\.output\.dense|temporal_attention\.output\.dense|temporal_dense)$"

# ---------------------------------------------------------------- data
class Bank(Dataset):
    """item i -> (ego clip, exo clip) uint8 tensors (8, 3, 224, 224), optional per-view augmentation."""
    def __init__(self, name, augment):
        self.ix = torch.load(BANK_DIR / f"{name}_index.pt", weights_only=False)
        self.path, self.augment, self.mm = BANK_DIR / f"{name}.bin", augment, None

    def __len__(self):
        return len(self.ix["item_take"])

    def _clip(self, i, cols, rng):
        if self.mm is None:
            self.mm = np.memmap(self.path, np.uint8, "r")
        frames = [cv2.imdecode(np.asarray(self.mm[o:o + l]), cv2.IMREAD_COLOR)[:, :, ::-1]
                  for o, l in zip(self.ix["off"][i, cols], self.ix["len"][i, cols])]
        x = np.stack(frames)
        if self.augment:
            s = rng.uniform(0.7, 1.0); r = math.exp(rng.uniform(math.log(3 / 4), math.log(4 / 3)))
            w = int(round(SIZE * math.sqrt(s * r))); h = int(round(SIZE * math.sqrt(s / r)))
            w, h = min(w, SIZE), min(h, SIZE)
            x0, y0 = rng.randint(0, SIZE - w), rng.randint(0, SIZE - h)
            x = np.stack([cv2.resize(f[y0:y0 + h, x0:x0 + w], (SIZE, SIZE), interpolation=cv2.INTER_LINEAR) for f in x])
            if rng.random() < 0.5:
                x = x[:, :, ::-1]
            b, c = rng.uniform(-20, 20), rng.uniform(0.8, 1.2)
            x = np.clip((x.astype(np.float32) - 128) * c + 128 + b, 0, 255).astype(np.uint8)
        return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2)

    def __getitem__(self, i):
        rng = random.Random()
        return self._clip(i, slice(0, 8), rng), self._clip(i, slice(8, 16), rng), i

class GroupBatches(Sampler):
    """P participants x T takes x C clips from one (university, scenario) group per batch."""
    def __init__(self, ix, P=4, T=2, C=2, min_gap=16, seed=0):
        self.P, self.T, self.C, self.gap = P, T, C, min_gap
        self.rng = random.Random(seed)
        meta, takes = ix["take_meta"], ix["takes"]
        clips = defaultdict(list)
        for i, (t, k0) in enumerate(zip(ix["item_take"], ix["item_k0"])):
            clips[int(t)].append((int(k0), i))
        self.clips = {t: sorted(v) for t, v in clips.items() if len(v) >= C}
        by_group = defaultdict(lambda: defaultdict(list))
        for t in self.clips:
            m = meta[takes[t]]
            by_group[(m["university"], m["scenario"])][m["participant"]].append(t)
        self.groups = {}
        for g, parts in by_group.items():
            ok = {p: ts for p, ts in parts.items() if len(ts) >= T and p >= 0}
            if len(ok) >= 2:
                self.groups[g] = ok
        self.by_scenario = defaultdict(list)
        for g in self.groups:
            self.by_scenario[g[1]].append(g)
        self.weights = [sum(len(self.clips[t]) for ts in self.groups[g].values() for t in ts) for g in self.groups]
        self.keys = list(self.groups)
        n_used = sum(len(self.clips[t]) for g in self.groups for ts in self.groups[g].values() for t in ts)
        self.n_clips = n_used
        print(f"sampler: {len(self.groups)} (site, scenario) groups, "
              f"{sum(len(v) for v in self.groups.values())} participants with >= {T} takes, "
              f"{n_used}/{len(ix['item_take'])} clips usable", flush=True)

    def _pick_clips(self, t):
        v = self.clips[t]
        for _ in range(20):
            pick = self.rng.sample(v, self.C)
            k = sorted(p[0] for p in pick)
            if all(b - a >= self.gap for a, b in zip(k, k[1:])):
                return [p[1] for p in pick]
        return [p[1] for p in self.rng.sample(v, self.C)]

    def __iter__(self):
        while True:
            g = self.rng.choices(self.keys, self.weights)[0]
            pool = [(g, p) for p in self.groups[g]]
            self.rng.shuffle(pool)
            if len(pool) < self.P:  # fill with participants from other sites of the same scenario
                extra = [(g2, p) for g2 in self.by_scenario[g[1]] if g2 != g for p in self.groups[g2]]
                self.rng.shuffle(extra)
                pool += extra
            batch = []
            for gg, p in pool[:self.P]:
                for t in self.rng.sample(self.groups[gg][p], self.T):
                    batch += self._pick_clips(t)
            yield batch

    def __len__(self):
        return 10**9

# ---------------------------------------------------------------- model
def build_model(lora_r):
    from peft import LoraConfig, get_peft_model
    from transformers import TimesformerModel
    base = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k600")
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, target_modules=LORA_TARGETS, bias="none")
    model = get_peft_model(base, cfg)
    # the input-require-grads hook (added for checkpointing) breaks on TimeSformer's tuple patch-embedding output and
    # is not needed with non-reentrant checkpointing
    try:
        base.disable_input_require_grads()
    except Exception:
        pass
    return model

def head(d_out=256):
    return nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 1024), nn.GELU(), nn.Linear(1024, d_out))

def normalize(x):
    """uint8 (B, 8, 3, 224, 224) -> normalized float"""
    return x.float().div_(255).sub_(MEAN).div_(STD)

def token_std(model, x):
    h = model(pixel_values=normalize(x)).last_hidden_state
    return h[:, 1:].float().std(1)

def sync_loss(ze, zx, tau):
    logits = ze @ zx.T / tau
    y = torch.arange(len(ze), device=ze.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))

def part_loss(z, part, take, tau):
    """SupCon; positives = same participant, other take; same-take pairs removed from the denominator."""
    sim = z @ z.T / tau
    same_take = take[:, None] == take[None, :]
    pos = (part[:, None] == part[None, :]) & ~same_take
    sim = sim.masked_fill(same_take, -1e4)
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    has = pos.any(1)
    if not has.any():
        return sim.new_zeros(())
    return -(logp * pos).sum(1)[has].div(pos.sum(1)[has]).mean()

# ---------------------------------------------------------------- probe
@torch.no_grad()
def probe_features(model, bank, batch=48):
    model.eval()
    loader = DataLoader(bank, batch_size=batch, shuffle=False, num_workers=6, persistent_workers=False)
    n_t = len(bank.ix["takes"])
    acc = {v: np.zeros((n_t, 768)) for v in ("ego", "exo")}
    cnt = np.zeros(n_t)
    for ego, exo, idx in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fe = token_std(model, ego.cuda(non_blocking=True)).cpu().numpy()
            fx = token_std(model, exo.cuda(non_blocking=True)).cpu().numpy()
        t = bank.ix["item_take"][idx.numpy()]
        np.add.at(acc["ego"], t, fe); np.add.at(acc["exo"], t, fx); np.add.at(cnt, t, 1)
    model.train()
    return {v: a / np.maximum(cnt, 1)[:, None] for v, a in acc.items()}

def level_probe(feats, takes, repeats=2):
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    names = [t for t in takes if t in OFFICIAL]
    rows = np.array([takes.index(t) for t in names])
    sc = np.array([OFFICIAL[t]["score"] for t in names], float)
    scen = np.array([OFFICIAL[t]["scenario"] for t in names])
    part = np.array([OFFICIAL[t]["participant"] for t in names])
    split = np.array([OFFICIAL[t]["split"] for t in names])
    S = np.unique(scen)
    lev_c = sc.copy()
    for s in S:
        lev_c[scen == s] -= sc[scen == s].mean()

    def design(blocks, fit):
        parts = [(scen[:, None] == S[None, :]).astype(float)]
        for v in blocks:
            X = feats[v][rows].copy()
            for s in S:
                m = scen == s
                X[m] -= X[fit & m].mean(0)
            parts.append(PCA(64, svd_solver="full").fit(X[fit]).transform(X))
        X = np.hstack(parts)
        return StandardScaler().fit(X[fit]).transform(X)

    out = {}
    for label, blocks in (("ego+exo", ("ego", "exo")), ("exo", ("exo",)), ("ego", ("ego",))):
        P = np.zeros((repeats, len(names)))
        for r in range(repeats):
            for fit_i, held in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(np.zeros(len(names)), sc, part):
                fit = np.zeros(len(names), bool); fit[fit_i] = True
                X = design(blocks, fit)
                p = RidgeCV(alphas=np.logspace(-2, 6, 17)).fit(X[fit], sc[fit]).predict(X[held])
                for s in S:
                    m = scen[held] == s
                    if m.any():
                        p[m] -= p[m].mean()
                P[r, held] = p
        out[f"cv_rho_{label}"] = float(spearmanr(P.mean(0), lev_c)[0])
    fit = split == "train"
    X = design(("ego", "exo"), fit)
    lr = LogisticRegression(C=0.01, max_iter=5000).fit(X[fit], sc[fit])
    out["val_acc_ego+exo"] = float((lr.predict(X[~fit]) == sc[~fit]).mean())
    pv = RidgeCV(alphas=np.logspace(-2, 6, 17)).fit(X[fit], sc[fit]).predict(X[~fit])
    for s in S:
        m = scen[~fit] == s
        if m.any():
            pv[m] -= pv[m].mean()
    out["val_rho_ego+exo"] = float(spearmanr(pv, lev_c[~fit])[0])
    return out

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--probe-every", type=int, default=3000)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--lam-part", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--train-bank", default="train")
    ap.add_argument("--probe-bank", default="probe")
    ap.add_argument("--smoke", action="store_true", help="few steps, tiny probe, no step-0 probe")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = RUN_DIR / a.run
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "args.json", "w"), indent=1)
    log = open(out / "log.jsonl", "a")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    train = Bank(a.train_bank, augment=True)
    sampler = GroupBatches(train.ix, a.P, a.T, a.C)
    loader = DataLoader(train, batch_sampler=sampler, num_workers=a.workers, pin_memory=True, persistent_workers=True,
                        prefetch_factor=4)
    probe = Bank(a.probe_bank, augment=False)

    model = build_model(a.lora_r).cuda()
    model.print_trainable_parameters()
    heads = nn.ModuleDict({"ego": head(), "exo": head(), "part": head()}).cuda()
    lora = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr, "weight_decay": 0.0},
                             {"params": heads.parameters(), "lr": a.head_lr, "weight_decay": 0.05}])
    warm = min(300, a.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    step = 0
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", weights_only=False)
        model.load_state_dict(ck["model"], strict=False); heads.load_state_dict(ck["heads"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); step = ck["step"]
        print("resumed at step", step, flush=True)

    def save(tag):
        lora_sd = {k: v for k, v in model.state_dict().items() if "lora_" in k}
        torch.save({"model": lora_sd, "heads": heads.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "args": vars(a)}, out / f"{tag}.pt")

    def run_probe():
        t0 = time.time()
        if a.smoke:
            probe.ix = {**probe.ix, "item_take": probe.ix["item_take"][:200]}
        f = probe_features(model, probe)
        res = {"step": step, "probe": level_probe(f, probe.ix["takes"]) if not a.smoke else {}, "probe_min": (time.time() - t0) / 60}
        print(json.dumps(res), flush=True)
        log.write(json.dumps(res) + "\n"); log.flush()
        return res

    if step == 0 and not a.smoke and not (out / "base_probe.json").exists():
        with model.disable_adapter():
            res = run_probe()
        json.dump(res, open(out / "base_probe.json", "w"))

    model.train()
    it = iter(loader)
    t0, hist = time.time(), defaultdict(list)
    while step < a.steps:
        ego, exo, idx = next(it)
        ego, exo = ego.cuda(non_blocking=True), exo.cuda(non_blocking=True)
        take = torch.from_numpy(train.ix["item_take"][idx.numpy()]).cuda()
        part = torch.tensor([train.ix["take_meta"][train.ix["takes"][t]]["participant"] for t in take.tolist()]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fe, fx = token_std(model, ego), token_std(model, exo)
            ze, zx = F.normalize(heads["ego"](fe).float(), dim=-1), F.normalize(heads["exo"](fx).float(), dim=-1)
            zp = F.normalize(heads["part"](torch.cat([fe, fx])).float(), dim=-1)
        ls = sync_loss(ze, zx, a.tau)
        lp = part_loss(zp, torch.cat([part, part]), torch.cat([take, take]), a.tau)
        loss = ls + a.lam_part * lp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora + list(heads.parameters()), 1.0)
        opt.step(); sched.step(); step += 1
        with torch.no_grad():
            acc = (ze @ zx.T).argmax(1).eq(torch.arange(len(ze), device=ze.device)).float().mean()
        for k, v in (("loss", loss), ("sync", ls), ("part", lp), ("sync_acc", acc)):
            hist[k].append(float(v))
        if step % 50 == 0 or a.smoke:
            el = time.time() - t0
            rec = {"step": step, **{k: float(np.mean(v)) for k, v in hist.items()}, "lr": sched.get_last_lr()[0],
                   "pairs_per_s": 50 * len(idx) / el if not a.smoke else len(idx) / el,
                   "mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            hist, t0 = defaultdict(list), time.time()
        if a.smoke and step >= 20:
            break
        if step % a.probe_every == 0 or step == a.steps:
            save("last")
            res = run_probe()
            save(f"step{step}")
    if a.smoke:
        run_probe()
        save("last")
    print("done", flush=True)

if __name__ == "__main__":
    main()
