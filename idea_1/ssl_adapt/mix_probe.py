"""Probe mixes of adapted / frozen views on the probe bank: python ssl_adapt/mix_probe.py RUN CKPT [RUN CKPT ...]"""
import json, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import RUN_DIR
from train import Bank, build_model, level_probe, probe_features


def main():
    bank = Bank("probe", augment=False)
    model = build_model(16).cuda()
    cache = Path("embeddings/ssl_bank/probe_feats.pt")
    F = torch.load(cache, weights_only=False) if cache.exists() else {}
    if "frozen" not in F:
        with model.disable_adapter():
            F["frozen"] = probe_features(model, bank)
        torch.save(F, cache)
    args = sys.argv[1:]
    for run, ck in zip(args[::2], args[1::2]):
        key = f"{run}/{ck}"
        if key not in F:
            sd = torch.load(RUN_DIR / run / f"{ck}.pt", weights_only=False)["model"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            assert not unexpected
            F[key] = probe_features(model, bank)
            torch.save(F, cache)
    takes = bank.ix["takes"]
    print("frozen", json.dumps({k: round(v, 3) for k, v in level_probe(F["frozen"], takes, repeats=4).items()}), flush=True)
    for key in [k for k in F if k != "frozen"]:
        for label, mix in (("adapted both", F[key]), ("adapted ego + frozen exo", {"ego": F[key]["ego"], "exo": F["frozen"]["exo"]})):
            print(key, label, json.dumps({k: round(v, 3) for k, v in level_probe(mix, takes, repeats=4).items()}), flush=True)

if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this module
    main()
