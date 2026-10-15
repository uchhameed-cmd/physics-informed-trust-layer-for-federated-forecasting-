"""Run the federated-learning experiment matrix. Each run is saved to results/fl/<key>.json
and skipped if present.  Usage: python exp_fl.py [main|ablation|fraction|onset|central|all]"""
import json, sys, time, itertools
from pathlib import Path
from multiprocessing import Pool
import numpy as np, torch

from qtrust import fl, data as D
from qtrust.model import Forecaster, get_vec, set_vec, local_train

OUT = Path(__file__).parent / "results" / "fl"
METHODS = ["trust", "fedavg", "median", "trimmed", "krum"]
ATTACKS = ["none", "signflip", "gauss", "alie", "pvinflate"]
SEEDS = [0, 1, 2]
ABL = {"trust-no_geom": dict(use_geom=False), "trust-no_temporal": dict(use_temporal=False),
       "trust-no_probe": dict(use_probe=False), "trust-no_meter": dict(use_meter=False),
       "trust-no_clip": dict(clip=False), "trust-no_gate": dict(gate=None),
       "trust-no_smooth": dict(smooth=0.0)}
FED = None


def init():
    global FED
    torch.set_num_threads(1)
    FED = fl.build()


def theta_for(variant):
    p = OUT / f"theta_{variant}.json"
    if p.exists():
        return json.loads(p.read_text())["theta"]
    cfg = fl.TrustCfg(**ABL.get(variant, {}))
    th = fl.calibrate(FED, cfg)
    p.write_text(json.dumps(dict(variant=variant, theta=th, calibration_seed=100, quantile=0.95)))
    return th


def job(spec):
    method, attack, n_mal, seed, onset, keep = spec
    key = f"{method}_{attack}_m{n_mal}_s{seed}_o{onset}"
    p = OUT / f"{key}.json"
    if p.exists():
        return key, "skip"
    t0 = time.time()
    cfg = None
    if method.startswith("trust"):
        cfg = fl.TrustCfg(**ABL.get(method, {}))
        cfg.theta = theta_for(method)
    o = fl.run(FED, method, attack, 0 if attack == "none" else n_mal, seed, cfg=cfg,
               keep_model=keep, onset=onset)
    o["seconds"] = time.time() - t0
    o.pop("components", None)
    p.write_text(json.dumps(o))
    return key, round(o["seconds"])


def central(seed):
    p = OUT / f"central_s{seed}.json"
    if p.exists():
        return
    torch.manual_seed(seed)
    X = torch.cat(FED.Xtr); Y = torch.cat(FED.Ytr)
    m = Forecaster(); g = torch.Generator().manual_seed(seed)
    for _ in range(10):
        local_train(m, X, Y, epochs=1, lr=1e-4, batch=256, gen=g)
    ev = fl.evaluate(m, FED, list(range(D.N_HOMES)))
    p.write_text(json.dumps(dict(method="central", seed=seed, final=ev)))


def baselines():
    """Persistence and weekly-profile references on the same test origins."""
    p = OUT / "naive_baselines.json"
    h = FED.homes
    res = {}
    for name in ["persistence", "weekly_profile"]:
        rm = []
        for i in range(D.N_HOMES):
            X, Y = FED.Xte[i], FED.Yte[i]
            if name == "persistence":
                L, P = X[:, :24], X[:, 24:48]
            else:
                L, P = X[:, -48:-24], X[:, -24:]
            nlh = L * h.load_scale[i] - P * h.pv_kw[i]
            nl = Y[:, :24] * h.load_scale[i] - Y[:, 24:] * h.pv_kw[i]
            rm.append(float(np.sqrt(((nlh - nl) ** 2).mean())))
        res[name] = dict(rmse=float(np.mean(rm)), worst_rmse=float(np.max(rm)))
    p.write_text(json.dumps(res))


def specs(which):
    S = []
    if which in ("main", "all"):
        for m, a, s in itertools.product(METHODS, ATTACKS, SEEDS):
            S.append((m, a, 4, s, fl.ONSET, s == 0))
    if which in ("ablation", "all"):
        for m, a, s in itertools.product(ABL, ATTACKS, SEEDS):
            S.append((m, a, 4, s, fl.ONSET, False))
    if which in ("fraction", "all"):
        for m, a, k, s in itertools.product(METHODS, ["signflip", "alie", "pvinflate"], [2, 6], SEEDS):
            S.append((m, a, k, s, fl.ONSET, False))
    if which in ("onset", "all"):
        for m, a, s in itertools.product(["median", "trust"], ["signflip", "alie", "pvinflate"], SEEDS):
            S.append((m, a, 4, s, 0, False))
    return S


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    init()
    baselines()
    for s in SEEDS:
        central(s)
    for v in ["trust", *ABL]:          # calibrate thresholds first (attack-free, seed 100)
        print(v, "theta", round(theta_for(v), 3), flush=True)
    with Pool(2, initializer=init) as pool:
        for key, sec in pool.imap_unordered(job, specs(which)):
            print(key, sec, flush=True)
