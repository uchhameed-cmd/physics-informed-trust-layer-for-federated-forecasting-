"""Second-round experiments: threshold-aware adaptive attack and threshold sensitivity.
Results -> results/fl2/*.json ; calibration scores -> results/fl2/calib_scores.json"""
import json, sys, time, itertools
from pathlib import Path
from multiprocessing import Pool
import numpy as np, torch
from qtrust import fl

OUT = Path(__file__).parent / "results" / "fl2"; OUT.mkdir(parents=True, exist_ok=True)
FED = None
THETA = json.loads((Path(__file__).parent / "results" / "fl" / "theta_trust.json").read_text())["theta"]


def init():
    global FED
    torch.set_num_threads(1)
    FED = fl.build()


def job(spec):
    kind, method, attack, seed, param = spec
    key = f"{kind}_{method}_{attack}_p{param}_s{seed}"
    p = OUT / f"{key}.json"
    if p.exists():
        return key, "skip"
    t0 = time.time()
    cfg = None
    if kind == "adaptive":
        fl.ADAPT_MARGIN = float(param); fl.ADAPT_THETA = THETA
        if method == "trust":
            cfg = fl.TrustCfg(theta=THETA)
    else:   # threshold sensitivity
        th = json.loads((OUT / f"theta_q{param}.json").read_text())["theta"]
        cfg = fl.TrustCfg(theta=th)
    o = fl.run(FED, method, attack, 0 if attack == "none" else 4, seed, cfg=cfg)
    o.update(kind=kind, param=param, seconds=time.time() - t0)
    p.write_text(json.dumps(o))
    return key, round(o["seconds"])


if __name__ == "__main__":
    init()
    # calibration scores of the attack-free monitor run (seed 100) -> thresholds for several q
    cs = OUT / "calib_scores.json"
    if not cs.exists():
        S = fl.calibrate(FED, fl.TrustCfg(), return_scores=True)
        cs.write_text(json.dumps(S.tolist()))
    S = np.array(json.loads(cs.read_text()))
    for q in ["0.90", "0.99"]:
        (OUT / f"theta_q{q}.json").write_text(json.dumps(dict(q=float(q), theta=float(np.quantile(S, float(q))))))
    specs = []
    for m, s, mg in itertools.product(["trust", "fedavg", "median", "trimmed", "krum"], [0, 1, 2], ["0.5", "1.0"]):
        specs.append(("adaptive", m, "adaptive", s, mg))
    for q, a, s in itertools.product(["0.90", "0.99"], ["none", "signflip", "alie", "pvinflate"], [0, 1, 2]):
        specs.append(("theta", "trust", a, s, q))
    with Pool(2, initializer=init) as pool:
        for key, sec in pool.imap_unordered(job, specs):
            print(key, sec, flush=True)
