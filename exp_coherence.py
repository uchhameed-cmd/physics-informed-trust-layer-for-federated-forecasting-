"""Detectability of PV falsification by the coherence test, on the measured PV data.
Step attacks (factor kappa from week 10) and slow 'creep' attacks (growth g per week), with the
rolling trusted baseline used in Q-TRUST and with a baseline anchored at commissioning
(weeks 0-3). Results -> results/coherence.json"""
import json
from pathlib import Path

import numpy as np

from qtrust import data as D

RES = Path(__file__).parent / "results"
TOL = 0.25
ONSET = 10


def weekly_ratios(h, mult):
    """mult: (17, n_weeks) multiplier applied to each home's reported PV in each week."""
    tr = np.where(np.isin(h.time.month.values, D.TRAIN_MONTHS))[0]
    weeks = tr[: (len(tr) // 168) * 168].reshape(-1, 168)
    R = np.zeros((D.N_HOMES, len(weeks)))
    for k, idx in enumerate(weeks):
        pv = h.pv_pu[:, idx] * mult[:, k:k + 1]
        peer = np.median(pv, axis=0)
        day = peer > 0.15
        R[:, k] = np.median(pv[:, day] / peer[day], axis=1)
    return R


def run_test(R, anchored=False):
    """Return flags (17, n_weeks) of the coherence test."""
    lr = np.log(R + 0.05)
    n, W = lr.shape
    flags = np.zeros((n, W), bool)
    for i in range(n):
        hist = []
        for w in range(W):
            if anchored:
                base = np.median(lr[i, :4]) if w >= 4 else lr[i, w]
            else:
                base = np.median(hist[-4:]) if hist else lr[i, w]
            q = abs(lr[i, w] - base) / TOL
            flags[i, w] = q > 1 and w >= 4
            if not flags[i, w]:
                hist.append(lr[i, w])
    return flags


def main():
    h = D.load_homes()
    W = weekly_ratios(h, np.ones((17, 40))).shape[1]
    out = dict(n_weeks=W, tol=TOL, onset=ONSET)
    honest = weekly_ratios(h, np.ones((17, W)))
    for anch in (False, True):
        f = run_test(honest, anch)
        out[f"honest_fpr_{'anchored' if anch else 'rolling'}"] = float(f[:, 4:].mean())
        out[f"honest_fpr_per_home_{'anchored' if anch else 'rolling'}"] = f[:, 4:].mean(1).tolist()
    # step attacks: each home alone as attacker
    step = {}
    for kappa in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
        rec = {"rolling": [], "anchored": [], "first_week": []}
        for i in range(17):
            m = np.ones((17, W)); m[i, ONSET:] = kappa
            R = weekly_ratios(h, m)
            for anch in (False, True):
                f = run_test(R, anch)
                rec["anchored" if anch else "rolling"].append(float(f[i, ONSET:].mean()))
            f = run_test(R, False)
            rec["first_week"].append(bool(f[i, ONSET]))
        step[str(kappa)] = {k: float(np.mean(v)) for k, v in rec.items()}
    out["step"] = step
    # creep attacks
    creep = {}
    for g in [0.02, 0.05, 0.10, 0.20]:
        res = {"rolling": [], "anchored": [], "max_factor_undetected_rolling": []}
        for i in range(17):
            m = np.ones((17, W))
            m[i, ONSET:] = np.minimum((1 + g) ** np.arange(1, W - ONSET + 1), 2.0)
            R = weekly_ratios(h, m)
            fr, fa = run_test(R, False), run_test(R, True)
            res["rolling"].append(float(fr[i, ONSET:].mean()))
            res["anchored"].append(float(fa[i, ONSET:].mean()))
            und = m[i, ONSET:][~fr[i, ONSET:]]
            res["max_factor_undetected_rolling"].append(float(und.max()) if len(und) else 1.0)
        creep[str(g)] = {k: float(np.mean(v)) for k, v in res.items()}
    out["creep"] = creep
    (RES / "coherence.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
