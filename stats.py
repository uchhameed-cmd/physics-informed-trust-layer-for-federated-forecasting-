"""Paired statistics on per-home errors (results/stats.json, paper/tables/stats.tex).
For each attack, Q-TRUST is compared with every baseline on the same honest homes and seeds:
mean difference of per-home RMSE with a 95% bootstrap CI and the Wilcoxon signed-rank p-value."""
import glob, json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RES = Path(__file__).parent / "results"
TAB = Path(__file__).parent / "paper" / "tables"
ATT = ["none", "signflip", "gauss", "alie", "pvinflate"]
AN = {"none": "No attack", "signflip": "Sign flip", "gauss": "Gaussian", "alie": "ALIE", "pvinflate": "PV inflation"}
BASE = ["fedavg", "median", "trimmed", "krum"]
BN = {"fedavg": "FedAvg", "median": "Median", "trimmed": "Trimmed", "krum": "Multi-Krum"}


def per_home(method, attack, seed, honest=None):
    o = json.load(open(RES / "fl" / f"{method}_{attack}_m4_s{seed}_o10.json"))
    pc = np.array(o["final"]["rmse_per_client"])
    hon = honest if honest is not None else [i for i in range(17) if i not in o["malicious"]]
    if attack == "none":
        return pc[hon], hon
    return pc, hon


def main():
    rng = np.random.default_rng(0)
    out = {}
    rows = []
    for a in ATT:
        cells = []
        for b in BASE:
            diffs = []
            for s in (0, 1, 2):
                ref = json.load(open(RES / "fl" / f"trust_signflip_m4_s{s}_o10.json"))["malicious"]
                hon = [i for i in range(17) if i not in ref]
                t, _ = per_home("trust", a, s, hon)
                c, _ = per_home(b, a, s, hon)
                diffs.append(t - c)
            d = np.concatenate(diffs)
            boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
            lo, hi = np.quantile(boot, [0.025, 0.975])
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            out[f"{a}_{b}"] = dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), p=p, n=len(d))
            star = "$^{*}$" if p < 0.05 else ""
            cells.append(f"{1000*d.mean():+.1f} [{1000*lo:+.1f}, {1000*hi:+.1f}]{star}")
        rows.append(AN[a] + " & " + " & ".join(cells) + r" \\")
    tex = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
           "Attack & " + " & ".join("vs " + BN[b] for b in BASE) + r" \\", r"\midrule", *rows,
           r"\bottomrule", r"\end{tabular}"]
    (TAB / "stats.tex").write_text("\n".join(tex))
    (RES / "stats.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, {a: round(b, 4) for a, b in v.items()})


if __name__ == "__main__":
    main()
