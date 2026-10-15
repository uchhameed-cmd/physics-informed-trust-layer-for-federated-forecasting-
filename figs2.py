"""Figures for the extended study (adaptive attack, calibration, detectability, re-dispatch, incentives)."""
import glob, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path(__file__).parent / "results"
FIG = Path(__file__).parent / "paper_els" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COL = {"trust": "#2a78d6", "fedavg": "#eb6834", "median": "#1baf7a", "trimmed": "#eda100", "krum": "#e87ba4"}
NAME = {"trust": "Q-TRUST", "fedavg": "FedAvg", "median": "Median", "trimmed": "Trimmed mean", "krum": "Multi-Krum"}
HATCH = {"trust": "", "fedavg": "//", "median": "..", "trimmed": "xx", "krum": "\\\\"}
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e4e0"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42})


def clean_on(method, seed, honest):
    o = json.load(open(RES / "fl" / f"{method}_none_m4_s{seed}_o10.json"))
    return float(np.mean(np.array(o["final"]["rmse_per_client"])[honest]))


def adaptive_summary():
    S = {}
    for m in ["fedavg", "median", "trimmed", "krum", "trust"]:
        for mg in ["0.5", "1.0"]:
            rel, lam, rec = [], [], []
            for s in (0, 1, 2):
                p = RES / "fl2" / f"adaptive_{m}_adaptive_p{mg}_s{s}.json"
                if not p.exists():
                    continue
                o = json.load(open(p))
                hon = [i for i in range(17) if i not in o["malicious"]]
                rel.append(o["final"]["rmse"] / clean_on(m, s, hon))
                lam.append(float(np.mean(o.get("adapt_lambda", [0]))))
                if o.get("detection"):
                    rec.append(o["detection"]["recall"])
            S[f"{m}_{mg}"] = dict(rel=rel, lam=lam, recall=rec)
    return S


def fig_adaptive(S):
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.3))
    meths = ["fedavg", "median", "trimmed", "krum", "trust"]
    for k, mg in enumerate(["0.5", "1.0"]):
        mu = [100 * (np.mean(S[f"{m}_{mg}"]["rel"]) - 1) for m in meths]
        sd = [100 * np.std(S[f"{m}_{mg}"]["rel"]) for m in meths]
        ax[k].bar(range(5), mu, 0.65, color=[COL[m] for m in meths], hatch=None, yerr=sd,
                  error_kw=dict(lw=0.6, ecolor=MUTED, capsize=2), edgecolor="white")
        for j, m in enumerate(meths):
            ax[k].text(j, max(mu[j], 0) + sd[j] + 0.08, f"{mu[j]:+.1f}%", ha="center", fontsize=7, color=INK)
        ax[k].set_xticks(range(5)); ax[k].set_xticklabels([NAME[m].replace(" ", "\n") for m in meths], fontsize=7)
        ax[k].set_title(f"({'ab'[k]}) margin $m={mg}$", fontsize=8, color=INK)
        ax[k].set_ylim(-1.6, 2.4)
        ax[k].axhline(0, color=INK, lw=0.6)
    ax[0].set_ylabel("RMSE increase vs attack-free (%)")
    fig.tight_layout(); fig.savefig(FIG / "fig_adaptive.pdf"); plt.close(fig)


def calib_summary():
    out = {}
    for q in ["0.90", "0.95", "0.99"]:
        row = {}
        for a in ["none", "signflip", "alie", "pvinflate"]:
            runs = []
            for s in (0, 1, 2):
                p = (RES / "fl" / f"trust_{a}_m4_s{s}_o10.json") if q == "0.95" else (RES / "fl2" / f"theta_trust_{a}_p{q}_s{s}.json")
                if p.exists():
                    runs.append(json.load(open(p)))
            D = [r["detection"] for r in runs]
            if not D:
                continue
            tp = sum(d["tp"] for d in D); fp = sum(d["fp"] for d in D); fn = sum(d["fn"] for d in D); tn = sum(d["tn"] for d in D)
            row[a] = dict(fpr=fp / (fp + tn), recall=(tp / (tp + fn) if tp + fn else None),
                          rmse=float(np.mean([r["final"]["rmse"] for r in runs])))
        out[q] = row
    return out


def fig_calib(C):
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.2))
    qs = ["0.90", "0.95", "0.99"]; nom = [0.10, 0.05, 0.01]
    obs = [C[q]["none"]["fpr"] for q in qs]
    ax[0].plot(nom, nom, ls="--", color=MUTED, lw=0.8, label="nominal $1-q$")
    ax[0].plot(nom, obs, marker="o", color=COL["trust"], lw=1.6, label="observed (no attack)")
    ax[0].set_xlabel("Nominal false-alarm rate $1-q$"); ax[0].set_ylabel("Observed FPR")
    ax[0].legend(fontsize=7); ax[0].set_title("(a) Calibration of false alarms", fontsize=8, color=INK)
    for a, lab, c in [("signflip", "Sign flip", "#2a78d6"), ("alie", "ALIE", "#eb6834"), ("pvinflate", "PV inflation", "#1baf7a")]:
        ax[1].plot([1 - float(q) for q in qs], [C[q][a]["recall"] for q in qs], marker="o", color=c, lw=1.6, label=lab)
    ax[1].set_xlabel("Nominal false-alarm rate $1-q$"); ax[1].set_ylabel("Recall")
    ax[1].set_ylim(-0.02, 1.05); ax[1].legend(fontsize=7)
    ax[1].set_title("(b) Recall at each threshold", fontsize=8, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "fig_calib.pdf"); plt.close(fig)


def fig_coherence():
    o = json.load(open(RES / "coherence.json"))
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.2))
    ks = sorted(o["step"], key=float)
    ax[0].plot([float(k) for k in ks], [o["step"][k]["rolling"] for k in ks], marker="o", color=COL["trust"], lw=1.6, label="rolling baseline")
    ax[0].plot([float(k) for k in ks], [o["step"][k]["anchored"] for k in ks], marker="s", color=COL["fedavg"], lw=1.6, label="anchored baseline")
    ax[0].axvline(np.exp(0.41), color=MUTED, ls=":", lw=0.8)
    ax[0].text(np.exp(0.41) + 0.01, 0.05, "Prop. 3\nguarantee", fontsize=6.5, color=MUTED)
    ax[0].set_xlabel(r"Step inflation factor $\kappa$"); ax[0].set_ylabel("Share of attack weeks flagged")
    ax[0].set_ylim(-0.02, 1.02); ax[0].legend(fontsize=7, loc="lower right")
    ax[0].set_title("(a) Step falsification", fontsize=8, color=INK)
    gs = sorted(o["creep"], key=float); x = np.arange(len(gs))
    ax[1].bar(x - 0.18, [o["creep"][g]["rolling"] for g in gs], 0.34, color=COL["trust"], label="rolling baseline", edgecolor="white")
    ax[1].bar(x + 0.18, [o["creep"][g]["anchored"] for g in gs], 0.34, color=COL["fedavg"], hatch="//", label="anchored baseline", edgecolor="white")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{100*float(g):.0f}%/week" for g in gs])
    ax[1].set_ylabel("Share of attack weeks flagged"); ax[1].set_ylim(0, 1.02); ax[1].legend(fontsize=7)
    ax[1].set_title("(b) Slow creep falsification", fontsize=8, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "fig_coherence.pdf"); plt.close(fig)


def fig_market2():
    M = json.load(open(RES / "market.json")); M2 = json.load(open(RES / "market2.json"))
    keys = [k for k in ["trust_none", "fedavg_none", "median_none", "fedavg_signflip", "trust_signflip"] if k in M2["rolling"]]
    lab = {"trust_none": "Q-TRUST", "fedavg_none": "FedAvg", "median_none": "Median",
           "fedavg_signflip": "FedAvg\n(sign flip)", "trust_signflip": "Q-TRUST\n(sign flip)"}
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.3))
    x = np.arange(len(keys))
    ax[0].bar(x - 0.18, [M[k]["violations"] for k in keys], 0.34, color="#b9b9b2", label="day-ahead only", edgecolor="white")
    ax[0].bar(x + 0.18, [M2["rolling"][k]["violations"] for k in keys], 0.34, color=COL["trust"], label="+ intraday re-dispatch", edgecolor="white")
    ax[0].axhline(M["oracle_aware"]["violations"], color=INK, ls="--", lw=0.8)
    ax[0].text(len(keys) - 0.5, M["oracle_aware"]["violations"] + 15, "perfect forecasts", fontsize=6.5, ha="right", color=INK)
    ax[0].set_xticks(x); ax[0].set_xticklabels([lab[k] for k in keys], fontsize=7)
    ax[0].set_ylabel("Bus-hours outside [0.95, 1.05] p.u."); ax[0].legend(fontsize=7)
    ax[0].set_title("(a) Voltage violations, 60 test days", fontsize=8, color=INK)
    inc = M2["incentives"]
    names = [("inflated", "unchecked"), ("inflated_physics", "physics cap"), ("inflated_trust_coupled", "trust-coupled cap")]
    x = np.arange(len(names))
    g0 = [inc[f"{n}_phi0.0"]["attacker_gain_usd"] for n, _ in names]
    g1 = [inc[f"{n}_phi1.0"]["attacker_gain_usd"] for n, _ in names]
    ax[1].bar(x - 0.18, g0, 0.34, color=COL["fedavg"], label=r"single price ($\phi=0$)", edgecolor="white")
    ax[1].bar(x + 0.18, g1, 0.34, color=COL["trust"], hatch="//", label=r"penalised ($\phi=1$)", edgecolor="white")
    ax[1].axhline(0, color=INK, lw=0.6)
    ax[1].set_xticks(x); ax[1].set_xticklabels([l for _, l in names], fontsize=7)
    ax[1].set_ylabel("Attackers' gain (US\\$)"); ax[1].set_ylim(-37000, 7000)
    ax[1].legend(fontsize=7, loc="upper right", ncol=2)
    ax[1].set_title("(b) Does inflating PV bids pay?", fontsize=8, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "fig_market2.pdf"); plt.close(fig)


if __name__ == "__main__":
    A = adaptive_summary(); C = calib_summary()
    (RES / "summary2.json").write_text(json.dumps(dict(adaptive=A, calibration=C), indent=1))
    fig_adaptive(A); fig_calib(C); fig_coherence(); fig_market2()
    print(json.dumps(dict(adaptive={k: dict(rel=np.round(v["rel"], 4).tolist(), lam=np.round(v["lam"], 2).tolist(), recall=v["recall"]) for k, v in A.items()}, calibration=C), indent=1))
