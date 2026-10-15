"""Publication figures (PDF) from the result files."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qtrust import data as D, fl

RES = Path(__file__).parent / "results"
FIG = Path(__file__).parent / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
# validated categorical palette (light): slot order fixed
COL = {"trust": "#2a78d6", "fedavg": "#eb6834", "median": "#1baf7a", "trimmed": "#eda100", "krum": "#e87ba4"}
NAME = {"trust": "Q-TRUST", "fedavg": "FedAvg", "median": "Median", "trimmed": "Trimmed mean", "krum": "Multi-Krum"}
HATCH = {"trust": "", "fedavg": "//", "median": "..", "trimmed": "xx", "krum": "\\\\"}
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e4e0"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42})


def fig_data():
    h = D.load_homes()
    fed = fl.build(h)
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.3))
    jun = (h.time.month.values == 6)
    hr = h.time.hour.values
    prof = np.array([[h.pv_pu[i, jun & (hr == k)].mean() for k in range(24)] for i in range(17)])
    env = np.array([h.envelope[jun & (hr == k)].mean() for k in range(24)])
    for i in range(17):
        if i in (13, 14):
            continue
        ax[0].plot(range(24), prof[i], color="#b9b9b2", lw=1)
    ax[0].plot(range(24), prof[13], color="#eb6834", lw=2, label="Home 14")
    ax[0].plot(range(24), prof[14], color="#6250d6", lw=2, label="Home 15")
    ax[0].plot([], [], color="#b9b9b2", lw=1, label="Other 15 homes")
    ax[0].plot(range(24), env, color=INK, lw=1.2, ls="--", label=r"Envelope $\bar e_h$")
    ax[0].set_xlabel("Hour of day (June)"); ax[0].set_ylabel("PV output (kW per kW)")
    ax[0].legend(loc="upper left", fontsize=7); ax[0].set_xlim(0, 23); ax[0].set_ylim(0, 2.0)
    ax[0].set_title("(a) Mean daily PV profile and physical envelope", fontsize=8, color=INK)
    r = fed.peer_ratio_week
    wk = np.arange(r.shape[1])
    for i in range(17):
        if i in (13, 14):
            continue
        ax[1].plot(wk, r[i], color="#b9b9b2", lw=1)
    ax[1].plot(wk, r[13], color="#eb6834", lw=2, label="Home 14")
    ax[1].plot(wk, r[14], color="#6250d6", lw=2, label="Home 15")
    ax[1].plot(wk[10:], 1.6 * r[0, 10:], color="#e34948", lw=1.5, ls=":", label="Home 1 inflated x1.6")
    ax[1].plot(wk, r[0], color=INK, lw=1.2, label="Home 1")
    ax[1].set_xlabel("Training week"); ax[1].set_ylabel(r"PV ratio to neighbours $\rho_i$")
    ax[1].set_title("(b) Weekly PV coherence with the neighbourhood", fontsize=8, color=INK)
    ax[1].legend(loc="upper left", fontsize=6.5, ncol=2); ax[1].set_ylim(0, 3.4)
    fig.tight_layout(); fig.savefig(FIG / "fig_data.pdf"); plt.close(fig)


def fig_main(S):
    attacks = ["none", "signflip", "gauss", "alie", "pvinflate"]
    an = ["No attack", "Sign flip", "Gaussian", "ALIE", "PV inflation"]
    meths = ["fedavg", "median", "trimmed", "krum", "trust"]
    fig, ax = plt.subplots(figsize=(7.16, 2.2))
    w = 0.16
    for j, m in enumerate(meths):
        mu = [np.mean(S["main"][f"{m}_{a}"]["rmse"]) for a in attacks]
        sd = [np.std(S["main"][f"{m}_{a}"]["rmse"]) for a in attacks]
        x = np.arange(len(attacks)) + (j - 2) * (w + 0.01)
        mu_c = np.minimum(mu, 1.3)
        ax.bar(x, np.array(mu_c) - 0.75, w, bottom=0.75, color=COL[m], hatch=HATCH[m], edgecolor="white",
               lw=0.4, label=NAME[m], yerr=sd, error_kw=dict(lw=0.6, ecolor=MUTED, capsize=1.5))
        for xi, v in zip(x, mu):
            if v > 1.3:
                ax.text(xi, 1.305, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.axhline(S["naive"]["weekly_profile"]["rmse"], color=INK, lw=0.8, ls="--")
    ax.text(4.55, S["naive"]["weekly_profile"]["rmse"] + 0.005, "weekly profile", fontsize=6.5, color=INK, ha="right")
    ax.set_xticks(range(len(attacks))); ax.set_xticklabels(an)
    ax.set_ylabel("Net-load RMSE, honest homes (kW)"); ax.set_ylim(0.75, 1.36)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig_main.pdf"); plt.close(fig)


def fig_fraction(S):
    meths = ["fedavg", "median", "trimmed", "krum", "trust"]
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.15))
    xt = [2, 4, 6]; xl = ["2\n(12%)", "4\n(24%)", "6\n(35%)"]
    for k, (a, t) in enumerate([("signflip", "(a) Sign flip"), ("alie", "(b) ALIE")]):
        for m in meths:
            ys = [np.mean(S["fraction"][f"{m}_{a}_{n}"]["rel"]) for n in xt]
            ax[k].plot(xt, ys, marker="o", ms=4, lw=1.6, color=COL[m], label=NAME[m])
        ax[k].set_xticks(xt); ax[k].set_xticklabels(xl); ax[k].set_title(t, fontsize=8, color=INK)
        ax[k].set_ylabel("RMSE / attack-free RMSE")
    ax[0].set_yscale("log")
    for a, lab, c in [("signflip", "Sign flip", "#2a78d6"), ("alie", "ALIE", "#eb6834"), ("pvinflate", "PV inflation", "#1baf7a")]:
        ys = [S["fraction"][f"trust_{a}_{n}"]["recall"] for n in xt]
        fp = [S["fraction"][f"trust_{a}_{n}"]["fpr"] for n in xt]
        ax[2].plot(xt, ys, marker="o", ms=4, lw=1.6, color=c, label=f"{lab} recall")
        ax[2].plot(xt, fp, marker="s", ms=3, lw=1, ls="--", color=c)
    ax[2].plot([], [], ls="--", color=MUTED, marker="s", ms=3, label="FPR (same colour)")
    ax[2].set_xticks(xt); ax[2].set_xticklabels(xl); ax[2].set_ylim(-0.02, 1.05)
    ax[2].set_title("(c) Q-TRUST detection", fontsize=8, color=INK); ax[2].set_ylabel("Rate")
    ax[1].set_xlabel("Malicious homes")
    ax[0].legend(fontsize=6.5, loc="upper left"); ax[2].legend(fontsize=6, loc="center right")
    fig.tight_layout(); fig.savefig(FIG / "fig_fraction.pdf"); plt.close(fig)


def fig_trajectory(T):
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.0), sharey=True)
    for k, (a, t) in enumerate([("signflip", "(a) Sign flip"), ("alie", "(b) ALIE"), ("pvinflate", "(c) PV inflation")]):
        o = T[a]; mal = o["malicious"]; th = o["theta"]
        S = np.array([c["s"] for c in o["components"]])
        Q = np.array([c["q"] for c in o["components"]])
        rounds = np.arange(1, len(S) + 1)
        for i in range(17):
            if i in mal:
                continue
            ax[k].plot(rounds, S[:, i], color="#b9b9b2", lw=0.8)
        for i in mal:
            ax[k].plot(rounds, S[:, i], color="#e34948", lw=1.4)
            if a == "pvinflate":
                fl_r = rounds[Q[:, i] > 1]
                ax[k].plot(fl_r, np.full(len(fl_r), 9.6), "|", color="#e34948", ms=5)
        ax[k].axhline(th, color=INK, ls="--", lw=0.8)
        ax[k].axvline(10.5, color=MUTED, ls=":", lw=0.8)
        ax[k].set_title(t, fontsize=8, color=INK); ax[k].set_xlabel("Round"); ax[k].set_ylim(0, 10)
    ax[0].set_ylabel(r"Smoothed score $s_i$")
    ax[0].plot([], [], color="#e34948", lw=1.4, label="malicious"); ax[0].plot([], [], color="#b9b9b2", label="honest")
    ax[0].plot([], [], color=INK, ls="--", lw=0.8, label=r"$\theta$")
    ax[0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig_trajectory.pdf"); plt.close(fig)


if __name__ == "__main__":
    S = json.load(open(RES / "summary.json"))
    fig_data()
    fig_main(S)
    fig_fraction(S)
    p = RES / "trajectories.json"
    if p.exists():
        fig_trajectory(json.load(open(p)))
    print("figures written")
