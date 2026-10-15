"""Market extensions (results -> results/market2.json):
  1. intraday receding-horizon re-dispatch vs day-ahead only, for several forecast sources;
  2. trust-coupled PV bid check (tight tolerance for nodes flagged by the FL coherence test);
  3. two-settlement accounting per node and the incentive to inflate PV bids under
     single-price and penalised deviation settlement."""
import json
from pathlib import Path

import numpy as np

from qtrust import grid as G, fl, data as D
from exp_market import setup, forecasts, verify_bids, N_HOMES_PER_NODE, INFLATE

RES = Path(__file__).parent / "results"


def detail(h, f, bg, com, days, Lhat, bidP):
    """Day-ahead clearing + realisation; return scheduled and actual node net loads."""
    S, A = [], []
    agg = dict(cost=0.0, violations=0, imbalance=0.0)
    for k, d0 in enumerate(days):
        sl = slice(d0, d0 + 24)
        L = N_HOMES_PER_NODE * h.load[:, sl]
        P = N_HOMES_PER_NODE * h.pv_pu[:, sl] * h.pv_kw[:, None]
        s = G.clear_day(f, com, Lhat[k], bidP[k], h.price[sl], bg[sl])
        r = G.realize(f, com, s, L, P, h.price[sl], bg[sl], bidP[k])
        sched = Lhat[k] - bidP[k] + s["ch"] - s["dis"] + s["cur"]
        S.append(sched); A.append(r["net"])
        agg["cost"] += r["cost"]; agg["violations"] += r["violations"]
        agg["imbalance"] += float(np.abs(r["net"] - sched).sum())
    return np.array(S), np.array(A), agg


def settle(S, A, price, fit, phi):
    """Two-settlement bills per node (positive = pays).  Day-ahead: hourly P2P pool at the
    mid price between import tariff and export price, pro rata; residual at grid prices.
    Real time: deviation d = actual - scheduled; d>0 pays (1+phi)*tariff, d<0 is paid fit."""
    days, N, T = S.shape
    bill = np.zeros(N)
    for k in range(days):
        for t in range(T):
            pi = price[k, t]; p2p = 0.5 * (pi + fit)
            n = S[k, :, t]
            buy = np.maximum(n, 0); sell = np.maximum(-n, 0)
            V = min(buy.sum(), sell.sum())
            fb = V / buy.sum() if buy.sum() > 0 else 0.0
            fs = V / sell.sum() if sell.sum() > 0 else 0.0
            bill += buy * (fb * p2p + (1 - fb) * pi)
            bill -= sell * (fs * p2p + (1 - fs) * fit)
            d = A[k, :, t] - n
            bill += np.where(d > 0, d * (1 + phi) * pi, d * fit)
    return bill


def main():
    h, f, bg, com, days = setup()
    fed = fl.build(h)
    price = np.stack([h.price[d0:d0 + 24] for d0 in days])
    out = {}
    # ---------------- 1. intraday re-dispatch
    roll = {}
    for key in ["trust_none", "fedavg_none", "median_none", "fedavg_signflip", "trust_signflip"]:
        m, a = key.split("_")
        o = json.loads((RES / "fl" / f"{m}_{a}_m4_s0_o10.json").read_text())
        Lh, Ph = forecasts(fed, o["w"])
        agg = dict(cost=0.0, violations=0, vmin=9.0)
        for k, d0 in enumerate(days):
            sl = slice(d0, d0 + 24)
            L = N_HOMES_PER_NODE * h.load[:, sl]
            P = N_HOMES_PER_NODE * h.pv_pu[:, sl] * h.pv_kw[:, None]
            r = G.rolling_day(f, com, Lh[k], Ph[k], L, P, h.price[sl], bg[sl])
            agg["cost"] += r["cost"]; agg["violations"] += r["violations"]; agg["vmin"] = min(agg["vmin"], r["vmin"])
        roll[key] = agg
        print("rolling", key, agg, flush=True)
    out["rolling"] = roll
    # ---------------- 2./3. bid inflation: settlement and trust coupling
    o = json.loads((RES / "fl" / "trust_none_m4_s0_o10.json").read_text())
    Lh, Ph = forecasts(fed, o["w"])
    mal = json.loads((RES / "fl" / "trust_pvinflate_m4_s0_o10.json").read_text())
    flag_rate = np.array(mal["flag_rate_per_client"]); M = mal["malicious"]
    flagged = flag_rate > 0.5                       # nodes the FL coherence test flags in most rounds
    ref = np.median(fed.peer_ratio_week, axis=1)
    bid = Ph.copy(); bid[:, M] *= INFLATE
    variants = {
        "honest": Ph,
        "inflated": bid,
        "inflated_physics": verify_bids(fed, bid, ref, 0.25),
    }
    # trust-coupled: tight tolerance (0) for flagged nodes, default for the others
    tc = verify_bids(fed, bid, ref, 0.25)
    tight = verify_bids(fed, bid, ref, 0.0)
    tc[:, flagged] = tight[:, flagged]
    variants["inflated_trust_coupled"] = tc
    hon_tc = verify_bids(fed, Ph, ref, 0.25); hon_tight = verify_bids(fed, Ph, ref, 0.0)
    hon_tc[:, flagged] = hon_tight[:, flagged]
    variants["honest_trust_coupled"] = hon_tc
    res = {}
    bills = {}
    for name, B in variants.items():
        S, A, agg = detail(h, f, bg, com, days, Lh, B)
        for phi in (0.0, 1.0):
            bills[(name, phi)] = settle(S, A, price, com.fit, phi)
        res[name] = dict(agg, phantom_kwh=float(np.maximum(B - Ph, 0)[:, M].sum()),
                         honest_trimmed_kwh=float(np.maximum(Ph - B, 0)[:, [i for i in range(17) if i not in M]].sum()))
        print(name, res[name], flush=True)
    honest_nodes = [i for i in range(17) if i not in M]
    inc = {}
    for name in ["inflated", "inflated_physics", "inflated_trust_coupled"]:
        base = "honest_trust_coupled" if name == "inflated_trust_coupled" else "honest"
        for phi in (0.0, 1.0):
            gain = bills[(base, phi)][M].sum() - bills[(name, phi)][M].sum()       # >0: cheating pays
            harm = bills[(name, phi)][honest_nodes].sum() - bills[(base, phi)][honest_nodes].sum()
            inc[f"{name}_phi{phi}"] = dict(attacker_gain_usd=float(gain), honest_extra_cost_usd=float(harm))
    out.update(bids=res, incentives=inc, flagged_nodes=np.where(flagged)[0].tolist(), malicious=M,
               honest_bill_total={f"{k[0]}_phi{k[1]}": float(v[honest_nodes].sum()) for k, v in bills.items()},
               attacker_bill_total={f"{k[0]}_phi{k[1]}": float(v[M].sum()) for k, v in bills.items()})
    print(json.dumps(inc, indent=1))
    (RES / "market2.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
