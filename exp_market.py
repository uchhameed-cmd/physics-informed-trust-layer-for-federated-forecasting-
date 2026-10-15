"""Grid-constrained day-ahead P2P clearing on the IEEE 33-bus feeder, driven by the forecasts
of each federated model (seed 0), plus the bid-manipulation experiment.
Results -> results/market.json"""
import json
from pathlib import Path

import numpy as np
import torch

from qtrust import data as D, grid as G, fl
from qtrust.model import Forecaster, set_vec, predict

RES = Path(__file__).parent / "results"
N_HOMES_PER_NODE = 30
INFLATE = 1.6


def setup():
    h = D.load_homes()
    f = G.ieee33()
    shape = (h.load / h.load_scale[:, None]).mean(0)
    shape = np.convolve(shape, np.ones(3) / 3, "same")
    bg = shape / np.quantile(shape, 0.999)
    n = N_HOMES_PER_NODE
    com = G.Community(n, n * h.pv_kw, n * h.batt_kwh, n * h.batt_kw)
    days = D.split_origins(h.time, D.TEST_MONTHS, hour=0)
    return h, f, bg, com, days


def forecasts(fed, w):
    """Day-ahead forecasts (kW per node) for every test day from global model w."""
    h = fed.homes
    m = Forecaster(); set_vec(m, torch.tensor(np.asarray(w, np.float32)))
    L, P = [], []
    for i in range(D.N_HOMES):
        out = predict(m, torch.tensor(fed.Xte[i]))
        L.append(np.maximum(out[:, :D.H], 0) * h.load_scale[i] * N_HOMES_PER_NODE)
        P.append(np.maximum(out[:, D.H:], 0) * h.pv_kw[i] * N_HOMES_PER_NODE)
    return np.stack(L, 1), np.stack(P, 1)            # (days, N, 24)


def simulate(h, f, bg, com, days, Lhat, Phat, grid_aware=True, bid_P=None):
    """bid_P: PV forecasts as submitted in bids (may be manipulated); default = Phat."""
    bid_P = Phat if bid_P is None else bid_P
    agg = dict(cost=0.0, violations=0, vmin=9.0, vmax=0.0, imbalance=0.0, p2p=0.0, imports=0.0,
               curtailed=0.0, slack_days=0)
    per_node_dev = np.zeros(D.N_HOMES)
    for k, d0 in enumerate(days):
        sl = slice(d0, d0 + 24)
        L = N_HOMES_PER_NODE * h.load[:, sl]
        P = N_HOMES_PER_NODE * h.pv_pu[:, sl] * h.pv_kw[:, None]
        s = G.clear_day(f, com, Lhat[k], bid_P[k], h.price[sl], bg[sl], grid_aware=grid_aware)
        if s is not None and s["slack"].max() > 1e-6:
            agg["slack_days"] += 1
        r = G.realize(f, com, s, L, P, h.price[sl], bg[sl], bid_P[k])
        sched_net = Lhat[k] - bid_P[k] + (s["ch"] - s["dis"] + s["cur"] if s is not None else 0)
        dev = np.abs(r["net"] - sched_net)
        agg["imbalance"] += float(dev.sum()); per_node_dev += dev.sum(1)
        for key in ("cost", "p2p", "imports", "curtailed"):
            agg[key] += r[key]
        agg["violations"] += r["violations"]
        agg["vmin"] = min(agg["vmin"], r["vmin"]); agg["vmax"] = max(agg["vmax"], r["vmax"])
    agg["per_node_imbalance"] = per_node_dev.tolist()
    return agg


def verify_bids(fed, Phat_bid, Phat_ref_ratio, tol=0.25):
    """Physics check of PV bids: (i) clear-sky envelope x installed capacity; (ii) coherence with
    the neighbourhood: a node's PV per kW may not exceed exp(tol) x its own trusted ratio x the
    median PV-per-kW forecast of the other nodes."""
    h = fed.homes
    days = D.split_origins(h.time, D.TEST_MONTHS, hour=0)
    out = Phat_bid.copy()
    cap = N_HOMES_PER_NODE * h.pv_kw
    for k, d0 in enumerate(days):
        env = h.envelope[d0:d0 + 24]
        per_kw = Phat_bid[k] / cap[:, None]
        for i in range(D.N_HOMES):
            others = np.delete(per_kw, i, axis=0)
            peer = np.median(others, 0)
            lim = np.minimum(env, np.exp(tol) * Phat_ref_ratio[i] * peer + 0.02)
            out[k, i] = np.minimum(Phat_bid[k, i], lim * cap[i])
    return out


def main():
    h, f, bg, com, days = setup()
    fed = fl.build(h)
    res = {}
    # oracle (perfect forecasts)
    L_true = np.stack([N_HOMES_PER_NODE * h.load[:, d0:d0 + 24] for d0 in days])
    P_true = np.stack([N_HOMES_PER_NODE * h.pv_pu[:, d0:d0 + 24] * h.pv_kw[:, None] for d0 in days])
    res["oracle_aware"] = simulate(h, f, bg, com, days, L_true, P_true, True)
    res["oracle_unaware"] = simulate(h, f, bg, com, days, L_true, P_true, False)
    print("oracle", {k: v for k, v in res["oracle_aware"].items() if k != "per_node_imbalance"}, flush=True)
    for meth in ["fedavg", "median", "trimmed", "krum", "trust"]:
        for att in ["none", "signflip", "gauss", "alie", "pvinflate"]:
            p = RES / "fl" / f"{meth}_{att}_m4_s0_o10.json"
            o = json.loads(p.read_text())
            Lh, Ph = forecasts(fed, o["w"])
            res[f"{meth}_{att}"] = simulate(h, f, bg, com, days, Lh, Ph, True)
            print(meth, att, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[f"{meth}_{att}"].items()
                              if k != "per_node_imbalance"}, flush=True)
            if meth == "trust" and att == "none":
                res["trust_none_unaware"] = simulate(h, f, bg, com, days, Lh, Ph, False)
                clean = (Lh, Ph)
    # bid manipulation: four nodes inflate their PV bids (forecasts from the clean trust model)
    o = json.loads((RES / "fl" / "trust_signflip_m4_s0_o10.json").read_text())
    mal = o["malicious"]
    Lh, Ph = clean
    bid = Ph.copy(); bid[:, mal] *= INFLATE
    ref_ratio = np.median(fed.peer_ratio_week, axis=1)          # trusted historical PV-per-kW ratios
    res["bid_honest"] = simulate(h, f, bg, com, days, Lh, Ph, True)
    res["bid_inflated"] = simulate(h, f, bg, com, days, Lh, Ph, True, bid_P=bid)
    ver = verify_bids(fed, bid, ref_ratio)
    res["bid_inflated_verified"] = simulate(h, f, bg, com, days, Lh, Ph, True, bid_P=ver)
    ver_h = verify_bids(fed, Ph, ref_ratio)
    res["bid_honest_verified"] = simulate(h, f, bg, com, days, Lh, Ph, True, bid_P=ver_h)
    cap = N_HOMES_PER_NODE * h.pv_kw
    res["bid_meta"] = dict(malicious=mal, inflate=INFLATE,
                           phantom_kwh_unverified=float((bid - Ph)[:, mal].sum()),
                           phantom_kwh_verified=float(np.maximum(ver - Ph, 0)[:, mal].sum()),
                           honest_pv_clipped_kwh=float((Ph - ver_h).sum()),
                           honest_pv_total_kwh=float(Ph.sum()),
                           malicious_capped_fraction=float(((bid - ver)[:, mal] > 1e-6).mean()))
    for k in ["bid_honest", "bid_inflated", "bid_inflated_verified", "bid_honest_verified"]:
        print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in res[k].items() if a != "per_node_imbalance"})
    print(res["bid_meta"])
    (RES / "market.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
