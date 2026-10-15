"""IEEE 33-bus feeder, LinDistFlow voltage model, AC backward/forward sweep, and the
grid-constrained day-ahead P2P clearing with community batteries."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandapower.networks as pn
import scipy.sparse as sp
from scipy.optimize import linprog

S_BASE_KVA = 1000.0
V_KV = 12.66
Z_BASE = V_KV ** 2 / (S_BASE_KVA / 1000.0)          # ohm
NODE_BUSES = [5, 8, 11, 13, 15, 16, 17, 19, 20, 21, 23, 24, 27, 28, 29, 31, 32]   # 0-based, 17 prosumer nodes
V_MIN, V_MAX = 0.95, 1.05
LIN_MARGIN = 0.002                                    # tighten the linear limits to absorb its error
V0 = 1.03                                             # substation set-point (OLTC), pu


@dataclass
class Feeder:
    n: int
    parent: np.ndarray
    r: np.ndarray        # pu, indexed by child bus
    x: np.ndarray
    p_nom: np.ndarray    # background active load kW per bus
    q_nom: np.ndarray    # background reactive load kvar per bus
    R: np.ndarray        # (n-1, n-1) common-path resistance, buses 1..n-1
    X: np.ndarray
    order: list          # buses in BFS order from the root


def ieee33() -> Feeder:
    net = pn.case33bw()
    n = len(net.bus)
    parent = -np.ones(n, int); r = np.zeros(n); x = np.zeros(n)
    for _, ln in net.line[net.line.in_service].iterrows():
        f, t = int(ln.from_bus), int(ln.to_bus)
        parent[t] = f
        r[t] = ln.r_ohm_per_km * ln.length_km / Z_BASE
        x[t] = ln.x_ohm_per_km * ln.length_km / Z_BASE
    p = np.zeros(n); q = np.zeros(n)
    for _, ld in net.load.iterrows():
        p[int(ld.bus)] += ld.p_mw * 1000; q[int(ld.bus)] += ld.q_mvar * 1000
    paths = []
    for j in range(n):
        s, k = set(), j
        while k > 0:
            s.add(k); k = parent[k]
        paths.append(s)
    R = np.zeros((n - 1, n - 1)); X = np.zeros((n - 1, n - 1))
    for a in range(1, n):
        for b in range(1, n):
            common = paths[a] & paths[b]
            R[a - 1, b - 1] = sum(r[k] for k in common)
            X[a - 1, b - 1] = sum(x[k] for k in common)
    order = [0]
    children = {k: [] for k in range(n)}
    for k in range(1, n):
        children[parent[k]].append(k)
    i = 0
    while i < len(order):
        order += children[order[i]]; i += 1
    return Feeder(n, parent, r, x, p, q, R, X, order)


def sweep(f: Feeder, p_kw: np.ndarray, q_kvar: np.ndarray, iters: int = 50, tol: float = 1e-9) -> np.ndarray:
    """AC backward/forward sweep. p, q: net load per bus (kW, kvar; negative = injection)."""
    s = (p_kw + 1j * q_kvar) / S_BASE_KVA
    V = np.full(f.n, V0, complex)
    z = f.r + 1j * f.x
    for _ in range(iters):
        I = np.conj(s / V)
        Ib = I.copy()
        for k in reversed(f.order[1:]):
            Ib[f.parent[k]] += Ib[k]
        Vn = V.copy(); Vn[0] = V0
        for k in f.order[1:]:
            Vn[k] = Vn[f.parent[k]] - z[k] * Ib[k]
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn; break
        V = Vn
    return np.abs(V)


def lin_voltage(f: Feeder, p_kw: np.ndarray, q_kvar: np.ndarray) -> np.ndarray:
    return V0 - (f.R @ p_kw[1:] + f.X @ q_kvar[1:]) / (S_BASE_KVA * V0)


def G_V0():
    return V0


# ----------------------------------------------------------------------------- market
@dataclass
class Community:
    n_homes: int                 # homes represented by each prosumer node
    pv_kw: np.ndarray            # per node, kW installed
    e_kwh: np.ndarray            # battery energy per node
    p_kw: np.ndarray             # battery power per node
    eta: float = 0.95            # one-way efficiency (round trip ~0.90)
    fit: float = 0.05            # export price $/kWh (assumption)
    wear: float = 0.01           # $/kWh battery throughput
    curt_cost: float = 0.05      # $/kWh curtailed PV (lost export value proxy)
    soc0: float = 0.5


VIOL_COST = 5.0e4                                     # $ per pu of voltage-limit slack


def clear_day(f: Feeder, com: Community, load_hat: np.ndarray, pv_hat: np.ndarray, price: np.ndarray,
              bg_scale: np.ndarray, grid_aware: bool = True, iters: int = 3, e0=None, e_term=None):
    """Grid-aware clearing with AC correction: solve the LinDistFlow LP, evaluate the planned
    schedule with the AC sweep, shift the linear voltages by the AC-linear error, re-solve."""
    corr = np.zeros((f.n - 1, load_hat.shape[1]))
    out = None
    for _ in range(iters if grid_aware else 1):
        out = _clear(f, com, load_hat, pv_hat, price, bg_scale, grid_aware, corr, e0, e_term)
        if not grid_aware:
            break
        nodes = np.array(NODE_BUSES)
        worst = 0.0
        for t in range(load_hat.shape[1]):
            p = f.p_nom * bg_scale[t]; q = f.q_nom * bg_scale[t]
            p = p.copy(); p[nodes] += load_hat[:, t] - pv_hat[:, t] + out["ch"][:, t] - out["dis"][:, t] + out["cur"][:, t]
            vac = sweep(f, p, q)[1:]; vl = lin_voltage(f, p, q)
            corr[:, t] = vac - vl
            worst = max(worst, np.max(vac - V_MAX), np.max(V_MIN - vac))
        if worst <= 1e-4:
            break
    return out


def _clear(f, com, load_hat, pv_hat, price, bg_scale, grid_aware, corr, e0=None, e_term=None):
    e0 = com.soc0 * com.e_kwh if e0 is None else np.asarray(e0)
    e_term = com.soc0 * com.e_kwh if e_term is None else np.asarray(e_term)
    """Day-ahead clearing for 24 h.  load_hat, pv_hat: (N, 24) kW per node.
    Returns schedule dict.  Variables per node and hour: ch, dis, soc, cur; plus community
    grid import/export per hour."""
    N, T = load_hat.shape
    nv_node = 4
    idx = lambda v, i, t: (v * N + i) * T + t            # v: 0 ch, 1 dis, 2 soc, 3 cur
    base = nv_node * N * T
    gi = lambda t: base + t
    ge = lambda t: base + T + t
    sl = lambda t: base + 2 * T + t                      # voltage slack per hour
    nvar = base + 3 * T
    c = np.zeros(nvar)
    lb = np.zeros(nvar); ub = np.full(nvar, np.inf)
    for i in range(N):
        for t in range(T):
            c[idx(0, i, t)] = com.wear / 2; c[idx(1, i, t)] = com.wear / 2
            c[idx(3, i, t)] = com.curt_cost
            ub[idx(0, i, t)] = com.p_kw[i]; ub[idx(1, i, t)] = com.p_kw[i]
            lb[idx(2, i, t)] = 0.1 * com.e_kwh[i]; ub[idx(2, i, t)] = com.e_kwh[i]
            ub[idx(3, i, t)] = max(pv_hat[i, t], 0.0)
    for t in range(T):
        c[gi(t)] = price[t]; c[ge(t)] = -com.fit; c[sl(t)] = VIOL_COST
    rows, cols, vals, beq = [], [], [], []
    r = 0
    for i in range(N):                                   # SoC dynamics
        for t in range(T):
            rows += [r, r, r]; cols += [idx(2, i, t), idx(0, i, t), idx(1, i, t)]
            vals += [1.0, -com.eta, 1.0 / com.eta]
            if t > 0:
                rows.append(r); cols.append(idx(2, i, t - 1)); vals.append(-1.0); beq.append(0.0)
            else:
                beq.append(float(e0[i]))
            r += 1
    for t in range(T):                                    # community balance
        rows += [r, r]; cols += [gi(t), ge(t)]; vals += [1.0, -1.0]
        for i in range(N):
            rows += [r, r, r]; cols += [idx(0, i, t), idx(1, i, t), idx(3, i, t)]; vals += [-1.0, 1.0, -1.0]
        beq.append(float(np.sum(load_hat[:, t] - pv_hat[:, t])))
        r += 1
    Aeq = sp.csr_matrix((vals, (rows, cols)), shape=(r, nvar))
    # terminal SoC >= initial
    Ar, Ac, Av, bub = [], [], [], []
    k = 0
    for i in range(N):
        Ar.append(k); Ac.append(idx(2, i, T - 1)); Av.append(-1.0); bub.append(-float(e_term[i])); k += 1
    if grid_aware:                                       # LinDistFlow voltage limits at every bus
        nodes = np.array(NODE_BUSES)
        for t in range(T):
            p0 = f.p_nom * bg_scale[t]; q0 = f.q_nom * bg_scale[t]
            p_fix = p0.copy(); p_fix[nodes] += load_hat[:, t] - pv_hat[:, t]
            v_fix = lin_voltage(f, p_fix, q0) + corr[:, t]   # batteries idle, no curtailment, AC-corrected
            # dV_j/dP_node = -R[j, node]/S ; node net load += ch - dis + cur
            sens = -f.R[:, nodes - 1] / (S_BASE_KVA * G_V0())      # (n-1, N)
            for j in range(f.n - 1):
                coeffs = sens[j]
                # upper: v_fix + sens@(ch - dis + cur) <= Vmax - m
                for sgn, bound in ((1.0, V_MAX - LIN_MARGIN - v_fix[j]), (-1.0, -(V_MIN + LIN_MARGIN) + v_fix[j])):
                    for i in range(N):
                        a = sgn * coeffs[i]
                        Ar += [k, k, k]; Ac += [idx(0, i, t), idx(1, i, t), idx(3, i, t)]; Av += [a, -a, a]
                    Ar.append(k); Ac.append(sl(t)); Av.append(-1.0)
                    bub.append(bound); k += 1
    Aub = sp.csr_matrix((Av, (Ar, Ac)), shape=(k, nvar))
    res = linprog(c, A_ub=Aub, b_ub=np.array(bub), A_eq=Aeq, b_eq=np.array(beq),
                  bounds=list(zip(lb, ub)), method="highs")
    if res.status != 0:
        return None
    xv = res.x
    get = lambda v: np.array([[xv[idx(v, i, t)] for t in range(T)] for i in range(N)])
    return dict(ch=get(0), dis=get(1), soc=get(2), cur=get(3), cost=res.fun,
                slack=np.array([xv[sl(t)] for t in range(T)]))


def realize(f: Feeder, com: Community, sched: dict | None, load: np.ndarray, pv: np.ndarray,
            price: np.ndarray, bg_scale: np.ndarray, pv_hat: np.ndarray):
    """Real-time execution with actual load/PV.  Batteries follow the schedule within SoC
    limits; scheduled curtailment is applied as an export cap on the PV inverter."""
    N, T = load.shape
    soc = com.soc0 * com.e_kwh.copy()
    imp = np.zeros(T); exp_ = np.zeros(T); vmax = np.zeros(T); vmin = np.zeros(T)
    viol = 0; net_nodes = np.zeros((N, T)); thr = 0.0; curtailed = 0.0
    for t in range(T):
        ch = sched["ch"][:, t] if sched else np.zeros(N)
        dis = sched["dis"][:, t] if sched else np.zeros(N)
        cap = (pv_hat[:, t] - sched["cur"][:, t]) if sched else np.full(N, np.inf)
        cap = np.where((sched["cur"][:, t] if sched else np.zeros(N)) > 1e-6, np.maximum(cap, 0), np.inf)
        pv_used = np.minimum(pv[:, t], cap)
        curtailed += float(np.sum(pv[:, t] - pv_used))
        ch = np.minimum(ch, (com.e_kwh - soc) / com.eta)
        dis = np.minimum(dis, (soc - 0.1 * com.e_kwh) * com.eta)
        dis = np.maximum(dis, 0); ch = np.maximum(ch, 0)
        soc = soc + com.eta * ch - dis / com.eta
        thr += float(np.sum(ch + dis))
        n_net = load[:, t] - pv_used + ch - dis
        net_nodes[:, t] = n_net
        tot = n_net.sum()
        imp[t] = max(tot, 0); exp_[t] = max(-tot, 0)
        p = f.p_nom * bg_scale[t]; q = f.q_nom * bg_scale[t]
        p = p.copy(); p[NODE_BUSES] += n_net
        V = sweep(f, p, q)
        vmax[t], vmin[t] = V.max(), V.min()
        viol += int(np.sum((V > V_MAX + 1e-9) | (V < V_MIN - 1e-9)))
    cost = float(np.sum(imp * price) - com.fit * np.sum(exp_) + com.wear * thr)
    # P2P volume: energy exchanged between nodes inside the community each hour
    pos = np.maximum(net_nodes, 0).sum(0); neg = np.maximum(-net_nodes, 0).sum(0)
    p2p = float(np.sum(np.minimum(pos, neg)))
    return dict(cost=cost, imports=float(imp.sum()), exports=float(exp_.sum()), vmax=float(vmax.max()),
                vmin=float(vmin.min()), violations=viol, p2p=p2p, curtailed=curtailed, net=net_nodes,
                soc_end=soc)


def rolling_day(f: Feeder, com: Community, load_hat: np.ndarray, pv_hat: np.ndarray, load: np.ndarray,
                pv: np.ndarray, price: np.ndarray, bg_scale: np.ndarray, alpha: float = 0.8, iters: int = 2):
    """Intraday receding-horizon re-dispatch. At each hour h the forecasts of the remaining hours
    are corrected with the last observed error (decaying with alpha^k), the LP is re-solved for
    [h, 24) from the actual state of charge, and only hour h is executed with actual load/PV."""
    N, T = load.shape
    soc = com.soc0 * com.e_kwh.copy()
    e_term = com.soc0 * com.e_kwh
    imp = np.zeros(T); exp_ = np.zeros(T); thr = 0.0; viol = 0; vmin, vmax = 9.0, 0.0
    net = np.zeros((N, T)); sched_first = np.zeros((N, T))
    err_l = np.zeros(N); err_p = np.zeros(N)
    for h in range(T):
        k = np.arange(1, T - h + 1)
        Lh = np.maximum(load_hat[:, h:] + np.outer(err_l, alpha ** k), 0)
        Ph = np.maximum(pv_hat[:, h:] + np.outer(err_p, alpha ** k), 0)
        s = clear_day(f, com, Lh, Ph, price[h:], bg_scale[h:], True, iters=iters, e0=soc, e_term=e_term)
        ch = s["ch"][:, 0] if s else np.zeros(N)
        dis = s["dis"][:, 0] if s else np.zeros(N)
        cur = s["cur"][:, 0] if s else np.zeros(N)
        cap = np.where(cur > 1e-6, np.maximum(Ph[:, 0] - cur, 0), np.inf)
        pv_used = np.minimum(pv[:, h], cap)
        ch = np.maximum(np.minimum(ch, (com.e_kwh - soc) / com.eta), 0)
        dis = np.maximum(np.minimum(dis, (soc - 0.1 * com.e_kwh) * com.eta), 0)
        soc = soc + com.eta * ch - dis / com.eta
        thr += float(np.sum(ch + dis))
        n_net = load[:, h] - pv_used + ch - dis
        net[:, h] = n_net
        sched_first[:, h] = Lh[:, 0] - Ph[:, 0] + ch - dis + cur
        tot = n_net.sum(); imp[h] = max(tot, 0); exp_[h] = max(-tot, 0)
        p = f.p_nom * bg_scale[h]; q = f.q_nom * bg_scale[h]
        p = p.copy(); p[NODE_BUSES] += n_net
        V = sweep(f, p, q)
        vmin, vmax = min(vmin, V.min()), max(vmax, V.max())
        viol += int(np.sum((V > V_MAX + 1e-9) | (V < V_MIN - 1e-9)))
        err_l = load[:, h] - load_hat[:, h]
        err_p = pv[:, h] - pv_hat[:, h]
    cost = float(np.sum(imp * price) - com.fit * np.sum(exp_) + com.wear * thr)
    return dict(cost=cost, violations=viol, vmin=float(vmin), vmax=float(vmax), net=net,
                imports=float(imp.sum()), exports=float(exp_.sum()))
