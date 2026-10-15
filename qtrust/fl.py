"""Federated training with attacks, robust baselines and the physics-informed trust rule."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.stats import norm

from . import data as D
from .model import Forecaster, get_vec, set_vec, local_train, predict

ROUNDS = 40
ONSET = 10                      # attacks start at this round (sleeper clients)
PV_INFLATE = 1.6                # falsification factor of the PV-inflation attacker
LAMBDA_ENV = 10.0               # weight of envelope violations in the probe loss


# ----------------------------------------------------------------------------- data
@dataclass
class Fed:
    homes: D.Homes
    Xtr: list                   # honest training tensors per client
    Ytr: list
    Xtr_bad: list               # falsified (PV x PV_INFLATE) training tensors per client
    Ytr_bad: list
    Xte: list                   # test (daily origins at 00:00)
    Yte: list
    Xpr: torch.Tensor           # DSO probe: aggregate feeder data, validation months
    Ypr: torch.Tensor
    env_pr: torch.Tensor        # physics envelope for the probe targets
    peer_ratio_week: np.ndarray  # (17, n_weeks) honest peer-consistency ratio per training week
    week_of_round: np.ndarray


def build(homes: D.Homes | None = None) -> Fed:
    h = homes or D.load_homes()
    cal = D.calendar(h.time)
    lp = h.load / h.load_scale[:, None]
    otr = D.split_origins(h.time, D.TRAIN_MONTHS)
    ote = D.split_origins(h.time, D.TEST_MONTHS, hour=0)
    opr = D.split_origins(h.time, D.VAL_MONTHS, step=6)
    Xtr, Ytr, Xb, Yb, Xte, Yte = [], [], [], [], [], []
    for i in range(D.N_HOMES):
        X, Y = D.windows(lp[i], h.pv_pu[i], h.envelope, cal, otr)
        Xtr.append(torch.tensor(X)); Ytr.append(torch.tensor(Y))
        X, Y = D.windows(lp[i], h.pv_pu[i] * PV_INFLATE, h.envelope, cal, otr)
        Xb.append(torch.tensor(X)); Yb.append(torch.tensor(Y))
        X, Y = D.windows(lp[i], h.pv_pu[i], h.envelope, cal, ote)
        Xte.append(X); Yte.append(Y)
    # DSO probe: transformer-level aggregate (sum of true power flows), per unit
    agg_load = h.load.sum(0) / h.load_scale.sum()
    agg_pv = (h.pv_pu * h.pv_kw[:, None]).sum(0) / h.pv_kw.sum()
    Xp, Yp = D.windows(agg_load, agg_pv, h.envelope, cal, opr)
    envp = np.stack([h.envelope[t:t + D.H] for t in opr]).astype(np.float32)
    # physics peer-consistency of reported PV (irradiance is common across the neighbourhood)
    tr_idx = np.where(np.isin(h.time.month.values, D.TRAIN_MONTHS))[0]
    weeks = tr_idx[: (len(tr_idx) // 168) * 168].reshape(-1, 168)
    peer = np.median(h.pv_pu, axis=0)
    ratio = np.zeros((D.N_HOMES, len(weeks)))
    for k, idx in enumerate(weeks):
        day = idx[peer[idx] > 0.15]
        ratio[:, k] = np.median(h.pv_pu[:, day] / peer[day], axis=1)
    wor = np.arange(ROUNDS) % len(weeks)
    return Fed(h, Xtr, Ytr, Xb, Yb, Xte, Yte, torch.tensor(Xp), torch.tensor(Yp),
               torch.tensor(envp), ratio, wor)


# ----------------------------------------------------------------------------- metrics
def probe_loss(model: Forecaster, fed: Fed) -> float:
    with torch.no_grad():
        P = model(fed.Xpr)
        mse = ((P - fed.Ypr) ** 2).mean()
        pv = P[:, D.H:]
        viol = (torch.relu(pv - fed.env_pr) + torch.relu(-pv)).mean()
    return float(mse + LAMBDA_ENV * viol)


def evaluate(model: Forecaster, fed: Fed, honest: list[int]) -> dict:
    h = fed.homes
    rm, ma, pvb, pvr = [], [], [], []
    for i in honest:
        P = predict(model, torch.tensor(fed.Xte[i])); Y = fed.Yte[i]
        nl_hat = P[:, :D.H] * h.load_scale[i] - P[:, D.H:] * h.pv_kw[i]
        nl = Y[:, :D.H] * h.load_scale[i] - Y[:, D.H:] * h.pv_kw[i]
        rm.append(float(np.sqrt(((nl_hat - nl) ** 2).mean())))
        ma.append(float(np.abs(nl_hat - nl).mean()))
        day = Y[:, D.H:] > 0.05
        pvb.append(float(((P[:, D.H:] - Y[:, D.H:]) * h.pv_kw[i])[day].mean()))
        pvr.append(float(np.sqrt((((P[:, D.H:] - Y[:, D.H:]) * h.pv_kw[i]) ** 2).mean())))
    return dict(rmse=float(np.mean(rm)), mae=float(np.mean(ma)), worst_rmse=float(np.max(rm)),
                pv_bias=float(np.mean(pvb)), pv_rmse=float(np.mean(pvr)), rmse_per_client=rm)


# ----------------------------------------------------------------------------- attacks
def alie_z(n: int, f: int) -> float:
    s = math.floor(n / 2 + 1) - f
    return float(norm.ppf((n - s) / n))


def craft(attack: str, U: torch.Tensor, mal: list[int], gen: torch.Generator) -> torch.Tensor:
    """Replace malicious rows of U (n x d) according to the attack."""
    if not mal or attack in ("none", "pvinflate"):
        return U
    U = U.clone()
    med_norm = U.norm(dim=1).median()
    if attack == "signflip":
        for i in mal:
            U[i] = -3.0 * U[i]
    elif attack == "gauss":
        for i in mal:
            g = torch.randn(U.shape[1], generator=gen)
            U[i] = g / g.norm() * 3.0 * med_norm
    elif attack == "alie":
        mu, sd = U.mean(0), U.std(0)
        z = alie_z(U.shape[0], len(mal))
        for i in mal:
            U[i] = mu - z * sd
    elif attack == "adaptive":
        # white-box, threshold-aware attacker: sends (1 - lam) * mu (mu = mean of the benign
        # updates), i.e. shrinks or reverses the honest step, with the largest lam for which
        # the server's cross-sectional geometry score of every malicious update stays below
        # ADAPT_MARGIN * theta (the probe and history channels are unknown to the attacker)
        mu = U.mean(0)
        best = None
        for lam in ADAPT_GRID:
            V = U.clone()
            for i in mal:
                V[i] = (1.0 - lam) * mu
            m = V.median(0).values
            nrm = V.norm(dim=1); medn = nrm.median()
            cos_m = torch.nn.functional.cosine_similarity(V, m.unsqueeze(0), dim=1).numpy()
            d = (1 - cos_m) + np.abs(np.log((nrm / (medn + 1e-12)).numpy() + 1e-12))
            z = np.maximum(robust_z(d), 0)
            if max(z[i] for i in mal) <= ADAPT_MARGIN * ADAPT_THETA:
                best = lam
        lam = 0.0 if best is None else best
        ADAPT_LOG.append(lam)
        for i in mal:
            U[i] = (1.0 - lam) * mu
    else:
        raise ValueError(attack)
    return U


ADAPT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
ADAPT_MARGIN = 0.5
ADAPT_THETA = 5.25
ADAPT_LOG: list = []


# ----------------------------------------------------------------------------- aggregators
def agg_mean(U, **_):
    return U.mean(0), None


def agg_median(U, **_):
    return U.median(0).values, None


def agg_trimmed(U, f, **_):
    k = max(1, f)
    S = U.sort(0).values
    return S[k:U.shape[0] - k].mean(0), None


def agg_krum(U, f, **_):
    n = U.shape[0]
    dist = torch.cdist(U, U) ** 2
    k = max(1, n - f - 2)
    score = dist.sort(1).values[:, 1:k + 1].sum(1)
    m = n - f                                     # multi-Krum: average the m best
    sel = score.argsort()[:m]
    flags = torch.ones(n, dtype=torch.bool); flags[sel] = False
    return U[sel].mean(0), flags.numpy()


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med)) + 1e-9
    return np.clip((x - med) / mad, -3, 6)


@dataclass
class TrustCfg:
    theta: float = float("inf")  # flag threshold on the smoothed score (calibrated, see calibrate())
    kappa: float = 1.0           # slope of the trust weight tau = sigmoid(kappa * (theta - s))
    smooth: float = 0.5          # EMA memory of the score (history evidence)
    use_geom: bool = True        # cross-sectional update geometry (cosine + norm ratio)
    use_temporal: bool = True    # inconsistency with the client's own update trajectory
    use_probe: bool = True       # physics evidence 1: DSO feeder-probe residual (own baseline)
    use_meter: bool = True       # physics evidence 2: PV coherence with neighbours
    clip: bool = True
    gate: float | None = 0.10    # reject the round if the probe loss rises by more than 10 %
    meter_tol: float = 0.25      # tolerated jump of the PV peer ratio (log scale)
    warmup: int = 5              # first rounds: everyone trusted, baselines start
    hist_window: int = 8


@dataclass
class TrustState:
    n: int
    ema: list = field(default_factory=list)
    s: np.ndarray = None
    meter_hist: list = None
    he: list = None
    last: dict = None

    def __post_init__(self):
        self.ema = [None] * self.n
        self.s = np.zeros(self.n)
        self.meter_hist = [[] for _ in range(self.n)]
        self.he = [[] for _ in range(self.n)]


def personal_z(x: np.ndarray, hist: list, window: int, min_len: int = 3, floor0: float = 0.0) -> np.ndarray:
    """Shift of each client's evidence from its OWN trusted history (fair to atypical homes)."""
    z = np.zeros(len(x))
    scales = []
    stats = []
    for hi in hist:
        if len(hi) >= min_len:
            hh = np.asarray(hi[-window:]); med = np.median(hh)
            mad = 1.4826 * np.median(np.abs(hh - med)); stats.append((med, mad)); scales.append(mad)
        else:
            stats.append(None)
    floor = max(0.25 * np.median(scales) if scales else 0.0, floor0)
    for i, st in enumerate(stats):
        if st is not None:
            z[i] = (x[i] - st[0]) / (max(st[1], floor) + 1e-9)
    return np.clip(z, -3, 10)


def trust_evidence(U, w, fed: Fed, st: TrustState, cfg: TrustCfg, reported_ratio: np.ndarray):
    n = U.shape[0]
    m = U.median(0).values
    norms = U.norm(dim=1); medn = norms.median()
    cos_m = torch.nn.functional.cosine_similarity(U, m.unsqueeze(0), dim=1).numpy()
    d = (1 - cos_m) + np.abs(np.log((norms / (medn + 1e-12)).numpy() + 1e-12))
    r = np.zeros(n)
    for i in range(n):
        if st.ema[i] is not None:
            r[i] = 1 - float(torch.nn.functional.cosine_similarity(U[i], st.ema[i], dim=0))
    tmp = Forecaster(); set_vec(tmp, w); base = probe_loss(tmp, fed)
    e = np.zeros(n)
    if cfg.use_probe:
        for i in range(n):
            set_vec(tmp, w + U[i]); e[i] = (probe_loss(tmp, fed) - base) / base
    lr = np.log(reported_ratio + 0.05)
    mb = np.array([np.median(b[-4:]) if b else x for b, x in zip(st.meter_hist, lr)])
    q = np.abs(lr - mb) / cfg.meter_tol
    score = np.zeros(n)
    if cfg.use_geom:
        score += np.maximum(robust_z(d), 0)
    if cfg.use_temporal:
        score += np.maximum(robust_z(r), 0)
    # remove the round-level drift (all residuals change as the model converges), then compare
    # each client with its own history; floor the scale at half the cross-sectional spread
    ec = e - np.median(e)
    mad_x = 1.4826 * np.median(np.abs(ec))
    pe = personal_z(ec, st.he, cfg.hist_window, floor0=0.5 * mad_x) if cfg.use_probe else np.zeros(n)
    score += np.maximum(pe, 0)
    meter_flag = (q > 1) if cfg.use_meter else np.zeros(n, bool)
    return dict(d=d, r=r, e=e, ec=ec, q=q, pe=pe, lr=lr, score=score, meter_flag=meter_flag, base=base, tmp=tmp,
                medn=medn)


def agg_trust(U, w, fed: Fed, st: TrustState, cfg: TrustCfg, reported_ratio: np.ndarray, t: int = 0, **_):
    n = U.shape[0]
    ev = trust_evidence(U, w, fed, st, cfg, reported_ratio)
    st.s = cfg.smooth * st.s + (1 - cfg.smooth) * ev["score"]
    if t < cfg.warmup:
        tau = np.ones(n); flag = np.zeros(n, bool)
    else:
        tau = 1 / (1 + np.exp(-cfg.kappa * (cfg.theta - st.s)))
        flag = (st.s > cfg.theta) | ev["meter_flag"]
        tau[ev["meter_flag"]] = 0.0
    elig = ~flag
    st.last = dict(score=ev["score"], s=st.s.copy(), d=ev["d"], r=ev["r"], e=ev["e"], pe=ev["pe"], q=ev["q"], tau=tau)
    if elig.sum() < max(2, n // 2):                 # too few trusted clients: fall back
        agg = U.median(0).values
    else:
        V = U[elig]
        if cfg.clip:
            sc = torch.clamp(ev["medn"] / (V.norm(dim=1) + 1e-12), max=1.0)
            V = V * sc.unsqueeze(1)
        wt = torch.tensor(tau[elig], dtype=V.dtype)
        agg = (wt.unsqueeze(1) * V).sum(0) / wt.sum()
    tmp = ev["tmp"]; set_vec(tmp, w + agg)
    if cfg.gate is not None and t >= cfg.warmup and probe_loss(tmp, fed) > (1 + cfg.gate) * ev["base"]:
        agg = torch.zeros_like(agg)                 # validation gate: keep the last model
    for i in range(n):
        st.ema[i] = U[i].clone() if st.ema[i] is None else 0.7 * st.ema[i] + 0.3 * U[i]
        if elig[i]:                                 # trusted rounds update the meter baseline
            st.meter_hist[i].append(ev["lr"][i])
        # physics-residual baseline: always updated, but winsorized so that an abrupt shift
        # enters the client's own baseline only slowly (re-admits honest clients after drift)
        hi = st.he[i]
        if len(hi) >= 3:
            hh = np.asarray(hi[-cfg.hist_window:]); med = np.median(hh)
            sc = max(1.4826 * np.median(np.abs(hh - med)), 1e-3)
            hi.append(float(np.clip(ev["ec"][i], med - 2 * sc, med + 2 * sc)))
        else:
            hi.append(float(ev["ec"][i]))
    return agg, flag


def calibrate(fed: "Fed", cfg: TrustCfg, seed: int = 100, rounds: int = ROUNDS, q: float = 0.95,
              return_scores: bool = False):
    """Attack-free calibration run in monitor mode: theta = q-quantile of honest smoothed scores."""
    mon = TrustCfg(**{**cfg.__dict__, "theta": float("inf"), "gate": None})
    o = run(fed, "trust", "none", 0, seed, rounds=rounds, cfg=mon, keep_model=True, log_every=rounds)
    S = np.array([c["s"] for c in o["components"][mon.warmup:]])
    if return_scores:
        return S
    return float(np.quantile(S, q))


# ----------------------------------------------------------------------------- one run
def run(fed: Fed, method: str, attack: str, n_mal: int, seed: int, rounds: int = ROUNDS,
        cfg: TrustCfg | None = None, keep_model: bool = False, log_every: int = 5,
        onset: int = ONSET) -> dict:
    rng = np.random.default_rng(seed)
    mal = sorted(rng.choice(D.N_HOMES, n_mal, replace=False).tolist()) if attack != "none" else []
    honest = [i for i in range(D.N_HOMES) if i not in mal]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    g = Forecaster(); w = get_vec(g)
    st = TrustState(D.N_HOMES)
    cfg = cfg or TrustCfg()
    f_assumed = max(n_mal, 1)
    hist, flags_log, comp_log = [], [], []
    ADAPT_LOG.clear()
    for t in range(rounds):
        active = t >= onset
        ups = []
        for i in range(D.N_HOMES):
            m = Forecaster(); set_vec(m, w)
            bad = active and attack == "pvinflate" and i in mal
            X, Y = (fed.Xtr_bad[i], fed.Ytr_bad[i]) if bad else (fed.Xtr[i], fed.Ytr[i])
            local_train(m, X, Y, gen=gen)
            ups.append(get_vec(m) - w)
        U = torch.stack(ups)
        if active:
            U = craft(attack, U, mal, gen)
        # reported PV meter data this round (falsified by PV-inflation attackers)
        rr = fed.peer_ratio_week[:, fed.week_of_round[t]].copy()
        if active and attack == "pvinflate":
            rr[mal] *= PV_INFLATE
        if method == "fedavg":
            agg, fl = agg_mean(U)
        elif method == "median":
            agg, fl = agg_median(U)
        elif method == "trimmed":
            agg, fl = agg_trimmed(U, f=f_assumed)
        elif method == "krum":
            agg, fl = agg_krum(U, f=f_assumed)
        elif method.startswith("trust"):
            agg, fl = agg_trust(U, w, fed, st, cfg, rr, t=t)
        else:
            raise ValueError(method)
        w = w + agg
        if fl is not None:
            flags_log.append((t, np.asarray(fl, bool).tolist()))
        if method.startswith("trust") and st.last is not None:
            comp_log.append({k: np.round(np.asarray(v, float), 3).tolist() for k, v in st.last.items()})
        if (t + 1) % log_every == 0 or t == rounds - 1:
            set_vec(g, w)
            ev = evaluate(g, fed, honest)
            hist.append(dict(round=t + 1, rmse=ev["rmse"], probe=probe_loss(g, fed)))
    set_vec(g, w)
    out = dict(method=method, attack=attack, n_mal=n_mal, seed=seed, malicious=mal, onset=onset,
               final=evaluate(g, fed, honest), history=hist)
    if flags_log:
        out["detection"] = detection(flags_log, mal, attack, onset)
        out["flag_rate_per_client"] = np.mean([f for t, f in flags_log if t >= max(onset, 1)], 0).tolist()
    if ADAPT_LOG:
        out["adapt_lambda"] = list(ADAPT_LOG)
    if comp_log and keep_model:
        out["components"] = comp_log
    if keep_model:
        out["w"] = w.numpy().tolist()
    return out


def detection(flags_log, mal, attack, onset=ONSET):
    tp = fp = fn = tn = 0
    for t, fl in flags_log:
        if t < onset:
            continue
        for i, f in enumerate(fl):
            bad = i in mal and attack != "none"
            tp += f and bad; fp += f and not bad; fn += (not f) and bad; tn += (not f) and not bad
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec,
                fpr=fp / (fp + tn) if fp + tn else float("nan"),
                f1=2 * prec * rec / (prec + rec) if tp else 0.0)
