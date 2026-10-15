"""Real household data (CityLearn Challenge 2022, Fontana, CA) and the physics envelope.

Data: 17 single-family homes, hourly, one year (rows start with the hour ending
24:00 on 31 July; Day Type 1 on 1 August matches Monday 1 August 2016).
Load  = 'Equipment Electric Power [kWh]' (non-shiftable electricity, kWh per hour = kW).
PV    = 'Solar Generation [W/kW]' (AC output per kW installed) x nominal PV kW (schema.json).
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

DATA = Path(__file__).resolve().parents[1] / "data" / "citylearn_challenge_2022_phase_all"
LAT, LON = 34.09, -117.44          # Fontana, California
N_HOMES = 17
H = 24                              # day-ahead horizon (h)
LAG = 24                            # recent history window (h)
WEEK = 168                          # weekly-profile window (h)

# chronological split by calendar month of the hour (data year Aug 2016 - Jul 2017)
TRAIN_MONTHS = [8, 9, 10, 11, 12, 1, 2, 3]
VAL_MONTHS = [4, 5]
TEST_MONTHS = [6, 7]


@dataclass
class Homes:
    time: pd.DatetimeIndex          # interval start, local standard time (UTC-8)
    load: np.ndarray                # (17, T) kW
    pv_pu: np.ndarray               # (17, T) kW per kW installed
    pv_kw: np.ndarray               # (17,) installed PV kW
    batt_kwh: np.ndarray
    batt_kw: np.ndarray
    batt_eff: np.ndarray
    price: np.ndarray               # (T,) $/kWh import tariff
    clearsky: np.ndarray            # (T,) clear-sky GHI / 1000 (kW/m2-equivalent)
    envelope: np.ndarray            # (T,) physics upper bound on PV per kW
    load_scale: np.ndarray          # (17,) training-period mean load (kW)
    env_k: float


def _clearsky(time: pd.DatetimeIndex) -> np.ndarray:
    loc = pvlib.location.Location(LAT, LON, tz="Etc/GMT+8", altitude=380)
    mid = time + pd.Timedelta(minutes=30)
    cs = loc.get_clearsky(mid.tz_localize("Etc/GMT+8"), model="ineichen")
    return (cs["ghi"].values / 1000.0).clip(0, None)


def load_homes(shift_h: int | None = None) -> Homes:
    schema = json.loads((DATA / "schema.json").read_text())
    loads, pvs, pvkw, bk, bp, be = [], [], [], [], [], []
    for i in range(1, N_HOMES + 1):
        df = pd.read_csv(DATA / f"Building_{i}.csv")
        loads.append(df["Equipment Electric Power [kWh]"].to_numpy(float))
        pvs.append(df["Solar Generation [W/kW]"].to_numpy(float) / 1000.0)
        b = schema["buildings"][f"Building_{i}"]
        pvkw.append(b["pv"]["attributes"]["nominal_power"])
        es = b["electrical_storage"]["attributes"]
        bk.append(es["capacity"]); bp.append(es["nominal_power"]); be.append(es["efficiency"])
    price = pd.read_csv(DATA / "pricing.csv").iloc[:, 0].to_numpy(float)
    T = len(price)
    time = pd.date_range("2016-07-31 23:00", periods=T, freq="h")
    load, pv = np.array(loads), np.array(pvs)
    cs = _clearsky(time)
    if shift_h is None:                       # align measured PV with solar geometry
        shift_h = best_shift(pv.mean(0), cs)
    cs = np.roll(cs, shift_h)
    months = time.month.values
    tr = np.isin(months, TRAIN_MONTHS)
    # envelope multiplier: smallest k such that 99.9% of training daytime readings of every home
    # satisfy pv <= k * clearsky + 0.02 (tilted arrays can exceed GHI/1000)
    # +-1 h tolerance for timestamp convention and array orientation
    cs_w = np.maximum(cs, np.maximum(np.roll(cs, 1), np.roll(cs, -1)))
    day = cs_w > 0.05
    ratio = (pv[:, tr & day] / cs_w[tr & day]).ravel()
    env_k = float(np.quantile(ratio, 0.999))
    envelope = env_k * cs_w + 0.02
    load_scale = load[:, tr].mean(1)
    return Homes(time, load, pv, np.array(pvkw, float), np.array(bk, float), np.array(bp, float),
                 np.array(be, float), price, cs, envelope, load_scale, env_k)


def best_shift(pv_mean: np.ndarray, cs: np.ndarray) -> int:
    best, arg = -1, 0
    for s in range(-3, 4):
        c = np.corrcoef(pv_mean, np.roll(cs, s))[0, 1]
        if c > best:
            best, arg = c, s
    return arg


def calendar(time: pd.DatetimeIndex) -> np.ndarray:
    h = time.hour.values; doy = time.dayofyear.values
    wk = (time.dayofweek.values >= 5).astype(float)
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                     np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), wk], 1)


def windows(load_pu: np.ndarray, pv_pu: np.ndarray, envelope: np.ndarray, cal: np.ndarray,
            origins: np.ndarray):
    """Inputs at origin t (first forecast hour): last 24 h of load and PV, next 24 h of the
    physics envelope, calendar of t, and the mean daily profile of load and PV over the
    previous 7 days.  Targets: next 24 h of load (pu) and PV (kW/kW)."""
    X, Y = [], []
    for t in origins:
        X.append(np.concatenate([load_pu[t - LAG:t], pv_pu[t - LAG:t], envelope[t:t + H], cal[t],
                                 load_pu[t - WEEK:t].reshape(7, 24).mean(0),
                                 pv_pu[t - WEEK:t].reshape(7, 24).mean(0)]))
        Y.append(np.concatenate([load_pu[t:t + H], pv_pu[t:t + H]]))
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def split_origins(time: pd.DatetimeIndex, months, step: int = 1, hour: int | None = None):
    T = len(time)
    m = time.month.values
    o = []
    for t in range(WEEK, T - H + 1, step):
        if hour is not None and time[t].hour != hour:
            continue
        if m[t] in months and m[t + H - 1] in months:
            o.append(t)
    return np.array(o)
