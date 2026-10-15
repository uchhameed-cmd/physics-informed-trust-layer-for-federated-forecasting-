"""Wall-clock cost of one Q-TRUST aggregation step (17 clients) -> results/aggtime.json"""
import json, time
from pathlib import Path
import numpy as np, torch
from qtrust import fl
from qtrust.model import Forecaster, get_vec
torch.set_num_threads(1)
fed = fl.build(); w = get_vec(Forecaster())
st = fl.TrustState(17); cfg = fl.TrustCfg(theta=5.25)
ts = []
for t in range(30):
    U = torch.randn(17, len(w)) * 1e-3
    rr = fed.peer_ratio_week[:, t % 34]
    t0 = time.perf_counter(); fl.agg_trust(U, w, fed, st, cfg, rr, t=t); ts.append(time.perf_counter() - t0)
out = dict(median_s=float(np.median(ts[5:])), p95_s=float(np.quantile(ts[5:], 0.95)))
Path("results/aggtime.json").write_text(json.dumps(out)); print(out)
