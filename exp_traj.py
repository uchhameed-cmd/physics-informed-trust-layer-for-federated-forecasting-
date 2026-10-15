"""Score trajectories of one illustrative run per attack (seed 0) -> results/trajectories.json"""
import json
from pathlib import Path
import torch
from qtrust import fl
torch.set_num_threads(1)
RES = Path(__file__).parent / "results"
fed = fl.build()
th = json.load(open(RES / "fl" / "theta_trust.json"))["theta"]
out = {}
for a in ["signflip", "alie", "pvinflate"]:
    cfg = fl.TrustCfg(theta=th)
    o = fl.run(fed, "trust", a, 4, 0, cfg=cfg, keep_model=True, log_every=40)
    out[a] = dict(malicious=o["malicious"], theta=th, components=[{k: c[k] for k in ("s", "q", "tau")} for c in o["components"]])
(RES / "trajectories.json").write_text(json.dumps(out))
print("ok")
