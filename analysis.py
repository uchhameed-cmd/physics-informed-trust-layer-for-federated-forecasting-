"""Aggregate result JSONs into the tables of the paper (LaTeX) and a summary JSON."""
import glob, json
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path(__file__).parent / "results"
TAB = Path(__file__).parent / "paper" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
METHODS = ["fedavg", "median", "trimmed", "krum", "trust"]
MNAME = {"fedavg": "FedAvg", "median": "Median", "trimmed": "Trimmed mean", "krum": "Multi-Krum",
         "trust": "\\textbf{Q-TRUST}"}
ATTACKS = ["none", "signflip", "gauss", "alie", "pvinflate"]
ANAME = {"none": "No attack", "signflip": "Sign flip", "gauss": "Gaussian", "alie": "ALIE", "pvinflate": "PV inflation"}
ABL = ["trust", "trust-no_geom", "trust-no_temporal", "trust-no_probe", "trust-no_meter", "trust-no_clip",
       "trust-no_gate", "trust-no_smooth"]
ABLNAME = {"trust": "Full Q-TRUST", "trust-no_geom": "without geometry", "trust-no_temporal": "without temporal",
           "trust-no_probe": "without probe residual", "trust-no_meter": "without PV coherence",
           "trust-no_clip": "without clipping", "trust-no_gate": "without gate", "trust-no_smooth": "without smoothing"}


def load():
    R = defaultdict(list)
    for f in glob.glob(str(RES / "fl" / "*_o*.json")):
        o = json.load(open(f))
        R[(o["method"], o["attack"], o["n_mal"] if o["attack"] != "none" else 4, o["onset"])].append(o)
    return R


def clean_on(R, method, seed, honest):
    """RMSE of the attack-free run of the same method and seed, on the given honest homes."""
    for r in R[(method, "none", 4, 10)]:
        if r["seed"] == seed:
            pc = np.array(r["final"]["rmse_per_client"])
            return float(pc[honest].mean())
    return float("nan")


def honest_of(r):
    return [i for i in range(17) if i not in r["malicious"]]


def ms(xs, nd=3):
    return f"{np.mean(xs):.{nd}f}$\\pm${np.std(xs):.{nd}f}"


def main():
    R = load()
    S = {}
    # reference forecasts
    S["naive"] = json.load(open(RES / "fl" / "naive_baselines.json"))
    cen = [json.load(open(f))["final"] for f in sorted(glob.glob(str(RES / "fl" / "central_s*.json")))]
    S["central_rmse"] = [c["rmse"] for c in cen]
    # ---------------- main table: RMSE and PV bias
    rows = []
    S["main"] = {}
    for m in METHODS:
        cells = []
        ref = {r["seed"]: honest_of(r) for r in R[(m, "signflip", 4, 10)]}   # honest homes per seed
        for a in ATTACKS:
            runs = sorted(R[(m, a, 4, 10)], key=lambda r: r["seed"])
            if a == "none":    # evaluate the clean run on the same 13 honest homes as the attacked runs
                rm = [clean_on(R, m, r["seed"], ref[r["seed"]]) for r in runs]
                wr = [float(np.max(np.array(r["final"]["rmse_per_client"])[ref[r["seed"]]])) for r in runs]
            else:
                rm = [r["final"]["rmse"] for r in runs]
                wr = [r["final"]["worst_rmse"] for r in runs]
            pb = [r["final"]["pv_bias"] for r in runs]
            S["main"][f"{m}_{a}"] = dict(rmse=rm, worst=wr, pv_bias=pb, n=len(runs),
                                         det=[r.get("detection") for r in runs])
            cells.append(ms(rm) if rm else "--")
        rows.append(MNAME[m] + " & " + " & ".join(cells) + r" \\")
    tex = [r"\begin{tabular}{@{}l" + "c" * len(ATTACKS) + "@{}}", r"\toprule",
           "Method & " + " & ".join(ANAME[a] for a in ATTACKS) + r" \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    (TAB / "main_rmse.tex").write_text("\n".join(tex))
    # worst-home and PV bias table
    rows = []
    for m in METHODS:
        c = []
        for a in ATTACKS:
            d = S["main"][f"{m}_{a}"]
            c.append(f"{np.mean(d['worst']):.2f}")
        pb = S["main"][f"{m}_pvinflate"]["pv_bias"]; pb0 = S["main"][f"{m}_none"]["pv_bias"]
        rows.append(MNAME[m] + " & " + " & ".join(c) + f" & {np.mean(pb0):+.3f} & {np.mean(pb):+.3f}" + r" \\")
    tex = [r"\begin{tabular}{@{}l" + "c" * len(ATTACKS) + "cc@{}}", r"\toprule",
           r"& \multicolumn{5}{c}{Worst honest home RMSE (kW)} & \multicolumn{2}{c}{PV bias (kW)}\\",
           r"\cmidrule(lr){2-6}\cmidrule(lr){7-8}",
           "Method & " + " & ".join(ANAME[a] for a in ATTACKS) + r" & clean & PV infl. \\", r"\midrule", *rows,
           r"\bottomrule", r"\end{tabular}"]
    (TAB / "worst_bias.tex").write_text("\n".join(tex))
    # ---------------- detection table (Q-TRUST and multi-Krum)
    rows = []
    S["detection"] = {}
    for m in ["trust", "krum"]:
        for a in ATTACKS:
            dets = [d for d in S["main"][f"{m}_{a}"]["det"] if d]
            if not dets:
                continue
            tp = sum(d["tp"] for d in dets); fp = sum(d["fp"] for d in dets)
            fn = sum(d["fn"] for d in dets); tn = sum(d["tn"] for d in dets)
            prec = tp / (tp + fp) if (tp + fp) and a != "none" else float("nan")
            rec = tp / (tp + fn) if tp + fn else float("nan")
            fpr = fp / (fp + tn)
            S["detection"][f"{m}_{a}"] = dict(precision=prec, recall=rec, fpr=fpr, tp=tp, fp=fp, fn=fn, tn=tn)
            f = lambda x: "--" if x != x else f"{x:.2f}"
            rows.append(f"{MNAME[m]} & {ANAME[a]} & {f(prec)} & {f(rec)} & {fpr:.3f}" + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    tex = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule", r"Method & Attack & Precision & Recall & FPR \\",
           r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    (TAB / "detection.tex").write_text("\n".join(tex))
    # per-home clean flag rates
    S["clean_flag_rate"] = {m: np.mean([r["flag_rate_per_client"] for r in R[(m, "none", 4, 10)]], 0).tolist()
                            for m in ["trust", "krum"]}
    # ---------------- ablations
    rows = []
    S["ablation"] = {}
    for v in ABL:
        cells = []
        th = json.load(open(RES / "fl" / f"theta_{v}.json"))["theta"]
        for a in ATTACKS:
            runs = R[(v, a, 4, 10)]
            rm = [r["final"]["rmse"] for r in runs]
            dets = [r["detection"] for r in runs]
            tp = sum(d["tp"] for d in dets); fp = sum(d["fp"] for d in dets)
            fn = sum(d["fn"] for d in dets); tn = sum(d["tn"] for d in dets)
            rec = tp / (tp + fn) if tp + fn else float("nan"); fpr = fp / (fp + tn) if fp + tn else float("nan")
            S["ablation"][f"{v}_{a}"] = dict(rmse=rm, recall=rec, fpr=fpr)
            if a == "none":
                cells.append(f"{np.mean(rm):.3f} / {fpr:.3f}")
            else:
                cells.append(f"{np.mean(rm):.3f} / {rec:.2f}")
        rows.append(f"{ABLNAME[v]} ($\\theta$={th:.2f}) & " + " & ".join(cells) + r" \\")
    tex = [r"\begin{tabular}{@{}l" + "c" * len(ATTACKS) + "@{}}", r"\toprule",
           "Variant & " + " & ".join(ANAME[a] for a in ATTACKS) + r" \\", r"\midrule", *rows, r"\bottomrule",
           r"\end{tabular}"]
    (TAB / "ablation.tex").write_text("\n".join(tex))
    # ---------------- fraction sweep
    S["fraction"] = {}
    for m in METHODS:
        for a in ["signflip", "alie", "pvinflate"]:
            for k in [2, 4, 6]:
                runs = R[(m, a, k, 10)]
                D_ = [r.get("detection") for r in runs if r.get("detection")]
                rec_ = (sum(d["tp"] for d in D_) / max(1, sum(d["tp"] + d["fn"] for d in D_))) if D_ else None
                fpr_ = (sum(d["fp"] for d in D_) / max(1, sum(d["fp"] + d["tn"] for d in D_))) if D_ else None
                S["fraction"][f"{m}_{a}_{k}"] = dict(recall=rec_, fpr=fpr_,
                    rmse=[r["final"]["rmse"] for r in runs],
                    rel=[r["final"]["rmse"] / clean_on(R, m, r["seed"], honest_of(r)) for r in runs],
                    pv_bias=[r["final"]["pv_bias"] for r in runs])
    # ---------------- onset at enrolment
    rows = []
    S["onset0"] = {}
    for m in ["median", "trust"]:
        cells = []
        for a in ["signflip", "alie", "pvinflate"]:
            r0 = [r["final"]["rmse"] for r in R[(m, a, 4, 0)]]
            r10 = [r["final"]["rmse"] for r in R[(m, a, 4, 10)]]
            b0 = [r["final"]["pv_bias"] for r in R[(m, a, 4, 0)]]
            d0 = [r.get("detection") for r in R[(m, a, 4, 0)]]
            S["onset0"][f"{m}_{a}"] = dict(rmse0=r0, rmse10=r10, pv_bias0=b0, det0=d0)
            cells.append(f"{np.mean(r10):.3f} & {np.mean(r0):.3f}")
        rows.append(MNAME[m] + " & " + " & ".join(cells) + r" \\")
    tex = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
           r"& \multicolumn{2}{c}{Sign flip} & \multicolumn{2}{c}{ALIE} & \multicolumn{2}{c}{PV inflation}\\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
           r"Method & $t_0{=}10$ & $t_0{=}0$ & $t_0{=}10$ & $t_0{=}0$ & $t_0{=}10$ & $t_0{=}0$ \\", r"\midrule",
           *rows, r"\bottomrule", r"\end{tabular}"]
    (TAB / "onset.tex").write_text("\n".join(tex))
    # ---------------- market tables
    M = json.load(open(RES / "market.json"))
    S["market"] = {k: {a: b for a, b in v.items() if a != "per_node_imbalance"} for k, v in M.items() if k != "bid_meta"}
    S["bid_meta"] = M["bid_meta"]
    rows = [r"Perfect forecasts & " + f"{M['oracle_aware']['cost']/1e3:.1f} & {M['oracle_aware']['violations']} & -- & -- & -- & --" + r" \\",
            r"\quad no voltage limits & " + f"{M['oracle_unaware']['cost']/1e3:.1f} & {M['oracle_unaware']['violations']} & -- & -- & -- & --" + r" \\",
            r"\midrule"]
    for m in METHODS:
        c = M[f"{m}_none"]; sf = M[f"{m}_signflip"]; pv = M[f"{m}_pvinflate"]
        rows.append(f"{MNAME[m]} & {c['cost']/1e3:.1f} & {c['violations']} & {c['imbalance']/1e3:.0f} & {sf['cost']/1e3:.1f} & {sf['violations']} & {pv['violations']}" + r" \\")
    tex = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
           r"& \multicolumn{3}{c}{No attack} & \multicolumn{2}{c}{Sign flip} & PV infl.\\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}",
           r"Forecasts from & Cost & Viol. & Imb. & Cost & Viol. & Viol.\\", r"\midrule", *rows,
           r"\midrule",
           r"\multicolumn{7}{@{}l}{\emph{Bid inflation by four nodes (Q-TRUST forecasts)}}\\",
           f"Honest bids & {M['bid_honest']['cost']/1e3:.1f} & {M['bid_honest']['violations']} & {M['bid_honest']['imbalance']/1e3:.0f} & & & " + r"\\",
           f"Inflated, unchecked & {M['bid_inflated']['cost']/1e3:.1f} & {M['bid_inflated']['violations']} & {M['bid_inflated']['imbalance']/1e3:.0f} & & & " + r"\\",
           f"Inflated, physics check & {M['bid_inflated_verified']['cost']/1e3:.1f} & {M['bid_inflated_verified']['violations']} & {M['bid_inflated_verified']['imbalance']/1e3:.0f} & & & " + r"\\",
           r"\bottomrule", r"\end{tabular}"]
    (TAB / "market.tex").write_text("\n".join(tex))
    B = json.load(open(RES / "bidcheck_tolerance.json"))
    S["bidtol"] = B
    rows = [f"{float(k):.2f} & {100*v['phantom_left']:.0f}\\% & {100*v['honest_clipped']:.1f}\\%" + r" \\" for k, v in B.items()]
    tex = [r"\begin{tabular}{@{}ccc@{}}", r"\toprule", r"$\epsilon_q$ & Phantom energy left & Honest PV trimmed\\",
           r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    (TAB / "bidtol.tex").write_text("\n".join(tex))
    O = json.load(open(RES / "overhead.json"))
    S["overhead"] = O
    p_, c_ = O["pqc"], O["classical"]
    rows = [f"ML-KEM-768 keygen / encaps / decaps & {p_['ml_kem_768_keygen_ms'][0]:.2f} / {p_['ml_kem_768_encaps_ms'][0]:.2f} / {p_['ml_kem_768_decaps_ms'][0]:.2f} ms" + r" \\",
            f"X25519 key exchange & {c_['x25519_exchange_ms'][0]:.2f} ms" + r" \\",
            f"ML-DSA-44 sign / verify & {p_['ml_dsa_44_sign_ms'][0]:.2f} / {p_['ml_dsa_44_verify_ms'][0]:.2f} ms" + r" \\",
            f"ECDSA P-256 sign / verify & {c_['ecdsa_p256_sign_ms'][0]:.2f} / {c_['ecdsa_p256_verify_ms'][0]:.2f} ms" + r" \\",
            f"AES-256-GCM, 104 kB update (enc / dec) & {p_['aes256gcm_encrypt_update_ms'][0]:.3f} / {p_['aes256gcm_decrypt_update_ms'][0]:.3f} ms" + r" \\",
            f"SHA3-256 of update & {p_['sha3_update_ms'][0]:.2f} ms" + r" \\",
            r"\midrule",
            f"Bytes per update, PQC / classical & {O['per_update_overhead_bytes_pqc']:,} / {O['per_update_overhead_bytes_classical']} B" + r" \\",
            f"Bytes per session, ML-KEM-768 & {O['per_session_setup_bytes_pqc']:,} B" + r" \\",
            f"Replays / forgeries rejected & {O['freshness_test']['replays_rejected']}/{O['freshness_test']['replays_sent']}, {O['freshness_test']['tampered_rejected']}/{O['freshness_test']['tampered_sent']}" + r" \\",
            r"\midrule",
            f"Trust evidence + aggregation per round & {json.load(open(RES / 'aggtime.json'))['median_s']*1e3:.0f} ms" + r" \\",
            f"Ledger block CPU, 4 validators (median / p95) & {O['ledger']['cpu_ms_median']:.1f} / {O['ledger']['cpu_ms_p95']:.1f} ms" + r" \\"]
    tex = [r"\begin{tabular}{@{}p{4.6cm}r@{}}", r"\toprule", r"Operation & Cost\\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    (TAB / "overhead.tex").write_text("\n".join(tex))
    (RES / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print("tables written; runs per main cell:", {k: v["n"] for k, v in S["main"].items()})


if __name__ == "__main__":
    main()
