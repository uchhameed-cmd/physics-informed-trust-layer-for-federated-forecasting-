# Q-TRUST: physics-informed trust for federated forecasting and grid-constrained P2P trading

Code, data pipeline and raw results behind the manuscript
*Physics-based trust evidence for federated energy communities: what it detects, what it costs,
and what it is worth in the market* (H. A. Muqeet, J. Ahmad, A. Hameed). Manuscript under review.
Every number, table and figure in the paper is produced by the scripts below from the files in `results/`.
`results/` contains the raw outputs of the runs reported in the paper, so all tables can be regenerated
with `python analysis.py && python stats.py` without re-running any experiment.

## Status of the evaluation

The paper states openly which claims its evidence supports and which it does not. Two points matter
for anyone reading this code:

* **Forecast accuracy does not separate the robust aggregation rules** on this forecaster. The
  coordinate-wise median matches Q-TRUST under every attack tested.
* **The baseline that would most plausibly be deployed is not yet in this repository**: median
  aggregation with the PV-coherence test run as a standalone meter-integrity monitor. See the
  paper's limitations section. Work in progress.

## Data
`data/citylearn_challenge_2022_phase_all/` – CityLearn Challenge 2022 data set (17 homes, Fontana, CA,
hourly, Aug 2016–Jul 2017; measured non-shiftable load and PV per kW; tariff; schema with PV/battery ratings).
Source: Nweye, Sankaranarayanan, Nagy, *The CityLearn Challenge 2022*, Texas Data Repository,
doi:10.18738/T8/0YLJ6Q. Please cite the data set if you reuse it.

## Layout
```
qtrust/data.py     loading, clear-sky envelope (pvlib), forecast windows, chronological splits
qtrust/model.py    residual MLP day-ahead forecaster (26,064 parameters), local training
qtrust/fl.py       FL loop, attacks (sign flip, Gaussian, ALIE, PV inflation), FedAvg/median/trimmed/
                   multi-Krum, Q-TRUST evidence, score, calibration, clipped aggregation, gate
qtrust/grid.py     IEEE 33-bus (pandapower data), LinDistFlow, AC backward/forward sweep,
                   grid-constrained day-ahead clearing (HiGHS), real-time settlement
exp_fl.py          FL experiment matrix: main (5 methods x 5 attacks x 3 seeds), 7 ablations,
                   attacker fractions 2/4/6, attacks from round 0      -> results/fl/*.json
exp_traj.py        score trajectories for Fig. 4                   -> results/trajectories.json
exp_market.py      market/feeder study and bid manipulation        -> results/market.json
exp_overhead.py    ML-KEM-768 / ML-DSA-44 / AES-GCM / ledger timings, replay test -> results/overhead.json
exp_aggtime.py     time of one Q-TRUST aggregation step            -> results/aggtime.json
exp_new.py         threshold-aware adaptive attack (margins 0.5/1.0) and calibration at
                   q=0.90/0.99                                      -> results/fl2/*.json
exp_coherence.py   detectability of PV falsification: step factors and slow creep,
                   rolling vs commissioning-anchored baseline       -> results/coherence.json
exp_market2.py     intraday receding-horizon re-dispatch, trust-coupled bid cap,
                   two-settlement incentives                        -> results/market2.json
stats.py           paired per-home tests with bootstrap CIs         -> results/stats.json, tables/stats.tex
analysis.py        tables (paper/tables/*.tex) and results/summary.json
figs.py, figs2.py  figures (paper/figures, paper_els/figures) and results/summary2.json
paper/             earlier IEEE-format version
paper_els/         Elsevier (elsarticle) version: main.tex, sections/, tables/, figures/
```

## Reproduce
```bash
pip install torch pandas numpy scipy pvlib pandapower matplotlib cryptography pqcrypto
export PYTHONPATH=.
python exp_fl.py all        # ~2.5 h on 2 CPU cores; skips runs whose JSON exists
python exp_traj.py
python exp_market.py
python exp_overhead.py && python exp_aggtime.py
python exp_new.py && python exp_coherence.py && python exp_market2.py
python analysis.py && python stats.py && python figs.py && python figs2.py
cd paper_els && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Settings: 40 rounds, attacks from round 10, seeds 0–2, Adam lr 1e-4, threshold calibrated on an
attack-free run with seed 100 (`results/fl/theta_*.json`). Timings depend on the machine.

## Notes for the authors
* The learning rate was chosen among 4 values by the attack-free FedAvg test error; this is disclosed in the paper.
* The market represents each prosumer node as 30 copies of one measured home (a stated limitation).
* The ledger is an in-process PBFT simulation with measured CPU cost and modelled network latency, not Hyperledger Fabric.
