#!/usr/bin/env python3
"""Full-pipeline known-answer test: 01_ranking -> 02_score -> 03_sim."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "04_sim.py")
WORK = os.path.join(HERE, "smoke_work_score2")
DATA = os.path.join(WORK, "data")
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


# stage 1+2: rebuild the 01/02 fixture pipeline
r = subprocess.run([sys.executable, os.path.join(HERE, "smoke_test_score2.py")],
                   capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-1200:])
    sys.exit("upstream pipeline smoke failed")

# index-control files (prepped-style so the loader tolerates them)
days = pd.bdate_range("2016-01-04", periods=60)
D = [str(x.date()) for x in days]
for sym, lo, hi in [("SPY", 100, 110), ("RSP", 80, 84)]:
    px = np.linspace(lo, hi, 60)
    pd.DataFrame([{"Date": D[i], "Open": px[i], "Close": px[i],
                   "Volume": 1_000_000, "mad": .1, "z": .2, "price_share_5": .5,
                   "sig_cross_up": False, "sig_rollback": False, "in_index": False}
                  for i in range(60)]).to_csv(os.path.join(DATA, f"{sym}.csv"),
                                              index=False)

env = {**os.environ, "PORTFOLIO_DATADIR": DATA, "SIM_SEEDS": "3",
       "SIM_TOP_Q": "1", "SIM_SPLIT": "2016-01-20", "SIM_END": "2016-12-31",
       "SCORE_MIN_TRAIN": "3", "SCORE_N_RAND": "3"}
r = subprocess.run([sys.executable, SIM], cwd=WORK, env=env,
                   capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-1500:])
    print(r.stderr[-3000:])
    sys.exit("04_sim.py FAILED")

summ = pd.read_csv(os.path.join(WORK, "results", "04_sim_summary.csv")).set_index("run")
tr = pd.read_csv(os.path.join(WORK, "results", "04_sim_trades.csv"))

c = 0.0005
check("SPY control to the penny",
      abs(summ.loc["SPY", "final_equity"]
          - (100000 / (100 * (1 + c))) * 110) < 0.02)
check("RSP control to the penny",
      abs(summ.loc["RSP", "final_equity"]
          - (100000 / (80 * (1 + c))) * 84) < 0.02)
check("all 18 runs present (SPY + RSP + 12 menu + 4 bench)",
      len(summ) == 18 and "RANKVOL_BENCH_N10" in summ.index
      and "SCORE_S_N10" in summ.index)
b = tr[tr.run == "RANKVOL_BENCH_N10"].sort_values("entry_date")
check("bench (TOP_Q=1) trades ONLY the top-volume name (AAA)",
      set(b.ticker) == {"AAA"} and len(b) >= 2)
check("bench empty before the first boundary: no January entries (cash held)",
      pd.to_datetime(b.entry_date).dt.month.min() >= 2)
nb = tr[tr.run == "RANKVOL_N10"]
check("unbenched book trades both signal names",
      set(nb.ticker) == {"AAA", "BBB"})

# hand-replicate the bench account from its own ledger (sequential AAA trades)
cash, mark = 100000.0, 0.0
for row in b.itertuples():
    alloc = min(cash / 10, cash)
    sh = alloc / (float(row.entry_px) * (1 + c))
    if row.reason == "open":
        cash -= alloc
        mark += sh * float(row.exit_px)            # marked at last close
    else:
        cash = cash - alloc + sh * float(row.exit_px) * (1 - c)
check("bench account final equity matches hand-replication to the penny",
      abs(summ.loc["RANKVOL_BENCH_N10", "final_equity"] - (cash + mark)) < 0.05)
check("split-half (oos_*) columns populated",
      np.isfinite(summ.loc["RANKVOL_N10", "oos_cagr"]))
check("placement columns present for menu runs",
      np.isfinite(summ.loc["RANKVOL_N10", "env_pctile"]))

print(f"\n{'ALL SIM CHECKS PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(1 if fails else 0)
