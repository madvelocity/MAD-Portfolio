#!/usr/bin/env python3
"""Regression test for the phantom ticker-reuse bug (the SPLS pattern):
a held name stops trading (acquisition), its symbol is reused months later at a
higher price inside the 5x splice band. The fixed pipeline must close the
position at its last real close shortly after trading stops -- no phantom P&L."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DIRP = os.path.join(HERE, "..")
WORK = os.path.join(HERE, "smoke_work_phantom")
DATA = os.path.join(WORK, "data")
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(DATA)
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


jan = pd.bdate_range("2016-01-04", periods=20)          # era A: real company
may = pd.bdate_range("2016-05-02", periods=10)          # era B: reused symbol, 4x
full = pd.bdate_range("2016-01-04", "2016-05-13")       # control name trades daily


def rows(dates, opens, closes, vol, cross=(), rollb=(), member=True):
    return [{"Date": str(d.date()), "Open": opens[i], "Close": closes[i],
             "Volume": vol, "mad": .3, "z": .6, "price_share_5": .8,
             "sig_cross_up": i in cross, "sig_rollback": i in rollb,
             "in_index": member} for i, d in enumerate(dates)]


# PHAN: cross at Jan bar 1 -> entry Jan bar 2 open @20; NO rollback ever; last
# era-A close 21.0; era B reopens at 84 (4x -- inside the old 5x blind spot)
phan = (rows(jan, [20.0] * 20, [20.0] * 2 + [20.5] * 17 + [21.0], 5000, cross=[1])
        + rows(may, [84.0] * 10, [84.0] * 10, 5000))
pd.DataFrame(phan).to_csv(os.path.join(DATA, "PHAN.csv"), index=False)
# AAA: trades every day, no signals (keeps the master calendar continuous)
n = len(full)
pd.DataFrame(rows(full, [50.0] * n, [50.0] * n, 9000)).to_csv(
    os.path.join(DATA, "AAA.csv"), index=False)
for sym, px in [("SPY", 100.0), ("RSP", 80.0)]:
    pd.DataFrame(rows(full, [px] * n, [px] * n, 1_000_000, member=False)).to_csv(
        os.path.join(DATA, f"{sym}.csv"), index=False)

env = {**os.environ, "PORTFOLIO_DATADIR": DATA, "SCORE_MIN_TRAIN": "3",
       "SCORE_N_RAND": "2", "SIM_SEEDS": "2", "SIM_TOP_Q": "100",
       "SIM_END": "2016-05-13"}
for script in ["02_ranking.py", "03_score.py", "04_sim.py"]:
    r = subprocess.run([sys.executable, os.path.join(DIRP, script)], cwd=WORK,
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:])
        print(r.stderr[-2500:])
        sys.exit(f"{script} FAILED")
    out = r.stdout

# loader flagged the era break
check("loader logs the era break as a broken bar",
      "SPLICE BARS BROKEN" in out and "PHAN" in out)

# 02 ledger: outcome excludes the 4x repricing, resolves at the era-A end
sc = pd.read_csv(os.path.join(WORK, "results", "03_scores.csv"))
p = sc[sc.ticker == "PHAN"].iloc[0]
check("ledger: PHAN resolves at era-A close (+5%), not the 4x phantom (+320%)",
      p.exit_reason == "data_break" and abs(p.trade_ret - (21.0 / 20.0 - 1)) < 1e-9
      and p.exit_date == str(jan[-1].date()))

# 03 book: position freed ~10 sessions after trading stops, at the stale close
tr = pd.read_csv(os.path.join(WORK, "results", "04_sim_trades.csv"))
bt = tr[(tr.run == "RANKVOL_N10") & (tr.ticker == "PHAN")].iloc[0]
c = 0.0005
sh = 10000 / (20.0 * (1 + c))
expected_final = 100000 - 10000 + sh * 21.0 * (1 - c)
summ = pd.read_csv(os.path.join(WORK, "results", "04_sim_summary.csv")).set_index("run")
check("book: PHAN closed as 'delisted' at the era-A close",
      bt.reason == "delisted" and abs(bt.exit_px - 21.0) < 1e-9)
check("book: exit lands within ~11 sessions of the last real bar",
      pd.Timestamp(bt.exit_date) <= jan[-1] + pd.Timedelta(days=25))
check("book: final equity carries no phantom P&L (to the penny)",
      abs(summ.loc["RANKVOL_N10", "final_equity"] - expected_final) < 0.05)
d = pd.read_csv(os.path.join(WORK, "results", "04_sim_daily.csv"))
check("SIM_END respected: no daily rows after the cap",
      d[d.run == "RANKVOL_N10"].date.max() <= "2016-05-13")
big = d[(d.run == "RANKVOL_N10")].copy()
big["jump"] = big.equity.diff().abs()
check("no single-day equity jump at the era-B reopening",
      big.jump.max() < 500)

print(f"\n{'ALL PHANTOM CHECKS PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(1 if fails else 0)
