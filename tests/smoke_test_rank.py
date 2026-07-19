#!/usr/bin/env python3
"""Known-answer tests for 01_ranking.py: hand-checkable ranks in every family."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "02_ranking.py")
WORK = os.path.join(HERE, "smoke_work_rank")
DATA = os.path.join(WORK, "data")
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(DATA)
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


# 40 trading days in Jan-Feb 2016 (no warmup needed: ret windows self-warm)
days = pd.bdate_range("2016-01-04", periods=40)
D = [str(x.date()) for x in days]


def write(tk, closes, volume, cross_days=(), rollb_days=(), member=True):
    rows = [{"Date": D[i], "Open": closes[i], "Close": closes[i],
             "Volume": volume, "price_share_5": 0.8,
             "sig_cross_up": i in cross_days, "sig_rollback": i in rollb_days,
             "in_index": member} for i in range(40)]
    pd.DataFrame(rows).to_csv(os.path.join(DATA, f"{tk}.csv"), index=False)


# volumes: AAA 4000 > BBB 3000 > CCC 2000 > DDD 1000  (constant -> constant ranks)
aaa = [100 * 1.01 ** i for i in range(40)]
bbb = [100.0] * 40
ccc = [100 * 0.995 ** i for i in range(40)]
ddd = [50.0] * 40
write("AAA", aaa, 4000, cross_days=[1, 6, 11], rollb_days=[4, 9, 14])
write("BBB", bbb, 3000, cross_days=[1, 6, 11], rollb_days=[4, 9, 14])
write("CCC", ccc, 2000, cross_days=[1, 6], rollb_days=[4, 9])
write("DDD", ddd, 1000)
# GHOST: highest volume of all, full data, but NEVER an index member --
# must be invisible to every ranking on every day
write("GHOST", [200.0] * 40, 9000, member=False)

env = {**os.environ, "PORTFOLIO_DATADIR": DATA}
r = subprocess.run([sys.executable, SCRIPT], cwd=WORK, env=env,
                   capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-1500:])
    print(r.stderr[-3000:])
    sys.exit("02_ranking.py FAILED")

ranks = pd.read_csv(os.path.join(WORK, "results", "02_ranks.csv"))
ranks["Date"] = ranks["Date"].astype(str)


def rk(day, tk, col):
    row = ranks[(ranks.Date == D[day]) & (ranks.ticker == tk)]
    v = row.iloc[0][col] if len(row) == 1 else None
    return None if v != v else int(v)


check("volume ranks AAA=1 BBB=2 CCC=3 DDD=4 on day 0 and day 39",
      all(rk(d, tk, "rank_vol") == i + 1
          for d in (0, 39) for i, tk in enumerate(["AAA", "BBB", "CCC", "DDD"])))
check("rank_ret_5 blank before window fills (day 3)",
      rk(3, "AAA", "rank_ret_5") is None)
check("rank_ret_5 day 39: AAA(rising)=1, DDD/BBB(flat) tie=2, CCC(falling)=4",
      rk(39, "AAA", "rank_ret_5") == 1 and rk(39, "BBB", "rank_ret_5") == 2
      and rk(39, "DDD", "rank_ret_5") == 2 and rk(39, "CCC", "rank_ret_5") == 4)
check("rank_ret_80 blank everywhere (window never fills in 40 days)",
      ranks["rank_ret_80"].isna().all())
check("rank_s2p_3 blank before the 3rd exit is known (day 14)",
      rk(14, "AAA", "rank_s2p_3") is None)
check("rank_s2p_3 from day 15: AAA(winners)=1, BBB(losers)=2, CCC/DDD blank",
      rk(15, "AAA", "rank_s2p_3") == 1 and rk(15, "BBB", "rank_s2p_3") == 2
      and rk(39, "AAA", "rank_s2p_3") == 1 and rk(39, "BBB", "rank_s2p_3") == 2
      and rk(39, "CCC", "rank_s2p_3") is None
      and rk(39, "DDD", "rank_s2p_3") is None)
check("rank_s2p_5 blank for everyone (nobody has 5 completed signals)",
      ranks["rank_s2p_5"].isna().all())
check("one row per member-day (4 names x 40 days)", len(ranks) == 160)
check("non-member with top volume is invisible to all rankings (no leakage)",
      len(ranks[ranks.ticker == "GHOST"]) == 0
      and rk(0, "AAA", "rank_vol") == 1)
check("latest-day file matches the last day",
      len(pd.read_csv(os.path.join(WORK, "results", "02_ranks_latest.csv"))) == 4)

print(f"\n{'ALL RANKING CHECKS PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(1 if fails else 0)
