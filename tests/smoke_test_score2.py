#!/usr/bin/env python3
"""Pipeline known-answer test: 01_ranking.py -> 02_score.py on one fixture."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RANK = os.path.join(HERE, "..", "02_ranking.py")
SCORE = os.path.join(HERE, "..", "03_score.py")
WORK = os.path.join(HERE, "smoke_work_score2")
DATA = os.path.join(WORK, "data")
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(DATA)
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


days = pd.bdate_range("2016-01-04", periods=60)
D = [str(x.date()) for x in days]
CROSS = [1, 6, 11, 30, 35]                        # Jan training waves + Feb scored
ROLLB = [4, 9, 14, 33, 38]


def write(tk, closes, volume, mad, signals=True):
    rows = [{"Date": D[i], "Open": closes[i], "Close": closes[i],
             "Volume": volume, "mad": mad, "z": mad * 2, "price_share_5": 0.8,
             "sig_cross_up": signals and i in CROSS,
             "sig_rollback": signals and i in ROLLB,
             "in_index": True} for i in range(60)]
    pd.DataFrame(rows).to_csv(os.path.join(DATA, f"{tk}.csv"), index=False)


write("AAA", [100 * 1.01 ** i for i in range(60)], 4000, .2)   # winners
write("BBB", [100 * 0.995 ** i for i in range(60)], 3000, .8)  # losers
write("CCC", [50.0] * 60, 2000, .5, signals=False)
write("DDD", [25.0] * 60, 1000, .5, signals=False)

env = {**os.environ, "PORTFOLIO_DATADIR": DATA,
       "SCORE_MIN_TRAIN": "3", "SCORE_N_RAND": "3"}
for script in (RANK, SCORE):
    r = subprocess.run([sys.executable, script], cwd=WORK, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:])
        print(r.stderr[-3000:])
        sys.exit(f"{os.path.basename(script)} FAILED")
out = r.stdout

sc = pd.read_csv(os.path.join(WORK, "results", "03_scores.csv"))
verdict = pd.read_csv(os.path.join(WORK, "results", "03_verdict.csv"))
summ = pd.read_csv(os.path.join(WORK, "results", "03_variant_summary.csv"))

check("pipeline: rank rows found for 100% of signals",
      "rank rows found for 100.0% of signals" in out)
check("10 signals enumerated (5 waves x AAA+BBB)", len(sc) == 10)
a1 = sc[(sc.ticker == "AAA")].iloc[0]
check("sig_date is the signal bar (entry bar minus one)",
      a1.entry_date == D[2] and a1.sig_date == D[1])
jan = sc[pd.to_datetime(sc.entry_date).dt.month == 1]
feb = sc[pd.to_datetime(sc.entry_date).dt.month == 2]
check("January waves unscored (burn-in); February waves scored with n_train >= 6",
      jan.score_RS.isna().all() and feb.score_RS.notna().all()
      and (feb.n_train >= 6).all())
fa = feb[feb.ticker == "AAA"]
fb = feb[feb.ticker == "BBB"]
check("rank features joined at signal close: AAA pct_vol .25, BBB .50",
      np.allclose(fa.pct_vol, .25) and np.allclose(fb.pct_vol, .50))
check("score_R separates: AAA (winner history, better ranks) > BBB",
      fa.score_R.min() > fb.score_R.max())
check("score_RS separates the same way", fa.score_RS.min() > fb.score_RS.max())
check("verdict table has all three models",
      sorted(verdict.model) == ["score_R", "score_RS", "score_S"])
check("book summary has 3 score variants + RAND row",
      sorted(summ.variant) == ["RAND", "SCORE_R", "SCORE_RS", "SCORE_S"])

print(f"\n{'ALL PIPELINE CHECKS PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(1 if fails else 0)
