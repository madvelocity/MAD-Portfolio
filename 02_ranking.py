#!/usr/bin/env python3
"""
02_ranking.py  --  the daily rank table: every stock, every day, three families.

For each trading day, every S&P 500 member is ranked (1 = best) on three
INDEPENDENT families:

  VOLUME            rank_vol        that day's raw share volume, highest first

  SIGNAL-TO-PROFIT  rank_s2p_{3,5,10,15,20}
                    the mean trade return of the stock's own PAST k signal
                    trades, best first.  A "signal trade" is the paper's rule:
                    regime -1 -> +1 cross with price_share_5 >= 0.50 at the
                    close -> buy next open; +2 -> +1 rollback -> sell next
                    open.  Strictly walk-forward: a trade enters the stock's
                    history only from the day its exit is known.  A stock with
                    fewer than k completed signal trades has no rank (blank).

  RETURNS           rank_ret_{5,10,15,20,40,80}
                    trailing k-BAR close-to-close return on the stock's own
                    bars, highest first (across a data gap, "k periods" means
                    k of the stock's own trading bars).

Ranks are recomputed every day over that day's rank universe: index members
with a bar that day.  Ties share the better rank (method='min').  Everything
is knowable at that day's close -- the table is legal input for next-open
decisions.  Splice bars in the archive (>5x moves, symbol-reuse artifacts)
break return chains and blank any return window that crosses them; signal
trades crossing one are closed at the prior close ("data_break").

PURPOSE: raw material for portfolio construction.  Hypothesis under test:
volume, signal-to-profit, and returns identify the best stocks to buy when
buy signals fire.

INPUT: the prepped per-ticker files (PORTFOLIO_DATADIR or ./data).
    python3 02_ranking.py
Outputs -> results/02_ranks.csv      one row per member-day: all 12 ranks
                                     (~1.2M rows, ~100MB)
           results/02_ranks_latest.csv   the final day's full table
Console: coverage stats and the final day's top 10 in each family.
"""
import glob
import os

import numpy as np
import pandas as pd

DATADIR = os.environ.get("PORTFOLIO_DATADIR", "data")
OUTDIR = "results"
ASOF = "2026-07-16"
WIN_START = pd.Timestamp("2016-01-01")
SHARE_MIN = 0.50
S2P_WINDOWS = [3, 5, 10, 15, 20]
RET_WINDOWS = [5, 10, 15, 20, 40, 80]
USECOLS = ["Date", "Open", "Close", "Volume", "price_share_5",
           "sig_cross_up", "sig_rollback", "in_index"]
os.makedirs(OUTDIR, exist_ok=True)


def truthy(series):
    return series.astype(str).str.strip().isin(["True", "true", "1", "1.0"]).to_numpy()


# ---------------------------------------------------------------- load
def load_universe():
    files = sorted(f for f in glob.glob(os.path.join(DATADIR, "*.csv"))
                   if not os.path.basename(f).startswith("_"))
    if not files:
        raise SystemExit(
            f"no prepped data found in {DATADIR}/. Point PORTFOLIO_DATADIR at "
            "the 04_data_prep.py output or symlink it here as data/")
    frames = {}
    n_bad = 0
    splices = []
    for i, f in enumerate(files, 1):
        tk = os.path.basename(f)[:-4]
        try:
            df = pd.read_csv(f, usecols=USECOLS)
        except (ValueError, KeyError):
            n_bad += 1
            continue
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[df["Date"] <= pd.Timestamp(ASOF)]
        df = (df.dropna(subset=["Open", "Close"]).sort_values("Date")
                .drop_duplicates(subset="Date", keep="first").set_index("Date"))
        if len(df) < 2:
            continue
        sig = truthy(df["sig_cross_up"]) & (df["price_share_5"].to_numpy() >= SHARE_MIN)
        df["enter_here"] = np.concatenate([[False], sig[:-1]])
        df["exit_here"] = np.concatenate([[False], truthy(df["sig_rollback"])[:-1]])
        df["prev_close"] = df["Close"].shift(1)
        ret1 = df["Close"] / df["prev_close"]
        gap_days = df.index.to_series().diff().dt.days
        bad = (ret1 > 5.0) | (ret1 < 0.2) | (gap_days > 30)
        for dt, rv in ret1[bad].items():
            splices.append((tk, str(dt.date()), round(float(rv), 1)))
        df.loc[bad, "prev_close"] = np.nan
        # trailing k-bar returns at the close, on the name's own bars;
        # windows that cross a splice bar are blanked
        b = bad.astype(float)
        for k in RET_WINDOWS:
            r = df["Close"] / df["Close"].shift(k) - 1.0
            if bad.any():
                # ret_k[t] chains the 1-day ratios at bars t-k+1..t: a length-k
                # window ending at t exactly covers the taint set
                cw = (b.rolling(k, min_periods=1).max() > 0)
                r[cw.astype(bool)] = np.nan
            df[f"ret_{k}"] = r
        df["member"] = truthy(df["in_index"])
        frames[tk] = df
        if i % 200 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] loaded")
    if n_bad > len(files) / 2:
        raise SystemExit(
            f"{n_bad}/{len(files)} files lack the prepped columns -- {DATADIR}/ "
            "looks like RAW data; point at the 04_data_prep.py output.")
    if splices:
        print(f"  SPLICE BARS BROKEN ({len(splices)}):")
        for tk, dt, rv in splices:
            print(f"    {tk:<8} {dt}  {rv}x")
    cal = sorted(set().union(*[set(d.index) for d in frames.values()]))
    cal = pd.DatetimeIndex([d for d in cal if d >= WIN_START])
    return frames, cal


# ---------------------------------------------------------------- signal trades
def signal_trades(frames):
    """Every gated signal walked to its outcome on the name's own bars.
    Returns rows: ticker, exit_date (when the outcome is KNOWN), trade_ret."""
    rows = []
    for tk, df in frames.items():
        O = df["Open"].to_numpy(float)
        C = df["Close"].to_numpy(float)
        PC = df["prev_close"].to_numpy(float)
        enter = df["enter_here"].to_numpy()
        exit_ = df["exit_here"].to_numpy()
        member = df["member"].to_numpy()
        dates = df.index
        n = len(df)
        for t in np.nonzero(enter & member)[0]:
            if not (np.isfinite(O[t]) and np.isfinite(C[t])):
                continue
            eq = C[t] / O[t]
            u, done, reason = t + 1, False, "open"
            while u < n:
                if not member[u]:                   # removal: sell at the open
                    if PC[u] == PC[u]:
                        eq *= O[u] / PC[u]
                    done, reason = True, "index_drop"
                    break
                if PC[u] != PC[u]:                  # splice/era break: resolve
                    done, reason = True, "data_break"   # at the pre-break bar
                    u = u - 1
                    break
                if exit_[u]:                        # rollback: sell at the open
                    eq *= O[u] / PC[u]
                    done, reason = True, "rollback"
                    break
                eq *= C[u] / PC[u]
                u += 1
            if not done:                            # series ended while open?
                if dates[-1] < pd.Timestamp(ASOF) - pd.Timedelta(days=10):
                    done, reason, u = True, "delisted", n - 1
                # else: still open at ASOF -- outcome unknown, never enters s2p
            if done:
                ex = dates[min(u, n - 1)]
                if reason == "delisted":
                    # the end of a series is only observable ~10 days after its
                    # last bar; stamp the outcome there so it can never surface
                    # on a published day (member_ok is False after the last bar)
                    ex = ex + pd.Timedelta(days=11)
                rows.append({"ticker": tk, "exit_date": ex,
                             "trade_ret": eq - 1.0, "reason": reason})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- run
print(f"loading universe from {DATADIR}/ (ASOF cap {ASOF})...")
frames, cal = load_universe()
tickers = sorted(frames)
print(f"{len(tickers)} names on a {len(cal)}-day calendar "
      f"({cal[0].date()} -> {cal[-1].date()})")

print("resolving signal trades for the signal-to-profit family...")
trades = signal_trades(frames)
print(f"  {len(trades)} completed signal trades "
      f"({int((trades.reason == 'rollback').sum())} rollback exits)")

# ---- matrices: day x name
print("building daily matrices...")
member_ok = pd.DataFrame({tk: frames[tk]["member"].reindex(cal).fillna(False)
                          & frames[tk]["Close"].reindex(cal).notna()
                          for tk in tickers})
vol = pd.DataFrame({tk: frames[tk]["Volume"].reindex(cal) for tk in tickers})
rets = {k: pd.DataFrame({tk: frames[tk][f"ret_{k}"].reindex(cal)
                         for tk in tickers}) for k in RET_WINDOWS}

s2p = {}
for k in S2P_WINDOWS:
    cols = {}
    for tk, g in trades.groupby("ticker"):
        g = g.sort_values("exit_date", kind="stable")   # same-day exits keep
                                                        # entry order
        roll = g["trade_ret"].rolling(k, min_periods=k).mean()
        s = pd.Series(roll.to_numpy(), index=g["exit_date"].to_numpy())
        s = s.groupby(level=0).last()               # same-day exits: latest state
        cols[tk] = s.reindex(cal, method="ffill")
    s2p[k] = pd.DataFrame(cols).reindex(columns=tickers)

# ---- rank (1 = best) within each day's member universe
def rank(mat):
    return mat.where(member_ok).rank(axis=1, ascending=False, method="min")

print("ranking...")
out = {"rank_vol": rank(vol)}
for k in S2P_WINDOWS:
    out[f"rank_s2p_{k}"] = rank(s2p[k])
for k in RET_WINDOWS:
    out[f"rank_ret_{k}"] = rank(rets[k])

# ---- long table: one row per member-day
print("writing the long table...")
long = member_ok.stack()
long = long[long].index.to_frame(index=False)
long.columns = ["Date", "ticker"]
for name, mat in out.items():
    long[name] = mat.stack().reindex(
        pd.MultiIndex.from_frame(long[["Date", "ticker"]])).to_numpy()
long["Date"] = pd.to_datetime(long["Date"]).dt.strftime("%Y-%m-%d")
long = long.sort_values(["Date", "ticker"]).reset_index(drop=True)
long.to_csv(os.path.join(OUTDIR, "02_ranks.csv"), index=False,
            float_format="%.0f")
last_day = long[long["Date"] == long["Date"].iloc[-1]]
last_day.to_csv(os.path.join(OUTDIR, "02_ranks_latest.csv"), index=False,
                float_format="%.0f")

# ---- console
print(f"\n{len(long):,} member-day rows -> results/02_ranks.csv")
for col in ["rank_vol", "rank_s2p_10", "rank_ret_20"]:
    cov = 100 * long[col].notna().mean()
    print(f"  {col:<14} coverage {cov:5.1f}%")
d = last_day
print(f"\n--- {long['Date'].iloc[-1]}: top 10 per family ---")
for col, label in [("rank_vol", "VOLUME"), ("rank_s2p_10", "SIGNAL-TO-PROFIT (10)"),
                   ("rank_ret_20", "RETURNS (20d)")]:
    top = d.dropna(subset=[col]).nsmallest(10, col)
    print(f"  {label}: " + "  ".join(f"{r.ticker}({int(getattr(r, col))})"
                                     for r in top.itertuples()))
print(f"\nwrote results/02_ranks.csv  results/02_ranks_latest.csv")
