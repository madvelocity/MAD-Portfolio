#!/usr/bin/env python3
"""
04_sim.py  --  the confirmation simulations: real-account dollars on the picking
rules the ledger already proved, against a random envelope, SPY, and a
split-half out-of-sample read.

DISCIPLINE.  The search for what works happened at the trade level (01_ranking's
family verdict, 02_score's nested models, proper statistics on 42k signals).
This script does NOT search: it runs a SHORT PRE-REGISTERED MENU in a realistic
account and reports what the winners are worth in dollars.  The 2022+ sub-period
is reported separately for every run -- the closest thing to out-of-sample we
own -- and a config that only worked 2016-2021 fails the phase.

THE MENU (every axis justified by prior evidence)
  pickers   SCORE_RS, SCORE_R    walk-forward scores read from 03_scores.csv
            SCORE_S              the signal-state model (established control)
            RANKVOL              raw volume rank from 02_ranks.csv ("the
                                 simplest thing that works")
  capacity  N in {5, 10, 20} slots (selectivity should fade as N grows)
  bench     at N=10 only: the cash-discipline variant -- signals tradeable only
            for names in the month's TOP_Q by volume rank (bench = names with
            rank_vol <= TOP_Q on the last published table before each month
            boundary; the rank table only contains live member-days, so dead
            names never hold a seat)
  controls  SPY buy-and-hold, and a RAND envelope per N (SIM_SEEDS random-pick
            books; exact Phipson & Smyth placement of every variant)

ACCOUNT RULES (the approved business rules)
  $SIM_CAPITAL start (default 100k), N equal slots sized equity/N at entry
  (cash-constrained, never rebalanced), market-on-open entries and exits,
  SIM_COST_BPS per side (default 5), fractional shares, cash earns 0.
  entry     gated signal at the name's close -> next open, ONE-day candidacy
            enforced on the master calendar (a name must have traded the
            immediately preceding session; halt-reopen stale signals are skipped)
  exit      +2 -> +1 rollback at a close -> sell at next open (no stop);
            index removal -> sell at that day's open; splice bar -> closed at
            the prior close; series end -> closed at the last bar's close
  honesty   prices split-adjusted, dividend-free (all lines measured alike);
            trade stats cover ALL closed trades; deterministic seeds
            SeedSequence([SEED, 6, N, s]), disjoint from every prior study.

INPUT: prepped files (PORTFOLIO_DATADIR or ./data) + results/02_ranks.csv +
results/03_scores.csv  (run 02_ranking.py and 03_score.py first).
    python3 04_sim.py
Outputs -> results/04_sim_summary.csv   every run: full-window AND 2022+ metrics
           results/04_sim_daily.csv     daily equity per named run
           results/04_sim_trades.csv    every trade per named run
           results/04_sim_rand.csv      the per-seed envelopes
SPY data: results/spy_cache.csv, else DATADIR/SPY.csv, else one yfinance pull
cached for deterministic reruns.
"""
import glob
import os

import numpy as np
import pandas as pd

DATADIR = os.environ.get("PORTFOLIO_DATADIR", "data")
OUTDIR = "results"
RANKS_CSV = os.path.join(OUTDIR, "02_ranks.csv")
SCORES_CSV = os.path.join(OUTDIR, "03_scores.csv")
ASOF = "2026-07-16"
WIN_START = pd.Timestamp("2016-01-01")
SPLIT = pd.Timestamp(os.environ.get("SIM_SPLIT", "2022-01-01"))
SIM_END = pd.Timestamp(os.environ.get("SIM_END", "2025-12-31"))
SHARE_MIN = 0.50
SEED = 7
SLOTS = [5, 10, 20]
BENCH_N = 10
TOP_Q = int(os.environ.get("SIM_TOP_Q", "100"))
COST = float(os.environ.get("SIM_COST_BPS", "5")) / 10000.0
CAPITAL = float(os.environ.get("SIM_CAPITAL", "100000"))
N_SEEDS = int(os.environ.get("SIM_SEEDS", "200"))
USECOLS = ["Date", "Open", "Close", "price_share_5",
           "sig_cross_up", "sig_rollback", "in_index"]
os.makedirs(OUTDIR, exist_ok=True)


def truthy(series):
    return series.astype(str).str.strip().isin(["True", "true", "1", "1.0"]).to_numpy()


# ---------------------------------------------------------------- load (verified)
def load_universe():
    files = sorted(f for f in glob.glob(os.path.join(DATADIR, "*.csv"))
                   if not os.path.basename(f).startswith("_"))
    if not files:
        raise SystemExit(f"no prepped data in {DATADIR}/ -- set PORTFOLIO_DATADIR")
    frames = {}
    n_bad = 0
    splices, no_volume = [], []
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
        df["member"] = truthy(df["in_index"])
        try:
            pd.read_csv(f, usecols=["Volume"], nrows=1)
        except (ValueError, KeyError):
            no_volume.append(tk)
        frames[tk] = df
        if i % 200 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] loaded")
    if no_volume:
        raise SystemExit(
            f"{len(no_volume)} files lack Volume (e.g. {', '.join(no_volume[:5])}) "
            "-- 01/02 rejected these; 03 must run on the same vintage.")
    if n_bad > len(files) / 2:
        raise SystemExit(f"{n_bad}/{len(files)} files unreadable -- wrong folder?")
    if splices:
        print(f"  SPLICE BARS BROKEN ({len(splices)}):")
        for tk, dt, rv in splices:
            print(f"    {tk:<8} {dt}  {rv}x")
    cal = sorted(set().union(*[set(d.index) for d in frames.values()]))
    cal = pd.DatetimeIndex([d for d in cal if WIN_START <= d <= SIM_END])
    U = {}
    for tk, df in frames.items():
        g = df.reindex(cal)
        has_bar = g["Close"].notna().to_numpy()
        if not has_bar.any():
            continue
        a = {c: g[c].to_numpy(float) for c in ["Open", "Close", "prev_close"]}
        for c in ["enter_here", "exit_here", "member"]:
            a[c] = g[c].fillna(False).astype(bool).to_numpy()
        a["has_bar"] = has_bar
        a["last_t"] = int(np.nonzero(has_bar)[0][-1])
        U[tk] = a
    return U, cal


# ---------------------------------------------------------------- pick values
def load_pickers(cal, tickers):
    """pick value per (ticker, entry calendar index); higher = taken first.
    Scores join on entry_date; the volume rank joins on the signal close."""
    if not os.path.exists(SCORES_CSV):
        raise SystemExit(f"{SCORES_CSV} not found -- run 03_score.py first.")
    if not os.path.exists(RANKS_CSV):
        raise SystemExit(f"{RANKS_CSV} not found -- run 02_ranking.py first.")
    t_of = {str(d.date()): t for t, d in enumerate(cal)}
    sc = pd.read_csv(SCORES_CSV,
                     usecols=["ticker", "entry_date", "sig_date",
                              "score_R", "score_RS", "score_S"])
    pick = {"SCORE_RS": {}, "SCORE_R": {}, "SCORE_S": {}, "RANKVOL": {}}
    for r in sc.itertuples():
        t = t_of.get(r.entry_date)
        if t is None:
            continue
        if r.score_RS == r.score_RS:
            pick["SCORE_RS"][(r.ticker, t)] = float(r.score_RS)
        if r.score_R == r.score_R:
            pick["SCORE_R"][(r.ticker, t)] = float(r.score_R)
        if r.score_S == r.score_S:
            pick["SCORE_S"][(r.ticker, t)] = float(r.score_S)
    rk = pd.read_csv(RANKS_CSV, usecols=["Date", "ticker", "rank_vol"])
    sc_sig = sc[["ticker", "entry_date", "sig_date"]].merge(
        rk, left_on=["sig_date", "ticker"], right_on=["Date", "ticker"],
        how="left")
    for r in sc_sig.itertuples():
        t = t_of.get(r.entry_date)
        if t is not None and r.rank_vol == r.rank_vol:
            pick["RANKVOL"][(r.ticker, t)] = -float(r.rank_vol)   # rank 1 first
    finite = {"SCORE_RS": int(sc.score_RS.notna().sum()),
              "SCORE_R": int(sc.score_R.notna().sum()),
              "SCORE_S": int(sc.score_S.notna().sum()),
              "RANKVOL": int(sc_sig.rank_vol.notna().sum())}
    for k in pick:
        cov = 100 * len(pick[k]) / max(finite[k], 1)
        flag = "" if cov >= 99 else "  <-- LOW JOIN COVERAGE, investigate"
        print(f"  picker {k}: {len(pick[k]):,} scored (ticker, day) pairs "
              f"({cov:.1f}% of finite upstream rows){flag}")
    return pick, rk


def volume_bench(rk, cal):
    """bench_by_day[t] = set of names with rank_vol <= TOP_Q on the last
    published table before each month boundary (walk-forward; the rank table
    holds only live member-days, so dead names never qualify)."""
    rk = rk.copy()
    rk["Date"] = pd.to_datetime(rk["Date"])
    per = cal.to_period("M")
    starts = [int(np.argmax(per == m)) for m in per.unique()]
    by_date = (rk[rk.rank_vol <= TOP_Q].groupby("Date")["ticker"]
               .apply(set).to_dict())
    dates_avail = sorted(by_date)
    bench_at = {}
    for s in starts:
        prior = [d for d in dates_avail if d < cal[s]]
        bench_at[s] = by_date[prior[-1]] if prior else set()
    out, cur, si = [], set(), 0
    for t in range(len(cal)):
        if si < len(starts) and t == starts[si]:
            cur = bench_at[starts[si]]
            si += 1
        out.append(cur)
    return out


# ---------------------------------------------------------------- the account
def run_account(U, cal, cand_by_day, n_slots, pickmap=None, rng=None,
                bench_by_day=None):
    cash, pos = CAPITAL, {}
    daily, trades = [], []
    for t in range(len(cal)):
        for tk in list(pos):
            a = U[tk]
            p = pos[tk]
            if not a["has_bar"][t]:
                if t > a["last_t"] or t - p.get("last_i", t) > 10:
                    cash += p["shares"] * p["last_close"] * (1 - COST)
                    trades.append({**p, "exit_date": p["last_date"],
                                   "exit_px": p["last_close"],
                                   "reason": "delisted"})
                    del pos[tk]
                continue
            if not a["member"][t]:                 # removal: sell at the open
                px = (a["Open"][t] if a["prev_close"][t] == a["prev_close"][t]
                      else p["last_close"])
                cash += p["shares"] * px * (1 - COST)
                trades.append({**p, "exit_date": str(cal[t].date()),
                               "exit_px": px, "reason": "index_drop"})
                del pos[tk]
                continue
            if a["prev_close"][t] != a["prev_close"][t]:
                cash += p["shares"] * p["last_close"] * (1 - COST)
                trades.append({**p, "exit_date": str(cal[t].date()),
                               "exit_px": p["last_close"], "reason": "data_break"})
                del pos[tk]
                continue
            if a["exit_here"][t]:
                px = a["Open"][t]
                cash += p["shares"] * px * (1 - COST)
                trades.append({**p, "exit_date": str(cal[t].date()),
                               "exit_px": px, "reason": "rollback"})
                del pos[tk]
        cands = [tk for tk in cand_by_day[t] if tk not in pos]
        if bench_by_day is not None:
            cands = [tk for tk in cands if tk in bench_by_day[t]]
        free = max(n_slots - len(pos), 0)
        if rng is not None:
            if len(cands) > 1:
                cands = list(rng.permutation(cands))
        elif pickmap is not None:
            cands.sort(key=lambda k: (-pickmap.get((k, t), -np.inf), k))
        for tk in cands[:free]:
            a = U[tk]
            px = a["Open"][t]
            mark = cash + sum(q["shares"] *
                              (U[k]["Open"][t] if U[k]["has_bar"][t]
                               else q["last_close"]) for k, q in pos.items())
            alloc = min(mark / n_slots, cash)
            if alloc < 1.0:
                continue
            shares = alloc / (px * (1 + COST))
            cash -= alloc
            pos[tk] = {"ticker": tk, "entry_date": str(cal[t].date()),
                       "entry_px": px, "shares": shares,
                       "last_close": a["Close"][t], "last_i": t,
                       "last_date": str(cal[t].date())}
        for tk, p in pos.items():
            a = U[tk]
            if a["has_bar"][t]:
                p["last_close"] = a["Close"][t]
                p["last_i"] = t
                p["last_date"] = str(cal[t].date())
        equity = cash + sum(p["shares"] * p["last_close"] for p in pos.values())
        daily.append({"date": cal[t], "equity": equity, "cash": cash,
                      "n_pos": len(pos)})
    for tk, p in list(pos.items()):
        trades.append({**p, "exit_date": "", "exit_px": p["last_close"],
                       "reason": "open"})
    return daily, trades


def spy_series(cal, symbol="SPY"):
    cache = os.path.join(OUTDIR, f"{symbol.lower()}_cache.csv")
    for path in [cache, os.path.join(DATADIR, f"{symbol}.csv")]:
        if os.path.exists(path):
            df = pd.read_csv(path).dropna(subset=["Open", "Close"])
            df["Date"] = pd.to_datetime(df["Date"])
            df = df[df["Date"] <= pd.Timestamp(ASOF)].set_index("Date")
            s = df.reindex(cal)[["Open", "Close"]]
            if s["Close"].notna().mean() > 0.9:
                print(f"  {symbol} from {path}")
                return s
    import yfinance as yf
    print(f"  {symbol}: pulling from yfinance (cached for reruns)...")
    df = yf.download(symbol, start="2015-06-01", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()[["Date", "Open", "Close"]]
    df.to_csv(cache, index=False)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] <= pd.Timestamp(ASOF)].set_index("Date")
    return df.reindex(cal)[["Open", "Close"]]


# ---------------------------------------------------------------- metrics
def window_stats(dates, eq, start=None):
    if start is None:
        e = eq
        base = CAPITAL
    else:
        m = dates >= start
        if m.sum() < 40:
            return {}
        i0 = int(np.argmax(m))
        base = eq[i0 - 1] if i0 > 0 else CAPITAL   # include the boundary day
        e = eq[m]
    path = np.concatenate([[base], e])
    r = np.diff(path) / path[:-1]
    years = len(e) / 252.0
    w = e[-1] / base
    sharpe = np.sqrt(252) * r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else np.nan
    dd = float((path / np.maximum.accumulate(path) - 1).min())
    return {"wealth_x": round(float(w), 3),
            "cagr": round(float(w ** (1 / years) - 1), 4),
            "sharpe": round(float(sharpe), 3) if sharpe == sharpe else np.nan,
            "maxdd": round(dd, 4)}


def metrics(daily, trades):
    dates = np.array([d["date"] for d in daily], dtype="datetime64[ns]")
    eq = np.array([d["equity"] for d in daily])
    npos = np.array([d["n_pos"] for d in daily])
    closed = [x for x in trades if x["reason"] != "open"]
    tr = np.array([x["exit_px"] * (1 - COST) / (x["entry_px"] * (1 + COST)) - 1
                   for x in closed]) if closed else np.array([])
    full = window_stats(dates, eq)
    late = {f"oos_{k}": v for k, v in
            window_stats(dates, eq, np.datetime64(SPLIT)).items()}
    return {"final_equity": round(float(eq[-1]), 2), **full, **late,
            "pct_days_invested": round(100 * float((npos > 0).mean()), 1),
            "avg_positions": round(float(npos.mean()), 2),
            "trades": len(closed),
            "win_rate": round(float((tr > 0).mean()), 3) if len(tr) else np.nan,
            "med_trade_ret": (round(float(np.median(tr)), 4)
                              if len(tr) else np.nan)}


# ---------------------------------------------------------------- run
print(f"loading universe from {DATADIR}/ (ASOF cap {ASOF})...")
U, cal = load_universe()
tickers = sorted(tk for tk in U if tk not in ("SPY", "RSP"))
print(f"{len(tickers)} names on a {len(cal)}-day calendar")

# one-day candidacy on the master calendar: the signal bar must be the
# immediately preceding session (halt-reopen stale signals skipped)
cand_by_day = [[] for _ in range(len(cal))]
for t in range(1, len(cal)):
    cand_by_day[t] = [tk for tk in tickers
                      if U[tk]["enter_here"][t] and U[tk]["member"][t]
                      and U[tk]["has_bar"][t - 1]
                      and np.isfinite(U[tk]["Open"][t])
                      and np.isfinite(U[tk]["Close"][t])]
n_sig = sum(len(c) for c in cand_by_day)
print(f"{n_sig} tradeable signal-days")

print("loading pickers...")
pick, rk = load_pickers(cal, tickers)
bench = volume_bench(rk, cal)
print(f"  volume bench: median size "
      f"{int(np.median([len(b) for b in bench if b]))} names")

summary, all_daily, all_trades = [], [], []
env_dists = {}


def record(name, daily, trades, keep=True):
    m = metrics(daily, trades)
    summary.append({"run": name, **m})
    oos = m.get("oos_cagr")
    print(f"  {name:<18} final ${m['final_equity']:>11,.0f}  "
          f"cagr {100*m['cagr']:>6.2f}%  shp {m['sharpe']:>6}  "
          f"dd {100*m['maxdd']:>6.1f}%  | 2022+ cagr "
          f"{100*oos if oos is not None else float('nan'):>6.2f}%  "
          f"shp {m.get('oos_sharpe', float('nan'))}")
    if keep:
        for d in daily:
            all_daily.append({"run": name, "date": str(d["date"].date()),
                              "equity": round(d["equity"], 2), "n_pos": d["n_pos"]})
        for x in trades:
            all_trades.append({"run": name,
                               **{k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in x.items()
                                  if k not in ("last_close", "last_i")}})
    return m


print("\n--- index controls (SPY cap-weight, RSP equal-weight) ---")
for symbol in ["SPY", "RSP"]:
    etf = spy_series(cal, symbol)
    op = etf["Open"].to_numpy(float)
    clx = etf["Close"].ffill().to_numpy(float)
    fin = np.nonzero(np.isfinite(op))[0]
    if not len(fin):
        raise SystemExit(f"{symbol} series empty -- provide results/"
                         f"{symbol.lower()}_cache.csv")
    first = int(fin[0])
    sh = CAPITAL / (op[first] * (1 + COST))
    etf_daily = [{"date": cal[t],
                  "cash": CAPITAL if t < first else 0.0,
                  "n_pos": 0 if t < first else 1,
                  "equity": CAPITAL if t < first else sh * clx[t]}
                 for t in range(len(cal))]
    record(symbol, etf_daily, [{"ticker": symbol,
                                "entry_date": str(cal[first].date()),
                                "entry_px": op[first], "shares": sh,
                                "exit_date": "", "exit_px": clx[-1],
                                "reason": "open",
                                "last_date": str(cal[-1].date())}])

print("\n--- the menu ---")
for n in SLOTS:
    for pname in ["SCORE_RS", "SCORE_R", "SCORE_S", "RANKVOL"]:
        record(f"{pname}_N{n}",
               *run_account(U, cal, cand_by_day, n, pickmap=pick[pname]))
for pname in ["SCORE_RS", "SCORE_R", "SCORE_S", "RANKVOL"]:
    record(f"{pname}_BENCH_N{BENCH_N}",
           *run_account(U, cal, cand_by_day, BENCH_N, pickmap=pick[pname],
                        bench_by_day=bench))

print(f"\n--- RAND envelopes ({N_SEEDS} seeds per N, + a benched envelope) ---")
rand_rows = []
for n in SLOTS:
    dist = []
    for s in range(N_SEEDS):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 6, n, s]))
        daily, trades = run_account(U, cal, cand_by_day, n, rng=rng)
        m = metrics(daily, trades)
        rand_rows.append({"n_slots": n, "bench": 0, "seed": s, **m})
        dist.append(m["final_equity"])
        if (s + 1) % 50 == 0:
            print(f"  N={n}: {s + 1}/{N_SEEDS}")
    env_dists[n] = np.array(dist, float)
    print(f"  N={n} envelope: median ${np.median(dist):,.0f}  "
          f"[p5 ${np.percentile(dist, 5):,.0f}, p95 ${np.percentile(dist, 95):,.0f}]")
# the BENCH runs get their own like-for-like null: random picks, same bench
dist = []
for s in range(N_SEEDS):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 6, BENCH_N, s, 1]))
    daily, trades = run_account(U, cal, cand_by_day, BENCH_N, rng=rng,
                                bench_by_day=bench)
    m = metrics(daily, trades)
    rand_rows.append({"n_slots": BENCH_N, "bench": 1, "seed": s, **m})
    dist.append(m["final_equity"])
    if (s + 1) % 50 == 0:
        print(f"  BENCH N={BENCH_N}: {s + 1}/{N_SEEDS}")
env_dists["bench"] = np.array(dist, float)
print(f"  BENCH envelope: median ${np.median(dist):,.0f}  "
      f"[p5 ${np.percentile(dist, 5):,.0f}, p95 ${np.percentile(dist, 95):,.0f}]")

print("\n--- placement vs the envelope (final equity, exact MC p) ---")
for row in summary:
    name = row["run"]
    n = None
    for cand_n in SLOTS:
        if name.endswith(f"_N{cand_n}"):
            n = cand_n
    if n is None or name in ("SPY", "RSP"):
        continue
    dist = env_dists["bench"] if "_BENCH_" in name else env_dists[n]
    v, K = row["final_equity"], len(dist)
    pct = float(((dist < v).sum() + 0.5 * (dist == v).sum()) / K)
    p = min(1.0, 2 * min(1 + (dist <= v).sum(), 1 + (dist >= v).sum()) / (K + 1))
    row["env_pctile"] = round(100 * pct, 1)
    row["env_p_two_sided"] = round(p, 4)
    print(f"  {name:<18} ${v:>11,.0f} -> pctile {row['env_pctile']:>5.1f}  p {p:.4f}")

pd.DataFrame(summary).to_csv(os.path.join(OUTDIR, "04_sim_summary.csv"), index=False)
pd.DataFrame(all_daily).to_csv(os.path.join(OUTDIR, "04_sim_daily.csv"), index=False)
pd.DataFrame(all_trades).to_csv(os.path.join(OUTDIR, "04_sim_trades.csv"), index=False)
pd.DataFrame(rand_rows).to_csv(os.path.join(OUTDIR, "04_sim_rand.csv"), index=False)
print(f"\nwrote results/04_sim_summary.csv  04_sim_daily.csv  04_sim_trades.csv  "
      f"04_sim_rand.csv")
