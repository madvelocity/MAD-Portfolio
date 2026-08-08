"""05_caprank_test.py -- pre-registered cap-rank selector test (capacity family, arm 3).

THE QUESTION
  Among simultaneous signal candidacies, does S&P-cap RANK (or a
  bench + velocity + cap composite) select better than raw volume rank --
  the only selector that survived Paper 4?

PRE-REGISTration (fixed before the run; nothing added mid-flight)
  eligibility  IDENTICAL for every arm: gated signal candidacy (one-day,
               same as 04_sim.py) + the walk-forward top-100 volume bench.
               Arms differ ONLY in the order candidates are taken.
  arms         RANKVOL    -rank_vol at the signal close   (the incumbent)
               CAPRANK    -rank_cap at the signal close   (cap size, 1=largest)
               SRANK      price_share_5 at the signal close (velocity ranking
                          beyond the gate -- the "hat-draw" question, settled)
               COMPOSITE  mean of same-day candidate percentiles of cap and s
                          (both required; missing-cap names rank last here too)
  capacity     N=10 slots, bench on -- the published headline config only.
  metrics      full-window and 2022+ OOS: CAGR, Sharpe, maxDD, final equity;
               win rate, median trade; % overlap of buys vs the RANKVOL run.
  null         the published benched RAND envelope (results/04_sim_rand.csv,
               bench=1 rows) -- an arm must clear its p95 final equity AND
               beat RANKVOL on BOTH full and OOS CAGR to displace volume.
  confound     names missing fundamentals rank LAST in CAPRANK/COMPOSITE --
               an implicit alive-in-2026 filter RANKVOL lacks. The
               covered-pair check re-runs RANKVOL and each cap arm on the
               cap-covered candidacy subset only; displacement ALSO requires
               the arm's margin to persist there. If the RAND file is absent
               the verdict says so explicitly -- an arm can beat volume but
               never displace it without the envelope.
  reconcile    before any arm prints, this script re-runs RANKVOL_BENCH_N10
               with its own loader and must land within RECON_TOL of the
               published $552,888 final equity. Fail -> SystemExit; the arms
               are not comparable and must not be read.

CAP CONSTRUCTION (the honest label is CAP RANK, not S&P float weight)
  shares   SEC-sourced share counts from Polygon financials (quarterly,
           annual fallback): diluted_average_shares else basic_average_shares.
           A count becomes usable the day AFTER its filing_date (missing
           filing_date -> end_date + 75 calendar days). Forward-filled to the
           next filing. NEVER backfilled: no name has a cap before its first
           filing is public.
  splits   as-filed counts are already split-adjusted through their FILING
           date (ASC 260 restates retroactively), so only later splits apply:
           shares_today_basis = reported x PRODUCT(to/from) over splits
           executed ON/AFTER the count's known date. Splits executed after
           the bar vintage (ASOF) are dropped -- the bars' back-adjustment
           stops there. A late filing for an OLDER period (amendment) never
           overrides a fresher count.
  cap      split-adjusted Close x shares_today_basis; rank 1 = largest cap
           among that day's index members. Names with no usable count rank
           last (-inf pick), never silently mid-list.
  coverage hard floor: if < MIN_COVER of signal-day candidacies carry a
           finite rank_cap, the run aborts rather than report a thin test.

RUN (from the repo root)
  prerequisite: a fundamentals store -- one CSV per ticker of SEC share
    counts and splits (see load_fundamentals for the expected columns), at
    $PORTFOLIO_FUNDDIR (default <DATADIR>/fundamentals).
  export PORTFOLIO_DATADIR=/path/to/prepped/data
  python3 05_caprank_test.py

DATA STORE (this script only READS it)
  <DATADIR>/fundamentals/<TICKER>.csv -- one file per ticker beside raw/ and
  the prepped files: rows kind=filing (end_date, filing_date, shares) and
  kind=split (execution_date, ratio). Override with PORTFOLIO_FUNDDIR.
  Vendor-derived; data files are not redistributed with this repo.

  needs: results/02_ranks.csv (volume ranks + bench source) and the prepped
  per-ticker files. results/04_sim_rand.csv optional (envelope comparison).
  outputs: results/05_capranks.csv (derived ranks, like 02_ranks.csv),
           05_test_summary.csv, 05_test_daily.csv, 05_test_trades.csv.
"""
import glob
import os

import numpy as np
import pandas as pd

DATADIR = os.environ.get("PORTFOLIO_DATADIR", "data")
OUTDIR = "results"
RANKS_CSV = os.path.join(OUTDIR, "02_ranks.csv")
RAND_CSV = os.path.join(OUTDIR, "04_sim_rand.csv")
ASOF = "2026-07-16"
WIN_START = pd.Timestamp("2016-01-01")
SPLIT = pd.Timestamp(os.environ.get("SIM_SPLIT", "2022-01-01"))
SIM_END = pd.Timestamp(os.environ.get("SIM_END", "2025-12-31"))
SHARE_MIN = 0.50
N_SLOTS = 10
TOP_Q = int(os.environ.get("SIM_TOP_Q", "100"))
COST = float(os.environ.get("SIM_COST_BPS", "5")) / 10000.0
CAPITAL = float(os.environ.get("SIM_CAPITAL", "100000"))
PUBLISHED_FINAL = 552888.0          # RANKVOL_BENCH_N10, Paper 4
RECON_TOL = 0.015                   # relative
FILING_LAG_FALLBACK = 75            # calendar days past period end
MIN_COVER = 0.80                    # finite rank_cap share of candidacies

FUND_DIR = os.environ.get("PORTFOLIO_FUNDDIR",
                          os.path.join(DATADIR, "fundamentals"))
CAPRANKS_CSV = os.path.join(OUTDIR, "05_capranks.csv")
os.makedirs(OUTDIR, exist_ok=True)


def _fpath(tk):
    return os.path.join(FUND_DIR, f"{tk}.csv")


def truthy(series):
    return series.astype(str).str.strip().isin(["True", "true", "1", "1.0"]).to_numpy()


# ---------------------------------------------------------------- fundamentals
def load_fundamentals(tickers):
    """Read the per-ticker store back into filings + splits tables."""
    sh, sp = [], []
    missing = 0
    for tk in tickers:
        p = _fpath(tk)
        if not os.path.exists(p):
            missing += 1
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        f = df[df.kind == "filing"]
        for r in f.itertuples():
            sh.append({"ticker": tk, "end_date": r.end_date,
                       "filing_date": r.filing_date, "shares": r.shares})
        s = df[df.kind == "split"]
        for r in s.itertuples():
            sp.append({"ticker": tk, "execution_date": r.execution_date,
                       "ratio": r.ratio})
    if missing:
        print(f"  {missing} tickers not in the store (fetch incomplete)")
    return (pd.DataFrame(sh, columns=["ticker", "end_date", "filing_date", "shares"]),
            pd.DataFrame(sp, columns=["ticker", "execution_date", "ratio"]))


# ---------------------------------------------------------------- cap ranks
def build_capranks(frames, cal):
    """results/05_capranks.csv: Date, ticker, rank_cap over that day's members."""
    sh, sp = load_fundamentals(sorted(frames))
    sh = sh.dropna(subset=["shares", "end_date"]).drop_duplicates(
        subset=["ticker", "end_date"], keep="last")
    sh["end_date"] = pd.to_datetime(sh["end_date"])
    sh["filing_date"] = pd.to_datetime(sh["filing_date"], errors="coerce")
    # a count is USABLE strictly after its filing date (fallback: end + lag)
    sh["known"] = sh["filing_date"] + pd.Timedelta(days=1)
    fb = sh["known"].isna()
    sh.loc[fb, "known"] = sh.loc[fb, "end_date"] + pd.Timedelta(
        days=FILING_LAG_FALLBACK)
    sp["execution_date"] = pd.to_datetime(sp["execution_date"])
    sp = sp[sp["execution_date"] <= pd.Timestamp(ASOF)]   # bars' vintage
    sp_by = {t: g.sort_values("execution_date") for t, g in sp.groupby("ticker")}

    caps = {}
    for tk, df in frames.items():
        g = sh[sh.ticker == tk]
        if g.empty:
            continue
        g = g.sort_values(["known", "end_date"])
        # a late filing for an OLDER period (amendment) never overrides
        g = g[~(g["end_date"] < g["end_date"].cummax())]
        # today-basis shares: as-filed counts already reflect splits through
        # the filing (ASC 260) -- only splits from the known date onward apply
        adj = []
        for r in g.itertuples():
            f = 1.0
            s = sp_by.get(tk)
            if s is not None:
                f = float(s.loc[s.execution_date >= r.known, "ratio"].prod())
            adj.append(r.shares * f)
        known = pd.Series(adj, index=g["known"].values).sort_index()
        known = known[~known.index.duplicated(keep="last")]
        daily = known.reindex(df.index, method="ffill")     # never backfill
        caps[tk] = df["Close"] * daily

    cap = pd.DataFrame(caps).reindex(cal)
    member = pd.DataFrame(
        {tk: frames[tk]["member"].reindex(cal, fill_value=False)
         for tk in frames if tk in cap.columns})
    cap = cap.where(member)
    rank_cap = cap.rank(axis=1, ascending=False, method="min")
    long = (rank_cap.stack().rename("rank_cap").reset_index()
            .rename(columns={"level_0": "Date", "level_1": "ticker"}))
    long["cap_dollars"] = cap.stack().reindex(
        long.set_index(["Date", "ticker"]).index).to_numpy()
    long.to_csv(CAPRANKS_CSV, index=False, float_format="%.0f")
    print(f"cap ranks: {len(long):,} member-days across "
          f"{long.ticker.nunique()} names -> {CAPRANKS_CSV}")
    return rank_cap


# ---------------------------------------------------------------- universe (04_sim mechanics, verbatim logic)
USECOLS = ["Date", "Open", "Close", "price_share_5",
           "sig_cross_up", "sig_rollback", "in_index"]


def load_universe():
    files = sorted(f for f in glob.glob(os.path.join(DATADIR, "*.csv"))
                   if not os.path.basename(f).startswith("_"))
    if not files:
        raise SystemExit(f"no prepped data in {DATADIR}/ -- set PORTFOLIO_DATADIR")
    frames, n_bad = {}, 0
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
        df.loc[bad, "prev_close"] = np.nan
        df["member"] = truthy(df["in_index"])
        frames[tk] = df
        if i % 200 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] loaded")
    if n_bad > len(files) / 2:
        raise SystemExit(f"{n_bad}/{len(files)} files unreadable -- wrong folder?")
    cal = sorted(set().union(*[set(d.index) for d in frames.values()]))
    cal = pd.DatetimeIndex([d for d in cal if WIN_START <= d <= SIM_END])
    U = {}
    for tk, df in frames.items():
        g = df.reindex(cal)
        has_bar = g["Close"].notna().to_numpy()
        if not has_bar.any():
            continue
        a = {c: g[c].to_numpy(float) for c in ["Open", "Close", "prev_close",
                                               "price_share_5"]}
        for c in ["enter_here", "exit_here", "member"]:
            a[c] = g[c].fillna(False).astype(bool).to_numpy()
        a["has_bar"] = has_bar
        a["last_t"] = int(np.nonzero(has_bar)[0][-1])
        U[tk] = a
    return frames, U, cal


def volume_bench(rk, cal):
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


def run_account(U, cal, cand_by_day, n_slots, pickmap=None, bench_by_day=None):
    cash, pos = CAPITAL, {}
    daily, trades = [], []
    for t in range(len(cal)):
        for tk in list(pos):
            a, p = U[tk], pos[tk]
            if not a["has_bar"][t]:
                if t > a["last_t"] or t - p.get("last_i", t) > 10:
                    cash += p["shares"] * p["last_close"] * (1 - COST)
                    trades.append({**p, "exit_date": p["last_date"],
                                   "exit_px": p["last_close"], "reason": "delisted"})
                    del pos[tk]
                continue
            if not a["member"][t]:
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
        if pickmap is not None:
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


def window_stats(dates, eq, start=None):
    if start is None:
        e, base = eq, CAPITAL
    else:
        m = dates >= start
        if m.sum() < 40:
            return {}
        i0 = int(np.argmax(m))
        base = eq[i0 - 1] if i0 > 0 else CAPITAL
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


# ---------------------------------------------------------------- main
print(f"loading universe from {DATADIR}/ (ASOF cap {ASOF})...")
frames, U, cal = load_universe()
tickers = sorted(tk for tk in U if tk not in ("SPY", "RSP"))
print(f"{len(tickers)} names on a {len(cal)}-day calendar")

cand_by_day = [[] for _ in range(len(cal))]
for t in range(1, len(cal)):
    cand_by_day[t] = [tk for tk in tickers
                      if U[tk]["enter_here"][t] and U[tk]["member"][t]
                      and U[tk]["has_bar"][t - 1]
                      and np.isfinite(U[tk]["Open"][t])
                      and np.isfinite(U[tk]["Close"][t])]
print(f"{sum(len(c) for c in cand_by_day)} tradeable signal-days")

if not os.path.exists(RANKS_CSV):
    raise SystemExit(f"{RANKS_CSV} not found -- run 02_ranking.py first.")
rk = pd.read_csv(RANKS_CSV, usecols=["Date", "ticker", "rank_vol"])
bench = volume_bench(rk, cal)

print("\nSTAGE A: fundamentals store check...")
have = sum(os.path.exists(_fpath(t)) for t in tickers)
print(f"  {have}/{len(tickers)} tickers present in {FUND_DIR}/")
if have < len(tickers) * 0.5:
    raise SystemExit("fundamentals store missing or thin -- populate it with "
                     "one CSV per ticker of SEC share counts and splits "
                     "(rows kind=filing: end_date, filing_date, shares; "
                     "kind=split: execution_date, ratio) first.")

print("\nSTAGE B: point-in-time cap ranks...")
member_frames = {tk: frames[tk] for tk in tickers}
rank_cap = build_capranks(member_frames, cal)

# pick values join at the signal close = cal[t-1] (guaranteed by the
# has_bar[t-1] candidacy condition above)
rk["Date"] = pd.to_datetime(rk["Date"])
rv = rk.set_index(["Date", "ticker"])["rank_vol"]
t_prev = {t: cal[t - 1] for t in range(1, len(cal))}
pick_vol, pick_cap, pick_s, pick_cmp = {}, {}, {}, {}
n_cand = n_cap_ok = 0
for t in range(1, len(cal)):
    day = t_prev[t]
    cands = cand_by_day[t]
    if not cands:
        continue
    caps, ss = {}, {}
    for tk in cands:
        n_cand += 1
        v = rv.get((day, tk))
        if v is not None and v == v:
            pick_vol[(tk, t)] = -float(v)
        c = rank_cap.at[day, tk] if tk in rank_cap.columns else np.nan
        if c == c:
            caps[tk] = -float(c)
            n_cap_ok += 1
        i = np.searchsorted(cal, day)
        s = U[tk]["price_share_5"][i]
        if s == s:
            ss[tk] = float(s)
    for tk, v in caps.items():
        pick_cap[(tk, t)] = v
    for tk, v in ss.items():
        pick_s[(tk, t)] = v
    # composite: mean of same-day candidate percentiles (scale-free)
    if caps or ss:
        def pctl(d):
            if not d:
                return {}
            ks = sorted(d, key=d.get)
            return {k: (i + 1) / len(ks) for i, k in enumerate(ks)}
        pc, ps = pctl(caps), pctl(ss)
        for tk in set(pc) & set(ps):     # both required -- no cap, no composite
            pick_cmp[(tk, t)] = (pc[tk] + ps[tk]) / 2

cover = n_cap_ok / max(n_cand, 1)
print(f"rank_cap coverage: {100 * cover:.1f}% of signal-day candidacies")
if cover < MIN_COVER:
    raise SystemExit(f"coverage {100 * cover:.1f}% < {100 * MIN_COVER:.0f}% floor "
                     "-- fetch is incomplete (rerun to resume) or fundamentals "
                     "are missing; refusing to report a thin test.")

print("\nSTAGE C: simulations (N=10, benched, published mechanics)...")
summary, all_daily, all_trades, buys = [], [], [], {}


def record(name, pickmap, cands=None):
    daily, trades = run_account(U, cal,
                                cands if cands is not None else cand_by_day,
                                N_SLOTS, pickmap=pickmap, bench_by_day=bench)
    m = metrics(daily, trades)
    summary.append({"run": name, **m})
    buys[name] = {(x["ticker"], x["entry_date"]) for x in trades}
    for d in daily:
        all_daily.append({"run": name, "date": str(d["date"].date()),
                          "equity": round(d["equity"], 2), "n_pos": d["n_pos"]})
    for x in trades:
        all_trades.append({"run": name,
                           **{k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in x.items()
                              if k not in ("last_close", "last_i")}})
    oos = m.get("oos_cagr", float("nan"))
    print(f"  {name:<12} final ${m['final_equity']:>11,.0f}  "
          f"cagr {100 * m['cagr']:>6.2f}%  shp {m['sharpe']:>6}  "
          f"dd {100 * m['maxdd']:>6.1f}%  | 2022+ cagr {100 * oos:>6.2f}%")
    return m


m_vol = record("RANKVOL", pick_vol)
gap = abs(m_vol["final_equity"] - PUBLISHED_FINAL) / PUBLISHED_FINAL
if gap > RECON_TOL:
    raise SystemExit(
        f"RECONCILIATION FAILED: RANKVOL final ${m_vol['final_equity']:,.0f} "
        f"vs published ${PUBLISHED_FINAL:,.0f} ({100 * gap:.2f}% off, tol "
        f"{100 * RECON_TOL:.1f}%). Mechanics or data vintage diverge from "
        "04_sim.py -- the arms below would not be comparable. Fix first.")
print(f"  reconciliation vs published: {100 * gap:.2f}% off -- PASS")

m_cap = record("CAPRANK", pick_cap)
m_s = record("SRANK", pick_s)
m_cmp = record("COMPOSITE", pick_cmp)

# covered-pair confound check: both selectors, cap-covered candidacies only
cov_cand = [[tk for tk in cand_by_day[t] if (tk, t) in pick_cap]
            for t in range(len(cal))]
print("  covered-pair check (cap-covered candidacies only):")
m_vol_cov = record("RANKVOL_COV", pick_vol, cov_cand)
m_cap_cov = record("CAPRANK_COV", pick_cap, cov_cand)
m_cmp_cov = record("COMPOSITE_COV", pick_cmp, cov_cand)

print("\nSTAGE D: diagnostics...")
vset = buys["RANKVOL"]
for name in ["CAPRANK", "SRANK", "COMPOSITE"]:
    b = buys[name]
    ov = 100 * len(b & vset) / max(len(b), 1)
    for row in summary:
        if row["run"] == name:
            row["buy_overlap_vs_RANKVOL_pct"] = round(ov, 1)
    print(f"  {name:<12} buy overlap vs RANKVOL: {ov:.1f}% "
          f"({len(b & vset)}/{len(b)} fills)")

# spearman between the two rank systems on candidacy days
pairs = [(-pick_vol[k], -pick_cap[k]) for k in pick_vol if k in pick_cap]
if len(pairs) > 100:
    a = pd.DataFrame(pairs, columns=["vol", "cap"])
    rho = a["vol"].corr(a["cap"], method="spearman")
    print(f"  spearman(rank_vol, rank_cap) on candidacies: {rho:.3f} "
          f"(n={len(a):,}) -- power lives where they disagree")

env_note = "04_sim_rand.csv missing or without bench rows -- envelope UNAVAILABLE"
if os.path.exists(RAND_CSV):
    rr = pd.read_csv(RAND_CSV)
    rr = rr[(rr.get("bench") == 1)]
    if len(rr):
        p95 = float(np.percentile(rr["final_equity"], 95))
        med = float(np.median(rr["final_equity"]))
        env_note = f"benched RAND envelope: median ${med:,.0f}, p95 ${p95:,.0f}"
        for row in summary:
            row["clears_rand_p95"] = bool(row["final_equity"] > p95)
print(f"  {env_note}")


def _beats(a, b):
    return (a["cagr"] > b["cagr"]
            and a.get("oos_cagr", -9) > b.get("oos_cagr", -9))


cov_pass = {"CAPRANK": _beats(m_cap_cov, m_vol_cov),
            "COMPOSITE": _beats(m_cmp_cov, m_vol_cov),
            "SRANK": True}                     # s has no coverage confound
print("\nVERDICT (pre-registered: displace volume only by beating RANKVOL on "
      "BOTH windows, clearing the RAND p95, AND surviving the covered-pair "
      "confound check):")
for row in summary:
    if row["run"] not in ("CAPRANK", "SRANK", "COMPOSITE"):
        continue
    beats = _beats(row, m_vol)
    clears = row.get("clears_rand_p95", None)
    conf = cov_pass[row["run"]]
    if beats and clears is True and conf:
        verdict = "DISPLACES VOLUME"
    elif beats and clears is None:
        verdict = ("beats volume -- envelope unavailable (need 04_sim_rand.csv); "
                   "NOT displaceable on this run")
    elif beats and not conf:
        verdict = "beats volume, FAILS the coverage-confound check"
    elif beats:
        verdict = "beats volume, fails the RAND envelope"
    else:
        verdict = "does not beat volume"
    print(f"  {row['run']:<12} {verdict}")

pd.DataFrame(summary).to_csv(os.path.join(OUTDIR, "05_test_summary.csv"), index=False)
pd.DataFrame(all_daily).to_csv(os.path.join(OUTDIR, "05_test_daily.csv"), index=False)
pd.DataFrame(all_trades).to_csv(os.path.join(OUTDIR, "05_test_trades.csv"), index=False)
print(f"\nwritten: {OUTDIR}/05_test_summary.csv, 05_test_daily.csv, "
      f"05_test_trades.csv, {CAPRANKS_CSV}")
print("NOTE: 'cap rank' is price x SEC share count, filing-date lagged, "
      "split-adjusted -- NOT S&P float-adjusted weight. Label it honestly.")
