"""07_c2_overlay_test.py -- pre-registered C2 skew-lean overlay on the buy signals.

THE QUESTION
  The C2 discovery (made on the pooled 5-offset anchor frame implemented
  in build_pool / load_name below): the manifold's innovation ASYMMETRY
  is forecastable out of sample while the mean is not.
  Does tilting the live book's entries toward benign-lean cells improve the
  book -- in the SHAPE channel where a distribution edge must show up?

HONEST EXPECTATION (pre-registered): the mean is closed, so the overlay
  should NOT be expected to raise CAGR. If C2 transfers, it shows up as a
  better median trade, shallower drawdowns, fewer air pockets. The success
  rule below therefore scores the LEFT TAIL of trade returns, with growth guardrails.

THE LEAN
  Anchor construction is the pooled 5-offset anchor frame implemented in
  build_pool / load_name below: non-overlapping K=5 grids (all 5 offsets
  pooled for map estimation), member-only, era-break guarded, frozen-sigma
  units.
    v    = (mad_t - mad_{t-5}) / sigma_t
    cell = zband(z_t) * 2 + (v > 0),  zbands (-inf,-3,-2,-1,0,1,2,3,inf)
    m1   = flat-price projection increment; e = y - m1 (the innovation),
           COMPLETED at t+5
    lean = Bowley skew of the cell's completed innovations:
           (q90 + q10 - 2*q50) / (q90 - q10);  LESS negative = benign
  WALK-FORWARD LAW: the map is recomputed each month from innovations whose
  COMPLETION date precedes the month start. No anchor completing on or after
  the month start touches that month's map. Applying the final map to the
  past is the leak this design exists to avoid.
  Cells with fewer than MIN_CELL_N completed innovations carry the pooled
  (all-cell) lean and are never eligible for the filter's worst-4 set.

ARMS (all N=10, volume bench; identical mechanics to the published sim)
  RANKVOL_BENCH_N10  the incumbent, reconciled to $552,888 before arms run.
  LEAN_TIEBREAK      PRIMARY (the gentle rule): candidates sort by volume
                     BAND (ranks 1-10, 11-20, ...) first, lean second,
                     rank third -- volume keeps the coarse order, lean
                     reorders locally within bands.
  LEAN_FILTER        volume order, but candidacies whose current cell is
                     among the map's worst-4 leans are skipped.
  LEAN_SIZER         volume order and selection unchanged; the slot
                     allocation is scaled by the lean's percentile among
                     the current map's 16 cells: x in [0.5, 1.5].
  LEAN_SELECT        the aggressive reference: candidates
                     ranked purely by lean, volume only as tiebreak. A full
                     selector faces the FULL selector bar: beat RANKVOL on
                     BOTH windows AND clear the published benched RAND
                     envelope's p95 (results/04_sim_rand.csv, bench=1) --
                     the same gauntlet cap rank and SRANK failed.
  NULL (all three overlay arms): PERSISTENT-PERMUTATION envelope. Per seed
  (SeedSequence([7, 9, s])), ONE permutation of the 16 cell identities is
  drawn and applied to EVERY monthly map, and all three overlay arms re-run
  under it. Identical gentleness, meaningless -- but PERSISTENT -- geography:
  the real maps are expanding-window and near-static, so a fair null must
  hold its wrong map for the decade too, not re-shuffle it monthly into
  self-cancelling noise. Each arm is priced against its own envelope.

SUCCESS RULE (pre-registered; the pass channel is the LEFT TAIL of trade
  returns -- the channel the C2 discovery actually scored (paired-tail
  pinball, q50 explicitly unscored). The median is REPORTED, never scored.
  "adopt-candidate" requires ALL of, per arm):
  1. p10_trade_ret  strictly better (less negative) than the baseline's
  2. maxdd          no worse than the baseline's
  3. full CAGR  >=  baseline - 1.0 pt  AND  OOS CAGR >= baseline - 1.0 pt
  4. p10_trade_ret  above the arm's own persistent-permutation p95
  A pass certifies the CELL GEOGRAPHY is causal; attribution to the skew
  channel specifically still requires the per-cell exhibit in the write-up.
  Anything less is reported as-is and does not touch the live book.
  Implementation picks (VOL_BAND=10, worst-4 filter, MIN_CELL_N) are
  stamped into results/07_c2_config.json at run time -- the one-shot claim
  is verifiable, not asserted.

RUN (from the repo root)
  export PORTFOLIO_DATADIR=/path/to/prepped/data
  python3 07_c2_overlay_test.py
  needs results/02_ranks.csv. outputs results/07_c2_summary.csv,
  07_c2_daily.csv, 07_c2_trades.csv, 07_c2_maps.csv, 07_c2_rand_maps.csv,
  07_c2_config.json.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

DATADIR = os.environ.get("PORTFOLIO_DATADIR", "data")
OUTDIR = "results"
RANKS_CSV = os.path.join(OUTDIR, "02_ranks.csv")
ASOF = "2026-07-16"
WIN_START = pd.Timestamp("2016-01-01")
SPLIT = pd.Timestamp(os.environ.get("SIM_SPLIT", "2022-01-01"))
SIM_END = pd.Timestamp(os.environ.get("SIM_END", "2025-12-31"))
SHARE_MIN = 0.50
N_SLOTS = 10
TOP_Q = int(os.environ.get("SIM_TOP_Q", "100"))
COST = float(os.environ.get("SIM_COST_BPS", "5")) / 10000.0
CAPITAL = float(os.environ.get("SIM_CAPITAL", "100000"))
PUBLISHED_FINAL = 552888.0
RECON_TOL = 0.015
SEED = 7
ENV_SEEDS = int(os.environ.get("C2_ENV_SEEDS", "100"))
MIN_CELL_N = int(os.environ.get("C2_MIN_CELL_N", "1000"))
# 1000 pooled anchors across 5 overlapping-in-time offsets ~ 200 effective
VOL_BAND = int(os.environ.get("C2_VOL_BAND", "10"))
K = 5                                   # anchor-frame velocity horizon
ERA_AGE = 275                           # anchor-frame within-era sigma warm-up
SMA_W = 20
Z_EDGES = np.array([-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf])
N_CELLS = (len(Z_EDGES) - 1) * 2
N_OFFSETS = 5
os.makedirs(OUTDIR, exist_ok=True)


def truthy(series):
    return series.astype(str).str.strip().isin(["True", "true", "1", "1.0"]).to_numpy()


# ================================================================ anchors
# load_name / build_anchors implement the pooled 5-offset anchor frame the
# C2 discovery was established on (the frame described in the docstring).
# Any edit here changes the frame the discovery was scored on -- don't.
def load_name(path):
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    ret1 = df["Close"] / df["Close"].shift(1)
    gap = df["Date"].diff().dt.days
    brk = (gap > 30) | (ret1 > 5.0) | (ret1 < 0.2)
    df["era"] = brk.fillna(False).cumsum()
    df["era_age"] = df.groupby("era").cumcount()
    return df


def build_anchors(df, ticker, offset):
    n = len(df)
    c = df["Close"].to_numpy(float)
    mad = df["mad"].to_numpy(float)
    sig = df["sigma"].to_numpy(float)
    z = df["z"].to_numpy(float)
    member = df["in_index"].to_numpy(bool)
    era = df["era"].to_numpy()
    age = df["era_age"].to_numpy()
    s15 = pd.Series(c).rolling(15).sum().to_numpy()
    rows = []
    for t in range(offset, n, K):
        if not member[t] or age[t] < ERA_AGE:
            continue
        if t - K < 0 or era[t - K] != era[t]:
            continue
        if not (np.isfinite(z[t]) and np.isfinite(mad[t]) and np.isfinite(sig[t])
                and sig[t] > 0 and np.isfinite(mad[t - K]) and np.isfinite(s15[t])):
            continue
        v_t = (mad[t] - mad[t - K]) / sig[t]
        if not np.isfinite(v_t):
            continue
        zb = int(np.digitize(z[t], Z_EDGES)) - 1
        cell = zb * 2 + (1 if v_t > 0 else 0)
        if t + K >= n or era[t + K] != era[t]:
            continue
        sma_flat = (s15[t] + K * c[t]) / SMA_W
        mad_flat = 100.0 * (c[t] - sma_flat) / c[t]
        rows.append({
            "ticker": ticker, "date": df["Date"].iloc[t],
            "comp": df["Date"].iloc[t + K],
            "cell": cell,
            "e": ((mad[t + K] - mad[t]) / sig[t])
                 - ((mad_flat - mad[t]) / sig[t]),
        })
    return rows


def build_pool(files):
    rows = []
    for i, f in enumerate(files, 1):
        tk = os.path.basename(f)[:-4]
        try:
            df = load_name(f)
        except Exception:
            continue
        need = {"mad", "sigma", "z", "in_index", "Close", "Date"}
        if not need.issubset(df.columns):
            continue
        df = df[df["Date"] <= pd.Timestamp(ASOF)].reset_index(drop=True)
        df["in_index"] = truthy(df["in_index"])
        for off in range(N_OFFSETS):
            rows.extend(build_anchors(df, tk, off))
        if i % 100 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] anchored ({len(rows):,} anchors)")
    pool = pd.DataFrame(rows)
    pool["comp"] = pd.to_datetime(pool["comp"])
    return pool.sort_values("comp").reset_index(drop=True)


# ================================================================ maps
def bowley(e):
    q10, q50, q90 = np.percentile(e, [10, 50, 90])
    d = q90 - q10
    return (q90 + q10 - 2 * q50) / d if d > 0 else np.nan


def monthly_maps(pool, cal):
    """lean[month_index m][cell] from anchors COMPLETED before month start.
    Thin cells (< MIN_CELL_N) carry the pooled lean and are flagged."""
    per = cal.to_period("M")
    starts = [int(np.argmax(per == m)) for m in per.unique()]
    comp = pool["comp"].to_numpy()
    cells = pool["cell"].to_numpy()
    e = pool["e"].to_numpy()
    maps, thin_flags, rows = [], [], []
    for s in starts:
        cutoff = np.datetime64(cal[s])
        k = np.searchsorted(comp, cutoff, side="left")   # strictly before
        leans = np.full(N_CELLS, np.nan)
        thin = np.zeros(N_CELLS, bool)
        if k > 0:
            e_k, c_k = e[:k], cells[:k]
            pooled = bowley(e_k)
            for cell in range(N_CELLS):
                m = c_k == cell
                if m.sum() >= MIN_CELL_N:
                    leans[cell] = bowley(e_k[m])
                else:
                    leans[cell] = pooled
                    thin[cell] = True
        maps.append(leans)
        thin_flags.append(thin)
        rows.append({"month_start": str(cal[s].date()), "n_completed": int(k),
                     **{f"lean_{c}": leans[c] for c in range(N_CELLS)}})
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "07_c2_maps.csv"),
                              index=False)
    month_of = np.zeros(len(cal), int)
    si = 0
    for t in range(len(cal)):
        if si + 1 < len(starts) and t >= starts[si + 1]:
            si += 1
        month_of[t] = si
    return maps, thin_flags, month_of, starts


# ================================================================ universe (04/06 mechanics)
USECOLS = ["Date", "Open", "Close", "price_share_5", "dmad_5", "z", "sigma",
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
        # the candidacy's cell at the signal close, shifted to the entry bar
        zz = df["z"].to_numpy(float)
        vv = df["dmad_5"].to_numpy(float) / df["sigma"].to_numpy(float)
        zb = np.digitize(zz, Z_EDGES) - 1
        cell = zb * 2 + (vv > 0).astype(int)
        cell = np.where(np.isfinite(zz) & np.isfinite(vv), cell, -1)
        df["cell_sig"] = np.concatenate([[-1], cell[:-1]]).astype(int)
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
        a = {c: g[c].to_numpy(float) for c in ["Open", "Close", "prev_close"]}
        for c in ["enter_here", "exit_here", "member"]:
            a[c] = g[c].fillna(False).astype(bool).to_numpy()
        a["cell_sig"] = g["cell_sig"].fillna(-1).astype(int).to_numpy()
        a["has_bar"] = has_bar
        a["last_t"] = int(np.nonzero(has_bar)[0][-1])
        U[tk] = a
    return U, cal, files


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


def run_account(U, cal, cand_by_day, n_slots, sortkey, bench_by_day=None,
                skipmap=None, sizemap=None):
    """04_sim mechanics; sortkey(tk, t) orders candidates (lower first);
    skipmap(tk, t) -> True drops a candidacy; sizemap(tk, t) -> multiplier
    scales the slot allocation (selection unchanged)."""
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
        if skipmap is not None:
            cands = [tk for tk in cands if not skipmap(tk, t)]
        free = max(n_slots - len(pos), 0)
        cands.sort(key=lambda k: sortkey(k, t))
        for tk in cands[:free]:
            a = U[tk]
            px = a["Open"][t]
            mark = cash + sum(q["shares"] *
                              (U[k]["Open"][t] if U[k]["has_bar"][t]
                               else q["last_close"]) for k, q in pos.items())
            mult = sizemap(tk, t) if sizemap is not None else 1.0
            alloc = min(mark / n_slots * mult, cash)
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
            "med_trade_ret": (round(float(np.median(tr)), 6)
                              if len(tr) else np.nan),
            "p10_trade_ret": (round(float(np.percentile(tr, 10)), 6)
                              if len(tr) else np.nan),
            "p90_trade_ret": (round(float(np.percentile(tr, 90)), 6)
                              if len(tr) else np.nan)}


# ================================================================ run
print(f"loading universe from {DATADIR}/ (ASOF cap {ASOF})...")
U, cal, files = load_universe()
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
rk["Date"] = pd.to_datetime(rk["Date"])
rv = rk.set_index(["Date", "ticker"])["rank_vol"]
rank_of = {}
for t in range(1, len(cal)):
    day = cal[t - 1]
    for tk in cand_by_day[t]:
        v = rv.get((day, tk))
        if v is not None and v == v:
            rank_of[(tk, t)] = float(v)

print("\nSTAGE A: anchor pool (registered frame, 5 offsets pooled)...")
pool = build_pool([f for f in files
                   if os.path.basename(f)[:-4] not in ("SPY", "RSP")])
print(f"  {len(pool):,} completed innovations, "
      f"{pool.comp.min().date()} .. {pool.comp.max().date()}")

print("STAGE B: walk-forward monthly maps (completion-gated)...")
maps, thin_flags, month_of, starts = monthly_maps(pool, cal)
n_thin = sum(f.sum() for f in thin_flags)
print(f"  {len(maps)} monthly maps; thin cell-months (pooled fallback): {n_thin}")
mstab = [pd.Series(maps[i]).corr(pd.Series(maps[i + 1]), method="spearman")
         for i in range(len(maps) - 1)]
print(f"  consecutive-map lean rank-corr: median "
      f"{np.nanmedian(mstab):.3f}, min {np.nanmin(mstab):.3f}")


def lean_at(tk, t, mp=None):
    cell = U[tk]["cell_sig"][t]
    if cell < 0:
        return None
    m = (mp if mp is not None else maps)[month_of[t]]
    v = m[cell]
    return None if v != v else float(v)


def worst4_at(t, mp=None, th_list=None):
    m = (mp if mp is not None else maps)[month_of[t]]
    th = (th_list if th_list is not None else thin_flags)[month_of[t]]
    ok = [(m[c], c) for c in range(N_CELLS) if m[c] == m[c] and not th[c]]
    ok.sort()
    return {c for _, c in ok[:4]}


def key_vol(tk, t):
    r = rank_of.get((tk, t), 1e9)
    return (r, tk)


def key_tiebreak_factory(mp=None):
    def key(tk, t):
        r = rank_of.get((tk, t), 1e9)
        lean = lean_at(tk, t, mp)
        return ((r - 1) // VOL_BAND if r < 1e9 else 1e9,
                -(lean if lean is not None else -1e9),
                r, tk)
    return key


summary, all_daily, all_trades = [], [], []


def record(name, daily, trades, keep=True):
    m = metrics(daily, trades)
    summary.append({"run": name, **m})
    if keep:
        for d in daily:
            all_daily.append({"run": name, "date": str(d["date"].date()),
                              "equity": round(d["equity"], 2),
                              "n_pos": d["n_pos"]})
        for x in trades:
            all_trades.append({"run": name,
                               **{k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in x.items()
                                  if k not in ("last_close", "last_i")}})
    oos = m.get("oos_cagr", float("nan"))
    print(f"  {name:<16} final ${m['final_equity']:>11,.0f}  "
          f"cagr {100 * m['cagr']:>6.2f}%  shp {m['sharpe']:>6}  "
          f"dd {100 * m['maxdd']:>6.1f}%  med {100 * m['med_trade_ret']:>5.2f}%  "
          f"| 2022+ cagr {100 * oos:>6.2f}%")
    return m


print("\nSTAGE C: arms...")
m_base = record("RANKVOL_BENCH_N10",
                *run_account(U, cal, cand_by_day, N_SLOTS, key_vol,
                             bench_by_day=bench))
gap = abs(m_base["final_equity"] - PUBLISHED_FINAL) / PUBLISHED_FINAL
if gap > RECON_TOL:
    raise SystemExit(
        f"RECONCILIATION FAILED: ${m_base['final_equity']:,.0f} vs published "
        f"${PUBLISHED_FINAL:,.0f} ({100 * gap:.2f}% off). Arms not comparable.")
print(f"  reconciliation vs published: {100 * gap:.2f}% off -- PASS")

m_tie = record("LEAN_TIEBREAK",
               *run_account(U, cal, cand_by_day, N_SLOTS,
                            key_tiebreak_factory(), bench_by_day=bench))
m_flt = record("LEAN_FILTER",
               *run_account(U, cal, cand_by_day, N_SLOTS, key_vol,
                            bench_by_day=bench,
                            skipmap=lambda tk, t:
                            U[tk]["cell_sig"][t] in worst4_at(t)))


def size_mult_factory(mp):
    def f(tk, t):
        m = mp[month_of[t]]
        lean = lean_at(tk, t, mp)
        ok = sorted(v for v in m if v == v)
        if lean is None or not ok:
            return 1.0
        return 0.5 + np.searchsorted(ok, lean, side="right") / len(ok)
    return f


m_sz = record("LEAN_SIZER",
              *run_account(U, cal, cand_by_day, N_SLOTS, key_vol,
                           bench_by_day=bench,
                           sizemap=size_mult_factory(maps)))


def key_select(tk, t):
    lean = lean_at(tk, t)
    return (-(lean if lean is not None else -1e9),
            rank_of.get((tk, t), 1e9), tk)


m_sel = record("LEAN_SELECT",
               *run_account(U, cal, cand_by_day, N_SLOTS, key_select,
                            bench_by_day=bench))

print(f"\nSTAGE D: persistent-permutation envelopes "
      f"({ENV_SEEDS} seeds x 3 overlay arms)...")
env_rows = []
for s in range(ENV_SEEDS):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 9, s]))
    perm = rng.permutation(N_CELLS)
    # ONE meaningless geography, held for the whole decade (the real maps
    # are near-static, so the null's must be too)
    shuf = [m[perm] for m in maps]
    shuf_thin = [f[perm] for f in thin_flags]
    for arm, runner in (
        ("LEAN_TIEBREAK", lambda: run_account(
            U, cal, cand_by_day, N_SLOTS, key_tiebreak_factory(shuf),
            bench_by_day=bench)),
        ("LEAN_FILTER", lambda: run_account(
            U, cal, cand_by_day, N_SLOTS, key_vol, bench_by_day=bench,
            skipmap=lambda tk, t:
            U[tk]["cell_sig"][t] in worst4_at(t, shuf, shuf_thin))),
        ("LEAN_SIZER", lambda: run_account(
            U, cal, cand_by_day, N_SLOTS, key_vol, bench_by_day=bench,
            sizemap=size_mult_factory(shuf))),
    ):
        d, tr = runner()
        mm = metrics(d, tr)
        env_rows.append({"seed": s, "arm": arm,
                         "final_equity": mm["final_equity"],
                         "p10_trade_ret": mm["p10_trade_ret"],
                         "med_trade_ret": mm["med_trade_ret"],
                         "maxdd": mm["maxdd"]})
    if (s + 1) % 10 == 0:
        print(f"    {s + 1}/{ENV_SEEDS}")
env = pd.DataFrame(env_rows)
p95_p10 = {arm: float(np.percentile(g["p10_trade_ret"], 95))
           for arm, g in env.groupby("arm")}
for arm, thr in sorted(p95_p10.items()):
    g = env[env.arm == arm]
    print(f"  {arm:<16} null p10-trade: median "
          f"{100 * g['p10_trade_ret'].median():.2f}%, p95 {100 * thr:.2f}%")

print("\nDIAGNOSTICS: entry-cell distribution (baseline buys)...")
base_trades = [x for x in all_trades if x["run"] == "RANKVOL_BENCH_N10"
               and x["reason"] != "open"]
t_of_date = {str(d.date()): t for t, d in enumerate(cal)}
cells = []
for x in base_trades:
    t = t_of_date.get(x["entry_date"])
    if t is not None:
        c = U[x["ticker"]]["cell_sig"][t]
        if c >= 0:
            cells.append(c)
hist = pd.Series(cells).value_counts().sort_index()
for c, n in hist.items():
    zlab = ["z<-3", "-3..-2", "-2..-1", "-1..0",
            "0..1", "1..2", "2..3", "z>3"][c // 2]
    print(f"    cell {c:>2} ({zlab:>6} {'v>0' if c % 2 else 'v<=0'}): {n}")

print("\nVERDICTS (pre-registered tail-channel rule; median reported, "
      "never scored):")


def judge(name, m):
    ok_tail = m["p10_trade_ret"] > m_base["p10_trade_ret"]
    ok_dd = m["maxdd"] >= m_base["maxdd"]
    ok_g = (m["cagr"] >= m_base["cagr"] - 0.01
            and m.get("oos_cagr", -9) >= m_base.get("oos_cagr", 9) - 0.01)
    thr = p95_p10.get(name)
    ok_env = thr is not None and m["p10_trade_ret"] > thr
    tags = [f"tail(p10) {'+' if ok_tail else 'x'}",
            f"dd {'+' if ok_dd else 'x'}", f"growth {'+' if ok_g else 'x'}",
            f"envelope {'+' if ok_env else 'x'}"]
    v = ("ADOPT-CANDIDATE" if (ok_tail and ok_dd and ok_g and ok_env)
         else "not adopted")
    print(f"  {name:<16} {v}  ({', '.join(tags)})  "
          f"[p10 {100 * m['p10_trade_ret']:.2f}% vs base "
          f"{100 * m_base['p10_trade_ret']:.2f}%; median "
          f"{100 * m['med_trade_ret']:.2f}% reported]")


judge("LEAN_TIEBREAK", m_tie)
judge("LEAN_FILTER", m_flt)
judge("LEAN_SIZER", m_sz)
print("  (a pass certifies the cell geography is causal; skew-channel "
      "attribution needs the per-cell exhibit)")

# LEAN_SELECT is a full selector: the displacement bar, not the shape rule
sel_beats = (m_sel["cagr"] > m_base["cagr"]
             and m_sel.get("oos_cagr", -9) > m_base.get("oos_cagr", 9))
rand04 = os.path.join(OUTDIR, "04_sim_rand.csv")
sel_env = None
if os.path.exists(rand04):
    rr = pd.read_csv(rand04)
    rr = rr[rr.get("bench") == 1]
    if len(rr):
        sel_env = m_sel["final_equity"] > float(
            np.percentile(rr["final_equity"], 95))
if sel_beats and sel_env:
    v = "DISPLACES VOLUME (both windows + envelope)"
elif sel_beats and sel_env is None:
    v = "beats volume; envelope unavailable -- NOT displaceable this run"
elif sel_beats:
    v = "beats volume, fails the RAND envelope"
else:
    v = "does not beat volume"
print(f"  {'LEAN_SELECT':<16} {v}  (full selector bar)")

pd.DataFrame(summary).to_csv(os.path.join(OUTDIR, "07_c2_summary.csv"),
                             index=False)
pd.DataFrame(all_daily).to_csv(os.path.join(OUTDIR, "07_c2_daily.csv"),
                               index=False)
pd.DataFrame(all_trades).to_csv(os.path.join(OUTDIR, "07_c2_trades.csv"),
                                index=False)
env.to_csv(os.path.join(OUTDIR, "07_c2_rand_maps.csv"), index=False)
with open(os.path.join(OUTDIR, "07_c2_config.json"), "w") as f:
    json.dump({"VOL_BAND": VOL_BAND, "MIN_CELL_N": MIN_CELL_N,
               "ENV_SEEDS": ENV_SEEDS, "SEED": SEED, "ASOF": ASOF,
               "TOP_Q": TOP_Q, "COST_BPS": COST * 10000,
               "SPLIT": str(SPLIT.date()), "SIM_END": str(SIM_END.date()),
               "WORST_K": 4, "N_OFFSETS": N_OFFSETS, "K": K,
               "pass_channel": "p10_trade_ret",
               "null": "persistent-permutation per seed, all overlay arms"},
              f, indent=1)
print(f"\nwritten: {OUTDIR}/07_c2_summary.csv, _daily, _trades, "
      f"07_c2_maps.csv, 07_c2_rand_maps.csv")
print("NOTE: the mean is closed -- adoption was never about CAGR. The "
      "pass channel is the left tail of trade returns, registered above.")
