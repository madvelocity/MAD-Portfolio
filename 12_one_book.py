#!/usr/bin/env python3
"""
12_one_book.py -- THE ONE BOOK: RANKVOL_BENCH_N10 + velocity adds into
weakness + margin bridging, in a single account. ENGINEERING RUN.

WHAT THIS IS (and is not)
  This is the wiring check for the mad-velocity.io production engine --
  the deployed configuration, simulated over the full decade so the
  combined machine is verified before it runs live. It is NOT a
  pre-registered study: no bars, no nulls, no tiers, no verdict. The
  registered records for the components live in the master ledger
  (studies 06/08 for ADDS, study 11 for MARGIN_BRIDGE).

THE THREE ANCHORS (hard gates -- a miss means the engine is miswired)
  BASELINE     adds off, margin off  -> must equal $552,887.57 (8th recon)
  ADDS_ONLY    adds on,  margin off  -> must equal $754,818.05 (08 ADDS)
  BRIDGE_ONLY  adds off, margin on   -> must equal $971,305.16 (11 valid run)
  Only after all three reproduce exactly does ONE_BOOK's novel tape print.

ONE_BOOK mechanics (the deployed spec; composition rules stated)
  - Session order: exits -> paydown (implicit in cash) -> forced
    de-leveraging (broker rule: close above 130% gross forces newest-first
    whole-position sales at next open) -> ADDS (priority, 08 convention)
    -> BUYS in RANKVOL order.
  - Adds: held name, one add per position, trigger z<0 & dmad_5>0 &
    price_share_5>=0.50 at the prior close, full slot unit (equity/10),
    funded free-cash-first then margin, gross cap 130% pro forma. In
    ONE_BOOK mode the 08 units<10 gate on adds is lifted (margin finances
    the add); the cap is the bound. Adds allowed on bridged positions
    (uniform rule).
  - Buys: regular fill while units < 10 (an added position counts 2
    units, 08 convention); at units >= 10 a signal is admitted BRIDGED
    on margin; every entry 10% of net equity via the waterfall, cap
    130%.
  - Financing: EFFR_month + 150 bps on the drawn balance, /252,
    capitalized daily (study-11 registered table and conventions).

RUN (from the repo root)
  export PORTFOLIO_DATADIR=/path/to/prepped/data
  python3 -u 12_one_book.py     # 4 deterministic sims; minutes
  outputs results/12_one_book_summary.csv, 12_one_book_daily.csv,
          12_one_book_trades.csv, 12_one_book_adds.csv,
          12_one_book_config.json
"""
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

DATADIR = os.environ.get("PORTFOLIO_DATADIR", "data")
OUTDIR = "results"
RANKS_CSV = os.path.join(OUTDIR, "02_ranks.csv")
SCORES_CSV = os.path.join(OUTDIR, "03_scores.csv")
ASOF = "2026-07-16"
WIN_START = pd.Timestamp("2016-01-01")
SPLIT = pd.Timestamp("2022-01-01")
SIM_END = pd.Timestamp("2025-12-31")
SHARE_MIN = 0.50
N_SLOTS = 10
TOP_Q = 100
COST = 5.0 / 10000.0
CAPITAL = 100000.0
GROSS_CAP = 1.30
SPREAD = 0.0150
MAINT_TRIP = 0.30
DRIFT_TRIP = 0.05
ANCHOR_BASELINE = 552887.57
ANCHOR_ADDS = 754818.05
ANCHOR_BRIDGE = 971305.16
# locked data vintage (same pins as study 11; anchors were minted on it)
VINTAGE = {
    "02_ranks.csv": "4ff98459729390aaf544f3d70499bdb4d56b8af6cb970e0ec227ab3c13d1fd31",
    "03_scores.csv": "1b852b4a339d0b500776ac5b87fbc6295cdeb2db2ea74d826fec369697e65bec",
}
EFFR = {
    2016: [0.36, 0.37, 0.36, 0.37, 0.37, 0.38, 0.40, 0.40, 0.39, 0.40, 0.40, 0.54],
    2017: [0.66, 0.66, 0.79, 0.91, 0.91, 1.04, 1.16, 1.16, 1.15, 1.16, 1.16, 1.30],
    2018: [1.42, 1.42, 1.51, 1.69, 1.70, 1.82, 1.91, 1.91, 1.95, 2.19, 2.20, 2.27],
    2019: [2.40, 2.40, 2.40, 2.42, 2.39, 2.38, 2.40, 2.12, 2.05, 1.83, 1.55, 1.55],
    2020: [1.55, 1.58, 0.63, 0.05, 0.05, 0.08, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
    2021: [0.09, 0.08, 0.07, 0.07, 0.06, 0.08, 0.10, 0.09, 0.08, 0.08, 0.08, 0.08],
    2022: [0.08, 0.08, 0.20, 0.33, 0.76, 1.19, 1.65, 2.33, 2.58, 3.08, 3.76, 4.09],
    2023: [4.33, 4.57, 4.65, 4.83, 5.05, 5.08, 5.12, 5.33, 5.33, 5.33, 5.33, 5.33],
    2024: [5.33, 5.33, 5.33, 5.33, 5.33, 5.33, 5.33, 5.33, 5.13, 4.83, 4.65, 4.48],
    2025: [4.33, 4.33, 4.33, 4.33, 4.33, 4.33, 4.33, 4.33, 4.23, 4.08, 3.88, 3.73],
}
USECOLS = ["Date", "Open", "Close", "price_share_5", "dmad_5", "z",
           "sig_cross_up", "sig_rollback", "in_index"]
os.makedirs(OUTDIR, exist_ok=True)

with open(__file__, "rb") as f:
    SCRIPT_SHA256 = hashlib.sha256(f.read()).hexdigest()


def truthy(series):
    return series.astype(str).str.strip().isin(["True", "true", "1", "1.0"]).to_numpy()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


BASE_COLS = ["Date", "Open", "Close", "price_share_5",
             "sig_cross_up", "sig_rollback", "in_index"]


# ---------------------------------------------------------------- load
def load_universe():
    files = sorted(f for f in glob.glob(os.path.join(DATADIR, "*.csv"))
                   if not os.path.basename(f).startswith("_"))
    if not files:
        raise SystemExit(f"no prepped data in {DATADIR}/ -- set PORTFOLIO_DATADIR")
    frames, n_bad = {}, 0
    no_volume, missing_sig_cols = [], []
    for i, f in enumerate(files, 1):
        tk = os.path.basename(f)[:-4]
        try:
            df = pd.read_csv(f, usecols=USECOLS)
        except (ValueError, KeyError):
            # distinguish: readable by 04/11 but lacking z/dmad_5 narrows
            # the anchor universe -- that is a data fault, not skippable
            try:
                pd.read_csv(f, usecols=BASE_COLS, nrows=1)
                missing_sig_cols.append(tk)
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
        zz = df["z"].to_numpy(float)
        dm = df["dmad_5"].to_numpy(float)
        ps = df["price_share_5"].to_numpy(float)
        base = (dm > 0) & (ps >= SHARE_MIN)
        df["add_here"] = np.concatenate([[False], ((zz < 0) & base)[:-1]])
        df["prev_close"] = df["Close"].shift(1)
        ret1 = df["Close"] / df["prev_close"]
        gap_days = df.index.to_series().diff().dt.days
        bad = (ret1 > 5.0) | (ret1 < 0.2) | (gap_days > 30)
        df.loc[bad, "prev_close"] = np.nan
        df["member"] = truthy(df["in_index"])
        try:
            pd.read_csv(f, usecols=["Volume"], nrows=1)
        except (ValueError, KeyError):
            no_volume.append(tk)
        frames[tk] = df
        if i % 200 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] loaded")
    if missing_sig_cols:
        raise SystemExit(
            f"{len(missing_sig_cols)} files lack z/dmad_5 (e.g. "
            f"{', '.join(missing_sig_cols[:5])}) -- universe would narrow vs "
            "the 04/11 anchor loaders; fix the store, do not run")
    if no_volume:
        raise SystemExit(
            f"{len(no_volume)} files lack Volume (e.g. {', '.join(no_volume[:5])}) "
            "-- 01/02 rejected these; store vintage is wrong")
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
        for c in ["enter_here", "exit_here", "member", "add_here"]:
            a[c] = g[c].fillna(False).astype(bool).to_numpy()
        a["has_bar"] = has_bar
        a["last_t"] = int(np.nonzero(has_bar)[0][-1])
        U[tk] = a
    return U, cal


def load_rankvol(cal):
    if not os.path.exists(SCORES_CSV) or not os.path.exists(RANKS_CSV):
        raise SystemExit("02_ranks.csv / 03_scores.csv required")
    t_of = {str(d.date()): t for t, d in enumerate(cal)}
    sc = pd.read_csv(SCORES_CSV, usecols=["ticker", "entry_date", "sig_date"])
    rk = pd.read_csv(RANKS_CSV, usecols=["Date", "ticker", "rank_vol"])
    sc_sig = sc.merge(rk, left_on=["sig_date", "ticker"],
                      right_on=["Date", "ticker"], how="left")
    pick = {}
    for r in sc_sig.itertuples():
        t = t_of.get(r.entry_date)
        if t is not None and r.rank_vol == r.rank_vol:
            pick[(r.ticker, t)] = -float(r.rank_vol)      # rank 1 first
    print(f"  RANKVOL picker: {len(pick):,} scored pairs")
    return pick, rk


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


# ---------------------------------------------------------------- account
def run_book(U, cal, cand_by_day, pickmap, bench, rate_by_t,
             adds_on=False, margin_on=False, want_ledger=False):
    """One decade. Mode matrix:
      adds_on=F margin_on=F : incumbent verbatim   (anchor 552,887.57)
      adds_on=T margin_on=F : 08 ADDS verbatim     (anchor 754,818.05)
      adds_on=F margin_on=T : 11 bridge verbatim   (anchor 971,305.16)
      adds_on=T margin_on=T : ONE_BOOK (the novel tape)"""
    cash, pos = CAPITAL, {}
    daily, trades, ledger = [], [], []
    interest_paid = 0.0
    n_adds = n_adds_drawing_margin = n_forced_delev = 0
    n_skip_cap_adds = n_skip_cap_buys = 0
    delever_pending = False

    def units():
        return sum(1 + p.get("added", 0) for p in pos.values())

    def mark_now(t):
        return cash + sum(q["shares"] *
                          (U[k]["Open"][t] if U[k]["has_bar"][t]
                           else q["last_close"]) for k, q in pos.items())

    def long_mv(t):
        return sum(q["shares"] * (U[k]["Open"][t] if U[k]["has_bar"][t]
                                  else q["last_close"]) for k, q in pos.items())

    def do_adds(t):
        nonlocal cash, n_adds, n_adds_drawing_margin, n_skip_cap_adds
        for tk in sorted(pos):
            p = pos[tk]
            a = U[tk]
            if p.get("added") or not a["has_bar"][t] or not a["member"][t]:
                continue
            if t == 0 or not a["has_bar"][t - 1]:
                continue
            if a["prev_close"][t] != a["prev_close"][t]:
                continue
            if not margin_on and units() >= N_SLOTS:
                break                       # 08 gate (cash world only)
            if not a["add_here"][t]:
                continue
            px = a["Open"][t]
            if not np.isfinite(px):
                continue
            if not margin_on:
                alloc = min(mark_now(t) / N_SLOTS, cash)
                if alloc < 1.0:
                    continue
            else:
                mv = long_mv(t)
                equity = cash + mv
                alloc = equity / N_SLOTS
                if alloc < 1.0:
                    continue
                if (mv + alloc) > GROSS_CAP * equity:
                    n_skip_cap_adds += 1
                    continue
            sh_new = alloc / (px * (1 + COST))
            if want_ledger:
                ledger.append({"ticker": tk, "add_date": str(cal[t].date()),
                               "add_px": round(px, 4),
                               "add_dollars": round(alloc, 2),
                               "margin_dollars": round(
                                   max(alloc - max(cash, 0.0), 0.0), 2),
                               "drew_margin": bool(margin_on and cash < alloc),
                               "orig_entry": p["entry_date"]})
            if margin_on and cash < alloc:
                n_adds_drawing_margin += 1
            p["entry_px"] = ((p["entry_px"] * p["shares"] + px * sh_new)
                             / (p["shares"] + sh_new))
            p["shares"] += sh_new
            p["added"] = 1
            p["add_date"] = str(cal[t].date())
            cash -= alloc
            n_adds += 1

    def do_buys(t):
        nonlocal cash, n_skip_cap_buys
        cands = [tk for tk in cand_by_day[t]
                 if tk not in pos and tk in bench[t]]
        cands.sort(key=lambda k: (-pickmap.get((k, t), -np.inf), k))
        if not margin_on:
            free_units = N_SLOTS - units()
            for tk in cands:
                if free_units < 1:
                    break
                a = U[tk]
                px = a["Open"][t]
                alloc = min(mark_now(t) / N_SLOTS, cash)
                if alloc < 1.0:
                    continue
                shares = alloc / (px * (1 + COST))
                cash -= alloc
                pos[tk] = {"ticker": tk, "entry_date": str(cal[t].date()),
                           "entry_px": px, "shares": shares, "entry_t": t,
                           "last_close": a["Close"][t], "last_i": t,
                           "last_date": str(cal[t].date()), "bridged": False}
                free_units -= 1
        else:
            for tk in cands:
                a = U[tk]
                px = a["Open"][t]
                mv = long_mv(t)
                equity = cash + mv
                alloc = equity / N_SLOTS
                if alloc < 1.0:
                    continue
                bridged = units() >= N_SLOTS
                if (mv + alloc) > GROSS_CAP * equity:
                    n_skip_cap_buys += 1
                    continue
                shares = alloc / (px * (1 + COST))
                cash -= alloc
                pos[tk] = {"ticker": tk, "entry_date": str(cal[t].date()),
                           "entry_px": px, "shares": shares, "entry_t": t,
                           "last_close": a["Close"][t], "last_i": t,
                           "last_date": str(cal[t].date()), "bridged": bridged}

    for t in range(len(cal)):
        # exits at the open (house machinery, verbatim)
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
        # forced de-leveraging (broker rule; margin modes only)
        if margin_on and delever_pending:
            while pos:
                mv = long_mv(t)
                equity = cash + mv
                if equity <= 0 or mv <= GROSS_CAP * equity:
                    break
                vtk, vp = max(pos.items(),
                              key=lambda kv: (kv[1]["entry_t"], kv[0]))
                av = U[vtk]
                px = (av["Open"][t] if av["has_bar"][t]
                      and np.isfinite(av["Open"][t]) else vp["last_close"])
                cash += vp["shares"] * px * (1 - COST)
                trades.append({**vp, "exit_date": str(cal[t].date()),
                               "exit_px": px, "reason": "forced_deleverage"})
                del pos[vtk]
                n_forced_delev += 1
        delever_pending = False
        # adds first (08 priority convention), then buys
        if adds_on:
            do_adds(t)
        do_buys(t)
        # upkeep
        for tk, p in pos.items():
            a = U[tk]
            if a["has_bar"][t]:
                p["last_close"] = a["Close"][t]
                p["last_i"] = t
                p["last_date"] = str(cal[t].date())
        # interest on the close drawn balance, capitalized
        drawn = max(-cash, 0.0)
        if margin_on and drawn > 0:
            day_int = drawn * (rate_by_t[t] + SPREAD) / 252.0
            cash -= day_int
            interest_paid += day_int
        gross = sum(p["shares"] * p["last_close"] for p in pos.values())
        equity = cash + gross
        delever_pending = bool(margin_on and equity > 0
                               and gross > GROSS_CAP * equity)
        daily.append({"date": cal[t], "equity": equity, "cash": cash,
                      "n_pos": len(pos), "units": units(),
                      "drawn": max(-cash, 0.0), "gross": gross})
    for tk, p in list(pos.items()):
        trades.append({**p, "exit_date": "", "exit_px": p["last_close"],
                       "reason": "open"})
    extras = {"interest_paid": interest_paid, "n_adds": n_adds,
              "n_adds_drawing_margin": n_adds_drawing_margin,
              "n_forced_delev": n_forced_delev,
              "n_skip_cap_adds": n_skip_cap_adds,
              "n_skip_cap_buys": n_skip_cap_buys}
    return daily, trades, ledger, extras


# ---------------------------------------------------------------- metrics
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
            "avg_positions": round(float(npos.mean()), 2),
            "trades": len(closed),
            "win_rate": round(float((tr > 0).mean()), 3) if len(tr) else np.nan,
            "mean_trade_ret": (round(float(tr.mean()), 4) if len(tr) else np.nan),
            "med_trade_ret": (round(float(np.median(tr)), 4)
                              if len(tr) else np.nan)}


def margin_extras(daily):
    drawn = np.array([d["drawn"] for d in daily])
    gross = np.array([d["gross"] for d in daily])
    eq = np.array([d["equity"] for d in daily])
    with np.errstate(divide="ignore", invalid="ignore"):
        me = np.where(gross > 0, eq / gross, np.inf)
        ge = np.where(eq > 0, gross / eq, np.inf)
    fin = np.isfinite(ge)
    drift = float((ge > GROSS_CAP + 1e-9).mean())
    return {"avg_drawn": round(float(drawn.mean()), 2),
            "max_drawn": round(float(drawn.max()), 2),
            "worst_margin_to_equity": round(float(me.min()), 4),
            "gross_p50": round(float(np.percentile(ge[fin], 50)), 4),
            "gross_p95": round(float(np.percentile(ge[fin], 95)), 4),
            "gross_max": round(float(ge[fin].max()), 4),
            "pct_sessions_gross_gt_cap": round(100 * drift, 2),
            "trip_maintenance": bool(me.min() < MAINT_TRIP),
            "trip_drift": bool(drift > DRIFT_TRIP)}


# ---------------------------------------------------------------- run
print("verifying data vintage against the study-11 pins...")
vintage_now = {}
for name in VINTAGE:
    p = os.path.join(OUTDIR, name)
    vintage_now[name] = sha256_file(p) if os.path.exists(p) else "MISSING"
    print(f"  {name}: {vintage_now[name][:16]}...")
if any(VINTAGE[k] != vintage_now[k] for k in VINTAGE):
    raise SystemExit("VINTAGE MISMATCH -- the anchors were minted on the "
                     "pinned inputs; this is a data problem, not an engine "
                     "problem. Fix the store before running.")

print(f"loading universe from {DATADIR}/ (ASOF cap {ASOF})...")
U, cal = load_universe()
tickers = sorted(tk for tk in U if tk not in ("SPY", "RSP"))
print(f"{len(tickers)} names on a {len(cal)}-day calendar")

cand_by_day = [[] for _ in range(len(cal))]
for t in range(1, len(cal)):
    cand_by_day[t] = [tk for tk in tickers
                      if U[tk]["enter_here"][t] and U[tk]["member"][t]
                      and U[tk]["has_bar"][t - 1]
                      and np.isfinite(U[tk]["Open"][t])
                      and np.isfinite(U[tk]["Close"][t])]

print("loading picker + bench...")
pick, rk = load_rankvol(cal)
bench = volume_bench(rk, cal)
rate_by_t = np.array([EFFR[d.year][d.month - 1] / 100.0 for d in cal])

runs = {}
print("\n--- ANCHOR 1: BASELINE (adds off, margin off) ---")
runs["BASELINE"] = run_book(U, cal, cand_by_day, pick, bench, rate_by_t)
m = metrics(runs["BASELINE"][0], runs["BASELINE"][1])
print(f"  ${m['final_equity']:,.2f} vs anchor ${ANCHOR_BASELINE:,.2f}")
if abs(m["final_equity"] - ANCHOR_BASELINE) >= 0.005:
    raise SystemExit("ANCHOR 1 MISS -- engine miswired; nothing else runs")

print("--- ANCHOR 2: ADDS_ONLY (adds on, margin off) ---")
runs["ADDS_ONLY"] = run_book(U, cal, cand_by_day, pick, bench, rate_by_t,
                             adds_on=True)
m = metrics(runs["ADDS_ONLY"][0], runs["ADDS_ONLY"][1])
print(f"  ${m['final_equity']:,.2f} vs anchor ${ANCHOR_ADDS:,.2f}")
if abs(m["final_equity"] - ANCHOR_ADDS) >= 0.005:
    raise SystemExit("ANCHOR 2 MISS -- adds path diverges from 08; fix first")

print("--- ANCHOR 3: BRIDGE_ONLY (adds off, margin on) ---")
runs["BRIDGE_ONLY"] = run_book(U, cal, cand_by_day, pick, bench, rate_by_t,
                               margin_on=True)
m = metrics(runs["BRIDGE_ONLY"][0], runs["BRIDGE_ONLY"][1])
print(f"  ${m['final_equity']:,.2f} vs anchor ${ANCHOR_BRIDGE:,.2f}")
if abs(m["final_equity"] - ANCHOR_BRIDGE) >= 0.005:
    raise SystemExit("ANCHOR 3 MISS -- bridge path diverges from 11; fix first")

print("--- THE ONE BOOK (adds on, margin on) ---")
runs["ONE_BOOK"] = run_book(U, cal, cand_by_day, pick, bench, rate_by_t,
                            adds_on=True, margin_on=True, want_ledger=True)

summary, all_daily, all_trades = [], [], []
for name in ["BASELINE", "ADDS_ONLY", "BRIDGE_ONLY", "ONE_BOOK"]:
    daily, trades, ledger, ex = runs[name]
    row = {"run": name, **metrics(daily, trades)}
    if name in ("BRIDGE_ONLY", "ONE_BOOK"):
        row.update(margin_extras(daily))
    row.update({"interest_paid": round(ex["interest_paid"], 2),
                "n_adds": ex["n_adds"],
                "n_adds_drawing_margin": ex["n_adds_drawing_margin"],
                "n_forced_delev": ex["n_forced_delev"],
                "n_skip_cap_adds": ex["n_skip_cap_adds"],
                "n_skip_cap_buys": ex["n_skip_cap_buys"]})
    n_bridges = sum(1 for x in trades if x.get("bridged"))
    row["n_bridges"] = n_bridges
    summary.append(row)
    print(f"  {name:<12} final ${row['final_equity']:>12,.2f}  "
          f"cagr {100*row['cagr']:5.2f}%  oos_wx {row.get('oos_wealth_x')}  "
          f"dd {100*row['maxdd']:6.1f}%  adds {ex['n_adds']}  "
          f"bridges {n_bridges}  int ${ex['interest_paid']:,.0f}")
    for d in daily:
        all_daily.append({"run": name, "date": str(d["date"].date()),
                          "equity": round(d["equity"], 2),
                          "n_pos": d["n_pos"], "units": d["units"],
                          "drawn": round(d["drawn"], 2),
                          "gross": round(d["gross"], 2)})
    for x in trades:
        all_trades.append({"run": name,
                           **{k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in x.items()
                              if k not in ("last_close", "last_i", "entry_t")}})

# margin-safety flags on the novel tape, evaluated before anything ships
ob_mx = margin_extras(runs["ONE_BOOK"][0])
tripped = ob_mx["trip_maintenance"] or ob_mx["trip_drift"]
safety = ("TRIPWIRE -- ONE_BOOK tape breaches margin-safety flags "
          f"(maintenance<{MAINT_TRIP}: {ob_mx['trip_maintenance']}, "
          f"drift>{DRIFT_TRIP:.0%}: {ob_mx['trip_drift']}); "
          "NOT safe to wire live without addressing"
          if tripped else "clean (maintenance and drift flags clear)")
print(f"\nmargin safety: {safety}")

pd.DataFrame(summary).to_csv(os.path.join(OUTDIR, "12_one_book_summary.csv"),
                             index=False)
pd.DataFrame(all_daily).to_csv(os.path.join(OUTDIR, "12_one_book_daily.csv"),
                               index=False)
pd.DataFrame(all_trades).to_csv(os.path.join(OUTDIR, "12_one_book_trades.csv"),
                                index=False)
pd.DataFrame(runs["ONE_BOOK"][2],
             columns=["ticker", "add_date", "add_px", "add_dollars",
                      "margin_dollars", "drew_margin", "orig_entry"]).to_csv(
    os.path.join(OUTDIR, "12_one_book_adds.csv"), index=False)
store_manifest = {os.path.basename(p): sha256_file(p)
                  for p in sorted(glob.glob(os.path.join(DATADIR, "*.csv")))}
with open(os.path.join(OUTDIR, "12_one_book_config.json"), "w") as f:
    json.dump({"purpose": "ENGINEERING RUN -- wiring check for the "
                          "mad-velocity.io production engine; no bars, no "
                          "verdict; component records live in the master "
                          "ledger (06/08 ADDS, 11 MARGIN_BRIDGE)",
               "script_sha256": SCRIPT_SHA256,
               "vintage": vintage_now,
               "anchors": {"BASELINE": ANCHOR_BASELINE,
                           "ADDS_ONLY": ANCHOR_ADDS,
                           "BRIDGE_ONLY": ANCHOR_BRIDGE,
                           "all_reproduced_exactly": True},
               "margin_safety": safety,
               "one_book_rules": {
                   "adds_units_gate": "lifted under margin (cap bounds adds)",
                   "adds_on_bridged_positions": "allowed (uniform rule)",
                   "bridge_threshold": "units >= 10 (added position = 2 units)",
                   "order": "exits, de-lev, adds, buys",
                   "no_cooldown": "a force-sold name may re-enter at the "
                                  "same open on a fresh signal (documented, "
                                  "no rule added)",
                   "cap": GROSS_CAP, "spread_bps": 150,
                   "MAINT_TRIP": MAINT_TRIP, "DRIFT_TRIP": DRIFT_TRIP},
               "picker_note": "anchor 2 certifies engine+picker jointly; on "
                              "any vintage refresh re-run the picker "
                              "cross-check before trusting anchors",
               "summary": {s["run"]: s["final_equity"] for s in summary},
               "store_manifest_sha256": store_manifest},
              f, indent=1)
print(f"\nwrote results/12_one_book_summary.csv  daily  trades  adds  config")
print(f"script sha256: {SCRIPT_SHA256}")
if tripped:
    raise SystemExit(1)
