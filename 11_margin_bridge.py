#!/usr/bin/env python3
"""
11_margin_bridge.py -- Study 11: MARGIN_BRIDGE. Margin-funded admission of
slot-forgone signals on the incumbent RANKVOL_BENCH_N10.

PRE-REGISTRATION
  MARGIN_BRIDGE_PROTOCOL.md (sha 977588b7..., locked before this script
  existed; independently reviewed pre-lock; all findings patched).
  Family: funding/capacity, look 1. ONE registered look, on arm A.

ARMS
  BASELINE  study-11 engine, bridge off, margin permanently zero; MUST
            reconcile $552,887.57 to the cent (7th) before any other arm
            simulates. Gate result written to config first. Re-running
            04_sim.py does not count.
  A         MARGIN_BRIDGE: signal at occupancy >= 10 admitted anyway at
            10% of net equity; free cash first, margin remainder; gross
            cap 130% pro forma at admission; exits pay debt first
            (cash representation: proceeds raise negative cash); interest
            compounds daily at EFFR_month + 150 bps on the close drawn
            balance.
  N         200-seed random-name bridge null (SeedSequence([7,16,s,0])),
            escalation to 500 if A >= p90 at 200 (same stream, no
            re-rolls). Full self-contained decades: regular book per the
            engine on its own state; bridge sessions pinned to A's
            bridge-tag list, names drawn uniformly (without replacement
            within session) from the session's bench-eligible set,
            excluding names A bridged that session and names already in
            the null's own book; drawn names run the FULL house exit
            machinery on their own tapes. Own cap can SKIP a pinned
            admission (logged).
  C         SPY financing-matched DISCLOSURE (not an arm of record; no
            verdict language may cite it): SPY bought at A's bridged
            admissions, closed at A's paired exit sessions.

BARS (evaluated in registered tier order)
  1. ADVERSE SELECTION: A <= p5 of N (full window).
  2. LIVE CANDIDATE: A > BASELINE on full final equity AND oos_wealth_x
     (strict), AND A >= p95 of N.
  3. SELECTION CONTENT NOT ESTABLISHED: bar 1 only.
  4. DEAD: anything else.

RUN (from the repo root)
  export PORTFOLIO_DATADIR=/path/to/prepped/data
  nohup python3 -u 11_margin_bridge.py > 11_run.log 2>&1 &
  Outputs: results/11_bridge_summary.csv, 11_bridge_daily.csv,
           11_bridge_trades.csv, 11_bridge_envelopes.csv,
           11_bridge_config.json
"""
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

# ---- registered constants (hardcoded; SIM_* presence recorded, inert -- Amd 01)
DATADIR = os.environ.get("PORTFOLIO_DATADIR", "data")
OUTDIR = "results"
RANKS_CSV = os.path.join(OUTDIR, "02_ranks.csv")
SCORES_CSV = os.path.join(OUTDIR, "03_scores.csv")
ASOF = "2026-07-16"
WIN_START = pd.Timestamp("2016-01-01")
SPLIT = pd.Timestamp("2022-01-01")
SIM_END = pd.Timestamp("2025-12-31")
SHARE_MIN = 0.50
SEED = 7
STUDY = 16  # registered seed-stream id (fixed at pre-registration; independent of script numbering)
N_SLOTS = 10
TOP_Q = 100
COST = 5.0 / 10000.0
CAPITAL = 100000.0
PUBLISHED_FINAL = 552887.57
GROSS_CAP = 1.30
SPREAD = 0.0150
SPREAD_SENS = 0.0400
N_SEEDS_SCREEN = 200
N_SEEDS_FULL = 500
ESCALATE_AT_P = 90
MAINT_TRIP = 0.30
DRIFT_TRIP = 0.05
BOOT_B = 2000
BOOT_BLOCK = 3.0
USECOLS = ["Date", "Open", "Close", "price_share_5",
           "sig_cross_up", "sig_rollback", "in_index"]
# locked data vintage (PROTOCOL_LOCK.json study_13 entry (registration id))
# -- must match at run
VINTAGE = {
    "02_ranks.csv": "4ff98459729390aaf544f3d70499bdb4d56b8af6cb970e0ec227ab3c13d1fd31",
    "03_scores.csv": "1b852b4a339d0b500776ac5b87fbc6295cdeb2db2ea74d826fec369697e65bec",
    "spy_cache.csv": "c18e22e75995ba7a2f111b65212b854ac5c473aa1f229cdea9989d8069a73aa2",
}
# EFFR_month registered table (%, NY Fed daily EFFR monthly means)
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
os.makedirs(OUTDIR, exist_ok=True)

with open(__file__, "rb") as f:
    SCRIPT_SHA256 = hashlib.sha256(f.read()).hexdigest()

sim_env = {k: v for k, v in os.environ.items() if k.startswith("SIM_")}
if sim_env:
    # amendment 01: presence is recorded and INERT (all study-11 constants
    # are hardcoded; the void clause applies to actual overrides only,
    # which are impossible here and detectable via script_sha256)
    print(f"NOTE: SIM_* variables present but INERT; recorded: {sim_env}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def truthy(series):
    return series.astype(str).str.strip().isin(["True", "true", "1", "1.0"]).to_numpy()


# ---------------------------------------------------------------- load
# (load_universe / load_pickers / volume_bench / spy_series verbatim from the
# audited 04_sim.py, constants hardcoded)
def load_universe():
    files = sorted(f for f in glob.glob(os.path.join(DATADIR, "*.csv"))
                   if not os.path.basename(f).startswith("_"))
    if not files:
        raise SystemExit(f"no prepped data in {DATADIR}/ -- set PORTFOLIO_DATADIR")
    frames = {}
    n_bad = 0
    no_volume = []
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


def load_pickers(cal):
    if not os.path.exists(SCORES_CSV):
        raise SystemExit(f"{SCORES_CSV} not found")
    if not os.path.exists(RANKS_CSV):
        raise SystemExit(f"{RANKS_CSV} not found")
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
    print(f"  RANKVOL picker: {len(pick):,} scored (ticker, day) pairs")
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


def spy_series(cal):
    for path in [os.path.join(OUTDIR, "spy_cache.csv"),
                 os.path.join(DATADIR, "SPY.csv")]:
        if os.path.exists(path):
            df = pd.read_csv(path).dropna(subset=["Open", "Close"])
            df["Date"] = pd.to_datetime(df["Date"])
            df = df[df["Date"] <= pd.Timestamp(ASOF)].set_index("Date")
            s = df.reindex(cal)[["Open", "Close"]]
            if s["Close"].notna().mean() > 0.9:
                print(f"  SPY from {path}")
                return s
    raise SystemExit("SPY series unavailable (results/spy_cache.csv or "
                     "DATADIR/SPY.csv required for arm C)")


# ---------------------------------------------------------------- the account
def run_margin_account(U, cal, cand_by_day, pickmap, bench_by_day, rate_by_t,
                       margin_on=False, spread=SPREAD,
                       pinned=None, a_bridge_names_by_t=None, rng=None,
                       spy=None, spy_events=None):
    """One decade. margin_on=False reproduces the incumbent byte-exactly
    (waterfall reduces to the min-with-cash clamp incl. the alloc<$1 skip;
    zero interest). margin_on=True implements the registered section-2
    mechanics. pinned = {t: count} of A's bridge events per session (arms
    N/C); a_bridge_names_by_t = {t: set of names A bridged} (excluded from
    null draws); rng draws null names; spy/spy_events drive arm C's SPY
    bridges (spy_events = list of (t_entry, t_exit_or_None))."""
    cash, pos = CAPITAL, {}
    daily, trades = [], []
    interest_paid = 0.0
    n_skip_cap, n_degenerate, n_collision, n_skip_cap_own = 0, 0, 0, 0
    n_forced_delev = 0
    delever_pending = False
    spy_book = []      # arm C: [{shares, t_exit, entry stuff}]
    # sort by admission session only: t_out may be None (open at SIM_END)
    # and int-vs-None comparison raises on tied t_in
    spy_ev = (sorted(spy_events, key=lambda e: e[0])
              if spy_events is not None else [])
    spy_ei = 0

    def long_mv(t):
        mv = sum(q["shares"] * (U[k]["Open"][t] if U[k]["has_bar"][t]
                                else q["last_close"]) for k, q in pos.items())
        if spy is not None:
            op = spy["Open"][t]
            for b in spy_book:
                mv += b["shares"] * (op if np.isfinite(op) else b["last_close"])
        return mv

    for t in range(len(cal)):
        # (1) exits at the open -- verbatim house machinery
        for tk in list(pos):
            a = U[tk]
            p = pos[tk]
            if not a["has_bar"][t]:
                if t > a["last_t"] or t - p.get("last_i", t) > 10:
                    cash += p["shares"] * p["last_close"] * (1 - COST)
                    trades.append({**p, "exit_date": p["last_date"],
                                   "exit_t": t,     # cash-recognition session
                                   "exit_px": p["last_close"],
                                   "reason": "delisted"})
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
        # arm C: close SPY bridges paired to A's exits
        if spy is not None:
            for b in list(spy_book):
                if b["t_exit"] is not None and t >= b["t_exit"]:
                    px = spy["Open"][t] if np.isfinite(spy["Open"][t]) \
                        else b["last_close"]
                    cash += b["shares"] * px * (1 - COST)
                    trades.append({"ticker": "SPY_BRIDGE",
                                   "entry_date": b["entry_date"],
                                   "entry_px": b["entry_px"],
                                   "shares": b["shares"],
                                   "exit_date": str(cal[t].date()),
                                   "exit_px": px, "reason": "paired",
                                   "last_date": str(cal[t].date()),
                                   "bridged": True})
                    spy_book.remove(b)
        # (2) paydown is implicit: proceeds raised cash toward or past zero

        # (2.5) amendment 02: forced de-leveraging -- if the PRIOR close
        # drifted gross above the cap, force-sell whole positions
        # newest-first at this open until compliant. Arms A and N only
        # (C's SPY closes are registration-paired; spy is not None there).
        if margin_on and delever_pending and spy is None:
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

        # (3) admissions in RANKVOL order
        cands = [tk for tk in cand_by_day[t] if tk not in pos]
        if bench_by_day is not None:
            cands = [tk for tk in cands if tk in bench_by_day[t]]
        cands.sort(key=lambda k: (-pickmap.get((k, t), -np.inf), k))
        # registered collision disclosure: a held null-drawn bridge name
        # blocking regular re-entry (side-effect count only; filter above
        # is byte-identical to the incumbent)
        if margin_on and pinned is not None:
            n_collision += sum(
                1 for tk in cand_by_day[t]
                if tk in pos and pos[tk].get("bridged")
                and (bench_by_day is None or tk in bench_by_day[t]))
        if not margin_on:
            free = max(N_SLOTS - len(pos), 0)
            for tk in cands[:free]:
                a = U[tk]
                px = a["Open"][t]
                mark = cash + sum(q["shares"] *
                                  (U[k]["Open"][t] if U[k]["has_bar"][t]
                                   else q["last_close"]) for k, q in pos.items())
                alloc = min(mark / N_SLOTS, cash)
                if alloc < 1.0:
                    continue
                shares = alloc / (px * (1 + COST))
                cash -= alloc
                pos[tk] = {"ticker": tk, "entry_date": str(cal[t].date()),
                           "entry_px": px, "shares": shares, "entry_t": t,
                           "last_close": a["Close"][t], "last_i": t,
                           "last_date": str(cal[t].date()), "bridged": False}
        else:
            for tk in cands:
                a = U[tk]
                px = a["Open"][t]
                mv = long_mv(t)
                equity = cash + mv
                alloc = equity / N_SLOTS
                if alloc < 1.0:
                    continue
                # amendment 01: SPY bridges count toward the 10 (arm C only;
                # spy_book is empty in every other arm)
                bridged = (len(pos) + len(spy_book)) >= N_SLOTS
                if bridged and pinned is not None:
                    continue          # arms N/C: no self-generated bridges
                if (mv + alloc) > GROSS_CAP * equity:
                    n_skip_cap_own += 1   # own-book cap skip (disclosure)
                    continue          # dropped whole; next candidate tested
                shares = alloc / (px * (1 + COST))
                cash -= alloc
                pos[tk] = {"ticker": tk, "entry_date": str(cal[t].date()),
                           "entry_px": px, "shares": shares, "entry_t": t,
                           "last_close": a["Close"][t], "last_i": t,
                           "last_date": str(cal[t].date()), "bridged": bridged}
            # pinned bridge admissions (arms N and C)
            n_pin = pinned.get(t, 0) if pinned is not None else 0
            if n_pin and spy is None:          # arm N: random-name bridges
                excl = a_bridge_names_by_t.get(t, set())
                used = set()
                for _ in range(n_pin):
                    elig = sorted(
                        tk for tk in cand_universe_by_day[t]
                        if tk not in pos and tk not in excl and tk not in used)
                    mv = long_mv(t)
                    equity = cash + mv
                    alloc = equity / N_SLOTS
                    if (mv + alloc) > GROSS_CAP * equity or alloc < 1.0:
                        n_skip_cap += 1
                        continue
                    if elig:
                        tk = elig[int(rng.integers(len(elig)))]
                    else:
                        # registered: empty eligible set keeps A's name
                        fallback = sorted(tk for tk in excl
                                          if tk not in pos and tk not in used)
                        if not fallback:
                            n_degenerate += 1
                            continue
                        tk = fallback[0]
                        n_degenerate += 1
                    used.add(tk)
                    a = U[tk]
                    px = a["Open"][t]
                    shares = alloc / (px * (1 + COST))
                    cash -= alloc
                    pos[tk] = {"ticker": tk, "entry_date": str(cal[t].date()),
                               "entry_px": px, "shares": shares, "entry_t": t,
                               "last_close": a["Close"][t], "last_i": t,
                               "last_date": str(cal[t].date()), "bridged": True}
            if spy is not None:                # arm C: SPY bridges
                while spy_ei < len(spy_ev) and spy_ev[spy_ei][0] == t:
                    _, t_exit = spy_ev[spy_ei]
                    spy_ei += 1
                    op = spy["Open"][t]
                    if not np.isfinite(op):
                        n_degenerate += 1
                        continue
                    mv = long_mv(t)
                    equity = cash + mv
                    alloc = equity / N_SLOTS
                    if (mv + alloc) > GROSS_CAP * equity or alloc < 1.0:
                        n_skip_cap += 1
                        continue
                    shares = alloc / (op * (1 + COST))
                    cash -= alloc
                    spy_book.append({"shares": shares, "t_exit": t_exit,
                                     "entry_date": str(cal[t].date()),
                                     "entry_px": op,
                                     "last_close": spy["Close"][t]
                                     if np.isfinite(spy["Close"][t]) else op})
        # position mark upkeep
        for tk, p in pos.items():
            a = U[tk]
            if a["has_bar"][t]:
                p["last_close"] = a["Close"][t]
                p["last_i"] = t
                p["last_date"] = str(cal[t].date())
        if spy is not None:
            for b in spy_book:
                if np.isfinite(spy["Close"][t]):
                    b["last_close"] = spy["Close"][t]
        # (4) interest on the close drawn balance, capitalized
        drawn = max(-cash, 0.0)
        if margin_on and drawn > 0:
            day_int = drawn * (rate_by_t[t] + spread) / 252.0
            cash -= day_int
            interest_paid += day_int
        # (5) mark at the close
        gross = sum(p["shares"] * p["last_close"] for p in pos.values()) \
            + sum(b["shares"] * b["last_close"] for b in spy_book)
        equity = cash + gross
        delever_pending = bool(margin_on and equity > 0
                               and gross > GROSS_CAP * equity)
        daily.append({"date": cal[t], "equity": equity, "cash": cash,
                      "n_pos": len(pos) + len(spy_book),
                      "drawn": max(-cash, 0.0), "gross": gross})
    for tk, p in list(pos.items()):
        trades.append({**p, "exit_date": "", "exit_px": p["last_close"],
                       "reason": "open"})
    for b in spy_book:
        trades.append({"ticker": "SPY_BRIDGE", "entry_date": b["entry_date"],
                       "entry_px": b["entry_px"], "shares": b["shares"],
                       "exit_date": "", "exit_px": b["last_close"],
                       "reason": "open", "last_date": str(cal[-1].date()),
                       "bridged": True})
    extras = {"interest_paid": interest_paid, "n_skip_cap": n_skip_cap,
              "n_degenerate": n_degenerate, "n_collision": n_collision,
              "n_skip_cap_own": n_skip_cap_own,
              "n_forced_delev": n_forced_delev}
    return daily, trades, extras


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
    cashv = np.array([d.get("cash", 0.0) for d in daily])
    closed = [x for x in trades if x["reason"] != "open"]
    tr = np.array([x["exit_px"] * (1 - COST) / (x["entry_px"] * (1 + COST)) - 1
                   for x in closed]) if closed else np.array([])
    full = window_stats(dates, eq)
    late = {f"oos_{k}": v for k, v in
            window_stats(dates, eq, np.datetime64(SPLIT)).items()}
    # exposure-adjusted disclosure block (registered, section 4)
    r = np.diff(eq) / eq[:-1]
    inv = npos[1:] > 0
    r_in = r[inv]
    in_sharpe = (np.sqrt(252) * r_in.mean() / r_in.std(ddof=1)
                 if len(r_in) > 2 and r_in.std(ddof=1) > 0 else np.nan)
    deployed_cagr = ((1 + r_in.mean()) ** 252 - 1 if len(r_in) > 2 else np.nan)
    return {"final_equity": round(float(eq[-1]), 2), **full, **late,
            "pct_days_invested": round(100 * float((npos > 0).mean()), 1),
            "avg_positions": round(float(npos.mean()), 2),
            "invested_frac": round(float(np.mean(1.0 - cashv / eq)), 4),
            "deployed_cagr": (round(float(deployed_cagr), 4)
                              if deployed_cagr == deployed_cagr else np.nan),
            "in_trade_sharpe": (round(float(in_sharpe), 3)
                                if in_sharpe == in_sharpe else np.nan),
            "in_trade_day_mean": (round(float(r_in.mean()), 6)
                                  if len(r_in) else np.nan),
            "in_trade_day_median": (round(float(np.median(r_in)), 6)
                                    if len(r_in) else np.nan),
            "trades": len(closed),
            "win_rate": round(float((tr > 0).mean()), 3) if len(tr) else np.nan,
            "mean_trade_ret": (round(float(tr.mean()), 4)
                               if len(tr) else np.nan),
            "med_trade_ret": (round(float(np.median(tr)), 4)
                              if len(tr) else np.nan)}


def margin_extras(daily):
    drawn = np.array([d["drawn"] for d in daily])
    gross = np.array([d["gross"] for d in daily])
    eq = np.array([d["equity"] for d in daily])
    with np.errstate(divide="ignore", invalid="ignore"):
        me = np.where(gross > 0, eq / gross, np.inf)
        ge = np.where(eq > 0, gross / eq, np.inf)
    drift = float((ge > GROSS_CAP + 1e-9).mean())
    return {"avg_drawn": round(float(drawn.mean()), 2),
            "max_drawn": round(float(drawn.max()), 2),
            "worst_margin_to_equity": round(float(me.min()), 4),
            "gross_p50": round(float(np.percentile(ge[np.isfinite(ge)], 50)), 4),
            "gross_p95": round(float(np.percentile(ge[np.isfinite(ge)], 95)), 4),
            "gross_max": round(float(ge[np.isfinite(ge)].max()), 4),
            "pct_sessions_gross_gt_cap": round(100 * drift, 2),
            "trip_maintenance": bool(me.min() < MAINT_TRIP),
            "trip_drift": bool(drift > DRIFT_TRIP)}


def trade_stats(trades, cal_idx):
    closed = [x for x in trades if x["reason"] != "open"]
    if not closed:
        return {"n": 0}
    tr = np.array([x["exit_px"] * (1 - COST) / (x["entry_px"] * (1 + COST)) - 1
                   for x in closed])
    hold = np.array([cal_idx.get(x["exit_date"], 0)
                     - cal_idx.get(x["entry_date"], 0) for x in closed])
    return {"n": len(closed), "mean_ret": round(float(tr.mean()), 4),
            "med_ret": round(float(np.median(tr)), 4),
            "win_rate": round(float((tr > 0).mean()), 3),
            "mean_hold_sessions": round(float(hold.mean()), 1),
            "med_hold_sessions": float(np.median(hold))}


def boot_two_sample_p(x, y):
    """two-sample stationary bootstrap on the mean difference (diagnostic
    only; registered seed [7,16,1000,0])."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5 or len(y) < 5:
        return np.nan
    obs = x.mean() - y.mean()
    xc, yc = x - x.mean(), y - y.mean()
    rng = np.random.default_rng(np.random.SeedSequence([SEED, STUDY, 1000, 0]))
    p_geo = 1.0 / BOOT_BLOCK
    hits = 0
    for _ in range(BOOT_B):
        def resample(z):
            n = len(z)
            idx = np.empty(n, dtype=int)
            i0 = rng.integers(n)
            for j in range(n):
                if j > 0 and rng.random() >= p_geo:
                    i0 = (i0 + 1) % n
                elif j > 0:
                    i0 = rng.integers(n)
                idx[j] = i0
            return z[idx]
        d = resample(xc).mean() - resample(yc).mean()
        if abs(d) >= abs(obs):
            hits += 1
    return (hits + 1.0) / (BOOT_B + 1.0)


# ---------------------------------------------------------------- run
print("verifying data vintage against the lock...")
vintage_now = {}
for name in ["02_ranks.csv", "03_scores.csv", "spy_cache.csv"]:
    p = os.path.join(OUTDIR, name)
    vintage_now[name] = sha256_file(p) if os.path.exists(p) else "MISSING"
    print(f"  {name}: {vintage_now[name][:16]}...")
vintage_ok = all(VINTAGE[k] == vintage_now[k] for k in VINTAGE)
if not vintage_ok:
    print("  VINTAGE MISMATCH vs lock -- run voids unless a pre-run amendment "
          "re-pins the vintage (registered relief path).")
    raise SystemExit(json.dumps({"locked": VINTAGE, "found": vintage_now},
                                indent=1))

print(f"loading universe from {DATADIR}/ (ASOF cap {ASOF})...")
U, cal = load_universe()
tickers = sorted(tk for tk in U if tk not in ("SPY", "RSP"))
print(f"{len(tickers)} names on a {len(cal)}-day calendar")
cal_idx = {str(d.date()): t for t, d in enumerate(cal)}

cand_by_day = [[] for _ in range(len(cal))]
for t in range(1, len(cal)):
    cand_by_day[t] = [tk for tk in tickers
                      if U[tk]["enter_here"][t] and U[tk]["member"][t]
                      and U[tk]["has_bar"][t - 1]
                      and np.isfinite(U[tk]["Open"][t])
                      and np.isfinite(U[tk]["Close"][t])]

print("loading picker + bench...")
pick, rk = load_pickers(cal)
bench = volume_bench(rk, cal)

# null eligible universe per day: bench-eligible, in-index, bar at t-1,
# finite O/C at t (the registered N pool; candidacy minus the entry signal)
cand_universe_by_day = [[] for _ in range(len(cal))]
for t in range(1, len(cal)):
    cand_universe_by_day[t] = [
        tk for tk in tickers
        if tk in bench[t] and U[tk]["member"][t] and U[tk]["has_bar"][t - 1]
        and np.isfinite(U[tk]["Open"][t]) and np.isfinite(U[tk]["Close"][t])]

# EFFR per session (decimal)
rate_by_t = np.array([EFFR[d.year][d.month - 1] / 100.0 for d in cal])

# ---- reconciliation gate (before ANY other arm simulates)
print("\n--- BASELINE (reconciliation gate) ---")
b_daily, b_trades, _ = run_margin_account(
    U, cal, cand_by_day, pick, bench, rate_by_t, margin_on=False)
b_m = metrics(b_daily, b_trades)
gate_pass = abs(b_m["final_equity"] - PUBLISHED_FINAL) < 0.005   # to the cent
within_tol = abs(b_m["final_equity"] - 552888.0) <= 0.015 * 552888.0
print(f"  BASELINE final ${b_m['final_equity']:,.2f}  "
      f"target ${PUBLISHED_FINAL:,.2f}  gate={'PASS' if gate_pass else 'FAIL'}")
gate_record = {"baseline_final": b_m["final_equity"],
               "target": PUBLISHED_FINAL, "gate_pass": bool(gate_pass),
               "within_house_tolerance": bool(within_tol)}
with open(os.path.join(OUTDIR, "11_bridge_config.json"), "w") as f:
    json.dump({"script_sha256": SCRIPT_SHA256, "gate": gate_record,
               "vintage": vintage_now, "status": "gate-only"}, f, indent=1)
if not gate_pass:
    msg = ("cent-exact FAIL but within 1.5% house tolerance -- halt for "
           "pre-run amendment" if within_tol else
           "reconciliation FAIL beyond house tolerance -- wrong data/env")
    raise SystemExit(f"GATE: {msg}. No arm was simulated; nothing consumed.")

# ---- arm A
print("\n--- ARM A: MARGIN_BRIDGE ---")
a_daily, a_trades, a_ex = run_margin_account(
    U, cal, cand_by_day, pick, bench, rate_by_t, margin_on=True)
a_m = metrics(a_daily, a_trades)
a_mx = margin_extras(a_daily)
a_bridges = [x for x in a_trades if x.get("bridged")]
a_regular = [x for x in a_trades if not x.get("bridged")]
print(f"  A final ${a_m['final_equity']:,.2f}  oos_wx {a_m.get('oos_wealth_x')}  "
      f"bridges {len(a_bridges)}  interest ${a_ex['interest_paid']:,.0f}  "
      f"maxdrawn ${a_mx['max_drawn']:,.0f}")

# A's bridge-tag list drives N and C
pinned = {}
a_names_by_t = {}
spy_events = []
for x in a_bridges:
    t_in = cal_idx[x["entry_date"]]
    pinned[t_in] = pinned.get(t_in, 0) + 1
    a_names_by_t.setdefault(t_in, set()).add(x["ticker"])
    # amendment 01: delisted A-bridges pair C's close to the CASH-RECOGNITION
    # session (exit_t), not the backdated last-bar ledger date
    if x["exit_date"]:
        t_out = (x.get("exit_t") if x.get("exit_t") is not None
                 else cal_idx.get(x["exit_date"]))
    else:
        t_out = None
    spy_events.append((t_in, t_out))
print(f"  bridge sessions: {len(pinned)}; events: {len(spy_events)}")

# amendment 01: vacuous-run guard -- zero bridge events means no bar can be
# evaluated; the look is NOT consumed; N/C/sensitivity are not simulated
if not a_bridges:
    pd.DataFrame([{"run": "BASELINE", **b_m},
                  {"run": "MARGIN_BRIDGE", **a_m, **a_mx}]).to_csv(
        os.path.join(OUTDIR, "11_bridge_summary.csv"), index=False)
    vac_daily = []
    for name, d in [("BASELINE", b_daily), ("MARGIN_BRIDGE", a_daily)]:
        for row in d:
            vac_daily.append({"run": name, "date": str(row["date"].date()),
                              "equity": round(row["equity"], 2),
                              "n_pos": row["n_pos"],
                              "drawn": round(row["drawn"], 2),
                              "gross": round(row["gross"], 2)})
    pd.DataFrame(vac_daily).to_csv(os.path.join(OUTDIR, "11_bridge_daily.csv"),
                                   index=False)
    vac_trades = [{"run": n, **{k: v for k, v in x.items()
                                if k not in ("last_close", "last_i", "exit_t", "entry_t")}}
                  for n, tr in [("BASELINE", b_trades),
                                ("MARGIN_BRIDGE", a_trades)] for x in tr]
    pd.DataFrame(vac_trades).to_csv(os.path.join(OUTDIR, "11_bridge_trades.csv"),
                                    index=False)
    with open(os.path.join(OUTDIR, "11_bridge_config.json"), "w") as f:
        json.dump({"script_sha256": SCRIPT_SHA256, "gate": gate_record,
                   "vintage": vintage_now, "sim_env_overrides": sim_env,
                   "status": ("VACUOUS -- arm A produced zero bridge events; "
                              "N, C, and the EFFR+400 sensitivity not "
                              "simulated; no tier awarded; the registered "
                              "look is NOT consumed (no bar evaluated)")},
                  f, indent=1)
    raise SystemExit("VACUOUS: zero bridges admitted; see 11_bridge_config.json")

# ---- A at EFFR+400 (registered sensitivity disclosure)
s_daily, s_trades, s_ex = run_margin_account(
    U, cal, cand_by_day, pick, bench, rate_by_t, margin_on=True,
    spread=SPREAD_SENS)
s_m = metrics(s_daily, s_trades)
print(f"  A @ EFFR+400: final ${s_m['final_equity']:,.2f}  "
      f"interest ${s_ex['interest_paid']:,.0f}")

# ---- arm C (disclosure only)
print("\n--- ARM C: SPY financing-matched disclosure ---")
spy = spy_series(cal)
spy_arr = {"Open": spy["Open"].to_numpy(float),
           "Close": spy["Close"].ffill().to_numpy(float)}
c_daily, c_trades, c_ex = run_margin_account(
    U, cal, cand_by_day, pick, bench, rate_by_t, margin_on=True,
    pinned={}, spy=spy_arr, spy_events=spy_events)
c_m = metrics(c_daily, c_trades)
c_mx = margin_extras(c_daily)
print(f"  C final ${c_m['final_equity']:,.2f}  oos_wx {c_m.get('oos_wealth_x')}")

# ---- arm N envelope
print(f"\n--- ARM N: random-name bridge null ({N_SEEDS_SCREEN} seeds, "
      f"escalate to {N_SEEDS_FULL} at p{ESCALATE_AT_P}) ---")
env_rows = []


def run_null_seed(s):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, STUDY, s, 0]))
    d, tr, ex = run_margin_account(
        U, cal, cand_by_day, pick, bench, rate_by_t, margin_on=True,
        pinned=pinned, a_bridge_names_by_t=a_names_by_t, rng=rng)
    m = metrics(d, tr)
    mx = margin_extras(d)
    n_admitted = sum(1 for x in tr if x.get("bridged"))
    return {"seed": s, "final_equity": m["final_equity"],
            "oos_wealth_x": m.get("oos_wealth_x", np.nan),
            "n_bridges_admitted": n_admitted,
            "n_skip_cap": ex["n_skip_cap"], "n_degenerate": ex["n_degenerate"],
            "n_collision": ex["n_collision"],
            "n_forced_delev": ex["n_forced_delev"],
            "worst_margin_to_equity": mx["worst_margin_to_equity"],
            "trip_maintenance": mx["trip_maintenance"],
            "trip_drift": mx["trip_drift"],
            "flag_low_admission": bool(n_admitted < 0.9 * len(spy_events))}


for s in range(N_SEEDS_SCREEN):
    env_rows.append(run_null_seed(s))
    if (s + 1) % 25 == 0:
        d = [r["final_equity"] for r in env_rows]
        print(f"  {s + 1}/{N_SEEDS_SCREEN}  median ${np.median(d):,.0f}")
dist = np.array([r["final_equity"] for r in env_rows])
escalated = a_m["final_equity"] >= np.percentile(dist, ESCALATE_AT_P)
if escalated:
    print(f"  A >= p{ESCALATE_AT_P} at {N_SEEDS_SCREEN} -- extending the "
          f"SAME stream to {N_SEEDS_FULL} (registered escalation)")
    for s in range(N_SEEDS_SCREEN, N_SEEDS_FULL):
        env_rows.append(run_null_seed(s))
        if (s + 1) % 50 == 0:
            print(f"  {s + 1}/{N_SEEDS_FULL}")
    dist = np.array([r["final_equity"] for r in env_rows])
K = len(dist)
p95 = float(np.percentile(dist, 95))
p5 = float(np.percentile(dist, 5))
v = a_m["final_equity"]
pctile = float(((dist < v).sum() + 0.5 * (dist == v).sum()) / K)
p_two = min(1.0, 2 * min(1 + (dist <= v).sum(), 1 + (dist >= v).sum()) / (K + 1))
oos_dist = np.array([r["oos_wealth_x"] for r in env_rows], float)
oos_v = a_m.get("oos_wealth_x", np.nan)
oos_pctile = (float(((oos_dist < oos_v).sum() + 0.5 * (oos_dist == oos_v).sum())
                    / np.isfinite(oos_dist).sum())
              if np.isfinite(oos_v) else np.nan)
print(f"  N envelope ({K}): median ${np.median(dist):,.0f}  "
      f"[p5 ${p5:,.0f}, p95 ${p95:,.0f}]  A pctile {100 * pctile:.1f}")

# ---- tripwires (registered: A and every N seed; C voids C only)
trip_a = a_mx["trip_maintenance"] or a_mx["trip_drift"]
trip_n = any(r["trip_maintenance"] or r["trip_drift"] for r in env_rows)
run_valid = not (trip_a or trip_n)
c_valid = not (c_mx["trip_maintenance"] or c_mx["trip_drift"])

# ---- bars and tier (registered order)
bar1 = (run_valid
        and a_m["final_equity"] > b_m["final_equity"]
        and np.isfinite(oos_v) and "oos_wealth_x" in b_m
        and a_m["oos_wealth_x"] > b_m["oos_wealth_x"])
bar2 = run_valid and v >= p95
adverse = run_valid and v <= p5
if not run_valid:
    tier = "INVALID (tripwire) -- no tier awarded pending pre-registered amendment"
elif adverse:
    tier = ("ADVERSE SELECTION -- the marginal forgone signal destroys value "
            "net of financing relative to the random-name financing null; "
            "the study-06 capacity warning is confirmed at the admission margin")
elif bar1 and bar2:
    tier = "LIVE CANDIDATE"
elif bar1:
    tier = (f"SELECTION CONTENT NOT ESTABLISHED -- A ${v:,.0f} vs p95 of N "
            f"${p95:,.0f} at {K} seeds; consistent with generic leverage in "
            f"a rising decade; no timing or selection claim attaches")
else:
    tier = "DEAD"
print(f"\nVERDICT: {tier}")

# ---- ledgers and outputs
bridge_st = trade_stats(a_bridges, cal_idx)
regular_st = trade_stats(a_regular, cal_idx)
tr_b = [x["exit_px"] * (1 - COST) / (x["entry_px"] * (1 + COST)) - 1
        for x in a_bridges if x["reason"] != "open"]
tr_r = [x["exit_px"] * (1 - COST) / (x["entry_px"] * (1 + COST)) - 1
        for x in a_regular if x["reason"] != "open"]
boot_p = boot_two_sample_p(tr_b, tr_r)

summary = []
for name, m, mx, ex in [("BASELINE", b_m, None, None),
                        ("MARGIN_BRIDGE", a_m, a_mx, a_ex),
                        ("MARGIN_BRIDGE_EFFR400", s_m, margin_extras(s_daily),
                         s_ex),
                        ("BRIDGE_SPY_DISCLOSURE", c_m, c_mx, c_ex)]:
    row = {"run": name, **m}
    if mx:
        row.update(mx)
    if ex:
        row.update({"interest_paid": round(ex["interest_paid"], 2),
                    "n_skip_cap": ex["n_skip_cap"],
                    "n_degenerate": ex["n_degenerate"],
                    "n_collision": ex["n_collision"],
                    "n_skip_cap_own": ex["n_skip_cap_own"],
                    "n_forced_delev": ex["n_forced_delev"]})
    summary.append(row)
summary[1].update({"env_pctile": round(100 * pctile, 1),
                   "env_p_two_sided": round(p_two, 4),
                   "env_p95": round(p95, 2), "env_p5": round(p5, 2),
                   "env_seeds": K,
                   "oos_env_pctile": (round(100 * oos_pctile, 1)
                                      if oos_pctile == oos_pctile else np.nan),
                   "n_bridges": len(a_bridges),
                   "tier": tier})
pd.DataFrame(summary).to_csv(os.path.join(OUTDIR, "11_bridge_summary.csv"),
                             index=False)

all_daily = []
for name, d in [("BASELINE", b_daily), ("MARGIN_BRIDGE", a_daily),
                ("BRIDGE_SPY_DISCLOSURE", c_daily)]:
    for row in d:
        all_daily.append({"run": name, "date": str(row["date"].date()),
                          "equity": round(row["equity"], 2),
                          "n_pos": row["n_pos"],
                          "drawn": round(row["drawn"], 2),
                          "gross": round(row["gross"], 2)})
pd.DataFrame(all_daily).to_csv(os.path.join(OUTDIR, "11_bridge_daily.csv"),
                               index=False)

all_trades = []
for name, tr in [("BASELINE", b_trades), ("MARGIN_BRIDGE", a_trades),
                 ("BRIDGE_SPY_DISCLOSURE", c_trades)]:
    for x in tr:
        all_trades.append({"run": name,
                           **{k: (round(vv, 4) if isinstance(vv, float) else vv)
                              for k, vv in x.items()
                              if k not in ("last_close", "last_i", "exit_t", "entry_t")}})
pd.DataFrame(all_trades).to_csv(os.path.join(OUTDIR, "11_bridge_trades.csv"),
                                index=False)
pd.DataFrame(env_rows).to_csv(os.path.join(OUTDIR, "11_bridge_envelopes.csv"),
                              index=False)

store_manifest = {os.path.basename(p): sha256_file(p)
                  for p in sorted(glob.glob(os.path.join(DATADIR, "*.csv")))}
with open(os.path.join(OUTDIR, "11_bridge_config.json"), "w") as f:
    json.dump({
        "protocol": "MARGIN_BRIDGE_PROTOCOL.md (study 11)",
        "script_sha256": SCRIPT_SHA256,
        "gate": gate_record, "vintage": vintage_now,
        "constants": {"N_SLOTS": N_SLOTS, "TOP_Q": TOP_Q, "COST_BPS": 5,
                      "CAPITAL": CAPITAL, "GROSS_CAP": GROSS_CAP,
                      "SPREAD_BPS": 150, "SPREAD_SENS_BPS": 400,
                      "SPLIT": str(SPLIT.date()), "SIM_END": str(SIM_END.date()),
                      "ASOF": ASOF, "SEED": SEED, "STUDY_STREAM": STUDY,
                      "N_SEEDS": [N_SEEDS_SCREEN, N_SEEDS_FULL],
                      "ESCALATE_AT_P": ESCALATE_AT_P,
                      "MAINT_TRIP": MAINT_TRIP, "DRIFT_TRIP": DRIFT_TRIP,
                      "day_count": "/252 (understates calendar/360 ~1.4%; "
                                   "disclosed, accepted)",
                      "dividends": "uncredited on bridged exposure; "
                                   "conservative, accepted",
                      "c_occupancy": "SPY bridges count toward the 10-slot "
                                     "occupancy (amendment 01, section-2 "
                                     "reading)",
                      "c_delist_pairing": "delisted A-bridges pair C's SPY "
                                          "close to the cash-recognition "
                                          "session (amendment 01)",
                      "forced_delev": "amendment 02: close-drift above the "
                                      "cap forces newest-first whole-position "
                                      "sales at the next open, arms A and N; "
                                      "C exempt (pairing preserved); "
                                      "tripwires unchanged"},
        "sim_env_overrides": sim_env,
        "escalated": bool(escalated), "env_seeds": K,
        "bars": {"bar1_beats_baseline_both_windows": bool(bar1),
                 "bar2_ge_p95": bool(bar2), "adverse_le_p5": bool(adverse),
                 "run_valid": bool(run_valid), "c_exhibit_valid": bool(c_valid)},
        "tier": tier,
        "bridged_trades": bridge_st, "regular_trades": regular_st,
        "bridged_vs_regular_boot_p_DIAGNOSTIC_ONLY": (round(boot_p, 5)
                                                      if boot_p == boot_p
                                                      else None),
        "store_manifest_sha256": store_manifest,
    }, f, indent=1)
print(f"\nwrote results/11_bridge_summary.csv  11_bridge_daily.csv  "
      f"11_bridge_trades.csv  11_bridge_envelopes.csv  11_bridge_config.json")
print(f"script sha256: {SCRIPT_SHA256}")
