"""09_asym_test.py -- Study 09: can the asymmetric map dictate weakness adds?

PRE-REGISTRATION
  ASYM_STRATEGY_PROTOCOL.md (sha256 in PROTOCOL_LOCK.json) as amended by
  PROTOCOL_AMENDMENT_01.md, both committed BEFORE this script first ran.
  Family bars (fixed): six weakness-family looks spent through this study.
    primary discovery  percentile >= 99.15 on BOTH nulls at 500 seeds
                       AND beats baseline on both windows (08 rule)
    suggestive         98.3 <= pct < 99.15  -> shadow log only
    otherwise          DEAD

ARMS
  RANKVOL      baseline, reconciliation gate ($552,888 within 1.5%)
  ADDS         08 arm re-run verbatim (H context; expect $754,818)
  A ADDS_LEAN_SELECT  lean IS the trigger: held name, z<0 at signal close,
               cell lean_hat>0 on this month's map; one add per day, the
               eligible name with the highest lean_hat wins.
  B ADDS_LEAN_SIZER   08 trigger verbatim; add sized by the 07 sizer
               multiplier (lean percentile among the month's cells, 0.5-1.5).
  C ADDS_LEAN_GATE    08 trigger verbatim, executed only if lean_hat>0.
  H HORSE_RACE        analysis, not a sim: exposure-adjusted comparison of
               ADDS vs baseline + displacement ledger (see amendment §2).

NULLS (per arm)
  N1 matched-intensity coin  SeedSequence([7, 12, s, arm_id])
     A: day-level coin, uniform pick among that day's weakness-eligible
     B: per-eligibility coin at realized p_hat, multiplier resampled from
        the arm's realized multipliers
     C: per-trigger coin at the realized lean-gate pass rate
  N2 persistent map permutation  SeedSequence([7, 12, s, arm_id + 10])
     ONE permutation of the 16 cell identities per seed, held for the
     entire decade of monthly maps (07 lesson: monthly reshuffles are
     anti-conservative).
  200-seed kill screen; arms at or above p95 on both nulls extend the SAME
  indexed seed stream to 500 (no re-rolls; the bar is evaluated at 500).

INHERITANCE
  Universe/engine: 08_adds_settle.py verbatim (loader, run_account, metrics).
  Maps: 07_c2_overlay_test.py verbatim (load_name/build_anchors from its
  registered anchor frame, completion-gated monthly maps, MIN_CELL_N=1000,
  thin cells carry pooled lean; sizer multiplier factory verbatim).

RUN (from the repo root -- needs results/02_ranks.csv)
  export PORTFOLIO_DATADIR=/path/to/prepped/data
  python3 09_asym_test.py
  outputs results/09_asym_summary.csv, 09_asym_envelopes.csv,
          09_ledger_<ARM>.csv, 09_asym_config.json
  SIM BUDGET: 1205 sims if all arms die at the 200-seed screen (~4-8h at
  08's pace); +600 sims per arm extending to 500 seeds (worst case 3005,
  ~10-20h). Fire under nohup and budget a full day -- the long case IS the
  pass case. Envelopes checkpoint to disk every 50 seeds.
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
PUBLISHED_ADDS = 754818.05
RECON_TOL = 0.015
SEED = 7
STUDY = 12                       # registered seed-stream id (fixed at pre-registration; independent of script numbering)
SEEDS_SCREEN = int(os.environ.get("ASYM_SEEDS_SCREEN", "200"))
SEEDS_FULL = int(os.environ.get("ASYM_SEEDS_FULL", "500"))
MIN_CELL_N = int(os.environ.get("C2_MIN_CELL_N", "1000"))
K = 5
ERA_AGE = 275
SMA_W = 20
Z_EDGES = np.array([-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf])
N_CELLS = (len(Z_EDGES) - 1) * 2
N_OFFSETS = 5
BAR_PRIMARY = 99.15
BAR_SUGGEST = 98.3
os.makedirs(OUTDIR, exist_ok=True)

# hash the code Python actually loaded, BEFORE any work (audit finding:
# an end-of-run hash would certify whatever sits on disk hours later)
with open(__file__, "rb") as f:
    SCRIPT_SHA256 = hashlib.sha256(f.read()).hexdigest()


def truthy(series):
    return series.astype(str).str.strip().isin(["True", "true", "1", "1.0"]).to_numpy()


# ================================================================ anchors
# load_name / build_anchors VERBATIM from 07_c2_overlay_test.py's anchor
# frame. Any edit breaks provenance.
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


def bowley(e):
    q10, q50, q90 = np.percentile(e, [10, 50, 90])
    d = q90 - q10
    return (q90 + q10 - 2 * q50) / d if d > 0 else np.nan


def monthly_maps(pool, cal):
    per = cal.to_period("M")
    starts = [int(np.argmax(per == m)) for m in per.unique()]
    comp = pool["comp"].to_numpy()
    cells = pool["cell"].to_numpy()
    e = pool["e"].to_numpy()
    maps, thin_flags, rows = [], [], []
    for s in starts:
        cutoff = np.datetime64(cal[s])
        k = np.searchsorted(comp, cutoff, side="left")
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
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "09_asym_maps.csv"),
                              index=False)
    month_of = np.zeros(len(cal), int)
    si = 0
    for t in range(len(cal)):
        if si + 1 < len(starts) and t >= starts[si + 1]:
            si += 1
        month_of[t] = si
    return maps, thin_flags, month_of


# ================================================================ universe
# 08 loader + 07's cell_sig, in one pass. sigma added to USECOLS for cells.
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
        zz = df["z"].to_numpy(float)
        dm = df["dmad_5"].to_numpy(float)
        ps = df["price_share_5"].to_numpy(float)
        base = (dm > 0) & (ps >= SHARE_MIN)
        df["add_here"] = np.concatenate([[False], ((zz < 0) & base)[:-1]])
        df["weak_here"] = np.concatenate([[False], (zz < 0)[:-1]])
        vv = dm / df["sigma"].to_numpy(float)
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
        for c in ["enter_here", "exit_here", "member", "add_here", "weak_here"]:
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


# ================================================================ engine
# 08 run_account extended with the lean-arm add modes. The 08 paths
# (adds=False; mode="adds08") are byte-equivalent in behavior to 08.
def run_account(U, cal, cand_by_day, pick_vol, bench, month_of=None,
                lean_maps=None, adds=False, mode="adds08", add_frac=1.0,
                adds_first=True, adds_rng=None, adds_prob=None,
                lean_gate=False, sizer=False, one_per_day=False,
                mult_pool=None, want_ledger=False):
    cash, pos = CAPITAL, {}
    daily, trades, ledger = [], [], []
    n_adds = n_elig = 0

    def units():
        return sum(1 + p.get("added", 0) * p.get("afrac", 1.0)
                   for p in pos.values())

    def mark_now(t):
        return cash + sum(q["shares"] *
                          (U[k]["Open"][t] if U[k]["has_bar"][t]
                           else q["last_close"]) for k, q in pos.items())

    def lean_of(tk, t, mp):
        cell = U[tk]["cell_sig"][t]
        if cell < 0:
            return None
        v = mp[month_of[t]][cell]
        return None if v != v else float(v)

    def sizer_mult(tk, t, mp):
        # 07 size_mult_factory verbatim: lean percentile among month cells
        m = mp[month_of[t]]
        lean = lean_of(tk, t, mp)
        ok = sorted(v for v in m if v == v)
        if lean is None or not ok:
            return 1.0
        return 0.5 + np.searchsorted(ok, lean, side="right") / len(ok)

    def addable(tk, t):
        p, a = pos[tk], U[tk]
        if p.get("added") or not a["has_bar"][t] or not a["member"][t]:
            return False
        if t == 0 or not a["has_bar"][t - 1]:
            return False
        if a["prev_close"][t] != a["prev_close"][t]:
            return False
        return np.isfinite(a["Open"][t])

    def execute_add(tk, t, frac, mult):
        nonlocal cash, n_adds
        p, a = pos[tk], U[tk]
        px = a["Open"][t]
        alloc = min(mark_now(t) / N_SLOTS * frac * mult, cash)
        if alloc < 1.0:
            return False
        sh_new = alloc / (px * (1 + COST))
        if want_ledger:
            ledger.append({"ticker": tk, "add_date": str(cal[t].date()),
                           "add_px": round(px, 4),
                           "add_dollars": round(alloc, 2),
                           "mult": round(float(mult), 4),
                           "lean": (round(lean_of(tk, t, lean_maps), 5)
                                    if lean_maps is not None
                                    and lean_of(tk, t, lean_maps) is not None
                                    else ""),
                           "orig_entry": p["entry_date"]})
        p["entry_px"] = ((p["entry_px"] * p["shares"] + px * sh_new)
                         / (p["shares"] + sh_new))
        p["shares"] += sh_new
        p["added"] = 1
        p["afrac"] = frac * mult
        p["add_date"] = str(cal[t].date())
        cash -= alloc
        n_adds += 1
        return True

    def do_adds_adds08(t):
        # 08 trigger family: B (sizer), C (gate), and their N1 coins
        nonlocal n_elig
        for tk in sorted(pos):
            if not addable(tk, t):
                continue
            if units() >= N_SLOTS:
                break
            a = U[tk]
            if not a["add_here"][t]:
                continue
            n_elig += 1
            if adds_rng is not None:
                if adds_rng.random() >= adds_prob:
                    continue
                mult = (float(adds_rng.choice(mult_pool))
                        if sizer and mult_pool is not None and len(mult_pool)
                        else 1.0)
            else:
                if lean_gate:
                    lv = lean_of(tk, t, lean_maps)
                    if lv is None or lv <= 0:
                        continue
                mult = sizer_mult(tk, t, lean_maps) if sizer else 1.0
            execute_add(tk, t, add_frac, mult)

    def do_adds_select(t):
        # A: lean IS the trigger; one add per day, highest lean wins.
        nonlocal n_elig
        if units() >= N_SLOTS:
            return
        elig = [tk for tk in sorted(pos)
                if addable(tk, t) and U[tk]["weak_here"][t]]
        if not elig:
            return
        n_elig += 1                      # day-level intensity (amendment 3)
        if adds_rng is not None:
            if adds_rng.random() >= adds_prob:
                return
            tk = elig[int(adds_rng.integers(len(elig)))]
            execute_add(tk, t, add_frac, 1.0)
            return
        scored = [(lean_of(tk, t, lean_maps), tk) for tk in elig]
        scored = [(lv, tk) for lv, tk in scored if lv is not None and lv > 0]
        if not scored:
            return
        scored.sort(key=lambda x: (-x[0], x[1]))
        execute_add(scored[0][1], t, add_frac, 1.0)

    def do_adds(t):
        if mode == "select":
            do_adds_select(t)
        else:
            do_adds_adds08(t)

    def do_buys(t):
        nonlocal cash
        cands = [tk for tk in cand_by_day[t]
                 if tk not in pos and tk in bench[t]]
        free_units = N_SLOTS - units()
        cands.sort(key=lambda k: (pick_vol.get((k, t), 1e9), k))
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
                       "entry_px": px, "shares": shares,
                       "last_close": a["Close"][t], "last_i": t,
                       "last_date": str(cal[t].date())}
            free_units -= 1

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
        if adds and adds_first:
            do_adds(t)
            do_buys(t)
        elif adds:
            do_buys(t)
            do_adds(t)
        else:
            do_buys(t)
        for tk, p in pos.items():
            a = U[tk]
            if a["has_bar"][t]:
                p["last_close"] = a["Close"][t]
                p["last_i"] = t
                p["last_date"] = str(cal[t].date())
        daily.append({"date": cal[t], "equity": cash + sum(
            p["shares"] * p["last_close"] for p in pos.values()),
            "cash": cash, "n_pos": len(pos)})
    for tk, p in list(pos.items()):
        trades.append({**p, "exit_date": "", "exit_px": p["last_close"],
                       "reason": "open"})
    return daily, trades, n_adds, n_elig, ledger


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
    cashv = np.array([d["cash"] for d in daily])
    npos = np.array([d["n_pos"] for d in daily])
    closed = [x for x in trades if x["reason"] != "open"]
    tr = np.array([x["exit_px"] * (1 - COST) / (x["entry_px"] * (1 + COST)) - 1
                   for x in closed]) if closed else np.array([])
    full = window_stats(dates, eq)
    late = {f"oos_{k}": v for k, v in
            window_stats(dates, eq, np.datetime64(SPLIT)).items()}
    # exposure-adjusted block (H; definitions stamped in config json)
    r = np.diff(eq) / eq[:-1]
    inv = npos[1:] > 0
    r_in = r[inv]
    invested_frac = float(np.mean(1.0 - cashv / eq))
    exp_adj = {
        "invested_frac": round(invested_frac, 4),
        "in_trade_day_mean": (round(float(r_in.mean()), 6)
                              if len(r_in) else np.nan),
        "in_trade_day_median": (round(float(np.median(r_in)), 6)
                                if len(r_in) else np.nan),
        "in_trade_sharpe": (round(float(np.sqrt(252) * r_in.mean()
                                        / r_in.std(ddof=1)), 3)
                            if len(r_in) > 2 and r_in.std(ddof=1) > 0
                            else np.nan),
        "deployed_cagr": (round(float((1 + r_in.mean()) ** 252 - 1), 4)
                          if len(r_in) else np.nan),
        "mean_trade_ret": (round(float(tr.mean()), 6) if len(tr) else np.nan),
    }
    return {"final_equity": round(float(eq[-1]), 2), **full, **late, **exp_adj,
            "trades": len(closed),
            "win_rate": round(float((tr > 0).mean()), 3) if len(tr) else np.nan,
            "med_trade_ret": (round(float(np.median(tr)), 6)
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

if not os.path.exists(RANKS_CSV):
    raise SystemExit(f"{RANKS_CSV} not found -- run 02_ranking.py first.")
rk = pd.read_csv(RANKS_CSV, usecols=["Date", "ticker", "rank_vol"])
bench = volume_bench(rk, cal)
rk["Date"] = pd.to_datetime(rk["Date"])
rv = rk.set_index(["Date", "ticker"])["rank_vol"]
pick_vol = {}
for t in range(1, len(cal)):
    day = cal[t - 1]
    for tk in cand_by_day[t]:
        v = rv.get((day, tk))
        if v is not None and v == v:
            pick_vol[(tk, t)] = float(v)

print("\nanchor pool (registered frame, 5 offsets pooled)...")
pool = build_pool([f for f in files
                   if os.path.basename(f)[:-4] not in ("SPY", "RSP")])
print(f"  {len(pool):,} completed innovations")
print("walk-forward monthly maps (completion-gated)...")
maps, thin_flags, month_of = monthly_maps(pool, cal)
print(f"  {len(maps)} monthly maps")

summary, env_rows = [], []


def record(name, res, ledger_name=None):
    daily, trades, n_adds, n_elig, ledger = res
    m = metrics(daily, trades)
    m["adds"] = n_adds
    summary.append({"run": name, **m})
    if ledger_name:
        # header-only CSV for zero-add arms (amendment 02; ledger contract)
        closed_by = {}
        for x in trades:
            closed_by[(x["ticker"], x.get("add_date"))] = x
        rows = []
        for led in ledger:
            row = dict(led)
            key = closed_by.get((led["ticker"], led["add_date"]))
            if key is not None:
                row["exit_date"] = key["exit_date"]
                row["exit_px"] = round(float(key["exit_px"]), 4)
                row["pos_ret"] = round(key["exit_px"] * (1 - COST)
                                       / (key["entry_px"] * (1 + COST)) - 1, 4)
                row["pos_pnl"] = round(key["shares"]
                                       * (key["exit_px"] - key["entry_px"]), 2)
            rows.append(row)
        cols = ["ticker", "add_date", "add_px", "add_dollars", "mult", "lean",
                "orig_entry", "exit_date", "exit_px", "pos_ret", "pos_pnl"]
        pd.DataFrame(rows, columns=cols).to_csv(
            os.path.join(OUTDIR, f"09_ledger_{ledger_name}.csv"), index=False)
        # protocol section 5: GPS/SMCI-style concentration -- top-2 add share
        # of arm P&L (pos_pnl = full doubled position, 08 ledger semantics)
        pnls = sorted((r["pos_pnl"] for r in rows
                       if r.get("pos_pnl") is not None), reverse=True)
        arm_pnl = m["final_equity"] - CAPITAL
        if pnls and arm_pnl > 0:
            share = sum(pnls[:2]) / arm_pnl
            summary[-1]["top2_add_share"] = round(share, 4)
            print(f"  {ledger_name} GPS/SMCI check: top-2 add share of arm "
                  f"P&L {100 * share:.1f}% ("
                  + " + ".join(f"${p:,.0f}" for p in pnls[:2]) + ")")
        pd.DataFrame(summary).to_csv(
            os.path.join(OUTDIR, "09_asym_summary.csv"), index=False)
    oos = m.get("oos_cagr", float("nan"))
    print(f"  {name:<18} final ${m['final_equity']:>11,.0f}  "
          f"cagr {100 * m['cagr']:>6.2f}%  | 2022+ {100 * oos:>6.2f}%  "
          f"adds {n_adds}")
    return m, n_adds, n_elig


print("\nbaseline + reconciliation...")
m_base, _, _ = record("RANKVOL", run_account(U, cal, cand_by_day, pick_vol,
                                             bench, month_of, maps))
gap = abs(m_base["final_equity"] - PUBLISHED_FINAL) / PUBLISHED_FINAL
if gap > RECON_TOL:
    raise SystemExit(f"RECONCILIATION FAILED ({100 * gap:.2f}% off).")
print(f"  reconciliation: {100 * gap:.2f}% off -- PASS")

print("\nADDS re-run (H context; expect ~$754,818)...")
m_adds, _, _ = record("ADDS", run_account(
    U, cal, cand_by_day, pick_vol, bench, month_of, maps,
    adds=True, mode="adds08", want_ledger=True), ledger_name="ADDS")
gap_a = abs(m_adds["final_equity"] - PUBLISHED_ADDS) / PUBLISHED_ADDS
print(f"  ADDS check vs 08: {100 * gap_a:.2f}% off "
      f"{'-- PASS' if gap_a < 0.001 else '-- WARN (investigate before trusting)'}")

print("\nH HORSE_RACE (analysis; exposure-adjusted, no bar):")
for nm, mm in (("RANKVOL", m_base), ("ADDS", m_adds)):
    print(f"  {nm:<8} invested_frac {mm['invested_frac']:.3f}  "
          f"in-trade/day mean {100 * mm['in_trade_day_mean']:.4f}% "
          f"median {100 * mm['in_trade_day_median']:.4f}%  "
          f"in-trade sharpe {mm['in_trade_sharpe']}  "
          f"deployed cagr {100 * mm['deployed_cagr']:.2f}%  "
          f"trade mean {100 * mm['mean_trade_ret']:.2f}% "
          f"med {100 * mm['med_trade_ret']:.2f}%")

ARMS = [
    ("ADDS_LEAN_SELECT", 1, dict(adds=True, mode="select"), dict()),
    ("ADDS_LEAN_SIZER", 2, dict(adds=True, mode="adds08", sizer=True), dict()),
    ("ADDS_LEAN_GATE", 3, dict(adds=True, mode="adds08", lean_gate=True), dict()),
]

verdicts = []
for name, arm_id, kw, _ in ARMS:
    print(f"\n--- {name} ---")
    res = run_account(U, cal, cand_by_day, pick_vol, bench, month_of, maps,
                      want_ledger=True, **kw)
    m_arm, n_add, n_el = record(name, res, ledger_name=name)
    if n_add == 0:
        verdicts.append((name, np.nan, np.nan, "EMPTY: zero adds fired"))
        continue
    p_hat = n_add / max(n_el, 1)
    mult_pool = np.array([r["mult"] for r in res[4]]) if kw.get("sizer") else None
    print(f"  intensity {n_add}/{n_el} (p={p_hat:.5f})")

    def null_pass(n_seeds, start_at, kind):
        rows = []
        for s in range(start_at, n_seeds):
            if kind == "N1":
                rng = np.random.default_rng(
                    np.random.SeedSequence([SEED, STUDY, s, arm_id]))
                d, tr, _, _, _ = run_account(
                    U, cal, cand_by_day, pick_vol, bench, month_of, maps,
                    adds=True, mode=kw.get("mode"), sizer=kw.get("sizer", False),
                    adds_rng=rng, adds_prob=p_hat, mult_pool=mult_pool)
            else:
                rng = np.random.default_rng(
                    np.random.SeedSequence([SEED, STUDY, s, arm_id + 10]))
                perm = rng.permutation(N_CELLS)
                shuf = [mp[perm] for mp in maps]
                d, tr, _, _, _ = run_account(
                    U, cal, cand_by_day, pick_vol, bench, month_of, shuf,
                    want_ledger=False, **kw)
            fe = metrics(d, tr)["final_equity"]
            rows.append(fe)
            env_rows.append({"arm": name, "null": kind, "seed": s,
                             "final_equity": fe})
            if (s + 1) % 50 == 0:
                print(f"    {kind} {s + 1}/{n_seeds}")
                # idempotent checkpoint (audit finding: a 15h run must not
                # hold every envelope in memory until the end)
                pd.DataFrame(env_rows).to_csv(
                    os.path.join(OUTDIR, "09_asym_envelopes.csv"), index=False)
        return rows

    dist1 = null_pass(SEEDS_SCREEN, 0, "N1")
    dist2 = null_pass(SEEDS_SCREEN, 0, "N2")
    pct1 = 100 * float((np.array(dist1) < m_arm["final_equity"]).mean())
    pct2 = 100 * float((np.array(dist2) < m_arm["final_equity"]).mean())
    print(f"  screen@{SEEDS_SCREEN}: N1 p{pct1:.1f}  N2 p{pct2:.1f}")
    if min(pct1, pct2) >= 95.0:
        print(f"  extending SAME seed stream to {SEEDS_FULL} (no re-rolls)...")
        dist1 += null_pass(SEEDS_FULL, SEEDS_SCREEN, "N1")
        dist2 += null_pass(SEEDS_FULL, SEEDS_SCREEN, "N2")
        pct1 = 100 * float((np.array(dist1) < m_arm["final_equity"]).mean())
        pct2 = 100 * float((np.array(dist2) < m_arm["final_equity"]).mean())
    pmin = min(pct1, pct2)
    beats = (m_arm["cagr"] > m_base["cagr"]
             and m_arm.get("oos_cagr", -9) > m_base.get("oos_cagr", 9))
    n_used = len(dist1)
    if beats and pmin >= BAR_PRIMARY and n_used >= SEEDS_FULL:
        v = f"PRIMARY DISCOVERY (p{pmin:.1f} both nulls @ {n_used}; family-safe)"
    elif pmin >= BAR_SUGGEST and n_used >= SEEDS_FULL:
        # suggestive tier is percentile-band-only per protocol section 6
        # (amendment 02): a >=p99.15 arm failing the beats rule lands HERE
        # (shadow log, never capital), not DEAD
        v = f"SUGGESTIVE (p{pmin:.1f}) -- shadow log only, never capital"
    elif beats:
        v = f"DEAD: beats baseline but fails family bars (p{pmin:.1f})"
    else:
        v = f"DEAD: does not beat baseline both windows (p{pmin:.1f})"
    verdicts.append((name, pct1, pct2, v))

print("\nVERDICTS (pre-registered; six-look family, bars fixed before run):")
for name, p1, p2, v in verdicts:
    print(f"  {name:<18} N1 p{p1:.1f}  N2 p{p2:.1f}  {v}")

pd.DataFrame(summary).to_csv(os.path.join(OUTDIR, "09_asym_summary.csv"),
                             index=False)
pd.DataFrame(env_rows).to_csv(os.path.join(OUTDIR, "09_asym_envelopes.csv"),
                              index=False)
run_hash = SCRIPT_SHA256
with open(os.path.join(OUTDIR, "09_asym_config.json"), "w") as f:
    json.dump({"protocol": "ASYM_STRATEGY_PROTOCOL.md + AMENDMENT_01",
               "script_sha256": run_hash, "SEED": SEED, "STUDY_STREAM": STUDY,
               "SEEDS_SCREEN": SEEDS_SCREEN, "SEEDS_FULL": SEEDS_FULL,
               "BAR_PRIMARY": BAR_PRIMARY, "BAR_SUGGEST": BAR_SUGGEST,
               "MIN_CELL_N": MIN_CELL_N, "ASOF": ASOF, "TOP_Q": TOP_Q,
               "COST_BPS": COST * 10000, "SPLIT": str(SPLIT.date()),
               "SIM_END": str(SIM_END.date()),
               "weakness": "z<0 at signal close (08 ADDS inheritance; amd 1)",
               "exposure_adjusted_defs": {
                   "invested_frac": "mean(1 - cash/equity)",
                   "in_trade_day": "daily equity return on days n_pos>0",
                   "in_trade_sharpe": "sqrt(252)*mean/std of in-trade days",
                   "deployed_cagr": "(1+mean in-trade day)^252 - 1"},
               "family_looks": ["ADDS p94.4", "ADDS_HALF p85.0",
                                "ADDS_LAST p91.0", "A", "B", "C"]},
              f, indent=1)
print(f"\nwritten: {OUTDIR}/09_asym_summary.csv, 09_asym_envelopes.csv, "
      f"09_ledger_*.csv, 09_asym_maps.csv, 09_asym_config.json")
print(f"script sha256: {run_hash}")
