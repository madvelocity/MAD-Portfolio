"""06_capacity_test.py -- pre-registered capacity/sizing family (arms 1-3).

THE QUESTIONS
  1. ADDS      does doubling into held weakness (tape-conditioned) beat
               spending the same slots on fresh queue names?
  2. 2% SLOTS  does the edge survive dilution to N=50 slots?
  3. BUY ALL   if we take EVERY signal and equal-weight the book, does
               selection matter at all -- or was slot scarcity the edge?

PRE-REGISTRATION (fixed before the run; family primary = ADDS)
  baseline   RANKVOL_BENCH_N10 -- the published config, re-run here and
             RECONCILED against $552,888 (tol 1.5%) before any arm prints.
             It doubles as the never-add control for the horse race.
  ADDS       held name's own tape at the signal close (cal[t-1], and the
             signal bar must BE cal[t-1] -- halt-reopen stale tape skipped,
             same freshness rule entries enforce):
               below trend        z < 0
               rising             dmad_5 > 0
               price-driven       price_share_5 >= 0.50
             -> add ONE slot unit at the next open. One add per position,
             lifetime (resets if the name re-enters later). Adds consume
             slots (a doubled position = 2 units) and take PRIORITY over new
             queue buys that day -- that is the doubling-down thesis. Exit
             unchanged: the whole position leaves on the rollback. NEVER
             conditioned on the position's own P&L.
             ADDS_DEEP robustness: same trigger with z <= -1.
             NULL: a matched-intensity RANDOM-ADD envelope (ENV_SEEDS seeds,
             SeedSequence([SEED, 8, 10, s, 2])) -- identical mechanics, adds
             fired at random eligible moments with the primary run's
             empirical intensity p = adds / eligible position-days. It asks
             whether the TAPE, not the extra deployment itself, carries any
             gain.
  N50        RANKVOL_BENCH_N50 -- 2% slots, same bench, same volume order.
             Judged against its own fresh benched RAND envelope (100 seeds,
             SeedSequence([SEED, 8, 50, s]) -- disjoint from every prior
             study) and against N10 exposure-adjusted.
  ALL_EW     every tradeable candidacy is bought (no slot cap, no selector);
             the whole book is RESIZED TO EQUAL WEIGHT at the open on every
             day the composition changes, and 5 bps is charged on every
             resize leg -- turnover is this arm's real cost. ALL_EW_BENCH
             (bench eligibility, primary) and ALL_EW_ALL (no bench,
             robustness). Controls: RSP (the market's own equal-weight
             everything) and SPY, from the cached series.
  metrics    full-window and 2022+ OOS CAGR, Sharpe, maxDD, final equity,
             win rate, median trade, adds count. Run names follow 04's
             published convention (*_BENCH_* = volume-bench eligibility).
             ALL_EW rows carry NO win-rate/median-trade values -- constant
             resizing makes those name-path numbers, not account-realized
             returns, so they are blanked rather than invite a false
             comparison. Verdicts:
               ADDS displaces the baseline only if it beats RANKVOL_BENCH_N10
               on BOTH windows AND its final equity clears the p95 of the
               matched-intensity random-add envelope.
               N50 is viable only if it clears its RAND N50 p95 AND its
               full+OOS Sharpe are no more than 0.1 BELOW N10 (one-sided:
               dilution tolerated, edge erosion not, better never penalized).
               ALL_EW verdict is descriptive: if it lands near N10 the slots
               were never load-bearing; if it craters toward RSP, scarcity +
               selection carry the record.

RUN (from the repo root)
  export PORTFOLIO_DATADIR=/path/to/prepped/data
  python3 06_capacity_test.py
  needs results/02_ranks.csv;
  spy/rsp caches in results/ or DATADIR. outputs results/06_capacity_*.csv.
"""
import glob
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
N_BIG = 50
TOP_Q = int(os.environ.get("SIM_TOP_Q", "100"))
COST = float(os.environ.get("SIM_COST_BPS", "5")) / 10000.0
CAPITAL = float(os.environ.get("SIM_CAPITAL", "100000"))
PUBLISHED_FINAL = 552888.0
RECON_TOL = 0.015
SEED = 7
ENV_SEEDS = int(os.environ.get("CAP_ENV_SEEDS", "100"))
os.makedirs(OUTDIR, exist_ok=True)


def truthy(series):
    return series.astype(str).str.strip().isin(["True", "true", "1", "1.0"]).to_numpy()


# ---------------------------------------------------------------- universe
USECOLS = ["Date", "Open", "Close", "price_share_5", "dmad_5", "z",
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
        # the ADDS tape condition at the signal close, acted on next open
        zz = df["z"].to_numpy(float)
        dm = df["dmad_5"].to_numpy(float)
        ps = df["price_share_5"].to_numpy(float)
        base = (dm > 0) & (ps >= SHARE_MIN)
        df["add_here"] = np.concatenate([[False], ((zz < 0) & base)[:-1]])
        df["add_deep_here"] = np.concatenate([[False], ((zz <= -1) & base)[:-1]])
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
        for c in ["enter_here", "exit_here", "member", "add_here",
                  "add_deep_here"]:
            a[c] = g[c].fillna(False).astype(bool).to_numpy()
        a["has_bar"] = has_bar
        a["last_t"] = int(np.nonzero(has_bar)[0][-1])
        U[tk] = a
    return U, cal


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


# ---------------------------------------------------------------- slot account (04_sim mechanics + optional adds)
def run_account(U, cal, cand_by_day, n_slots, pickmap=None, rng=None,
                bench_by_day=None, adds_key=None, adds_rng=None,
                adds_prob=0.0):
    cash, pos = CAPITAL, {}
    daily, trades = [], []
    n_adds = n_elig = 0
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

        def units():
            return sum(1 + p.get("added", 0) for p in pos.values())

        def mark_now():
            return cash + sum(q["shares"] *
                              (U[k]["Open"][t] if U[k]["has_bar"][t]
                               else q["last_close"]) for k, q in pos.items())

        # adds first -- the doubling-down thesis gives held weakness priority
        if adds_key is not None or adds_rng is not None:
            for tk in sorted(pos):
                p = pos[tk]
                a = U[tk]
                if p.get("added") or not a["has_bar"][t] or not a["member"][t]:
                    continue
                if t == 0 or not a["has_bar"][t - 1]:   # halt-reopen stale tape
                    continue
                if a["prev_close"][t] != a["prev_close"][t]:
                    continue
                if units() >= n_slots:
                    break
                n_elig += 1
                fire = (adds_rng.random() < adds_prob) if adds_rng is not None \
                    else a[adds_key][t]
                if not fire:
                    continue
                px = a["Open"][t]
                if not np.isfinite(px):
                    continue
                alloc = min(mark_now() / n_slots, cash)
                if alloc < 1.0:
                    continue
                sh_new = alloc / (px * (1 + COST))
                p["entry_px"] = ((p["entry_px"] * p["shares"] + px * sh_new)
                                 / (p["shares"] + sh_new))
                p["shares"] += sh_new
                p["added"] = 1
                p["add_date"] = str(cal[t].date())
                cash -= alloc
                n_adds += 1

        cands = [tk for tk in cand_by_day[t] if tk not in pos]
        if bench_by_day is not None:
            cands = [tk for tk in cands if tk in bench_by_day[t]]
        free = max(n_slots - units(), 0)
        if rng is not None:
            if len(cands) > 1:
                cands = list(rng.permutation(cands))
        elif pickmap is not None:
            cands.sort(key=lambda k: (-pickmap.get((k, t), -np.inf), k))
        for tk in cands[:free]:
            a = U[tk]
            px = a["Open"][t]
            alloc = min(mark_now() / n_slots, cash)
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
    return daily, trades, n_adds, n_elig


# ---------------------------------------------------------------- rebalancing account (buy everything, equal weight)
def run_account_rebal(U, cal, cand_by_day, bench_by_day=None):
    """No slot cap, no selector: every tradeable candidacy is bought, and on
    any day the composition changes the whole book is resized to equal
    weight at the open, COST charged on every resize leg."""
    cash, pos = CAPITAL, {}
    daily, trades = [], []
    for t in range(len(cal)):
        changed = False
        for tk in list(pos):
            a, p = U[tk], pos[tk]
            if not a["has_bar"][t]:
                if t > a["last_t"] or t - p.get("last_i", t) > 10:
                    cash += p["shares"] * p["last_close"] * (1 - COST)
                    trades.append({**p, "exit_date": p["last_date"],
                                   "exit_px": p["last_close"], "reason": "delisted"})
                    del pos[tk]
                    changed = True
                continue
            if not a["member"][t]:
                px = (a["Open"][t] if a["prev_close"][t] == a["prev_close"][t]
                      else p["last_close"])
                cash += p["shares"] * px * (1 - COST)
                trades.append({**p, "exit_date": str(cal[t].date()),
                               "exit_px": px, "reason": "index_drop"})
                del pos[tk]
                changed = True
                continue
            if a["prev_close"][t] != a["prev_close"][t]:
                cash += p["shares"] * p["last_close"] * (1 - COST)
                trades.append({**p, "exit_date": str(cal[t].date()),
                               "exit_px": p["last_close"], "reason": "data_break"})
                del pos[tk]
                changed = True
                continue
            if a["exit_here"][t]:
                px = a["Open"][t]
                cash += p["shares"] * px * (1 - COST)
                trades.append({**p, "exit_date": str(cal[t].date()),
                               "exit_px": px, "reason": "rollback"})
                del pos[tk]
                changed = True
        cands = [tk for tk in cand_by_day[t] if tk not in pos]
        if bench_by_day is not None:
            cands = [tk for tk in cands if tk in bench_by_day[t]]
        for tk in cands:
            a = U[tk]
            if not np.isfinite(a["Open"][t]):
                continue
            pos[tk] = {"ticker": tk, "entry_date": str(cal[t].date()),
                       "entry_px": a["Open"][t], "shares": 0.0,
                       "last_close": a["Close"][t], "last_i": t,
                       "last_date": str(cal[t].date())}
            changed = True
        if changed and pos:
            tradeable = {tk: p for tk, p in pos.items()
                         if U[tk]["has_bar"][t] and np.isfinite(U[tk]["Open"][t])}
            mark = cash + sum(
                p["shares"] * (U[tk]["Open"][t] if tk in tradeable
                               else p["last_close"]) for tk, p in pos.items())
            frozen = sum(p["shares"] * p["last_close"] for tk, p in pos.items()
                         if tk not in tradeable)
            target = max(mark - frozen, 0.0) / max(len(tradeable), 1)
            # sells first, then buys, so cash never goes negative
            legs = []
            for tk, p in tradeable.items():
                px = U[tk]["Open"][t]
                dv = target - p["shares"] * px
                legs.append((dv, tk, px))
            for dv, tk, px in sorted(legs):
                p = pos[tk]
                if dv < -1.0:
                    sell = min(-dv, p["shares"] * px)
                    p["shares"] -= sell / px
                    cash += sell * (1 - COST)
                elif dv > 1.0:
                    spend = min(dv, max(cash, 0.0))
                    if spend > 1.0:
                        p["shares"] += spend / (px * (1 + COST))
                        cash -= spend
            for tk in [k for k, p in pos.items() if p["shares"] <= 0]:
                del pos[tk]
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
    return daily, trades, 0


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
            "pct_days_invested": round(100 * float((npos > 0).mean()), 1),
            "avg_positions": round(float(npos.mean()), 2),
            "trades": len(closed),
            "win_rate": round(float((tr > 0).mean()), 3) if len(tr) else np.nan,
            "med_trade_ret": (round(float(np.median(tr)), 4)
                              if len(tr) else np.nan)}


def etf_series(cal, symbol):
    for path in [os.path.join(OUTDIR, f"{symbol.lower()}_cache.csv"),
                 os.path.join(DATADIR, f"{symbol}.csv")]:
        if os.path.exists(path):
            df = pd.read_csv(path).dropna(subset=["Open", "Close"])
            df["Date"] = pd.to_datetime(df["Date"])
            df = df[df["Date"] <= pd.Timestamp(ASOF)].set_index("Date")
            s = df.reindex(cal)[["Open", "Close"]]
            if s["Close"].notna().mean() > 0.9:
                return s
    return None


# ---------------------------------------------------------------- run
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
print(f"{sum(len(c) for c in cand_by_day)} tradeable signal-days")

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
            pick_vol[(tk, t)] = -float(v)

summary, all_daily, all_trades = [], [], []


def record(name, daily, trades, n_adds=0, n_elig=0, blank_trade_stats=False):
    m = metrics(daily, trades)
    m["adds"] = n_adds
    if blank_trade_stats:      # resizes make these name-path, not account, returns
        m["win_rate"] = np.nan
        m["med_trade_ret"] = np.nan
    summary.append({"run": name, **m})
    for d in daily:
        all_daily.append({"run": name, "date": str(d["date"].date()),
                          "equity": round(d["equity"], 2), "n_pos": d["n_pos"]})
    for x in trades:
        all_trades.append({"run": name,
                           **{k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in x.items()
                              if k not in ("last_close", "last_i")}})
    oos = m.get("oos_cagr", float("nan"))
    print(f"  {name:<16} final ${m['final_equity']:>11,.0f}  "
          f"cagr {100 * m['cagr']:>6.2f}%  shp {m['sharpe']:>6}  "
          f"dd {100 * m['maxdd']:>6.1f}%  | 2022+ cagr {100 * oos:>6.2f}%  "
          f"adds {n_adds}")
    return m


print("\n--- controls ---")
for sym in ["SPY", "RSP"]:
    s = etf_series(cal, sym)
    if s is None:
        print(f"  {sym}: no cached series -- control skipped")
        continue
    op = s["Open"].to_numpy(float)
    clx = s["Close"].ffill().to_numpy(float)
    fin = np.nonzero(np.isfinite(op))[0]
    first = int(fin[0])
    sh = CAPITAL / (op[first] * (1 + COST))
    d = [{"date": cal[t], "cash": CAPITAL if t < first else 0.0,
          "n_pos": 0 if t < first else 1,
          "equity": CAPITAL if t < first else sh * clx[t]}
         for t in range(len(cal))]
    record(sym, d, [{"ticker": sym, "entry_date": str(cal[first].date()),
                     "entry_px": op[first], "shares": sh, "exit_date": "",
                     "exit_px": clx[-1], "reason": "open",
                     "last_date": str(cal[-1].date())}])

print("\n--- baseline + reconciliation ---")
m_base = record("RANKVOL_BENCH_N10", *run_account(U, cal, cand_by_day, N_SLOTS,
                                                  pickmap=pick_vol,
                                                  bench_by_day=bench))
gap = abs(m_base["final_equity"] - PUBLISHED_FINAL) / PUBLISHED_FINAL
if gap > RECON_TOL:
    raise SystemExit(
        f"RECONCILIATION FAILED: ${m_base['final_equity']:,.0f} vs published "
        f"${PUBLISHED_FINAL:,.0f} ({100 * gap:.2f}% off). Arms not comparable.")
print(f"  reconciliation vs published: {100 * gap:.2f}% off -- PASS")

print("\n--- ARM 1: adds (family primary) ---")
d_a, t_a, n_add, n_el = run_account(U, cal, cand_by_day, N_SLOTS,
                                    pickmap=pick_vol, bench_by_day=bench,
                                    adds_key="add_here")
m_adds = record("ADDS_BENCH_N10", d_a, t_a, n_add, n_el)
m_deep = record("ADDS_DEEP_BENCH_N10",
                *run_account(U, cal, cand_by_day, N_SLOTS, pickmap=pick_vol,
                             bench_by_day=bench, adds_key="add_deep_here"))
p_hat = n_add / max(n_el, 1)
print(f"  adds intensity: {n_add}/{n_el} eligible position-days "
      f"(p = {p_hat:.5f})")
print(f"  random-add envelope ({ENV_SEEDS} seeds, matched intensity)...")
add_dist = []
for s in range(ENV_SEEDS):
    arng = np.random.default_rng(np.random.SeedSequence([SEED, 8, 10, s, 2]))
    d, tr, _, _ = run_account(U, cal, cand_by_day, N_SLOTS, pickmap=pick_vol,
                              bench_by_day=bench, adds_rng=arng,
                              adds_prob=p_hat)
    add_dist.append(metrics(d, tr)["final_equity"])
    if (s + 1) % 25 == 0:
        print(f"    {s + 1}/{ENV_SEEDS}")
p95_add = float(np.percentile(add_dist, 95))
print(f"  random-add envelope: median ${np.median(add_dist):,.0f}, "
      f"p95 ${p95_add:,.0f}")

print("\n--- ARM 2: 2% slots ---")
m_50 = record("RANKVOL_BENCH_N50", *run_account(U, cal, cand_by_day, N_BIG,
                                                pickmap=pick_vol,
                                                bench_by_day=bench))
print(f"  RAND N50 envelope ({ENV_SEEDS} seeds)...")
dist = []
for s in range(ENV_SEEDS):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 8, N_BIG, s]))
    d, tr, _, _ = run_account(U, cal, cand_by_day, N_BIG, rng=rng,
                              bench_by_day=bench)
    dist.append(metrics(d, tr)["final_equity"])
    if (s + 1) % 25 == 0:
        print(f"    {s + 1}/{ENV_SEEDS}")
p95_50 = float(np.percentile(dist, 95))
print(f"  N50 envelope: median ${np.median(dist):,.0f}, p95 ${p95_50:,.0f}")

print("\n--- ARM 3: buy everything, equal-weight resize ---")
m_ew_b = record("ALL_EW_BENCH", *run_account_rebal(U, cal, cand_by_day,
                                                   bench_by_day=bench),
                blank_trade_stats=True)
m_ew_a = record("ALL_EW_ALL", *run_account_rebal(U, cal, cand_by_day),
                blank_trade_stats=True)
print("  (ALL_EW win-rate/median-trade blanked: resizes make them "
      "name-path, not account-realized, returns)")

print("\nVERDICTS (pre-registered):")
beats = (m_adds["cagr"] > m_base["cagr"]
         and m_adds.get("oos_cagr", -9) > m_base.get("oos_cagr", -9))
clears_add = m_adds["final_equity"] > p95_add
if beats and clears_add:
    v_adds = "DISPLACES the never-add baseline (beats both windows, clears the random-add p95)"
elif beats:
    v_adds = "beats the baseline but NOT the random-add envelope -- deployment, not tape"
else:
    v_adds = "does not beat the never-add baseline"
deep_beats = (m_deep["cagr"] > m_base["cagr"]
              and m_deep.get("oos_cagr", -9) > m_base.get("oos_cagr", -9))
print(f"  ADDS         {v_adds}  (deep variant "
      f"{'beats' if deep_beats else 'does not beat'} the baseline)")
n50_ok = (m_50["final_equity"] > p95_50
          and m_50["sharpe"] >= m_base["sharpe"] - 0.1
          and m_50.get("oos_sharpe", -9) >= m_base.get("oos_sharpe", 9) - 0.1)
print(f"  N50 (2%)     {'VIABLE (clears envelope; Sharpe no more than 0.1 below N10)' if n50_ok else 'fails the viability bar'}")
print(f"  ALL_EW       descriptive -- benched ${m_ew_b['final_equity']:,.0f} vs "
      f"N10 ${m_base['final_equity']:,.0f} vs RSP-style everything; read the table")

pd.DataFrame(summary).to_csv(os.path.join(OUTDIR, "06_capacity_summary.csv"),
                             index=False)
pd.DataFrame(all_daily).to_csv(os.path.join(OUTDIR, "06_capacity_daily.csv"),
                               index=False)
pd.DataFrame(all_trades).to_csv(os.path.join(OUTDIR, "06_capacity_trades.csv"),
                                index=False)
pd.DataFrame({"n_slots": N_BIG, "seed": range(len(dist)),
              "final_equity": dist}).to_csv(
    os.path.join(OUTDIR, "06_capacity_rand50.csv"), index=False)
pd.DataFrame({"seed": range(len(add_dist)), "p_hat": p_hat,
              "final_equity": add_dist}).to_csv(
    os.path.join(OUTDIR, "06_capacity_rand_adds.csv"), index=False)
print(f"\nwritten: {OUTDIR}/06_capacity_summary.csv, _daily, _trades, "
      f"_rand50, _rand_adds")
