"""08_adds_settle.py -- the ADDS settlement: 500-seed null + the variant family.

WHY THIS EXISTS
  06_capacity_test.py left ADDS at p92 of a 100-seed matched-intensity
  random-add envelope, with attribution showing the margin concentrated in
  two doubled positions. At 100 seeds the p95 line itself carries sampling
  error. This run settles it: same arm, same null, five times the seeds --
  the ruler sharpens, the arm does not move.

PRE-REGISTRATION (fixed before the run)
  family    THREE arms, declared here, all of them; nothing added after:
    ADDS       the original: held name's tape shows z < 0, dmad_5 > 0,
               price_share_5 >= 0.50 at a fresh signal close -> add ONE slot
               unit at the next open; one add per position; adds take
               priority over queue buys. (Identical to 06.)
    ADDS_HALF  same trigger, HALF a slot unit per add -- the "tape is right,
               size is wrong" refinement.
    ADDS_LAST  same trigger, full unit, but adds come AFTER queue buys --
               fresh signals outrank doubling. (The redeploy-first fork.)
  nulls     matched-intensity random-add envelopes per arm, identical
            mechanics with the tape replaced by a coin of the same empirical
            intensity: ADDS 500 seeds (SeedSequence([7, 10, s, 0])),
            variants 200 seeds each ([7, 10, s, 1] half, [7, 10, s, 2] last).
  verdict   an arm is CROWNED only if it beats RANKVOL_BENCH_N10 on BOTH
            windows AND its final equity clears its own envelope's p95.
            FAMILY ACCOUNTING: three attempts at p95 carry ~14% family
            false-positive odds; a single crowned arm is therefore reported
            with its family-adjusted standing (clears p98.3 = family-safe;
            p95-p98.3 = crowned with the multiplicity caveat printed).
  attribution  per-add ledger written for every arm (results/08_adds_ledger
            .csv): ticker, add date, position outcome, dollar contribution --
            the two-trade-engine question answered in a table, per arm.
  reconcile RANKVOL_BENCH_N10 must reproduce $552,888 within 1.5% first.

RUN (from the repo root)
  export PORTFOLIO_DATADIR=/path/to/prepped/data
  python3 08_adds_settle.py            # ~900 sims; expect several hours
  outputs results/08_settle_summary.csv, 08_settle_envelopes.csv,
          08_adds_ledger.csv
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
TOP_Q = int(os.environ.get("SIM_TOP_Q", "100"))
COST = float(os.environ.get("SIM_COST_BPS", "5")) / 10000.0
CAPITAL = float(os.environ.get("SIM_CAPITAL", "100000"))
PUBLISHED_FINAL = 552888.0
RECON_TOL = 0.015
SEED = 7
SEEDS_MAIN = int(os.environ.get("SETTLE_SEEDS_MAIN", "500"))
SEEDS_VAR = int(os.environ.get("SETTLE_SEEDS_VAR", "200"))
os.makedirs(OUTDIR, exist_ok=True)


def truthy(series):
    return series.astype(str).str.strip().isin(["True", "true", "1", "1.0"]).to_numpy()


# ---------------------------------------------------------------- universe (06 mechanics verbatim)
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
        for c in ["enter_here", "exit_here", "member", "add_here"]:
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


def run_account(U, cal, cand_by_day, pick_vol, bench, adds=False,
                add_frac=1.0, adds_first=True, adds_rng=None, adds_prob=0.0,
                want_ledger=False):
    """06 mechanics; adds sized add_frac of a slot; adds_first toggles
    priority vs queue buys."""
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

    def do_adds(t):
        nonlocal cash, n_adds, n_elig
        for tk in sorted(pos):
            p = pos[tk]
            a = U[tk]
            if p.get("added") or not a["has_bar"][t] or not a["member"][t]:
                continue
            if t == 0 or not a["has_bar"][t - 1]:
                continue
            if a["prev_close"][t] != a["prev_close"][t]:
                continue
            if units() >= N_SLOTS:
                break
            n_elig += 1
            fire = (adds_rng.random() < adds_prob) if adds_rng is not None \
                else a["add_here"][t]
            if not fire:
                continue
            px = a["Open"][t]
            if not np.isfinite(px):
                continue
            alloc = min(mark_now(t) / N_SLOTS * add_frac, cash)
            if alloc < 1.0:
                continue
            sh_new = alloc / (px * (1 + COST))
            if want_ledger:
                ledger.append({"ticker": tk, "add_date": str(cal[t].date()),
                               "add_px": round(px, 4),
                               "add_dollars": round(alloc, 2),
                               "orig_entry": p["entry_date"]})
            p["entry_px"] = ((p["entry_px"] * p["shares"] + px * sh_new)
                             / (p["shares"] + sh_new))
            p["shares"] += sh_new
            p["added"] = 1
            p["afrac"] = add_frac
            p["add_date"] = str(cal[t].date())
            cash -= alloc
            n_adds += 1

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
    closed = [x for x in trades if x["reason"] != "open"]
    tr = np.array([x["exit_px"] * (1 - COST) / (x["entry_px"] * (1 + COST)) - 1
                   for x in closed]) if closed else np.array([])
    full = window_stats(dates, eq)
    late = {f"oos_{k}": v for k, v in
            window_stats(dates, eq, np.datetime64(SPLIT)).items()}
    return {"final_equity": round(float(eq[-1]), 2), **full, **late,
            "trades": len(closed),
            "win_rate": round(float((tr > 0).mean()), 3) if len(tr) else np.nan,
            "med_trade_ret": (round(float(np.median(tr)), 6)
                              if len(tr) else np.nan)}


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

summary, env_rows, ledger_rows = [], [], []


def record(name, res, keep_ledger=False):
    daily, trades, n_adds, n_elig, ledger = res
    m = metrics(daily, trades)
    m["adds"] = n_adds
    summary.append({"run": name, **m})
    if keep_ledger:
        t_of = {x["ticker"] + x.get("add_date", ""): x for x in ledger}
        closed = {x["ticker"] + x["entry_date"]: x for x in trades}
        for led in ledger:
            key = None
            for x in trades:
                if (x["ticker"] == led["ticker"]
                        and x.get("add_date") == led["add_date"]):
                    key = x
                    break
            row = {"arm": name, **led}
            if key is not None:
                row["exit_date"] = key["exit_date"]
                row["exit_px"] = round(float(key["exit_px"]), 4)
                row["pos_ret"] = round(key["exit_px"] * (1 - COST)
                                       / (key["entry_px"] * (1 + COST)) - 1, 4)
                row["pos_pnl"] = round(key["shares"]
                                       * (key["exit_px"] - key["entry_px"]), 2)
            ledger_rows.append(row)
    oos = m.get("oos_cagr", float("nan"))
    print(f"  {name:<12} final ${m['final_equity']:>11,.0f}  "
          f"cagr {100 * m['cagr']:>6.2f}%  | 2022+ {100 * oos:>6.2f}%  "
          f"adds {n_adds}")
    return m, n_adds, n_elig


print("\nbaseline + reconciliation...")
m_base, _, _ = record("RANKVOL", run_account(U, cal, cand_by_day, pick_vol,
                                             bench))
gap = abs(m_base["final_equity"] - PUBLISHED_FINAL) / PUBLISHED_FINAL
if gap > RECON_TOL:
    raise SystemExit(f"RECONCILIATION FAILED ({100 * gap:.2f}% off).")
print(f"  reconciliation: {100 * gap:.2f}% off -- PASS")

ARMS = [("ADDS", dict(adds=True, add_frac=1.0, adds_first=True), SEEDS_MAIN, 0),
        ("ADDS_HALF", dict(adds=True, add_frac=0.5, adds_first=True), SEEDS_VAR, 1),
        ("ADDS_LAST", dict(adds=True, add_frac=1.0, adds_first=False), SEEDS_VAR, 2)]

verdicts = []
for name, kw, n_seeds, sid in ARMS:
    print(f"\n--- {name} ---")
    m_arm, n_add, n_el = record(name, run_account(
        U, cal, cand_by_day, pick_vol, bench, want_ledger=True, **kw),
        keep_ledger=True)
    p_hat = n_add / max(n_el, 1)
    print(f"  intensity {n_add}/{n_el} (p={p_hat:.5f}); envelope {n_seeds} seeds...")
    dist = []
    for s in range(n_seeds):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 10, s, sid]))
        d, tr, _, _, _ = run_account(U, cal, cand_by_day, pick_vol, bench,
                                     adds_rng=rng, adds_prob=p_hat,
                                     **{k: v for k, v in kw.items()
                                        if k != "adds"}, adds=True)
        fe = metrics(d, tr)["final_equity"]
        dist.append(fe)
        env_rows.append({"arm": name, "seed": s, "final_equity": fe})
        if (s + 1) % 50 == 0:
            print(f"    {s + 1}/{n_seeds}")
    d = np.array(dist)
    p95 = float(np.percentile(d, 95))
    p983 = float(np.percentile(d, 98.3))
    pct = 100 * float((d < m_arm["final_equity"]).mean())
    beats = (m_arm["cagr"] > m_base["cagr"]
             and m_arm.get("oos_cagr", -9) > m_base.get("oos_cagr", 9))
    if beats and m_arm["final_equity"] > p983:
        v = "CROWNED (family-safe: clears p98.3)"
    elif beats and m_arm["final_equity"] > p95:
        v = "crowned at p95 WITH multiplicity caveat (below family-safe p98.3)"
    elif beats:
        v = "beats baseline, fails its envelope"
    else:
        v = "does not beat baseline"
    verdicts.append((name, pct, p95, v))
    print(f"  arm sits at p{pct:.1f}; null p95 ${p95:,.0f}, p98.3 ${p983:,.0f}")

print("\nVERDICTS (pre-registered):")
for name, pct, p95, v in verdicts:
    print(f"  {name:<12} p{pct:.1f}  {v}")

pd.DataFrame(summary).to_csv(os.path.join(OUTDIR, "08_settle_summary.csv"),
                             index=False)
pd.DataFrame(env_rows).to_csv(os.path.join(OUTDIR, "08_settle_envelopes.csv"),
                              index=False)
pd.DataFrame(ledger_rows).to_csv(os.path.join(OUTDIR, "08_adds_ledger.csv"),
                                 index=False)
print(f"\nwritten: {OUTDIR}/08_settle_summary.csv, 08_settle_envelopes.csv, "
      f"08_adds_ledger.csv")
