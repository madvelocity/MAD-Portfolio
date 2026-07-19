#!/usr/bin/env python3
"""
03_score.py  --  the rank-based score: walk-forward models fit on 02_ranking.py's
daily rank table, judged at the trade level and in a 10-slot book.

02_ranking.py's verdict: when signals fire, VOLUME rank picks winners (top-3 edge
+2.8 bps/day, t ~ 4.6), SIGNAL-TO-PROFIT rank adds modestly, RETURN ranks are
dead.  This script turns that verdict into a score: one number per candidate,
every day, fit strictly out of sample.

THREE NESTED MODELS, fit side by side in the same monthly walk-forward (ridge on
bps/day held, winsorized +/-250; refits on the first trading day of each month
using only trades resolved before it; MIN_TRAIN resolved trades before any score;
delisting resolutions lagged ~10 days):
    score_R    RANK FEATURES ONLY, from results/02_ranks.csv at the signal
               close, as daily percentiles (rank / that day's universe size):
               pct_vol, pct_s2p_3, pct_s2p_5, pct_s2p_10, pct_s2p_15, pct_s2p_20
               (return ranks excluded per 01's verdict).
               -> can the rank table ALONE carry the picking layer?
    score_S    the signal-state five: share_5, mad, z, rel_vol (5d/60d), ret_20
               -> the established baseline.
    score_RS   both.
               -> does the table add to what the signal's geometry knows?
Rank percentiles are outlier-proof and self-normalizing across eras; NaN features
(s2p burn-in etc.) impute to the training-pool mean, standardized per refit.

RANK JOIN LEGALITY: a signal prints at a name's close; the rank row used is that
SAME bar's row (the table is published at that close), looked up by the signal
bar's own date -- never the entry day's.  Everything else is knowable at the
close before entry.

VERDICT FRAMES (both printed):
  trade level   same-day-cohort Spearman IC and top-3-per-day edge vs realized
                bps/day, t-stats on monthly means -- the powered frame
  the book      N=10 slots, next-open mechanics inherited verbatim from the
                verified engine; SCORE_RS / SCORE_R / SCORE_S books vs a
                1000-seed random-composition null (exact Phipson & Smyth p;
                seeds SeedSequence([SEED, 5, 10, s]), disjoint from all priors)

INPUT: prepped per-ticker files (PORTFOLIO_DATADIR or ./data) + results/02_ranks.csv
(run 02_ranking.py first).
    python3 03_score.py
Outputs -> results/03_scores.csv           every signal: features, outcome, scores
           results/03_verdict.csv          trade-level IC / top-3 verdict table
           results/03_variant_summary.csv  the book results + null placement
           results/03_rand_dist.csv        1000-seed null metrics
           results/03_trades.csv           book trades (non-random variants)
           results/03_daily.csv            daily book series (non-random variants)
Console: coverage, all three models' final standardized coefficients, both
verdict frames.  Reruns deterministic per machine.
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
SHARE_MIN = 0.50
N_BOOK = 10
N_RAND = int(os.environ.get("SCORE_N_RAND", "1000"))
SEED = 7
COST = 0.0                              # book frame is GROSS, as 01-03 were
MIN_TRAIN = int(os.environ.get("SCORE_MIN_TRAIN", "300"))
RIDGE = 1e-3
BPS_WINSOR = 250.0
FEAT_R = ["pct_vol", "pct_s2p_3", "pct_s2p_5", "pct_s2p_10",
          "pct_s2p_15", "pct_s2p_20"]                                 # FROZEN
FEAT_S = ["share_sig", "mad_sig", "z_sig", "relv_sig", "r20_sig"]     # FROZEN
FEAT_RS = FEAT_R + FEAT_S
RANKCOLS = {"rank_vol": "pct_vol", "rank_s2p_3": "pct_s2p_3",
            "rank_s2p_5": "pct_s2p_5", "rank_s2p_10": "pct_s2p_10",
            "rank_s2p_15": "pct_s2p_15", "rank_s2p_20": "pct_s2p_20"}
USECOLS = ["Date", "Open", "Close", "Volume", "mad", "z", "price_share_5",
           "sig_cross_up", "sig_rollback", "in_index"]
os.makedirs(OUTDIR, exist_ok=True)


def truthy(series):
    return series.astype(str).str.strip().isin(["True", "true", "1", "1.0"]).to_numpy()


# ---------------------------------------------------------------- loader (verified)
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
        vol = df["Volume"].astype(float)
        relv = vol.rolling(5).mean() / vol.rolling(60).mean()
        r20 = df["Close"] / df["Close"].shift(20) - 1.0
        for col, feat in [("mad_sig", df["mad"]), ("z_sig", df["z"]),
                          ("share_sig", df["price_share_5"]),
                          ("relv_sig", relv), ("r20_sig", r20)]:
            df[col] = feat.shift(1)
        df["prev_close"] = df["Close"].shift(1)
        ret1 = df["Close"] / df["prev_close"]
        gap_days = df.index.to_series().diff().dt.days
        bad = (ret1 > 5.0) | (ret1 < 0.2) | (gap_days > 30)
        for dt, rv in ret1[bad].items():
            splices.append((tk, str(dt.date()), round(float(rv), 1)))
        df.loc[bad, "prev_close"] = np.nan
        if bad.any():
            b = bad.astype(float)
            for w, col in [(21, "r20_sig"), (61, "relv_sig")]:
                cw = (b.rolling(w, min_periods=1).max() > 0).shift(1).fillna(False)
                df.loc[cw.astype(bool), col] = np.nan
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
    cols = ["Open", "Close", "prev_close", "share_sig", "mad_sig", "z_sig",
            "relv_sig", "r20_sig"]
    U = {}
    for tk, df in frames.items():
        g = df.reindex(cal)
        has_bar = g["Close"].notna().to_numpy()
        if not has_bar.any():
            continue
        a = {c: g[c].to_numpy(float) for c in cols}
        for c in ["enter_here", "exit_here", "member"]:
            a[c] = g[c].fillna(False).astype(bool).to_numpy()
        a["has_bar"] = has_bar
        a["last_t"] = int(np.nonzero(has_bar)[0][-1])
        U[tk] = a
    return U, cal


# ---------------------------------------------------------------- signal ledger
def signal_outcomes(U, cal, tickers):
    """Every gated firing walked independently (verified mechanics); records the
    SIGNAL BAR's date (sig_date) for the rank join."""
    end_t = len(cal) - 1
    rows = []
    for tk in tickers:
        a = U[tk]
        bars = np.nonzero(a["has_bar"])[0]
        prev_bar = {}                              # entry bar -> its previous bar
        for i in range(1, len(bars)):
            prev_bar[bars[i]] = bars[i - 1]
        idxs = np.nonzero(a["enter_here"] & a["member"]
                          & np.isfinite(a["Open"]) & np.isfinite(a["Close"]))[0]
        for t in idxs:
            sb = prev_bar.get(t)                   # None: entry on the name's
            eq = a["Close"][t] / a["Open"][t]      # first in-window bar -- kept,
                                                   # rank features just impute
            days, resolved, reason, exit_t = 1, False, "open", None
            u = t + 1
            while u <= a["last_t"]:
                if not a["has_bar"][u]:
                    u += 1
                    continue
                if not a["member"][u]:             # removal: sell at the open
                    if a["prev_close"][u] == a["prev_close"][u]:
                        eq *= a["Open"][u] / a["prev_close"][u]
                        days += 1
                    resolved, reason, exit_t = True, "index_drop", u
                    break
                if a["prev_close"][u] != a["prev_close"][u]:
                    lb = u - 1
                    while lb > t and not a["has_bar"][lb]:
                        lb -= 1
                    resolved, reason, exit_t = True, "data_break", lb
                    break
                if a["exit_here"][u]:
                    eq *= a["Open"][u] / a["prev_close"][u]
                    days += 1
                    resolved, reason, exit_t = True, "rollback", u
                    break
                eq *= a["Close"][u] / a["prev_close"][u]
                days += 1
                u += 1
            if not resolved and a["last_t"] < end_t:
                resolved, reason, exit_t = True, "delisted", a["last_t"]
            ret = eq - 1.0
            rows.append({"ticker": tk, "entry_t": t,
                         "entry_date": str(cal[t].date()),
                         "sig_date": str(cal[sb].date()) if sb is not None else "",
                         "exit_date": (str(cal[exit_t].date())
                                       if exit_t is not None else ""),
                         "exit_reason": reason, "resolved": resolved,
                         "trade_ret": ret, "days": days,
                         "bps_day": 10000 * np.log(1 + ret) / days,
                         **{f: float(a[f][t]) for f in FEAT_S}})
    return pd.DataFrame(rows).sort_values(["entry_t", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------- rank features
def join_ranks(sig):
    if not os.path.exists(RANKS_CSV):
        raise SystemExit(f"{RANKS_CSV} not found -- run 02_ranking.py first.")
    ranks = pd.read_csv(RANKS_CSV, usecols=["Date", "ticker"] + list(RANKCOLS))
    n_day = ranks.groupby("Date")["ticker"].transform("size")
    for rc, pc in RANKCOLS.items():
        ranks[pc] = ranks[rc] / n_day              # percentile: low = best
    ranks = ranks[["Date", "ticker"] + list(RANKCOLS.values())]
    out = sig.merge(ranks, left_on=["sig_date", "ticker"],
                    right_on=["Date", "ticker"], how="left").drop(columns=["Date"])
    hit = out["pct_vol"].notna().mean()
    print(f"  rank rows found for {100 * hit:.1f}% of signals")
    return out


# ---------------------------------------------------------------- fits (verified)
def design(df, feats, mu=None, sd=None):
    X = df[feats].to_numpy(float)
    if mu is None:
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        mu[~np.isfinite(mu)] = 0.0
    idx = np.where(np.isnan(X))
    X[idx] = np.take(mu, idx[1])
    Z = (X - mu) / sd
    return np.hstack([np.ones((len(Z), 1)), Z]), mu, sd


def fit_ridge(X, y):
    return np.linalg.solve(X.T @ X + RIDGE * np.eye(X.shape[1]), X.T @ y)


MODELS = [("score_R", FEAT_R), ("score_S", FEAT_S), ("score_RS", FEAT_RS)]


def walk_forward_scores(sig):
    sig = sig.copy()
    for name, _ in MODELS:
        sig[name] = np.nan
    sig["n_train"] = 0
    months = pd.to_datetime(sig["entry_date"]).dt.to_period("M")
    exit_map = pd.to_datetime(sig["exit_date"], errors="coerce")
    avail = exit_map + pd.to_timedelta(
        np.where(sig["exit_reason"] == "delisted", 11, 0), unit="D")
    last_coefs = None
    for m in months.unique():
        cur = months == m
        boundary = pd.Timestamp(m.start_time)
        pool = sig[(sig["resolved"]) & (avail < boundary)]
        pool = pool[np.isfinite(pool["bps_day"])]
        if len(pool) < MIN_TRAIN:
            continue
        y = np.clip(pool["bps_day"].to_numpy(), -BPS_WINSOR, BPS_WINSOR)
        coefs = {}
        for name, feats in MODELS:
            Xtr, mu, sd = design(pool, feats)
            b = fit_ridge(Xtr, y)
            Xcur, _, _ = design(sig[cur], feats, mu, sd)
            sig.loc[cur, name] = Xcur @ b
            coefs[name] = b
        sig.loc[cur, "n_train"] = len(pool)
        last_coefs = (coefs, len(pool))
    return sig, last_coefs


# ---------------------------------------------------------------- trade-level verdict
def trade_verdict(sig):
    d = sig[sig["resolved"] & np.isfinite(sig["bps_day"])]
    rows = []
    for name, _ in MODELS:
        ics, eds = [], []
        for day, g in d.groupby("entry_date"):
            g = g.dropna(subset=[name, "bps_day"])
            if len(g) < 5:
                continue
            ic = g[name].rank().corr(g["bps_day"].rank())
            if ic == ic:
                ics.append((day[:7], ic))
            gg = g.sort_values(name, ascending=False)
            eds.append((day[:7], gg.head(3)["bps_day"].mean() - g["bps_day"].mean()))
        fi = pd.DataFrame(ics, columns=["mo", "v"]).groupby("mo").v.mean()
        fe = pd.DataFrame(eds, columns=["mo", "v"]).groupby("mo").v.mean()
        rows.append({"model": name,
                     "ic": round(float(np.mean([v for _, v in ics])), 4),
                     "t_ic": round(float(fi.mean() / (fi.std(ddof=1) / np.sqrt(len(fi)))), 2),
                     "top3_edge_bps": round(float(np.mean([v for _, v in eds])), 2),
                     "t_edge": round(float(fe.mean() / (fe.std(ddof=1) / np.sqrt(len(fe)))), 2),
                     "coverage_pct": round(100 * float(sig[name].notna().mean()), 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the book (verified)
def simulate(U, cal, cand_by_day, rule, rng=None, scores=None):
    held = {}
    daily, trades = [], []
    for t in range(len(cal)):
        rets, n_exit_rb, n_exit = [], 0, 0
        for tk in list(held):
            a = U[tk]
            if not a["has_bar"][t]:
                if t > a["last_t"]:
                    tr = held.pop(tk)
                    trades.append({**tr, "exit_date": tr["last_date"],
                                   "exit_reason": "delisted",
                                   "trade_ret": tr["eq"] - 1.0})
                    n_exit += 1
                continue
            if not a["member"][t]:
                booked = a["prev_close"][t] == a["prev_close"][t]
                r = a["Open"][t] / a["prev_close"][t] - 1.0 if booked else 0.0
                tr = held.pop(tk)
                tr["eq"] *= (1.0 + r)
                if booked:
                    tr["days"] += 1                # the open leg is a held day
                trades.append({**tr, "exit_date": str(cal[t].date()),
                               "exit_reason": "index_drop", "trade_ret": tr["eq"] - 1.0})
                if booked:
                    rets.append(r)
                    n_exit_rb += 1
                n_exit += 1
                continue
            if a["prev_close"][t] != a["prev_close"][t]:
                tr = held.pop(tk)
                trades.append({**tr, "exit_date": str(cal[t].date()),
                               "exit_reason": "data_break", "trade_ret": tr["eq"] - 1.0})
                n_exit += 1
                continue
            if a["exit_here"][t]:
                r = a["Open"][t] / a["prev_close"][t] - 1.0 - COST
                tr = held.pop(tk)
                tr["eq"] *= (1.0 + r)
                tr["days"] += 1
                trades.append({**tr, "exit_date": str(cal[t].date()),
                               "exit_reason": "rollback", "trade_ret": tr["eq"] - 1.0})
                rets.append(r)
                n_exit_rb += 1
                n_exit += 1
        cands = [tk for tk in cand_by_day[t] if tk not in held]
        free = max(N_BOOK - len(held), 0)
        if len(cands) > free:
            if rule == "rand":
                cands = list(rng.permutation(cands))
            else:
                def skey(k):
                    s = scores.get((k, t), np.nan)
                    return (-s if s == s else np.inf, k)
                cands.sort(key=skey)
        picks = cands[:free]
        picks_set = set(picks)
        for tk in picks:
            a = U[tk]
            r = a["Close"][t] / a["Open"][t] - 1.0 - COST
            held[tk] = {"ticker": tk, "entry_date": str(cal[t].date()),
                        "eq": 1.0 + r, "days": 1, "last_date": str(cal[t].date())}
            rets.append(r)
        for tk, tr in held.items():
            if tk in picks_set:
                continue
            a = U[tk]
            if not a["has_bar"][t]:
                continue
            r = a["Close"][t] / a["prev_close"][t] - 1.0
            tr["eq"] *= (1.0 + r)
            tr["days"] += 1
            tr["last_date"] = str(cal[t].date())
            rets.append(r)
        daily.append({"date": cal[t], "port_ret": sum(rets) / N_BOOK if rets else 0.0,
                      "n_held": len(held), "n_deployed": len(held) + n_exit_rb,
                      "n_exited": n_exit})
    for tk, tr in held.items():
        trades.append({**tr, "exit_date": "", "exit_reason": "open",
                       "trade_ret": tr["eq"] - 1.0})
    return daily, trades


def metrics(daily, trades):
    r = np.array([d["port_ret"] for d in daily])
    dep = np.array([d["n_deployed"] for d in daily], float)
    eq = np.cumprod(1 + r)
    wealth = float(eq[-1])
    years = len(r) / 252.0
    sharpe = np.sqrt(252) * r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else np.nan
    eq1 = np.concatenate([[1.0], eq])
    dd = float((eq1 / np.maximum.accumulate(eq1) - 1).min())
    slot_days = float((dep / N_BOOK).sum())
    bps = 10000 * np.log(wealth) / slot_days if slot_days > 0 and wealth > 0 else np.nan
    closed = [x for x in trades if x["exit_reason"] == "rollback"]
    tr = np.array([x["trade_ret"] for x in closed])
    return {"wealth": round(wealth, 4),
            "cagr": round(wealth ** (1 / years) - 1, 4),
            "sharpe": round(sharpe, 3) if sharpe == sharpe else np.nan,
            "maxdd": round(dd, 4),
            "bps_per_slot_day": round(bps, 2) if bps == bps else np.nan,
            "bps_raw": bps, "round_trips": len(closed),
            "win_rate": round(float((tr > 0).mean()), 3) if len(tr) else np.nan,
            "med_trade_ret": round(float(np.median(tr)), 4) if len(tr) else np.nan}


# ---------------------------------------------------------------- run
print(f"loading universe from {DATADIR}/ (ASOF cap {ASOF})...")
U, cal = load_universe()
tickers = sorted(U)
print(f"{len(tickers)} names on a {len(cal)}-day calendar")

print("resolving every gated signal...")
sig = signal_outcomes(U, cal, tickers)
print(f"  {len(sig)} signals ({int(sig['resolved'].sum())} resolved)")

print("joining rank features from 02_ranks.csv...")
sig = join_ranks(sig)

print(f"walk-forward scoring: 3 nested models (MIN_TRAIN={MIN_TRAIN})...")
sig, last_coefs = walk_forward_scores(sig)
print(f"  coverage: {100 * sig['score_RS'].notna().mean():.1f}% of signals")
if last_coefs is not None:
    coefs, npool = last_coefs
    print(f"  final refit (n_train={npool}), standardized coefficients:")
    for name, feats in MODELS:
        print(f"    {name}: " + "  ".join(
            f"{n}={v:+.2f}" for n, v in zip(["const"] + feats, coefs[name])))
sig.drop(columns=["entry_t"]).to_csv(os.path.join(OUTDIR, "03_scores.csv"),
                                     index=False)

print("\n--- TRADE-LEVEL VERDICT (the powered frame) ---")
verdict = trade_verdict(sig)
print(verdict.to_string(index=False))
verdict.to_csv(os.path.join(OUTDIR, "03_verdict.csv"), index=False)

# ---- the book
cand_by_day = []
for t in range(len(cal)):
    cand_by_day.append([tk for tk in tickers
                        if U[tk]["enter_here"][t] and U[tk]["member"][t]
                        and np.isfinite(U[tk]["Open"][t])
                        and np.isfinite(U[tk]["Close"][t])])

summary, all_trades, all_daily, rand_rows = [], [], [], []
score_maps = {name: {(r.ticker, r.entry_t): getattr(r, name)
                     for r in sig.itertuples()} for name, _ in MODELS}

print(f"\n--- THE BOOK (N={N_BOOK}, {N_RAND}-seed null) ---")
for name, _ in [("score_RS", None), ("score_R", None), ("score_S", None)]:
    daily, trades = simulate(U, cal, cand_by_day, "score",
                             scores=score_maps[name])
    m = metrics(daily, trades)
    label = name.upper()
    summary.append({"variant": label, **m})
    print(f"  {label:<10} wealth {m['wealth']:>7.3f}  cagr {100*m['cagr']:>6.2f}%  "
          f"shp {m['sharpe']:>6}  bps/sd {m['bps_per_slot_day']:>6}  "
          f"trips {m['round_trips']:>4}")
    for x in trades:
        all_trades.append({"variant": label, **{k: v for k, v in x.items()
                                                if k != "eq"}})
    for d in daily:
        all_daily.append({"variant": label, "date": str(d["date"].date()),
                          **{k: v for k, v in d.items() if k != "date"}})

bps_dist = []
for s in range(N_RAND):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 5, N_BOOK, s]))
    daily, trades = simulate(U, cal, cand_by_day, "rand", rng=rng)
    m = metrics(daily, trades)
    rand_rows.append({"seed": s, **m})
    bps_dist.append(m["bps_raw"])
    if (s + 1) % 200 == 0:
        print(f"    RAND {s + 1}/{N_RAND}")
bps_dist = np.array([b for b in bps_dist if b == b])
summary.append({"variant": "RAND", "bps_mean": round(float(bps_dist.mean()), 2),
                "bps_p5": round(float(np.percentile(bps_dist, 5)), 2),
                "bps_p95": round(float(np.percentile(bps_dist, 95)), 2)})
print(f"  RAND ({len(bps_dist)} seeds)  bps/sd mean {summary[-1]['bps_mean']}  "
      f"[p5 {summary[-1]['bps_p5']}, p95 {summary[-1]['bps_p95']}]")
K = len(bps_dist)
for row in summary:
    if row["variant"].startswith("SCORE"):
        b = row["bps_raw"]
        pct = float(((bps_dist < b).sum() + 0.5 * (bps_dist == b).sum()) / K)
        p = min(1.0, 2 * min(1 + (bps_dist <= b).sum(),
                             1 + (bps_dist >= b).sum()) / (K + 1))
        row["rand_pctile"] = round(100 * pct, 1)
        row["rand_p_two_sided"] = round(p, 4)
        print(f"    {row['variant']:<10} bps/sd {row['bps_per_slot_day']:>6} -> "
              f"pctile {row['rand_pctile']:>5.1f}  p {p:.4f}")

pd.DataFrame(summary).to_csv(os.path.join(OUTDIR, "03_variant_summary.csv"),
                             index=False)
pd.DataFrame(rand_rows).to_csv(os.path.join(OUTDIR, "03_rand_dist.csv"), index=False)
pd.DataFrame(all_trades).to_csv(os.path.join(OUTDIR, "03_trades.csv"), index=False)
pd.DataFrame(all_daily).to_csv(os.path.join(OUTDIR, "03_daily.csv"), index=False)
print(f"\nwrote results/03_scores.csv  03_verdict.csv  03_variant_summary.csv  "
      f"03_rand_dist.csv  03_trades.csv  03_daily.csv")
