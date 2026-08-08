"""10_area_test.py -- Study 10: negative moving-average AREA as a
forward-return conditioner.

PRE-REGISTRATION
  AREA_PROTOCOL.md (sha 8396828a..., locked before this script existed).
  Area family, four-look primary: {AREA_EP, AREA_RW} x {5d, 21d}.
  Bars (Sidak-4): |NW-t| >= 2.50 AND boot p <= .0127; robustness legs
  (best-month exclusion, halves, crisis/calm disclosed); NOVELTY GAUNTLET:
  paired dominance vs point-z and vs trailing-21d-return conditioning,
  NW-t >= 2.0 each, else "repackaged reversal / point-z".

THE OBJECT
  AREA_EP: episode sum of mad/sigma while MAD < 0 (resets at zero-cross).
  AREA_RW: trailing-21-session sum of min(mad,0)/sigma.
  Formation: month-end sessions 2016-01..2025-12, index members.
  Primary per cell: monthly mean forward return of D1 (most negative
  decile) MINUS the equal-weight eligible-universe mean. Gross; 10 bps
  round trip (5 bps/side, long-D1 leg only) disclosed alongside.

SURVIVORSHIP HANDLING (fixed by design, not a knob)
  A name delisting inside the forward window is valued at its LAST
  available close (liquidation-at-last-price assumption, count disclosed)
  rather than excluded -- exclusion would hand the reversal claim a free
  survivorship tailwind exactly in D1.

AUDIT FIXES (pre-run review; deviations from draft disclosed here)
  - unregistered |fwd|>4 outcome guard REMOVED; replaced by the house
    era-break convention (single-day >5x / <0.2x or >30d gap inside the
    forward path excludes the name-month, counted in n_break_excluded)
  - delisting at the session after formation now delist-valued at last
    close like every other inside-window delisting (was dropped)
  - AREA_EP resets ONLY at a finite MAD >= 0 close (the sole registered
    reset); undefined mad/sigma sessions poison the episode NaN until the
    next zero-cross, routing the name through the finiteness gate
  - AREA_RW windows containing an undefined session yield NaN
  - zero-area tie blocks straddling the D1 boundary are filled with the
    tie-mass mean (expectation of an unbiased fill), never alphabetically;
    occurrences counted per cell
  - formation-day lookup uses day.to_datetime64() (numpy<2.2 hashes
    datetime64[us] probes differently from [ns] keys: silent all-miss)
  - monthly csv persists the Z/REV benchmark legs; D1-minus-D10 added;
    registered three-tier verdict vocabulary; direction-aware best-month
    exclusion; net column named for its cost structure

RUN (from the repo root)
  export PORTFOLIO_DATADIR=/path/to/prepped/data
  python3 -u 10_area_test.py          # light: minutes, prepped daily csvs only
  outputs results/10_area_summary.csv, 10_area_monthly.csv,
          10_area_config.json
"""
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

DATADIR = os.environ.get("PORTFOLIO_DATADIR", "data")
OUTDIR = "results"
ASOF = "2026-07-16"
WIN_START = pd.Timestamp("2016-01-01")
SIM_END = pd.Timestamp("2025-12-31")
SEED = 7
STUDY = 15  # registered seed-stream id (fixed at pre-registration; independent of script numbering)
RW_WIN = 21
HORIZONS = (5, 21)
NW_LAG = 3
BOOT_B = 2000
BOOT_BLOCK = 3.0
BAR_T = 2.50
BAR_P = 0.0127
DOM_T = 2.00
COST_BPS = 5.0
CRISIS = [("2020-02", "2020-06"), ("2022-01", "2022-12")]
os.makedirs(OUTDIR, exist_ok=True)

with open(__file__, "rb") as f:
    SCRIPT_SHA256 = hashlib.sha256(f.read()).hexdigest()


def truthy(series):
    return series.astype(str).str.strip().isin(["True", "true", "1", "1.0"]).to_numpy()


# ================================================================ load
print(f"loading prepped store from {DATADIR}/ ...")
files = sorted(f for f in glob.glob(os.path.join(DATADIR, "*.csv"))
               if not os.path.basename(f).startswith("_"))
if not files:
    raise SystemExit(f"no prepped csvs in {DATADIR}/")

NAMES = {}
all_days = set()
n_delist_valued = [0]
n_break_excluded = [0]
n_area_nan_excl = [0]
for i, fpath in enumerate(files, 1):
    tk = os.path.basename(fpath)[:-4]
    if tk in ("SPY", "RSP"):
        continue
    try:
        df = pd.read_csv(fpath, usecols=["Date", "Close", "mad", "sigma",
                                         "z", "in_index"])
    except (ValueError, KeyError):
        continue
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] <= pd.Timestamp(ASOF)]
    df = (df.dropna(subset=["Close"]).sort_values("Date")
            .drop_duplicates(subset="Date").reset_index(drop=True))
    if len(df) < RW_WIN + 25:
        continue
    c = df["Close"].to_numpy(float)
    mad = df["mad"].to_numpy(float)
    sig = df["sigma"].to_numpy(float)
    z = df["z"].to_numpy(float)
    member = truthy(df["in_index"])
    dates = df["Date"].to_numpy()
    n = len(df)
    r = np.where(np.isfinite(mad) & np.isfinite(sig) & (sig > 0),
                 mad / np.where(sig > 0, sig, np.nan), np.nan)
    # era vector: house data-break convention --
    # single-day >5x / <0.2x moves or >30 calendar-day gaps split eras
    ret1 = c[1:] / c[:-1]
    gap = (dates[1:] - dates[:-1]) / np.timedelta64(1, "D")
    brk = (gap > 30) | (ret1 > 5.0) | (ret1 < 0.2)
    era = np.concatenate([[0], np.cumsum(brk)])
    # AREA_EP: episode accumulator; the ONE registered reset is a finite
    # MAD >= 0 close. An undefined mad/sigma session mid-episode makes the
    # episode sum unknown -> NaN until the next zero-cross (finiteness
    # gate then excludes the name-month at formation)
    a_ep = np.full(n, np.nan)
    acc = 0.0
    for t in range(n):
        if np.isfinite(mad[t]) and mad[t] >= 0:
            acc = 0.0
        elif np.isfinite(r[t]) and mad[t] < 0:
            acc = acc + r[t]        # NaN acc propagates
        else:
            acc = np.nan
        a_ep[t] = acc
    # AREA_RW: trailing sum of min(r,0); a window containing an undefined
    # session yields NaN (min_periods counts non-NaN observations)
    neg = np.where(np.isfinite(r), np.minimum(r, 0.0), np.nan)
    a_rw = pd.Series(neg).rolling(RW_WIN, min_periods=RW_WIN).sum().to_numpy()
    # trailing 21-session return (the reversal benchmark conditioner)
    rev = np.full(n, np.nan)
    rev[RW_WIN:] = c[RW_WIN:] / c[:-RW_WIN] - 1.0
    NAMES[tk] = {"dates": dates, "c": c, "z": z, "member": member,
                 "a_ep": a_ep, "a_rw": a_rw, "rev": rev, "era": era,
                 "didx": {d: t for t, d in enumerate(dates)}}
    all_days.update(dates)
    if i % 200 == 0 or i == len(files):
        print(f"  [{i}/{len(files)}] loaded")
print(f"{len(NAMES)} names")

cal = pd.DatetimeIndex(sorted(d for d in all_days
                              if WIN_START <= pd.Timestamp(d)))
per = cal.to_period("M")
form_days = []
for m in per.unique():
    if pd.Period(m, "M") > pd.Period(SIM_END, "M"):
        continue
    idx = np.where(per == m)[0]
    form_days.append(cal[idx[-1]])
months = [pd.Period(d, "M") for d in form_days]
n_m = len(form_days)
print(f"{n_m} formation month-ends, {form_days[0].date()} .. "
      f"{form_days[-1].date()}")


def fwd_ret(rec, t, h):
    """forward h-session simple return; delisting inside the window is
    valued at the LAST available close (counted). Reachable only for
    genuine delistings: live names carry data through ASOF, past the
    last formation month-end."""
    c = rec["c"]
    if t + h < len(c):
        return c[t + h] / c[t] - 1.0, False
    return c[-1] / c[t] - 1.0, True


# ================================================================ panels
# per (month, horizon): one common eligible set -- all conditioners finite,
# member at formation, forward return computable (delist-valued included)
CELLS = [("AREA_EP", "a_ep"), ("AREA_RW", "a_rw")]
BENCH = [("Z", "z"), ("REV", "rev")]
panel = {}          # (mi, h) -> dict of arrays
for mi, day in enumerate(form_days):
    d64 = day.to_datetime64()   # ns-unit probe: numpy<2.2 hashes [us] != [ns]
    rows = {"tk": [], "a_ep": [], "a_rw": [], "z": [], "rev": [],
            "f5": [], "f21": []}
    for tk, rec in NAMES.items():
        t = rec["didx"].get(d64)
        if t is None or not rec["member"][t]:
            continue
        vals = (rec["a_ep"][t], rec["a_rw"][t], rec["z"][t], rec["rev"][t])
        if not (np.isfinite(vals[0]) and np.isfinite(vals[1])):
            n_area_nan_excl[0] += 1
            continue
        if not (np.isfinite(vals[2]) and np.isfinite(vals[3])):
            continue
        # house era-break hygiene: exclude iff a data break (not a real
        # multi-day move) lies inside the valued forward path
        t_end = min(t + 21, len(rec["c"]) - 1)
        if rec["era"][t_end] != rec["era"][t]:
            n_break_excluded[0] += 1
            continue
        f5, dl5 = fwd_ret(rec, t, 5)
        f21, dl21 = fwd_ret(rec, t, 21)
        if dl5 or dl21:
            n_delist_valued[0] += 1     # once per delist-valued name-month
        rows["tk"].append(tk)
        rows["a_ep"].append(vals[0])
        rows["a_rw"].append(vals[1])
        rows["z"].append(vals[2])
        rows["rev"].append(vals[3])
        rows["f5"].append(f5)
        rows["f21"].append(f21)
    panel[mi] = {k: np.array(v) for k, v in rows.items()}

sizes = [len(panel[mi]["tk"]) for mi in range(n_m)]
print(f"panel sizes: median {int(np.median(sizes))}, "
      f"min {min(sizes)}, max {max(sizes)}; "
      f"delist-valued forward returns: {n_delist_valued[0]}; "
      f"era-break excluded: {n_break_excluded[0]}; "
      f"NaN-area excluded: {n_area_nan_excl[0]}")


TIE_FILL = {}   # (cond_key, h) -> months where a tie straddles the D1 edge


def decile_low_mean(f, cond):
    """mean fwd ret of the k most-negative names. A tie block straddling
    the decile boundary contributes its tie-mass MEAN for the remaining
    slots (the expectation of an unbiased fill) -- never a name-order
    fill, which would inject a fixed alphabetical subset. Identical to a
    plain order[:k] mean whenever no tie straddles the boundary (always,
    for continuous conditioners like Z/REV)."""
    n = len(cond)
    k = max(1, n // 10)
    order = np.argsort(cond, kind="stable")
    kth = cond[order[k - 1]]
    strict = cond < kth
    tied = cond == kth
    n_fill = k - int(strict.sum())
    straddles = n_fill < int(tied.sum())
    d1 = (f[strict].sum() + n_fill * float(f[tied].mean())) / k
    return d1, straddles


def d1_excess(mi, cond_key, h):
    """D1 (most negative conditioner decile) mean fwd ret minus EW mean."""
    p = panel[mi]
    n = len(p["tk"])
    if n < 20:
        return np.nan
    f = p[f"f{h}"]
    d1, straddles = decile_low_mean(f, p[cond_key])
    if straddles:
        TIE_FILL[(cond_key, h)] = TIE_FILL.get((cond_key, h), 0) + 1
    return float(d1 - f.mean())


def d1_minus_d10(mi, cond_key, h):
    """disclosure only (no bar): D1 mean fwd ret minus D10 (least negative
    decile; never-underwater names rank last per protocol). Both decile
    ends tie-mass mean-filled."""
    p = panel[mi]
    n = len(p["tk"])
    if n < 20:
        return np.nan
    f = p[f"f{h}"]
    cond = p[cond_key]
    d1, _ = decile_low_mean(f, cond)
    d10, _ = decile_low_mean(f, -cond)      # top decile via negation
    return float(d1 - d10)


def series_for(cond_key, h):
    return np.array([d1_excess(mi, cond_key, h) for mi in range(n_m)])


def nw_t(x):
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan, n
    xc = x - x.mean()
    L = min(NW_LAG, n - 1)
    v = float(xc @ xc) / n
    for l in range(1, L + 1):
        v += 2.0 * (1.0 - l / (L + 1.0)) * float(xc[l:] @ xc[:-l]) / n
    return (x.mean() / np.sqrt(v / n) if v > 0 else np.nan), n


def boot_p(x):
    x = x[np.isfinite(x)]
    n = len(x)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, STUDY, 999, 9]))
    p_geo = 1.0 / BOOT_BLOCK
    xc = x - x.mean()
    hits = 0
    for b in range(BOOT_B):
        idx = np.empty(n, dtype=int)
        i0 = rng.integers(n)
        for j in range(n):
            if j > 0 and rng.random() >= p_geo:
                i0 = (i0 + 1) % n
            elif j > 0:
                i0 = rng.integers(n)
            idx[j] = i0
        if abs(xc[idx].mean()) >= abs(x.mean()):
            hits += 1
    return (hits + 1.0) / (BOOT_B + 1.0)


crisis_mask = np.zeros(n_m, bool)
for lo, hi in CRISIS:
    for mi, m in enumerate(months):
        if pd.Period(lo, "M") <= m <= pd.Period(hi, "M"):
            crisis_mask[mi] = True

tks_all = sorted({tk for mi in range(n_m) for tk in panel[mi]["tk"]})
half_a = set(tks_all[0::2])


def d1_excess_half(mi, cond_key, h, in_a):
    p = panel[mi]
    msk = np.isin(p["tk"], list(half_a)) if in_a \
        else ~np.isin(p["tk"], list(half_a))
    n = int(msk.sum())
    if n < 20:
        return np.nan
    f = p[f"f{h}"][msk]
    d1, _ = decile_low_mean(f, p[cond_key][msk])
    return float(d1 - f.mean())


# ================================================================ score
summary, monthly_rows = [], []
print("\n==== FOUR PRIMARY CELLS ====")
for cname, ckey in CELLS:
    for h in HORIZONS:
        X = series_for(ckey, h) * 1e4              # bps
        D10 = np.array([d1_minus_d10(mi, ckey, h)
                        for mi in range(n_m)]) * 1e4    # disclosure only
        t_stat, n_used = nw_t(X)
        p_val = boot_p(X)
        exist = (np.isfinite(t_stat) and abs(t_stat) >= BAR_T
                 and p_val <= BAR_P)
        mean_x = float(np.nanmean(X))
        # robustness: exclude the single most influential month in the
        # direction of the mean (max for positive means, min for negative)
        excl = X[np.isfinite(X)]
        excl = np.delete(excl, np.argmax(excl) if mean_x > 0
                         else np.argmin(excl))
        sign_excl = np.sign(excl.mean()) == np.sign(mean_x)
        hA = np.nanmean([d1_excess_half(mi, ckey, h, True)
                         for mi in range(n_m)]) * 1e4
        hB = np.nanmean([d1_excess_half(mi, ckey, h, False)
                         for mi in range(n_m)]) * 1e4
        sign_halves = (np.sign(hA) == np.sign(hB) == np.sign(mean_x))
        cr = float(np.nanmean(np.where(crisis_mask, X, np.nan)))
        ca = float(np.nanmean(np.where(~crisis_mask, X, np.nan)))
        # novelty gauntlet: paired dominance vs each benchmark
        dom, bench_series = {}, {}
        for bname, bkey in BENCH:
            B = series_for(bkey, h) * 1e4
            bench_series[bname] = B
            tdom, _ = nw_t(X - B)
            dom[bname] = {"bench_mean": float(np.nanmean(B)),
                          "paired_t": float(tdom) if np.isfinite(tdom)
                          else np.nan,
                          "pass": bool(np.isfinite(tdom) and tdom >= DOM_T)}
        robust = bool(sign_excl and sign_halves)
        novel = all(d["pass"] for d in dom.values())
        # registered three-tier vocabulary (protocol sections 3-4):
        # existence + robustness + novelty -> DISCOVERY CANDIDATE;
        # existence failing a dominance leg -> known-effect recovery with
        # the mandated phrase, regardless of its own alpha; else dead
        if exist and robust and novel:
            tier = "DISCOVERY CANDIDATE"
        elif exist and not novel:
            tier = ("KNOWN-EFFECT RECOVERY "
                    "(repackaged reversal / point-z, no novelty)")
        else:
            tier = "dead"
        cell = f"{cname}_f{h}"
        print(f"  {cell:<14} mean {mean_x:+8.1f} bps  NW-t {t_stat:+5.2f}  "
              f"p {p_val:.4f}  | vs Z t {dom['Z']['paired_t']:+5.2f}  "
              f"vs REV t {dom['REV']['paired_t']:+5.2f}  -> {tier}")
        summary.append({
            "cell": cell, "mean_bps": round(mean_x, 2),
            "nw_t": round(float(t_stat), 3), "boot_p": round(p_val, 5),
            "n_months": n_used, "sign_ex_best": bool(sign_excl),
            "halfA_bps": round(float(hA), 2), "halfB_bps": round(float(hB), 2),
            "halves_consistent": bool(sign_halves),
            "crisis_bps": round(cr, 2), "calm_bps": round(ca, 2),
            "z_bench_bps": round(dom["Z"]["bench_mean"], 2),
            "vs_z_paired_t": round(dom["Z"]["paired_t"], 3),
            "rev_bench_bps": round(dom["REV"]["bench_mean"], 2),
            "vs_rev_paired_t": round(dom["REV"]["paired_t"], 3),
            "d1_minus_d10_bps": round(float(np.nanmean(D10)), 2),
            "n_tie_fill_months": int(TIE_FILL.get((ckey, h), 0)),
            "net_long_d1_roundtrip_bps": round(mean_x - 2 * COST_BPS, 2),
            "tier": tier})
        for mi in range(n_m):
            monthly_rows.append({"cell": cell, "month": str(months[mi]),
                                 "X_bps": X[mi],
                                 "Z_bench_bps": bench_series["Z"][mi],
                                 "REV_bench_bps": bench_series["REV"][mi],
                                 "d1_d10_bps": D10[mi],
                                 "crisis": bool(crisis_mask[mi])})

# ================================================================ SPY exhibit
spy_path = os.path.join(DATADIR, "SPY.csv")
spy_note = "SPY.csv not in store -- exhibit skipped"
if os.path.exists(spy_path):
    try:
        s = pd.read_csv(spy_path, usecols=["Date", "Close", "mad", "sigma"])
        s["Date"] = pd.to_datetime(s["Date"])
        s = s.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
        sc = s["Close"].to_numpy(float)
        sm = s["mad"].to_numpy(float)
        ss = s["sigma"].to_numpy(float)
        sr = np.where(np.isfinite(sm) & np.isfinite(ss) & (ss > 0),
                      sm / np.where(ss > 0, ss, np.nan), np.nan)
        a = np.full(len(s), np.nan)
        acc = 0.0
        for t in range(len(s)):
            if np.isfinite(sm[t]) and sm[t] >= 0:
                acc = 0.0
            elif np.isfinite(sr[t]) and sm[t] < 0:
                acc = acc + sr[t]
            else:
                acc = np.nan
            a[t] = acc
        pct = np.full(len(s), np.nan)
        for t in range(500, len(s)):
            hist = a[:t]
            hist = hist[np.isfinite(hist)]
            if np.isfinite(a[t]) and len(hist) >= 250:
                pct[t] = (hist <= a[t]).mean()
        rows = []
        for lo, hi, lab in ((0, 1 / 3, "deepest"), (1 / 3, 2 / 3, "mid"),
                            (2 / 3, 1.01, "shallow")):
            msk = (pct >= lo) & (pct < hi)
            f5s = [sc[t + 5] / sc[t] - 1 for t in np.where(msk)[0]
                   if t + 5 < len(sc)]
            f21s = [sc[t + 21] / sc[t] - 1 for t in np.where(msk)[0]
                    if t + 21 < len(sc)]
            rows.append(f"{lab}: f5 {1e4 * np.mean(f5s):+.1f} bps, "
                        f"f21 {1e4 * np.mean(f21s):+.1f} bps (n={msk.sum()})")
        spy_note = " | ".join(rows)
    except Exception as e:
        spy_note = f"SPY exhibit failed: {e}"
print(f"\nSPY exhibit (unscored): {spy_note}")

# ================================================================ outputs
pd.DataFrame(summary).to_csv(os.path.join(OUTDIR, "10_area_summary.csv"),
                             index=False)
pd.DataFrame(monthly_rows).to_csv(os.path.join(OUTDIR, "10_area_monthly.csv"),
                                  index=False)
with open(os.path.join(OUTDIR, "10_area_config.json"), "w") as f:
    json.dump({"protocol": "AREA_PROTOCOL.md (study 10)",
               "script_sha256": SCRIPT_SHA256, "SEED": SEED,
               "STUDY_STREAM": STUDY, "RW_WIN": RW_WIN,
               "HORIZONS": list(HORIZONS), "NW_LAG": NW_LAG,
               "BOOT_B": BOOT_B, "BOOT_BLOCK": BOOT_BLOCK,
               "BAR_T": BAR_T, "BAR_P": BAR_P, "DOM_T": DOM_T,
               "WIN_START": str(WIN_START.date()),
               "SIM_END": str(SIM_END.date()), "ASOF": ASOF,
               "CRISIS": CRISIS, "COST_BPS": COST_BPS,
               "COST_MODEL": ("5 bps per side x 2 sides = 10 bps, one "
                              "long-D1 round trip; EW benchmark leg is "
                              "paper (untraded)"),
               "n_months": n_m,
               "panel_median": int(np.median(sizes)),
               "n_delist_valued": int(n_delist_valued[0]),
               "delist_rule": "valued at last close (liquidation-at-last)",
               "n_break_excluded": int(n_break_excluded[0]),
               "break_rule": ("house era convention: single-day >5x/<0.2x "
                              "or >30d gap inside the 21-session forward "
                              "path excludes the name-month"),
               "n_area_nan_excl": int(n_area_nan_excl[0]),
               "tie_fill_months": {f"{k[0]}_f{k[1]}": v
                                   for k, v in sorted(TIE_FILL.items())},
               "tie_rule": ("D1-boundary tie blocks filled with tie-mass "
                            "mean (unbiased-fill expectation)"),
               "spy_exhibit": spy_note,
               "cells": [s["cell"] for s in summary],
               "tiers": {s["cell"]: s["tier"] for s in summary}},
              f, indent=1)
print(f"\nwritten: {OUTDIR}/10_area_summary.csv, 10_area_monthly.csv, "
      f"10_area_config.json")
print(f"script sha256: {SCRIPT_SHA256}")
