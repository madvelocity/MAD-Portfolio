#!/usr/bin/env python3
"""
01_data.py  --  raw daily prices from polygon.io + the backtest-ready per-ticker
files for the MAD-Portfolio pipeline.

Builds the ticker universe (every S&P 500 constituent 2016-2025, from the fja05680
point-in-time membership history), pulls the LAST 20 YEARS of daily aggregates per
name, then computes the exact columns scripts 02-04 consume: the MAD indicator
stack, the velocity-gated signal flags, and day-by-day index membership. Polygon
serves DELISTED securities, so failed and acquired members are included -- the
studies run on a survivorship-free universe.

The paper's published runs consumed files built by the companion repository's
pipeline (MAD-Velocity-Signal, scripts 02-04); this script consolidates those
steps and reproduces the same columns from scratch, minus the walk-forward P(up)
estimator, which this paper never reads. Rerunning it is only necessary to
rebuild the archive: scripts 02-04 run unchanged on any folder of files carrying
these columns.

For each historical ticker the script tries, in order, and MERGES by date:
  1. the historical symbol itself           (Polygon keeps delisted history under it)
  2. its dot/dash variant                    (class shares: BRK.B / BRK-B)
  3. its PREDECESSOR symbol                  (the membership file records renamed
     companies under their new symbol; Polygon stores the old era under the old one)
  4. its renamed successor                   (FB -> META)
Earlier candidates win on duplicate dates. source_ticker records provenance.

NOTES
  - adjusted=true on Polygon adjusts for SPLITS ONLY (no dividend reinvestment).
  - Ticker-symbol reuse exists in history (one symbol, two companies). Downstream
    scripts break every price chain at splice bars (>5x one-day moves) and at
    era gaps (>30 calendar days between consecutive bars); see the paper's
    Section IV.
  - RESUMABLE: tickers whose raw csv already exists in data/raw/ are skipped;
    the prep pass overwrites deterministically.
  - The membership file is freshness-guarded: if its last snapshot is stale the
    script refuses to run rather than silently build a survivorship-biased
    universe.

API key: read from the POLYGON_API_KEY environment variable
(export POLYGON_API_KEY=...). Never printed or logged. The free tier works (the
pull auto-throttles on 429s); a paid plan finishes in minutes.

    python3 01_data.py
Output -> data/raw/<TICKER>.csv   raw bars (Date,Open,High,Low,Close,Volume,
                                  VWAP,Transactions,source_ticker)
          data/<TICKER>.csv       backtest-ready: raw columns + sma_20, mad,
                                  sigma, z, regime, price_share_5, sig_cross_up,
                                  sig_rollback, in_index
          data/raw/_missing.csv   tickers with no data + reason
"""
import csv
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

RAWDIR = os.path.join("data", "raw")
OUTDIR = "data"
MISSING = os.path.join(RAWDIR, "_missing.csv")
MEMBERSHIP_SRC = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                  "S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv")
UA = "Mozilla/5.0"
YEAR_LO, YEAR_HI = 2016, 2025
FRESH_MIN = "2025-06-01"     # membership file must be at least this current
YEARS_BACK = 20
SMA_W, VOL_W, K = 20, 255, 5
SLEEP = 0.15                 # pacing between requests (paid plans tolerate far more)
RETRIES = [2, 5, 15, 60]     # backoff on 429/5xx
RENAME = {                   # historical symbol -> current fetchable symbol
    "ABC": "COR", "ANTM": "ELV", "BHGE": "BKR", "ADS": "BFH", "FB": "META",
    "WLTW": "WTW", "UTX": "RTX", "HFC": "DINO", "FISV": "FI", "FBHS": "FBIN",
    "COG": "CTRA", "PKI": "RVTY", "RE": "EG",
    "FLT": "CPAY", "TMK": "GL", "CTL": "LUMN", "KORS": "CPRI", "WYND": "TNL", "HCP": "DOC",
}
PREDECESSOR = {              # membership records the NEW symbol; Polygon stores the old era
    "AABA": "YHOO", "CCEP": "CCE", "WYND": "WYN", "JEF": "LUK", "ANDV": "TSO",
    "KDP": "DPS", "BHGE": "BHI", "CBRE": "CBG", "WELL": "HCN", "BKNG": "PCLN",
    "APTV": "DLPH", "TPR": "COH", "UAA": "UA", "CPRI": "KORS",
}
os.makedirs(RAWDIR, exist_ok=True)


# ---------------------------------------------------------------- API key
def load_key():
    key = os.environ.get("POLYGON_API_KEY", "").strip()
    if key:
        return key
    sys.exit("POLYGON_API_KEY not set. Get a key at polygon.io, then:  "
             "export POLYGON_API_KEY=yourkey")


KEY = load_key()
END = date.today()
START = END - timedelta(days=round(YEARS_BACK * 365.25))


# ---------------------------------------------------------------- universe (fja05680)
def load_membership():
    """Snapshot rows, sorted; freshness-guarded."""
    req = urllib.request.Request(MEMBERSHIP_SRC, headers={"User-Agent": UA})
    raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(raw)))
    rows.sort(key=lambda r: r["date"])
    last = rows[-1]["date"][:10]
    if last < FRESH_MIN:
        sys.exit(f"STALE MEMBERSHIP FILE: last snapshot {last} < {FRESH_MIN} -- "
                 "refusing to build a survivorship-biased universe.")
    return rows, last


def build_universe(rows):
    """Unique historical tickers touching the S&P 500 during [YEAR_LO, YEAR_HI]."""
    lo, hi = f"{YEAR_LO}-01-01", f"{YEAR_HI}-12-31"
    universe = set()
    for i, r in enumerate(rows):
        d = r["date"][:10]
        nxt = rows[i + 1]["date"][:10] if i + 1 < len(rows) else "9999-12-31"
        if nxt >= lo and d <= hi:                       # snapshot overlaps the window
            universe |= {t.strip().upper() for t in r["tickers"].split(",") if t.strip()}
    print(f"membership: {len(rows)} snapshots; "
          f"universe {YEAR_LO}-{YEAR_HI}: {len(universe)} tickers")
    return sorted(universe)


def membership_intervals(rows):
    """ticker -> list of [start, end) date-string intervals, day-by-day exact,
    clipped to the study window."""
    lo, hi = f"{YEAR_LO}-01-01", f"{YEAR_HI}-12-31"
    spans = {}
    for i, r in enumerate(rows):
        d = r["date"][:10]
        nxt = rows[i + 1]["date"][:10] if i + 1 < len(rows) else "9999-12-31"
        a, b = max(d, lo), min(nxt, f"{YEAR_HI}-12-31" if nxt > hi else nxt)
        if a > hi or nxt < lo:
            continue
        for t in {x.strip().upper() for x in r["tickers"].split(",") if x.strip()}:
            iv = spans.setdefault(t, [])
            if iv and iv[-1][1] == a:                  # extend contiguous span
                iv[-1][1] = min(nxt, "9999-12-31")
            else:
                iv.append([a, nxt])
    return spans


# ---------------------------------------------------------------- fetch one symbol
def fetch_symbol(sym):
    """All daily bars for sym over [START, END]. Follows next_url pagination.
    Retries on 429/5xx; 404/no-data -> empty."""
    url = (f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/{START}/{END}"
           f"?adjusted=true&sort=asc&limit=50000&apiKey={KEY}")
    rows = []
    while url:
        payload = None
        for wait in RETRIES + [None]:
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    payload = json.loads(resp.read().decode())
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return rows
                if e.code in (429, 500, 502, 503, 504) and wait is not None:
                    time.sleep(wait)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                if wait is None:
                    raise
                time.sleep(wait)
        for b in payload.get("results") or []:
            d = datetime.fromtimestamp(b["t"] / 1000, tz=timezone.utc).date()
            rows.append({"Date": d.isoformat(),
                         "Open": b.get("o"), "High": b.get("h"), "Low": b.get("l"),
                         "Close": b.get("c"), "Volume": b.get("v"),
                         "VWAP": b.get("vw"), "Transactions": b.get("n"),
                         "source_ticker": sym})
        url = payload.get("next_url")
        if url:
            url += f"&apiKey={KEY}"
        time.sleep(SLEEP)
    return rows


def candidates(tk):
    """Symbols to try, in priority order, deduped."""
    pred = PREDECESSOR.get(tk, "")
    succ = RENAME.get(tk, "")
    cands = [tk]
    for v in (tk.replace(".", "-") if "." in tk else tk.replace("-", "."),
              pred, pred.replace(".", "-"),
              succ, succ.replace(".", "-")):
        if v and v not in cands:
            cands.append(v)
    return cands


# ---------------------------------------------------------------- prep one ticker
def prep(tk, spans):
    """Raw bars -> the backtest-ready columns scripts 02-04 read. Identical math
    to the companion pipeline (papers 1-2): eqs (1)-(6) of the paper."""
    raw = pd.read_csv(os.path.join(RAWDIR, f"{tk}.csv"))
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = (raw.dropna(subset=["Open", "Close"]).sort_values("Date")
              .drop_duplicates(subset="Date", keep="first").reset_index(drop=True))
    C = raw["Close"].astype(float)
    sma = C.rolling(SMA_W).mean()
    mad = 100 * (C - sma) / C
    sigma = mad.rolling(VOL_W).std()
    z = (mad / sigma).to_numpy()
    regime = np.select([z > 2, z > 1, z > 0, z > -1, z > -2],
                       [3, 2, 1, -1, -2], default=-3).astype(float)
    regime[np.isnan(z)] = np.nan
    dP = C - C.shift(K)
    dSMA = sma - sma.shift(K)
    price_c = (100.0 * (sma / C**2) * dP).to_numpy()
    sma_c = (-100.0 * (1.0 / C) * dSMA).to_numpy()
    denom = np.abs(price_c) + np.abs(sma_c)
    pshare = np.where(denom > 0, price_c / denom, np.nan)
    n = len(raw)
    cross = np.zeros(n, bool)
    rollb = np.zeros(n, bool)
    for t in range(1, n):
        if np.isnan(regime[t]) or np.isnan(regime[t - 1]):
            continue
        if regime[t] == 1 and regime[t - 1] == -1:
            cross[t] = True
        if regime[t] == 1 and regime[t - 1] == 2:
            rollb[t] = True
    member = np.zeros(n, bool)
    ds = raw["Date"].dt.strftime("%Y-%m-%d")
    for a, b in spans.get(tk, []):
        member |= ((ds >= a) & (ds < b)).to_numpy()
    out = raw.copy()
    out["Date"] = ds
    out["sma_20"] = sma.round(4)
    out["mad"] = mad.round(4)
    out["sigma"] = sigma.round(4)
    out["z"] = np.round(z, 4)
    out["regime"] = [int(x) if x == x else np.nan for x in regime]
    out["price_share_5"] = np.round(pshare, 4)
    out["sig_cross_up"] = cross
    out["sig_rollback"] = rollb
    out["in_index"] = member
    out.to_csv(os.path.join(OUTDIR, f"{tk}.csv"), index=False)
    return int(member.sum()), int(cross.sum())


# ---------------------------------------------------------------- run
mrows, mlast = load_membership()
todo = build_universe(mrows)
spans = membership_intervals(mrows)
print(f"window {START} -> {END}   out {RAWDIR}/ and {OUTDIR}/")

done = skipped = empty = 0
missing = []
t0 = time.time()
for i, tk in enumerate(todo, 1):
    path = os.path.join(RAWDIR, f"{tk}.csv")
    if os.path.exists(path):
        skipped += 1
        continue
    merged = {}                                        # date -> row (first candidate wins)
    used = []
    for sym in candidates(tk):
        try:
            rows = fetch_symbol(sym)
        except Exception as e:                         # hard failure on this candidate
            missing.append({"ticker": tk, "symbol": sym, "reason": f"error: {e}"})
            continue
        fresh = 0
        for r in rows:
            if r["Date"] not in merged:
                merged[r["Date"]] = r
                fresh += 1
        if fresh:
            used.append(f"{sym}:{fresh}")
    if not merged:
        empty += 1
        missing.append({"ticker": tk, "symbol": "|".join(candidates(tk)),
                        "reason": "no data returned"})
        print(f"  [{i}/{len(todo)}] {tk:<8} EMPTY")
        continue
    rows = [merged[d] for d in sorted(merged)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Date", "Open", "High", "Low", "Close",
                                          "Volume", "VWAP", "Transactions", "source_ticker"])
        w.writeheader()
        w.writerows(rows)
    done += 1
    if done % 25 == 0 or i == len(todo):
        rate = done / max(time.time() - t0, 1)
        print(f"  [{i}/{len(todo)}] {tk:<8} {len(rows):>5} rows "
              f"({', '.join(used)})   {rate:.1f} tickers/s")

with open(MISSING, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ticker", "symbol", "reason"])
    w.writeheader()
    w.writerows(missing)

print("\nprep pass: indicators, signal flags, membership...")
prepped = 0
for i, tk in enumerate(todo, 1):
    if not os.path.exists(os.path.join(RAWDIR, f"{tk}.csv")):
        continue
    md, cu = prep(tk, spans)
    prepped += 1
    if prepped % 100 == 0:
        print(f"  [{i}/{len(todo)}] prepped {prepped}")

print(f"\nfetched {done}   skipped(existing) {skipped}   empty {empty}   "
      f"prepped {prepped}   elapsed {(time.time()-t0)/60:.1f} min")
print(f"missing log -> {MISSING} ({len(missing)} entries)")
