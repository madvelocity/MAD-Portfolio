"""Assemble final_summary.csv -- every configuration across studies
05-12, one row per run, derived strictly from the per-study summary
CSVs in results/ with standing/verdict text joined from
MAD-Portfolio_tests_master.csv. Re-runnable: no hand-entered numbers.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
LEDGER = os.path.join(HERE, "MAD-Portfolio_tests_master.csv")

SOURCES = [
    (5,  "05_caprank_test.py",   "05_test_summary.csv"),
    (6,  "06_capacity_test.py",  "06_capacity_summary.csv"),
    (7,  "07_c2_overlay_test.py","07_c2_summary.csv"),
    (8,  "08_adds_settle.py",    "08_settle_summary.csv"),
    (9,  "09_asym_test.py",      "09_asym_summary.csv"),
    (10, "10_area_test.py",      "10_area_summary.csv"),
    (11, "11_margin_bridge.py",  "11_bridge_summary.csv"),
    (12, "12_one_book.py",       "12_one_book_summary.csv"),
]

# summary run name -> ledger run name where they differ
LEDGER_ALIAS = {(11, "BASELINE"): "RANKVOL"}

# rows absent from the ledger (anchors / sensitivity arms) get fixed text
EXTRA_VERDICT = {
    (11, "MARGIN_BRIDGE_EFFR400"):
        "funding-stress sensitivity (EFFR+400bps); not verdict-bearing",
    (12, "BASELINE"):    "anchor: the incumbent, reproduced to the cent",
    (12, "ADDS_ONLY"):   "anchor: study 06/08 adds config, reproduced to the cent",
    (12, "BRIDGE_ONLY"): "anchor: study 11 bridge config, reproduced to the cent",
}

COLS = ["study", "script", "run", "final_equity", "wealth_x", "cagr",
        "oos_cagr", "sharpe", "oos_sharpe", "maxdd", "oos_maxdd",
        "trades", "adds", "null_standing", "verdict"]

ledger = {}
with open(LEDGER, newline="") as f:
    for r in csv.DictReader(f):
        ledger[(int(r["study"]), r["run"])] = (r["null_standing"], r["verdict"])

out = []
for study, script, fname in SOURCES:
    with open(os.path.join(RES, fname), newline="") as f:
        rows = list(csv.DictReader(f))
    if study == 10:  # stat-track study: cells, not accounts
        for r in rows:
            std, ver = ledger.get((10, r["cell"]), ("", r["tier"]))
            out.append({"study": study, "script": script, "run": r["cell"],
                        "null_standing": std or
                        f"mean {r['mean_bps']} bps/mo, NW-t {r['nw_t']}, boot p {r['boot_p']}",
                        "verdict": ver})
        continue
    rows.sort(key=lambda r: -float(r["final_equity"]))
    for r in rows:
        key = (study, LEDGER_ALIAS.get((study, r["run"]), r["run"]))
        std, ver = ledger.get(key, ("", EXTRA_VERDICT.get((study, r["run"]), "")))
        if not ver and r["run"] in ("SPY", "RSP"):
            ver = "index control"
        out.append({"study": study, "script": script, "run": r["run"],
                    "final_equity": r.get("final_equity", ""),
                    "wealth_x": r.get("wealth_x", ""),
                    "cagr": r.get("cagr", ""),
                    "oos_cagr": r.get("oos_cagr", ""),
                    "sharpe": r.get("sharpe", ""),
                    "oos_sharpe": r.get("oos_sharpe", ""),
                    "maxdd": r.get("maxdd", ""),
                    "oos_maxdd": r.get("oos_maxdd", ""),
                    "trades": r.get("trades", ""),
                    "adds": r.get("adds", r.get("n_adds", "")),
                    "null_standing": std, "verdict": ver})

with open(os.path.join(HERE, "final_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    for r in out:
        w.writerow({c: r.get(c, "") for c in COLS})

print(f"wrote final_summary.csv -- {len(out)} rows")
n_no_verdict = sum(1 for r in out if not r["verdict"])
print(f"rows without verdict text: {n_no_verdict}")
