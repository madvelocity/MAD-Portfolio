# MAD-Portfolio

**Portfolio Construction from the Moving-Average-Distance Signal**

The fourth paper in the MAD series. The earlier papers built a trading signal;
this one builds the portfolio the signal implies — and finds that nearly
everything of consequence happens in the step between them. Applied to 728
point-in-time S&P 500 constituents, the gated entry fires roughly twelve times
per trading day, far more than a concentrated book can act on, so realized
performance is decided by one question: which candidacies deserve capital.

The answer that survives every test is the simplest column on the tape.

**The trading rule, in two sentences:** when the gated signal fires, trade only
names in the current top hundred by volume, accepting the most-traded first into
ten equal slots. Sell each position at the first +2 → +1 rollback.

📄 **Paper:** `MAD-Portfolio.pdf` (included)

---

## Headline results

| | Full window (2016–2025) | Confirmation window (2022–2025) |
|---|---|---|
| **The strategy** | $100,000 → $552,888 · 18.7%/yr · Sharpe 0.90 | **19.7%/yr · Sharpe 0.97 · max drawdown −19.1%** |
| SPY (same dividend-free basis) | 13.1%/yr · Sharpe 0.77 | 9.5%/yr · −25.4% drawdown |
| RSP (equal-weight control) | 9.8%/yr | 4.2%/yr |
| Random-selection books (200 seeds) | median $242,100 | — |

The strategy is the only simulated configuration whose confirmation window
exceeds its full window, and it beats 100% of the 200 random-selection books
run under identical mechanics. In the window's two down years it was the best
performer on the board: +3.2% in 2018 (SPY −6.3%) and −3.4% in 2022
(SPY −19.5%).

The negatives are half the paper: trailing returns predict nothing at any
horizon from one week to four months; an intuitive displacement filter selects
no better than chance; and fitted scoring models that win the statistical
contest fail to keep their advantage in out-of-sample dollar simulation.

## Repository contents

| File | What it does |
|---|---|
| `01_data.py` | Rebuilds the archive: point-in-time universe + 20 years of Polygon daily bars + the backtest-ready columns (optional — needs a free Polygon key) |
| `02_ranking.py` | The daily rank table: every constituent, every day, twelve ranks in three families |
| `03_score.py` | The signal ledger (every candidacy resolved through the trade mechanics) + three nested walk-forward scoring models |
| `04_sim.py` | The simulation menu, random-selection envelopes, and SPY/RSP controls under realistic account mechanics |
| `05–12` | The post-publication study program — see the next section |
| `MAD-Portfolio_tests_master.csv` | The master ledger: every configuration run since publication, one row per run, with its null standing and verdict |
| `tests/` | Known-answer fixture suites for the 01–04 pipeline: trade mechanics, walk-forward knowability, membership gating, era-break handling, and dollar accounting verified to the cent |
| `results/` | The CSVs cited in the paper, the outputs of studies 05–12, and the cached index-control series that pin the benchmark vintage |

## Post-publication studies (scripts 05–12)

After the paper shipped, a pre-registered challenger and extension program
asked whether anything could beat or extend the published configuration.
Nothing amends the paper. Each study ran against explicit null models with
seeds fixed at registration, and each run reproduces the published baseline
($552,887.57) as a gate before printing its own result. The full run-by-run
record is `MAD-Portfolio_tests_master.csv`.

| Script | Question | Verdict |
|---|---|---|
| `05_caprank_test.py` | Does selecting by market cap beat selecting by volume? | Dead — loses both windows and the covered-pair check |
| `06_capacity_test.py` | Capacity family: adds into held weakness; dilution to fifty slots; taking every signal equal-weight | Adds promising at p92 (100 seeds), settled by study 08; fifty slots collapse to SPY; taking everything gives back half the alpha |
| `07_c2_overlay_test.py` | Does a conditioning overlay on the entry improve selection? | Dead — ties its own permutation null |
| `08_adds_settle.py` | The adds question settled at 500 seeds | Dead — p94.4 against a p95 bar |
| `09_asym_test.py` | Three lean-map variants of the adds idea | Dead — all three arms |
| `10_area_test.py` | Does depth × duration below the average (area) predict a bounce? | Refuted — the deepest decile continues to fall; the inverted effect is not novel against point-depth (`10_AREA_RESULTS.md`) |
| `11_margin_bridge.py` | Admitting overflow signals on borrowed money, capped at 130% | Dead — fails the out-of-sample window under the registered forced-liquidation remedy |
| `12_one_book.py` | All components wired together, as deployed | Engineering run, no verdict — $100,000 → $1,060,456.71 over the decade, reproducing all three component anchors to the cent |

`final_summary.csv` is the combined table: every configuration
across the eight studies, one row per run with its null standing and verdict,
assembled by `build_final_summary.py` strictly from the per-study summary
files — no hand-entered numbers.

Two of these components — the adds (08) and the margin bridge (11) — run in
the deployed engine documented at [mad-velocity.io](https://mad-velocity.io)
even though neither cleared its pre-registered bar. That choice is disclosed
there and here: the validated result is the published $552,888 baseline; the
engine's $1,060,457 is a simulation anchor, not a finding. The live book is
the ongoing test.

## Reproducibility

The data vintage is pinned (July 16, 2026) and all randomness derives from named
seed sequences, so every number in the paper — including each envelope
percentile and Monte Carlo p-value — reproduces exactly; reruns are
byte-identical on a given machine. Raw price data is not redistributed:
`01_data.py` rebuilds it from polygon.io under a user-supplied key
(`export POLYGON_API_KEY=...`), then scripts 02–04 run unchanged:

```bash
python3 02_ranking.py
python3 03_score.py
python3 04_sim.py
```

The studies run in numeric order after 04, from the repo root, with
`PORTFOLIO_DATADIR` pointing at the per-name store `01_data.py` builds:

```bash
python3 05_caprank_test.py   # also needs a fundamentals store: per-ticker
python3 06_capacity_test.py  # CSVs of SEC share counts and splits (see
python3 07_c2_overlay_test.py  # load_fundamentals in 05 for the columns)
python3 08_adds_settle.py
python3 09_asym_test.py
python3 10_area_test.py
python3 11_margin_bridge.py
python3 12_one_book.py
```

Three caveats. Studies 08, 09, and 11 are heavy — hundreds of full decade-long
account simulations each; hours, not minutes. Each study hard-exits unless its
inputs reproduce the published baseline within tolerance, so a store built from
a different data vintage fails loudly rather than silently drifting. And the
internal seed-stream ids inside the scripts were fixed at registration and are
independent of script numbering — they are what make the published null
envelopes reproducible, and they do not renumber.

## The MAD series

1. **[MAD-Markov](https://github.com/madvelocity/MAD-Markov-Model)** — position within the displacement band forecasts the next
   regime transition.
2. **[MAD-Velocity](https://github.com/madvelocity/MAD-Velocity-Signal)** — the
   composition of displacement change converts the forecast into an entry gate.
3. **[MAD-Manifold](https://github.com/madvelocity/MAD-Manifold)** — the joint
   displacement–velocity dynamics are stationary and shared across the index.
4. **MAD-Portfolio** (this repo) — the signal converts into a portfolio through
   liquidity, not optimization.

## Citation

> Arrington, L. (2026). *MAD-Portfolio: Portfolio Construction from the
> Moving-Average-Distance Signal.*

## License

MIT — see `LICENSE`.

*Nothing in this repository is investment advice.*
