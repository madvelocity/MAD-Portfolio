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
| `tests/` | Known-answer fixture suites: trade mechanics, walk-forward knowability, membership gating, era-break handling, and dollar accounting verified to the cent |
| `results/` | Every CSV cited in the paper, plus the cached index-control series that pin the benchmark vintage |

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

Membership history: the fja05680 point-in-time S&P 500 dataset
(github.com/fja05680/sp500). Prices are split-adjusted and dividend-free; all
strategy and control series are measured on the identical basis.

## The MAD series

1. **MAD-Markov** — position within the displacement band forecasts the next
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
