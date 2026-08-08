# Study 10 — Negative Moving-Average Area: Results Summary

**Run:** 2026-08-08, script sha `281d1b06…` — the publication re-run of
the registered 2026-08-02 run; fully seeded, statistics byte-identical.
Protocol: AREA_PROTOCOL.md (sha `8396828a…`), locked before code existed. Origin of hypothesis: the author — when the negative
MA area is "too" negative, it should indicate forward returns.

## The question

Point-in-time depth (z) is tested ground. This study tested the
integral — depth × duration of submersion, in sigma-days — as a
cross-sectional conditioner. Two definitions (episode sum since the last
zero-cross; trailing-21-session rolling sum), two horizons (5d, 21d),
monthly formation 2016-01 .. 2025-12, ~500 members per panel, 120 months.
Primary statistic per cell: D1 (deepest decile) mean forward return minus
the equal-weight universe mean. Bars fixed at Šidák-4: |NW-t| ≥ 2.50 and
bootstrap p ≤ .0127.

## The four cells

| cell | mean bps/mo | NW-t | boot p | verdict |
|---|---|---|---|---|
| AREA_EP_f5 | −20.8 | −1.44 | .142 | dead |
| AREA_EP_f21 | +1.1 | +0.06 | .951 | dead |
| AREA_RW_f5 | −39.1 | −3.13 | .0015 | KNOWN-EFFECT RECOVERY |
| AREA_RW_f21 | −3.1 | −0.17 | .875 | dead |

## What the live cell says

The bounce hypothesis is refuted. The deepest-area decile does not recover;
it continues to underperform, by 39 bps over the five sessions after
formation. The effect passes existence with room and survives every
registered robustness leg — worst-month exclusion (direction-aware), both
name-halves negative (−41.5 / −35.1), calm months negative on their own
(−27.1, crisis −111.4). The sign is simply opposite to the registered
hypothesis.

It then fails the novelty gauntlet. Paired against point-z conditioning the
dominance t is −1.33; against trailing-21-day-return conditioning, −1.89.
The registered legs were positive-direction, but the failure is not an
artifact of the lock — read in the inverted direction the magnitudes still
miss the 2.0 bar. Area cannot be statistically separated from depth, and
when two conditioners are indistinguishable the verdict goes to the
simpler, older one.

## Disclosure notes

- Neither benchmark clears existence on its own (Z: t −1.23; REV: t −0.29).
  Classic short-term reversal is absent in this universe and decade. Area
  is the only conditioner of the three that clears the bars. This is
  suggestive only, sits outside the registered bars, and cannot be
  re-tested against these tapes.
- The episode formulation carries nothing; the fixed 21-day window carries
  the entire signal, and only at the 5-day horizon.
- Counters: 82 delist-valued name-months (valued at last close), 26
  era-break exclusions, 546 NaN-area exclusions, 4 tie-fill months (EP
  cells only). Panel median 500.
- D1-minus-D10 disclosure column present in 10_area_summary.csv; monthly
  file persists both benchmark legs, so every dominance statistic is
  recomputable from outputs alone (verified independently after the run).
- SPY exhibit skipped — SPY.csv absent from PORTFOLIO_DATADIR. The run is
  fully seeded; a re-run with the file present reproduces the scored
  numbers identically and fills the exhibit.

## Program context

Third independent confirmation that buying into deep weakness loses in
this universe (the ADDS gate, the study-09 arms, now area), corroborating
the incumbent's confirmed −1→+1 entry, which buys after emergence rather
than into deterioration. The area family is spent — four looks counted,
no re-rolls. A follow-up on the inverted reading (area as an avoid/exit
signal) requires a fresh registration that counts these looks.
