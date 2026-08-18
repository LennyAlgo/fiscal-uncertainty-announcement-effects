# Fiscal uncertainty and macro announcement effects

Event study of ES (S&P 500) and ZN (10-year Treasury) intraday reactions to U.S.
macroeconomic announcements under fiscal policy uncertainty and government
shutdowns. MSc thesis, Ghent University.

## Research question

Does the market's reaction to a macroeconomic surprise depend on the prevailing
level of fiscal policy uncertainty? Two identification strategies are used side
by side: a continuous monthly fiscal policy uncertainty (FPU) index, and the
discrete episodes of U.S. federal government shutdowns.

The dependent variable is the intraday futures return around the release,
measured in basis points over a 5-minute window from the announcement bar
(`ret_sym5` for ES, `zn_sym5` for ZN), with 10-, 15-, 30-minute and symmetric
10-minute windows retained as robustness.

## Data

| Input | Source | In repo |
|---|---|---|
| ES 5-min bars, 392,463 bars, 2010-06-07 to 2026-05-18 | Licensed vendor data | No |
| ZN 5-min bars, 393,664 bars, same span | Licensed vendor data | No |
| Announcement panel — 2,931 events, 16 release types | Hand-built; surprises from public actuals vs. Reuters consensus | Yes |
| Shutdown panel — same 2,931 events, 68 delayed releases | Hand-built from shutdown chronology | Yes |

The two announcement mastersheets in `data/` carry release date and time, type,
raw and standardised surprise, the FPU level, and the shutdown / pre-window /
post-window / delayed-release flags. They contain no price data.

Surprises are computed against Reuters consensus forecasts. The underlying
`Actual` and `Forecast` columns are vendor data and are **not** redistributed
here — the published panels carry only the derived surprise measures, which is
what the analysis consumes. Nothing in the pipeline reads the raw forecast
columns, so their removal does not affect reproducibility.

`FPU_raw` enters as a pre-constructed monthly composite. Its construction is
documented in the thesis and is upstream of this script — the code standardises
it but does not build it.

Sample handling worth knowing about:

- **COVID excluded.** 62 event rows dropped (Mar–Dec 2020). Surprise
  standardisation uses non-COVID σ per release type, computed uniformly in code.
  Section 10c re-runs the main specification with full-sample σ to show the
  standardisation basis does not drive the results.
- **Event-time restructuring.** Announcements sharing a date and minute share one
  return window, so they are collapsed to a single `event_id` and estimated in
  wide form. This is why the main specifications report N ≈ 2,367 rather than
  the 2,855 long-format rows — regressing overlapping releases as independent
  observations would double-count the same price move.

## Method

A specification ladder rather than a single preferred model, so the reader can
see how the estimate moves as structure is added:

| Family | Models | What it adds |
|---|---|---|
| Continuous FPU | MC1 → MC5 | baseline → pooled FPU interaction → type FE → per-type interactions → year FE |
| Dummy FPU | M2 → M5 | same ladder with a fiscal-stress dummy instead of the continuous index |
| Shutdowns | M6 → M13c | active shutdown, pre-window (20d), post-window (45d), episode, mechanisms, delayed releases |
| Asymmetry | MA1, MA2 | positive vs negative surprise split |
| Robustness | §10–11b | return windows, lagged FPU, COVID σ basis, winsorised sample, FPU thresholds at the 70/75/80/90th percentile |

Standard errors are reported two ways throughout — clustered on `event_id` (main)
and HC3 (robustness). Per-type regressions carry Bonferroni-adjusted p-values
alongside the raw ones.

## Findings

**Pooled effects are absent.** MC2 (pooled continuous FPU) gives −0.31 bps,
p = 0.44, R² = 0.007. The dummy equivalent M2 gives 0.82 bps, p = 0.38. There is
no average FPU effect on announcement reactions to be found in this sample.

**Type-specific effects exist but cut both ways.** In the main specification
(MC4, event-clustered, N = 2,367, R² = 0.157), per 1-SD increase in FPU:

| Release | ES reaction | p |
|---|---|---|
| NFP | +9.71 bps | 0.027 |
| ISM PMI | +6.16 bps | 0.038 |
| Existing home sales | +3.05 bps | 0.032 |
| Consumer credit | −1.20 bps | 0.005 |
| Retail sales | −3.92 bps | 0.023 |
| Construction spending | −4.77 bps | 0.008 |
| Unemployment rate | −9.04 bps | 0.005 |

Several of these are *dampening* — the market reacts less to a given surprise
when fiscal uncertainty is high — which is the opposite sign to the
amplification hypothesised. Reported as found.

**Almost nothing survives multiple-testing correction.** With 16 release types
the Bonferroni threshold is p < 0.0031. In the per-type regressions only
construction spending clears it (β = −7.49, p = 0.0018, Bonferroni p = 0.029).
Unemployment rate, consumer credit and retail sales do not. The type-specific
results above should be read as suggestive, not established.

**The most robust single result is on delayed releases.** M13c: for releases
postponed by a shutdown, the interaction of absolute surprise with the delayed
flag is −2.37 bps, p = 0.0008 — stale news moves the market less. This clears
Bonferroni comfortably and is the one finding the sample supports strongly. It
does not replicate on ZN (−0.52 bps, p = 0.35).

**Shutdown level effects are null.** Active shutdown: −1.06 bps, p = 0.61.
Post-window: 0.47 bps, p = 0.87. Episode: 0.63 bps, p = 0.75. Only the
pre-window shows anything (−2.95 bps, p = 0.017, on 72 observations — too few to
lean on).

**No asymmetry detected.** MA1/MA2 find no release type where positive and
negative surprises are priced differently.

## Contents

```
fpu_event_study.py                  full pipeline: load → returns → clean → estimate → save
data/mastersheet_epu.xlsx           announcement panel with FPU index
data/mastersheet_shutdowns.xlsx     announcement panel with shutdown flags
results/regression_results.xlsx     44 sheets: all coefficients, descriptives, thesis tables 1–5
requirements.txt
```

## Reproducing

```bash
pip install -r requirements.txt
python fpu_event_study.py
```

The announcement panels resolve automatically. The ES/ZN bars are licensed and
not distributed; the script exits with an explicit message if they are absent.
To supply your own, place them in `data/futures/` as
`ES_5min_full(2010-2026).xlsx` and `ZN_5min_full(2010-2026)_v2.csv`, or point
elsewhere:

```bash
FUTURES_DIR=/path/to/bars python fpu_event_study.py
```

Both files need columns `ts_event, open, high, low, close, volume, return` at
5-minute frequency. ES timestamps are read as naive New York time; ZN timestamps
are read as UTC and converted to New York.

Section toggles at the top of the script (`RUN_DESCRIPTIVES`,
`RUN_SHUTDOWN_MODELS`, …) run subsets. A full run takes a few minutes, mostly
spent reading the 20 MB ES workbook, and rewrites
`results/regression_results.xlsx`.

## Limitations

- Single-country, single-decade sample; 16 release types, several with few events.
- Consensus forecasts are vendor point estimates, so surprise is measured with error.
- The shutdown sub-samples are small (41–152 observations) and the pre-window
  result in particular is fragile.
- FPU is monthly while the outcome is intraday, so the regime variable is coarse
  relative to the event.
- The type-specific results do not survive Bonferroni correction, as noted above.
