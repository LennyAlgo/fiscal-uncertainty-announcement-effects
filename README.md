# Fiscal uncertainty and macro announcement effects

Event study of ES (S&P 500) and ZN (10-year Treasury) intraday reactions to U.S.
macroeconomic announcements under fiscal policy uncertainty and government
shutdowns.

Code and results for the dissertation *Macroeconomic Announcement Surprises
under Fiscal Uncertainty: Evidence from Fiscal Policy Uncertainty and U.S.
Government Shutdowns* — Lennert Van Steen, Ghent University, supervised by
Prof. Dr. Mikael Petitjean, academic year 2025–2026.

## Research question

Does the market's reaction to a macroeconomic surprise depend on the prevailing
level of fiscal policy uncertainty? Two identification strategies are used side
by side: a continuous monthly fiscal policy uncertainty (FPU) index built from
the fiscal-policy component of the Baker, Bloom & Davis (2016) index, and the
discrete episodes of U.S. federal government shutdowns.

The headline result is that **fiscal uncertainty does not scale announcement
reactions uniformly — it reshapes them selectively**, activating equity
reactions to releases that are ignored under normal conditions, amplifying a
few others, dampening the bond market's response to real activity, and
producing an opposite-signed cross-market reaction for a subset of releases.

Hypotheses tested: H1 surprises move markets; H2 FPU amplifies the average
reaction; H2a the effect differs across release types; H3a–H3d shutdown
windows and delayed releases; H4 equity and Treasury markets move in opposite
directions under fiscal stress (flight-to-safety).

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

Model 4 — per-type surprise slopes βⱼ with per-type FPU interactions δⱼ, signed
returns, announcement fixed effects, event-time clustered errors — is the main
specification. Because the contemporaneous monthly FPU incorporates news
published after early-month releases, it is an ex-post regime classification;
the prior-month (lagged) specification is therefore treated as **co-main**
rather than as robustness.

Standard errors are reported two ways throughout — clustered on `event_id` (main)
and HC3 (robustness).

## Findings

**H1 confirmed — surprises move markets.** Pooled absolute surprise on absolute
return: +1.93 bps, p < 0.01 (N = 2,855). Adjusted R² is 0.005, as expected when
one common slope is imposed across sixteen heterogeneous releases.

**H2 not supported at the pooled level.** Adding FPU and its interaction leaves
the surprise effect unchanged and the interaction insignificant (−0.52, ns;
−0.31, ns with type FE). FPU does carry a significant *level* effect of +1.18 bps
(p < 0.01) — reactions are larger in magnitude when fiscal uncertainty is high —
but it does not amplify the average surprise sensitivity. This null is
informative rather than empty: a single pooled slope averages amplification in
some releases against dampening in others.

**H2a confirmed — the effect is selective, not uniform.** Once each release gets
its own slope (Model 4, N = 2,367 ES / 2,365 ZN), three distinct patterns appear.
δⱼ is the effect of a one-SD increase in FPU, in bps:

*Activation* — insignificant baseline slope, significant interaction; these
releases start to matter for equities only under fiscal stress:

| Release | βⱼ (ES) | δⱼ (ES) |
|---|---|---|
| NFP | 3.97 ns | **+9.71** ** |
| Unemployment rate | 2.30 ns | **−9.04** *** |
| Consumer credit | 0.47 ns | **−1.20** *** |

The opposite signs are economically coherent: a positive NFP surprise is a
strong labour market, a positive unemployment surprise is a weak one.

*Amplification* — interaction shares the sign of an already-significant baseline:
Existing home sales (β = +2.44**, δ = +3.05**) and ISM PMI (β = +11.77***,
δ = +6.16**).

*Bond-market dampening* — for real-activity releases the ZN interaction runs
against its baseline, pulling the reaction toward zero: Industrial production
(β = −1.12***, δ = +0.60*) and GDP (β = −3.78***, δ = +2.47*).

**H4 confirmed — flight-to-safety.** For a subset of releases, equities and
Treasuries move in *opposite* directions on the same surprise under fiscal
stress. Clearest for Retail sales:

| Release | δⱼ ES | δⱼ ZN |
|---|---|---|
| Retail sales | −3.92 ** | **+4.80** ** |
| Construction spending | −4.77 *** | +1.59 * |

Equities sell off while Treasury prices rise — the configuration Baele et al.
(2020) associate with risk-off reallocation. Because it operates across two
markets at once, it is hard to reconcile with a pure noise or attention story,
which would not predict a coordinated opposite-signed move.

For context, the largest *baseline* reactions are CPI (ES −23.80***, ZN
−12.49***) and, on the bond side, NFP (ZN −15.80***) — the payroll component
dominates the employment report for Treasuries, consistent with Balduzzi et al.
(2001).

**The main equity results hold under the real-time-valid lagged FPU.** NFP
(+8.63**), unemployment rate (−5.41**), consumer credit (−0.93*), retail sales
(−4.98***) and ISM PMI (+5.29**) all keep sign and significance. Existing home
sales and construction spending lose significance and are treated with more
caution. A 75th-percentile regime dummy corroborates the core effects
independently.

**H3b rejected — in the opposite direction.** Equity sensitivity is *dampened*
in the 20-day pre-shutdown window (−2.95 bps, p < 0.05), where amplification was
predicted. Active and post windows show no pooled effect (−1.06 ns, +0.47 ns),
so H3c is unsupported and H3a is uninformative given the limited power.

**H3d rejected — also in the opposite direction.** Releases delayed by a
shutdown produce a *weaker* equity reaction to their surprise content, not the
stronger one an information-backlog mechanism predicts: interaction −2.37 bps,
p < 0.01, against a full-sample surprise slope of +2.07 bps. The effect is
equity-specific — ZN shows nothing (−0.52, ns). Candidate explanations: the
information has already reached the market by other routes, the release is
partly anticipated once the shutdown resolves, or data compiled under disrupted
conditions is discounted as lower quality.

**Robustness.** Signs and significance of the principal interactions survive
HC3 errors, winsorising forecast errors at the 1st/99th percentiles, alternative
event windows ([0,+10], [0,+15], [0,+30], [−5,+5]), and computing surprise σ
over the full sample including COVID.

### How much weight the shutdown results carry

Deliberately less than the FPU results. The identifying variation comes from a
handful of episodes: the pre-window rests on four to six observations per
release, the post-window on eight to twelve, and the active window on roughly
sixty spread very unevenly (one for GDP). Type-specific shutdown coefficients
are correspondingly large and imprecise — the −69.64 bps pre-window
unemployment-rate estimate comes from four observations and exceeds every
baseline reaction in the sample. Those estimates are reported to document
heterogeneity, not as stable effects; the pooled shutdown models are the
trustworthy layer.

As a further check, the standalone per-type regressions carry Bonferroni-adjusted
p-values for the sixteen simultaneous tests (threshold p < 0.0031). Only
construction spending clears it there. The Model 4 estimates above are jointly
estimated with fixed effects and clustered errors rather than sixteen separate
regressions, but the adjustment is reported alongside them in the workbook.

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

- **Few shutdown episodes.** The binding constraint. Effective degrees of freedom
  are small and the type-specific shutdown estimates are imprecise, which is why
  they are read as suggestive throughout. Extending the sample backwards using
  Money Market Services survey forecasts, which predate this sample, would add
  earlier shutdown episodes and sharpen the event-based identification.
- **No business-cycle control.** Reactions are conditioned on fiscal uncertainty
  alone, yet announcement effects are known to be state-dependent — the same
  surprise can move markets in opposite directions in expansion versus
  contraction (McQueen & Roley 1993; Boyd et al. 2005). If fiscal stress
  coincides with particular cycle phases, part of the estimated FPU effect may
  reflect the underlying economic state.
- **Coarse regime variable.** FPU is monthly while the outcome is intraday.
- **One measure, two asset classes.** A single newspaper-based fiscal
  uncertainty index, and only equity and Treasury futures.
- Consensus forecasts are vendor point estimates, so surprise is measured with error.
- Single-country sample; 16 release types, several with few events.
