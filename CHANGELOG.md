# Changelog

## 0.4.0

Benchmark release. `benchmarks/run_benchmark.py` measures false-positive and
detection rates across hundreds of seeds and five sample sizes, with Evidently
run on the same samples. Its first run found two faults in Valorian's drift
detection, both fixed here. Results and their reading are in `benchmarks/`.

### Changed behaviour

- **PSI must clear a noise floor to count as drift.** On 200-row batches the
  0.10 threshold fired on 33% of unchanged samples; it is now 6%. The floor is
  shared across the features in a batch, so twenty unchanged features raise at
  least one alarm in 11% of 200-row batches instead of all of them. No effect
  at 1 000 rows and above: every figure in the README's demo and QLFS tables is
  unchanged. The PSI value itself is not altered, only the severity.
- **Values outside the reference range can raise severity on their own.**
  2% of values beyond the reference min/max (or in unseen categories) is
  moderate, 10% significant. Raw PSI missed 5% of rows moving far into the
  tail at every sample size from 1 000 up; detection is now 80-100%.

### Added

- `psi_noise_floor` and `out_of_range` on every feature drift result, persisted
  and named in alerts.
- `benchmarks/`: the script, `results.csv`, generated `RESULTS.md`, and a
  README covering what was found, including where Valorian is weaker than a KS
  test or than Evidently.
- 19 new tests (129 -> 148).

## 0.3.0

Evidence release: how sure is each finding, and which claims survive a harder test.

### Changed behaviour

- **Multivariate attribution now defaults to permutation importance on held-out
  rows.** The forest's built-in importances (still available as
  `attribution="impurity"`) are computed on training rows and favour features
  with many distinct values. Contributions will differ from 0.2.0, and are all
  zero when the batches are not separable, instead of an arbitrary split.
- `demo/permutation_check.py` runs 500 permutations instead of 20.

### Added

- **Bootstrap confidence intervals on performance** (`ci_low`, `ci_high`,
  `confirmed`). `degraded` is unchanged; `confirmed` says whether a flagged
  drop is clear of sampling noise, and the alert wording follows it.
- **`ModelMonitor(multivariate=True)`** runs the joint-distribution test inside
  `check`, stores it, and shows it in the summary and dashboard. Clear
  separability with every feature individually stable lifts input shift to
  `watch`. Off by default.
- Intervals and multivariate results are persisted; older databases are
  upgraded in place.
- 21 new tests (108 -> 129).

### Findings re-tested on the QLFS data

- Province as the driver of the multivariate shift: **held**, and stronger
  (74-100% of held-out separability, against "over half" before).
- Multivariate shift in 2023 Q1 is not noise: **held** at 500 permutations
  (p = 0.002, against p = 0.048 from 20).
- New caveats documented in the README: QLFS rows are clustered by household
  and the quarters overlap through the rotating panel, so both the p-value and
  "stable across four quarters" are weaker than they read.

## 0.2.0

Correctness release. The statistics are unchanged — every number in the README's
synthetic and QLFS tables reproduces exactly — but what a report *says* about
those numbers has changed in two places.

### Changed behaviour

- **`report.status` no longer reports input drift alone as `critical`.**
  Significant input drift with performance at baseline is now `warning`. The
  synthetic demo's 2026-07 batch moves from `critical` to `warning`; no other
  demo or QLFS batch changes. The old rule is available as
  `report.legacy_status`.
- **Performance on a batch too small to judge is no longer "not degraded".**
  `compare_performance` returns `evaluable=False` and
  `status == "insufficient_evidence"` for fewer than 100 labelled rows, or (for
  classification metrics) fewer than 10 rows of the rarest class. Both floors
  are arguments. Monitors that score very small batches will now see `warning`
  where they saw `ok`.
- **A column that arrives entirely null or as the wrong type no longer crashes
  `ModelMonitor.check`.** It is reported as a data-quality failure and left out
  of the drift table.

### Fixed

- A single-class batch made ROC-AUC NaN, which compared as "not degraded" and
  scored 100 on the DriftScore component carrying 70% of the weight. Unmeasured
  performance now falls back to `mode="inputs_only"`.
- `rmse` and `mae` were listed as lower-is-better metrics but not implemented.
- `y_true` and `y_pred_proba` of different lengths now raise a clear error.

### Added

- `report.health`: separate `data_quality`, `input_shift` and `performance`
  findings.
- `valorian.quality.check_quality`: missing, extra, type-changed and all-null
  columns, and null-rate changes against the reference.
- `missing_reference` / `missing_current` on every feature drift result.
- Health dimensions, quality issues, null rates and evaluability are persisted.
  Databases written by 0.1.0 are upgraded in place on open (columns added,
  nothing rewritten).
- 30 new tests (78 -> 108). One existing assertion changed, for the status rule
  above; the old expectation is still asserted via `legacy_status`.

## 0.1.0

Initial release: PSI / KS / chi-square drift, domain-classifier multivariate
drift, baseline performance comparison, DriftScore, SQLite store, Streamlit
dashboard.
