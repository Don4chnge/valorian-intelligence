# Changelog

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
