"""
The orchestrator. This is the object a user actually touches.

    monitor = ModelMonitor("churn-v3", reference_df, categorical=["region"])
    report  = monitor.check(live_df, batch_label="2026-08-20")
    print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from .drift import FeatureDrift, detect_drift
from .driftscore import DriftScore, score_report
from .multivariate import MultivariateDrift, multivariate_drift
from .performance import (
    MIN_PER_CLASS,
    MIN_SAMPLES,
    N_BOOTSTRAP,
    PerformanceDrift,
    cohort_performance,
    compare_performance,
)
from .quality import (
    NULL_RATE_FAILED,
    NULL_RATE_WARNING,
    DataQualityReport,
    check_quality,
)
from .store import MonitoringStore

_DRIFT_COLUMNS = list(FeatureDrift.__dataclass_fields__) + ["drifted"]

# Below this share of a batch's predictions having outcomes, performance is
# reported but not judged. A convention; see ModelMonitor.evaluate.
MIN_LABEL_COVERAGE = 0.8


def rollup_status(health: dict[str, str]) -> str:
    """The one-word status for a set of health dimensions. See MonitoringReport.status."""
    if health["performance"] == "degraded" or health["data_quality"] == "failed":
        return "critical"
    if (
        health["input_shift"] != "stable"
        or health["data_quality"] == "warning"
        or health["performance"] == "insufficient_evidence"
    ):
        return "warning"
    return "ok"


def _performance_state(performance: dict | None) -> str:
    if not performance:
        return "not_observed"
    if not performance.get("evaluable", True):
        return "insufficient_evidence"
    return "degraded" if performance["degraded"] else "within_tolerance"


def _cohort_alerts(cohorts: pd.DataFrame | None, performance: dict | None) -> list[str]:
    """A cohort that got worse while the overall number held is the point of cohorts."""
    if cohorts is None or not len(cohorts):
        return []
    overall_ok = _performance_state(performance) == "within_tolerance"
    alerts = []
    for _, r in cohorts[cohorts["status"] == "degraded"].iterrows():
        prefix = "Overall performance within tolerance, but " if overall_ok else ""
        noise = "" if r["confirmed"] else " (interval still includes its baseline — may be noise)"
        alerts.append(
            f"{prefix}cohort '{r['cohort']}' {r['metric']} is {r['relative_change']:+.1%} "
            f"vs its own baseline on {int(r['n'])} rows{noise}"
        )
    return alerts


@dataclass
class OutcomeReport:
    """Performance for a batch, measured on whichever outcomes have arrived so far."""

    model_name: str
    batch_label: str | None
    run_id: int | None
    n_predictions: int
    n_labelled: int
    coverage: float
    median_delay_days: float | None
    performance: dict
    status: str | None = None
    cohorts: pd.DataFrame | None = None
    alerts: list[str] = field(default_factory=list)

    def summary(self) -> str:
        p = self.performance
        delay = (
            f", median delay {self.median_delay_days:.0f} days"
            if self.median_delay_days is not None else ""
        )
        lines = [
            f"Model:  {self.model_name}",
            f"Batch:  {self.batch_label or 'unlabelled'}",
            f"Labels: {self.n_labelled:,} of {self.n_predictions:,} predictions "
            f"({self.coverage:.0%}){delay}",
        ]
        if p.get("evaluable", True):
            lines.append(
                f"{p['metric']}: {p['baseline']:.4f} -> {p['current']:.4f} "
                f"({p['relative_change']:+.1%})"
                + ("  [DEGRADED]" if p["degraded"] else "")
            )
        else:
            shown = "" if p["current"] != p["current"] else f" (so far {p['current']:.4f})"
            lines.append(f"{p['metric']}: not judged — {p['reason']}{shown}")
        if self.status:
            lines.append(f"Run status: {self.status.upper()}")
        if self.cohorts is not None and len(self.cohorts):
            lines += ["", f"{'cohort':<18}{'n':>7}{'score':>9}  status"]
            for _, r in self.cohorts.iterrows():
                value = "      —" if r["current"] != r["current"] else f"{r['current']:>7.4f}"
                lines.append(f"{r['cohort']:<18}{int(r['n']):>7}  {value}  {r['status']}")
        if self.alerts:
            lines += ["", "Alerts:"] + [f"  ! {a}" for a in self.alerts]
        return "\n".join(lines)


@dataclass
class MonitoringReport:
    model_name: str
    batch_label: str | None
    drift: pd.DataFrame
    performance: dict | None = None
    alerts: list[str] = field(default_factory=list)
    run_id: int | None = None
    quality: DataQualityReport | None = None
    multivariate: MultivariateDrift | None = None
    cohorts: pd.DataFrame | None = None

    @property
    def driftscore(self) -> DriftScore:
        """Composite 0-100 health score for this run."""
        return score_report(self.drift, self.performance)

    @property
    def health(self) -> dict[str, str]:
        """
        Three separate findings, because they are three separate questions.

            data_quality   healthy | warning | failed
                           did the batch arrive in the shape the model expects?
            input_shift    stable | watch | significant
                           do the inputs still look like the reference data?
            performance    within_tolerance | degraded |
                           insufficient_evidence | not_observed
                           is the model still right, as measured on outcomes?

        Only the last one is evidence about the model. `not_observed` means no
        labels were supplied; `insufficient_evidence` means labels were
        supplied but there were too few to support a verdict. Neither is a
        clean bill of health.
        """
        if (self.drift["severity"] == "significant").any():
            shift = "significant"
        elif (self.drift["severity"] == "moderate").any():
            shift = "watch"
        else:
            shift = "stable"

        # A joint-distribution shift with every marginal stable is exactly
        # what the per-feature tests cannot see. Clear separability lifts a
        # stable verdict to "watch"; mild separability (AUC under 0.65) is
        # reported but does not, since at production sample sizes a
        # classifier finds that much in almost any two real batches.
        mv = self.multivariate
        if shift == "stable" and mv is not None and mv.separability in ("clear", "trivial"):
            shift = "watch"

        return {
            "data_quality": self.quality.status if self.quality else "healthy",
            "input_shift": shift,
            "performance": _performance_state(self.performance),
        }

    @property
    def status(self) -> str:
        """
        One-word roll-up of `health`.

            critical   measured performance has degraded, or the batch failed
                       its data-quality checks and cannot be trusted
            warning    something needs a human look: inputs shifted, a quality
                       warning, or labels too thin to judge performance
            ok         none of the above

        Input drift alone never reaches `critical`, however large. Drift is a
        reason to investigate, not evidence the model is wrong — the demo's
        2026-07 batch has a PSI of 0.90 and a model performing at baseline.
        Before 0.2.0 that batch was `critical`; `legacy_status` keeps the old
        rule for anyone who depends on it.
        """
        return rollup_status(self.health)

    @property
    def legacy_status(self) -> str:
        """The pre-0.2.0 rule, where significant input drift alone was `critical`."""
        if self.performance and self.performance["degraded"]:
            return "critical"
        if (self.drift["severity"] == "significant").any():
            return "critical"
        if (self.drift["severity"] == "moderate").any():
            return "warning"
        return "ok"

    @property
    def drifted_features(self) -> list[str]:
        return self.drift.loc[self.drift["drifted"], "feature"].tolist()

    def summary(self) -> str:
        ds = self.driftscore
        lines = [
            f"Model:  {self.model_name}",
            f"Batch:  {self.batch_label or 'unlabelled'}",
            f"Status: {self.status.upper()}",
            f"Score:  {ds.score:.0f}/100 ({ds.band})"
            + ("  [inputs only]" if ds.mode == "inputs_only" else ""),
            "Health: "
            + " | ".join(f"{k.replace('_', ' ')}: {v.replace('_', ' ')}" for k, v in self.health.items()),
            "",
            f"{len(self.drift)} features checked, {len(self.drifted_features)} drifted",
        ]

        top = self.drift.head(5)
        if len(top):
            lines.append("")
            lines.append(f"{'feature':<22}{'psi':>8}  {'severity':<13}{'p-value':>10}")
            lines.append("-" * 55)
            for _, r in top.iterrows():
                lines.append(
                    f"{r['feature']:<22}{r['psi']:>8.4f}  {r['severity']:<13}{r['p_value']:>10.2e}"
                )

        if self.performance:
            p = self.performance
            if p.get("evaluable", True):
                interval = ""
                if p.get("ci_low") is not None:
                    interval = (
                        f"  {p.get('confidence', 0.95):.0%} CI "
                        f"[{p['ci_low']:.4f}, {p['ci_high']:.4f}]"
                    )
                lines += [
                    "",
                    f"{p['metric']}: {p['baseline']:.4f} -> {p['current']:.4f} "
                    f"({p['relative_change']:+.1%})"
                    + ("  [DEGRADED]" if p["degraded"] else "")
                    + interval,
                ]
            else:
                lines += ["", f"{p['metric']}: not evaluated — {p.get('reason')}"]

        if self.multivariate is not None:
            lines += ["", str(self.multivariate)]

        if self.alerts:
            lines.append("")
            lines.append("Alerts:")
            lines += [f"  ! {a}" for a in self.alerts]

        return "\n".join(lines)


class ModelMonitor:
    """
    Wraps a reference dataset and watches incoming batches against it.

    The reference set should be the data the model was *trained* on, captured
    once and then frozen. Re-baselining against recent data is the classic way
    to make a monitor blind to slow drift.
    """

    def __init__(
        self,
        model_name: str,
        reference: pd.DataFrame,
        categorical: list[str] | None = None,
        baseline_score: float | None = None,
        metric: str = "roc_auc",
        performance_threshold: float = 0.05,
        bins: int = 10,
        db_path: str | Path | None = "valorian.db",
        min_samples: int = MIN_SAMPLES,
        min_per_class: int = MIN_PER_CLASS,
        null_rate_warning: float = NULL_RATE_WARNING,
        null_rate_failed: float = NULL_RATE_FAILED,
        n_bootstrap: int = N_BOOTSTRAP,
        multivariate: bool = False,
        multivariate_permutations: int = 0,
        cohort_baselines: dict | None = None,
        min_label_coverage: float = MIN_LABEL_COVERAGE,
    ):
        if reference.empty:
            raise ValueError("reference dataframe is empty")

        self.model_name = model_name
        self.reference = reference.copy()
        self.categorical = categorical or []
        self.baseline_score = baseline_score
        self.metric = metric
        self.performance_threshold = performance_threshold
        self.bins = bins
        self.min_samples = min_samples
        self.min_per_class = min_per_class
        self.null_rate_warning = null_rate_warning
        self.null_rate_failed = null_rate_failed
        self.n_bootstrap = n_bootstrap
        # Off by default: it fits a forest per batch, which is seconds rather
        # than milliseconds, and nobody should pay that without asking.
        self.multivariate = multivariate
        self.multivariate_permutations = multivariate_permutations
        # Each cohort's own baseline score. A cohort without one is measured
        # but not judged — the overall baseline is not a fair yardstick for it.
        self.cohort_baselines = cohort_baselines or {}
        self.min_label_coverage = min_label_coverage
        self.store = MonitoringStore(db_path) if db_path else None

    def check(
        self,
        current: pd.DataFrame,
        batch_label: str | None = None,
        y_true=None,
        y_pred_proba=None,
        persist: bool = True,
        prediction_ids=None,
        cohorts=None,
    ) -> MonitoringReport:
        """
        Run one monitoring pass over a batch of production data.

        Outcomes rarely exist when a batch is scored. Pass `prediction_ids`
        with `y_pred_proba` and no `y_true` to record the predictions now; the
        run is stored with performance `not_observed`. When outcomes arrive,
        `record_outcomes` stores them and `evaluate` measures the batch.

        `cohorts` is one label per row (a region, a product). With labels it
        adds per-cohort performance to the report; with `prediction_ids` it is
        stored for use when the outcomes come in.
        """
        if prediction_ids is not None:
            if y_pred_proba is None:
                raise ValueError("prediction_ids need y_pred_proba: the scores to log")
            if not (persist and self.store):
                raise ValueError("logging predictions needs a store (db_path) and persist=True")
        if not any(c in current.columns for c in self.reference.columns):
            raise ValueError("reference and current share no columns")

        quality = check_quality(
            self.reference,
            current,
            categorical=self.categorical,
            null_rate_warning=self.null_rate_warning,
            null_rate_failed=self.null_rate_failed,
        )

        # A column that arrived entirely null or as the wrong type cannot be
        # drift-tested. It is already recorded as a quality failure, so leave
        # it out here rather than let it crash the run or report a fake PSI.
        usable = current.drop(columns=quality.unusable_columns)
        if any(c in usable.columns for c in self.reference.columns):
            drift = detect_drift(
                self.reference,
                usable,
                categorical=self.categorical,
                bins=self.bins,
            )
        else:
            drift = pd.DataFrame(columns=_DRIFT_COLUMNS)

        performance = None
        if y_true is not None and y_pred_proba is not None:
            if self.baseline_score is None:
                raise ValueError(
                    "baseline_score must be set on the monitor to compare performance"
                )
            performance = compare_performance(
                baseline=self.baseline_score,
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                metric=self.metric,
                threshold=self.performance_threshold,
                min_samples=self.min_samples,
                min_per_class=self.min_per_class,
                n_bootstrap=self.n_bootstrap,
            ).to_dict()

        mv = None
        if self.multivariate and len(drift):
            columns = drift["feature"].tolist()
            try:
                mv = multivariate_drift(
                    self.reference[columns],
                    usable[columns],
                    categorical=[c for c in self.categorical if c in columns],
                    n_permutations=self.multivariate_permutations,
                )
            except ValueError:
                mv = None  # too few rows to cross-validate; nothing to report

        cohort_table = None
        if cohorts is not None and performance is not None:
            cohort_table = self._cohort_table(y_true, y_pred_proba, cohorts)

        report = MonitoringReport(
            model_name=self.model_name,
            batch_label=batch_label,
            drift=drift,
            performance=performance,
            alerts=self._build_alerts(drift, performance, quality, mv)
            + _cohort_alerts(cohort_table, performance),
            quality=quality,
            multivariate=mv,
            cohorts=cohort_table,
        )

        if persist and self.store:
            report.run_id = self.store.save_run(
                model_name=self.model_name,
                drift_frame=drift,
                batch_label=batch_label,
                performance=performance,
                status=report.status,
                health=report.health,
                quality_issues=quality.issues,
                multivariate=mv.to_dict() if mv else None,
                cohorts=cohort_table,
            )

        if prediction_ids is not None:
            self.store.log_predictions(
                self.model_name,
                prediction_ids,
                y_pred_proba,
                batch_label=batch_label,
                run_id=report.run_id,
                cohorts=cohorts,
            )
            if y_true is not None:
                self.store.log_outcomes(self.model_name, prediction_ids, y_true)

        return report

    def _cohort_table(self, y_true, y_pred_proba, cohorts) -> pd.DataFrame:
        return cohort_performance(
            y_true,
            y_pred_proba,
            cohorts,
            metric=self.metric,
            baselines=self.cohort_baselines,
            threshold=self.performance_threshold,
            min_samples=self.min_samples,
            min_per_class=self.min_per_class,
            n_bootstrap=self.n_bootstrap,
        )

    def record_outcomes(
        self, prediction_ids, outcomes, observed_at=None, overwrite: bool = False
    ) -> dict:
        """
        Store what actually happened for predictions logged by `check`.

        Returns counts of what was recorded, already on file, in conflict with
        an earlier value, or unknown. Safe to re-send the same file.
        """
        if not self.store:
            raise ValueError("recording outcomes needs a store (db_path)")
        return self.store.log_outcomes(
            self.model_name, prediction_ids, outcomes,
            observed_at=observed_at, overwrite=overwrite,
        )

    def evaluate(self, batch_label: str | None) -> OutcomeReport:
        """
        Measure a batch on the outcomes that have arrived so far, and update
        its stored run.

        The outcomes that arrive first are rarely a fair sample of the batch.
        In credit a default is known the month it happens, while a loan is
        only known to be good when it matures, so early labels are mostly the
        failures — and the earliest failures are the ones the model found
        easiest to rank. A metric on those rows says little about the batch.
        So until `min_label_coverage` of the batch has an outcome (0.8 by
        default) the metric is reported as "so far" and the verdict is
        `insufficient_evidence`, never `within_tolerance` or `degraded`.
        Coverage is a guard against that bias, not a cure for it: at 80% the
        missing fifth can still differ from the rest.
        """
        if not self.store:
            raise ValueError("evaluating outcomes needs a store (db_path)")
        if self.baseline_score is None:
            raise ValueError("baseline_score must be set on the monitor to compare performance")

        frame = self.store.labelled_predictions(self.model_name, batch_label)
        if frame.empty:
            raise KeyError(
                f"no predictions logged for model '{self.model_name}', batch {batch_label!r}"
            )

        labelled = frame.dropna(subset=["outcome"])
        coverage = len(labelled) / len(frame)
        delay = None
        if len(labelled):
            waited = (
                pd.to_datetime(labelled["observed_at"], utc=True)
                - pd.to_datetime(labelled["predicted_at"], utc=True)
            )
            delay = float(waited.dt.total_seconds().median() / 86_400)

        if len(labelled):
            result = compare_performance(
                baseline=self.baseline_score,
                y_true=labelled["outcome"].to_numpy(),
                y_pred_proba=labelled["score"].to_numpy(),
                metric=self.metric,
                threshold=self.performance_threshold,
                min_samples=self.min_samples,
                min_per_class=self.min_per_class,
                n_bootstrap=self.n_bootstrap,
            )
        else:
            result = PerformanceDrift(
                metric=self.metric, baseline=float(self.baseline_score),
                current=float("nan"), delta=float("nan"), relative_change=float("nan"),
                degraded=False, threshold=float(self.performance_threshold),
                n=0, evaluable=False, reason="no outcomes recorded yet",
            )

        if result.evaluable and coverage < self.min_label_coverage:
            result = replace(
                result,
                evaluable=False,
                degraded=False,
                reason=(
                    f"outcomes in for {coverage:.0%} of the batch "
                    f"(minimum {self.min_label_coverage:.0%}); early outcomes "
                    "are rarely a fair sample"
                ),
            )
        performance = result.to_dict()

        cohort_table = None
        if len(labelled) and labelled["cohort"].notna().any():
            cohort_table = self._cohort_table(
                labelled["outcome"].to_numpy(), labelled["score"].to_numpy(),
                labelled["cohort"].to_numpy(),
            )
            if coverage < self.min_label_coverage:
                # Same reasoning as the overall metric: measured, not judged.
                judged = cohort_table["status"].isin(["within_tolerance", "degraded"])
                cohort_table.loc[judged, "status"] = "insufficient_evidence"

        run_id = frame["run_id"].dropna()
        run_id = int(run_id.iloc[0]) if len(run_id) else None
        status = None
        if run_id is not None:
            stored = self.store.run_detail(run_id)["run"]
            status = rollup_status({
                "data_quality": stored.get("data_quality") or "healthy",
                "input_shift": stored.get("input_shift") or "stable",
                "performance": _performance_state(performance),
            })
            self.store.update_performance(
                run_id, performance, status, _performance_state(performance),
                coverage=coverage, median_delay_days=delay, cohorts=cohort_table,
            )

        alerts = []
        if performance["degraded"]:
            alerts.append(
                f"{performance['metric']} degraded {performance['relative_change']:+.1%} vs baseline"
                + ("" if performance.get("confirmed") is not False
                   else " — interval still includes the baseline, may be sampling noise")
            )
        alerts += _cohort_alerts(cohort_table, performance)

        return OutcomeReport(
            model_name=self.model_name,
            batch_label=batch_label,
            run_id=run_id,
            n_predictions=len(frame),
            n_labelled=len(labelled),
            coverage=coverage,
            median_delay_days=delay,
            performance=performance,
            status=status,
            cohorts=cohort_table,
            alerts=alerts,
        )

    def _build_alerts(
        self,
        drift: pd.DataFrame,
        performance: dict | None,
        quality: DataQualityReport | None = None,
        multivariate: MultivariateDrift | None = None,
    ) -> list[str]:
        alerts = []

        if quality:
            alerts += [f"Data quality: {issue}" for issue in quality.issues]

        if performance and not performance.get("evaluable", True):
            alerts.append(
                f"{performance['metric']} not evaluated — {performance['reason']}. "
                "Performance is unverified for this batch, not confirmed healthy"
            )

        significant = drift[drift["severity"] == "significant"]
        for _, r in significant.iterrows():
            detail = f"PSI {r['psi']:.3f}"
            if r.get("out_of_range", 0.0) > 0:
                detail += f", {r['out_of_range']:.1%} of values outside the reference range"
            alerts.append(f"{r['feature']}: significant input drift ({detail})")

        moderate = drift[drift["severity"] == "moderate"]
        if len(moderate):
            names = ", ".join(moderate["feature"].head(3))
            more = f" (+{len(moderate) - 3} more)" if len(moderate) > 3 else ""
            alerts.append(f"Moderate drift on: {names}{more}")

        if (
            multivariate is not None
            and multivariate.separability in ("clear", "trivial")
            and significant.empty
            and moderate.empty
        ):
            drivers = ", ".join(k for k, v in multivariate.top_contributors(3) if v > 0)
            alerts.append(
                f"Joint distribution shifted (multivariate AUC {multivariate.auc:.3f}) "
                "with every feature individually stable"
                + (f" — driven by: {drivers}" if drivers else "")
            )

        if performance and performance["degraded"]:
            if performance.get("confirmed") is False:
                alerts.append(
                    f"{performance['metric']} degraded {performance['relative_change']:+.1%} "
                    f"vs baseline, but its {performance.get('confidence', 0.95):.0%} interval "
                    f"[{performance['ci_low']:.4f}, {performance['ci_high']:.4f}] still "
                    "includes the baseline — may be sampling noise; confirm on the next batch"
                )
            else:
                alerts.append(
                    f"{performance['metric']} degraded {performance['relative_change']:+.1%} "
                    f"vs baseline — retraining likely needed"
                )

        # The genuinely dangerous case: inputs look fine, outputs are wrong.
        # Usually means the target relationship moved, not the feature space.
        if performance and performance["degraded"] and significant.empty:
            alerts.append(
                "Performance dropped without significant input drift — "
                "suspect concept drift or a data pipeline/labelling fault"
            )

        return alerts
