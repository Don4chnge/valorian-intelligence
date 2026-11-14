"""
The orchestrator. This is the object a user actually touches.

    monitor = ModelMonitor("churn-v3", reference_df, categorical=["region"])
    report  = monitor.check(live_df, batch_label="2026-08-20")
    print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .drift import FeatureDrift, detect_drift
from .driftscore import DriftScore, score_report
from .multivariate import MultivariateDrift, multivariate_drift
from .performance import MIN_PER_CLASS, MIN_SAMPLES, N_BOOTSTRAP, compare_performance
from .quality import (
    NULL_RATE_FAILED,
    NULL_RATE_WARNING,
    DataQualityReport,
    check_quality,
)
from .store import MonitoringStore

_DRIFT_COLUMNS = list(FeatureDrift.__dataclass_fields__) + ["drifted"]


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

        if not self.performance:
            perf = "not_observed"
        elif not self.performance.get("evaluable", True):
            perf = "insufficient_evidence"
        elif self.performance["degraded"]:
            perf = "degraded"
        else:
            perf = "within_tolerance"

        return {
            "data_quality": self.quality.status if self.quality else "healthy",
            "input_shift": shift,
            "performance": perf,
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
        h = self.health
        if h["performance"] == "degraded" or h["data_quality"] == "failed":
            return "critical"
        if (
            h["input_shift"] != "stable"
            or h["data_quality"] == "warning"
            or h["performance"] == "insufficient_evidence"
        ):
            return "warning"
        return "ok"

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
        self.store = MonitoringStore(db_path) if db_path else None

    def check(
        self,
        current: pd.DataFrame,
        batch_label: str | None = None,
        y_true=None,
        y_pred_proba=None,
        persist: bool = True,
    ) -> MonitoringReport:
        """Run one monitoring pass over a batch of production data."""
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

        report = MonitoringReport(
            model_name=self.model_name,
            batch_label=batch_label,
            drift=drift,
            performance=performance,
            alerts=self._build_alerts(drift, performance, quality, mv),
            quality=quality,
            multivariate=mv,
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
            )

        return report

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
            alerts.append(
                f"{r['feature']}: significant input drift (PSI {r['psi']:.3f})"
            )

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
