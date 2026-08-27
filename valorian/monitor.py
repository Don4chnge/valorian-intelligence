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

from .drift import detect_drift
from .driftscore import DriftScore, score_report
from .performance import compare_performance
from .store import MonitoringStore


@dataclass
class MonitoringReport:
    model_name: str
    batch_label: str | None
    drift: pd.DataFrame
    performance: dict | None = None
    alerts: list[str] = field(default_factory=list)
    run_id: int | None = None

    @property
    def driftscore(self) -> DriftScore:
        """Composite 0-100 health score for this run."""
        return score_report(self.drift, self.performance)

    @property
    def status(self) -> str:
        """`critical` if the model is probably broken, `warning` if watch it, else `ok`."""
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
            f"Score:  {ds.score:.0f}/100 ({ds.band})",
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
            lines += [
                "",
                f"{p['metric']}: {p['baseline']:.4f} -> {p['current']:.4f} "
                f"({p['relative_change']:+.1%})"
                + ("  [DEGRADED]" if p["degraded"] else ""),
            ]

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
        drift = detect_drift(
            self.reference,
            current,
            categorical=self.categorical,
            bins=self.bins,
        )

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
            ).to_dict()

        report = MonitoringReport(
            model_name=self.model_name,
            batch_label=batch_label,
            drift=drift,
            performance=performance,
            alerts=self._build_alerts(drift, performance),
        )

        if persist and self.store:
            report.run_id = self.store.save_run(
                model_name=self.model_name,
                drift_frame=drift,
                batch_label=batch_label,
                performance=performance,
                status=report.status,
            )

        return report

    def _build_alerts(self, drift: pd.DataFrame, performance: dict | None) -> list[str]:
        alerts = []

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

        if performance and performance["degraded"]:
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
