"""
Valorian Intelligence — ML observability and drift monitoring.

Quick start:

    from valorian import ModelMonitor

    monitor = ModelMonitor("churn-v3", reference_df, categorical=["region"])
    report = monitor.check(live_df, batch_label="2026-08-20")
    print(report.summary())
"""

from .drift import (
    FeatureDrift,
    categorical_drift,
    detect_drift,
    numeric_drift,
    PSI_MODERATE,
    PSI_SIGNIFICANT,
)
from .driftscore import DriftScore, band_for, compute_driftscore, score_report
from .monitor import ModelMonitor, MonitoringReport
from .performance import PerformanceDrift, compare_performance, rolling_performance, score
from .store import MonitoringStore

__version__ = "0.1.0"

__all__ = [
    "ModelMonitor",
    "MonitoringReport",
    "MonitoringStore",
    "DriftScore",
    "compute_driftscore",
    "score_report",
    "band_for",
    "FeatureDrift",
    "PerformanceDrift",
    "detect_drift",
    "numeric_drift",
    "categorical_drift",
    "compare_performance",
    "rolling_performance",
    "score",
    "PSI_MODERATE",
    "PSI_SIGNIFICANT",
]
