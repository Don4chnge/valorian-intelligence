"""
Concept drift detection.

Data drift asks "have the inputs changed?". Concept drift asks the harder
question: "has the relationship between inputs and target changed?" — which
shows up as model performance decaying even when the inputs look normal.

This module needs ground-truth labels, so in practice it runs on a lag (you
find out last month's predictions were wrong once last month's outcomes land).
That lag is a real limitation and worth stating plainly in any write-up.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn import metrics

# Metrics where a *lower* score is better, so degradation is an increase.
_LOWER_IS_BETTER = {"log_loss", "mae", "rmse"}

_METRICS = {
    "accuracy": lambda y, p: metrics.accuracy_score(y, (np.asarray(p) >= 0.5).astype(int)),
    "f1": lambda y, p: metrics.f1_score(y, (np.asarray(p) >= 0.5).astype(int), zero_division=0),
    "roc_auc": lambda y, p: metrics.roc_auc_score(y, p),
    "log_loss": lambda y, p: metrics.log_loss(y, np.clip(p, 1e-9, 1 - 1e-9)),
}


@dataclass
class PerformanceDrift:
    metric: str
    baseline: float
    current: float
    delta: float
    relative_change: float
    degraded: bool
    threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


def score(y_true, y_pred_proba, metric: str = "roc_auc") -> float:
    """Score a batch of predictions with the named metric."""
    if metric not in _METRICS:
        raise ValueError(f"unknown metric '{metric}'; choose from {sorted(_METRICS)}")
    return float(_METRICS[metric](np.asarray(y_true), np.asarray(y_pred_proba)))


def compare_performance(
    baseline: float,
    y_true,
    y_pred_proba,
    metric: str = "roc_auc",
    threshold: float = 0.05,
) -> PerformanceDrift:
    """
    Compare a live batch against the model's baseline score.

    `threshold` is a *relative* tolerance: 0.05 means "flag if the metric moves
    more than 5% in the bad direction". Relative rather than absolute because a
    0.02 AUC drop means something very different at 0.95 than at 0.60.
    """
    current = score(y_true, y_pred_proba, metric=metric)
    delta = current - baseline

    if baseline == 0:
        relative = 0.0
    else:
        relative = delta / abs(baseline)

    if metric in _LOWER_IS_BETTER:
        degraded = relative > threshold
    else:
        degraded = relative < -threshold

    return PerformanceDrift(
        metric=metric,
        baseline=float(baseline),
        current=current,
        delta=float(delta),
        relative_change=float(relative),
        degraded=bool(degraded),
        threshold=float(threshold),
    )


def rolling_performance(
    frame: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    batch_col: str,
    metric: str = "roc_auc",
) -> pd.DataFrame:
    """
    Score performance per batch (per day, per week, per deployment — whatever
    `batch_col` encodes) so you can plot the decay curve rather than seeing a
    single before/after number.
    """
    rows = []
    for batch, chunk in frame.groupby(batch_col, sort=True):
        try:
            value = score(chunk[y_true_col], chunk[y_pred_col], metric=metric)
        except ValueError:
            # e.g. roc_auc on a batch containing a single class
            value = float("nan")
        rows.append({"batch": batch, "metric": metric, "score": value, "n": len(chunk)})

    return pd.DataFrame(rows)
