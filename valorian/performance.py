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

import math
import warnings
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
import pandas as pd
from sklearn import metrics

# Metrics where a *lower* score is better, so degradation is an increase.
_LOWER_IS_BETTER = {"log_loss", "mae", "rmse"}

# Metrics computed on class labels, which therefore need enough of each class
# to mean anything. The rest are regression errors on a continuous target.
_CLASSIFICATION = {"accuracy", "f1", "roc_auc", "log_loss"}

_METRICS = {
    "accuracy": lambda y, p: metrics.accuracy_score(y, (np.asarray(p) >= 0.5).astype(int)),
    "f1": lambda y, p: metrics.f1_score(y, (np.asarray(p) >= 0.5).astype(int), zero_division=0),
    "roc_auc": lambda y, p: metrics.roc_auc_score(y, p),
    "log_loss": lambda y, p: metrics.log_loss(y, np.clip(p, 1e-9, 1 - 1e-9)),
    "mae": lambda y, p: metrics.mean_absolute_error(y, p),
    "rmse": lambda y, p: float(np.sqrt(metrics.mean_squared_error(y, p))),
}

# Evidence floors. Below these a metric is still a number, but its sampling
# noise is wider than the 5% tolerance it is being compared against, so a
# verdict in either direction would be a coin toss presented as a finding.
# Conventions, not derived constants: both are arguments to
# `compare_performance` and should be tightened for rare-event models.
MIN_SAMPLES = 100
MIN_PER_CLASS = 10

# Bootstrap resamples behind the confidence interval. 200 is enough for a
# stable 95% interval to two decimal places and costs about two seconds on a
# 40 000-row batch; pass n_bootstrap=0 to skip it.
N_BOOTSTRAP = 200

PerformanceStatus = Literal["within_tolerance", "degraded", "insufficient_evidence"]


@dataclass
class PerformanceDrift:
    metric: str
    baseline: float
    current: float
    delta: float
    relative_change: float
    degraded: bool
    threshold: float
    n: int = 0
    # False when the batch cannot support a verdict. `degraded` is then False
    # because nothing was shown, NOT because the model is fine — read `status`.
    evaluable: bool = True
    reason: str | None = None
    # Bootstrap interval for `current`. None when the batch was not evaluable
    # or the interval was not requested.
    ci_low: float | None = None
    ci_high: float | None = None
    confidence: float = 0.95

    @property
    def confirmed(self) -> bool | None:
        """
        Whether a flagged drop is distinguishable from sampling noise.

        `degraded` compares a point estimate to a threshold. On a modest batch
        that estimate moves by several percent from resampling alone, so a
        drop just past the threshold may be nothing. This is True when the
        whole interval lies on the bad side of the baseline, False when the
        interval still includes it, and None when there is no drop to confirm
        or no interval to judge it by.

        The baseline is treated as a fixed number. It is itself an estimate,
        and if it was measured on the training data it is optimistic; neither
        is accounted for here.
        """
        if not self.degraded or self.ci_low is None or self.ci_high is None:
            return None
        if self.metric in _LOWER_IS_BETTER:
            return self.ci_low > self.baseline
        return self.ci_high < self.baseline

    @property
    def status(self) -> PerformanceStatus:
        if not self.evaluable:
            return "insufficient_evidence"
        return "degraded" if self.degraded else "within_tolerance"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status
        d["confirmed"] = self.confirmed
        return d


def _bootstrap_interval(
    y: np.ndarray,
    p: np.ndarray,
    metric: str,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap interval for the metric on this batch."""
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    values = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            try:
                value = score(y[idx], p[idx], metric=metric)
            except ValueError:
                continue  # a resample that drew a single class
            if not math.isnan(value):
                values.append(value)
    # If most resamples were unusable the interval would describe only the
    # lucky ones; better to report none.
    if len(values) < max(20, n_bootstrap // 2):
        return None, None
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def _evidence_gap(y: np.ndarray, metric: str, min_samples: int, min_per_class: int) -> str | None:
    """Why this batch cannot support a verdict, or None if it can."""
    if y.size < min_samples:
        return f"only {y.size} labelled rows (minimum {min_samples})"
    if metric in _CLASSIFICATION:
        _, counts = np.unique(y, return_counts=True)
        if counts.size < 2:
            return "labels contain a single class"
        if counts.min() < min_per_class:
            return (
                f"rarest class has {int(counts.min())} rows "
                f"(minimum {min_per_class})"
            )
    return None


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
    min_samples: int = MIN_SAMPLES,
    min_per_class: int = MIN_PER_CLASS,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence: float = 0.95,
    random_state: int = 0,
) -> PerformanceDrift:
    """
    Compare a live batch against the model's baseline score.

    `threshold` is a *relative* tolerance: 0.05 means "flag if the metric moves
    more than 5% in the bad direction". Relative rather than absolute because a
    0.02 AUC drop means something very different at 0.95 than at 0.60.

    A batch with fewer than `min_samples` labelled rows, or (for classification
    metrics) fewer than `min_per_class` rows of its rarest class, is returned
    with `evaluable=False` and `status == "insufficient_evidence"`. It is never
    reported as within tolerance: too little data to detect a drop is not the
    same as no drop.

    An evaluable batch also gets a percentile bootstrap interval for the
    metric (`ci_low`, `ci_high`). The interval does not change `degraded`,
    which stays a comparison of the point estimate to the threshold; it feeds
    `confirmed`, which says whether a flagged drop is clear of sampling noise.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_pred_proba)
    if y.shape[0] != p.shape[0]:
        raise ValueError(
            f"y_true has {y.shape[0]} rows but y_pred_proba has {p.shape[0]}"
        )
    if metric not in _METRICS:
        raise ValueError(f"unknown metric '{metric}'; choose from {sorted(_METRICS)}")

    reason = _evidence_gap(y, metric, min_samples, min_per_class)

    try:
        with warnings.catch_warnings():
            # sklearn warns when a metric is undefined (single-class ROC-AUC).
            # That case is reported through `reason`, so the warning is noise.
            warnings.simplefilter("ignore")
            current = score(y, p, metric=metric)
    except ValueError:
        # e.g. roc_auc on a single-class batch. Only tolerable when the batch
        # was already going to be reported as unevaluable.
        if reason is None:
            raise
        current = float("nan")

    if reason is None and math.isnan(current):
        reason = f"{metric} could not be computed on this batch"

    delta = current - baseline

    if baseline == 0:
        relative = 0.0
    else:
        relative = delta / abs(baseline)

    if reason is not None:
        degraded = False
    elif metric in _LOWER_IS_BETTER:
        degraded = relative > threshold
    else:
        degraded = relative < -threshold

    ci_low = ci_high = None
    if reason is None and n_bootstrap > 0:
        ci_low, ci_high = _bootstrap_interval(
            y, p, metric, n_bootstrap, confidence, random_state
        )

    return PerformanceDrift(
        metric=metric,
        baseline=float(baseline),
        current=current,
        delta=float(delta),
        relative_change=float(relative),
        degraded=bool(degraded),
        threshold=float(threshold),
        n=int(y.shape[0]),
        evaluable=reason is None,
        reason=reason,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=float(confidence),
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
