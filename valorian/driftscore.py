"""
DriftScore — a single 0-100 model health number.

The rest of Valorian reports evidence: a PSI per feature, a p-value, a
performance delta. That is the right output for someone doing the diagnosis
and the wrong output for someone deciding whether to care. DriftScore
collapses the evidence into one number, where 100 is a model behaving exactly
as it did at training time and 0 is a model that has fallen apart.

Three components, weighted:

    performance   0.70   how far the metric has moved from baseline
    max PSI       0.20   the worst single feature shift
    breadth       0.10   what share of features drifted at all

Why performance dominates
-------------------------
Input drift is a *predictor* of trouble, not evidence of it. The demo dataset
contains the counterexample directly: one month has the worst input drift in
the series (PSI 0.90) and a model performing exactly at baseline, while a
later month has no detectable input drift and a model that has lost 11% of
its discriminative power. A score that weighted those equally would call the
healthy month a crisis and the broken one mild.

Why performance does not dominate further
-----------------------------------------
Above roughly 0.80 the input terms stop moving the number enough to matter,
and the formula becomes a performance metric with two decorative terms
attached. 0.70 is about the highest weight that leaves drift doing real work.

The labels problem
------------------
Ground truth arrives late — often months late for credit or churn. When there
are no labels yet the performance component cannot be computed, so the score
falls back to the input components alone and reports `mode="inputs_only"`.
That score answers a genuinely different question ("has the world changed?"
rather than "is the model still right?") and the two should not be compared
or plotted on the same line without saying so.

Calibration status
------------------
The weights and cutoffs below were set against the synthetic demo series.
That is circular — the same data that motivated the design was used to check
it. Treat them as a starting hypothesis to be re-tuned against real data with
known outcomes, not as validated constants.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

# Component weights when performance is available.
W_PERFORMANCE = 0.70
W_MAX_PSI = 0.20
W_BREADTH = 0.10

# Saturation points: the value at which a component scores 0 rather than
# continuing to fall. Beyond these the model is already unusable and further
# degradation carries no extra decision-relevant information.
PSI_FLOOR = 0.50          # PSI at or above this scores 0 on the input term
PERFORMANCE_FLOOR = 0.20  # a 20% relative drop scores 0 on the performance term

# Band cutoffs.
BAND_HEALTHY = 85
BAND_WATCH = 70
BAND_DEGRADED = 50

Band = Literal["healthy", "watch", "degraded", "critical"]
Mode = Literal["full", "inputs_only"]


def band_for(score: float) -> Band:
    """
    Map a score onto an action.

        healthy   >= 85   nothing to do
        watch     >= 70   something moved; check again next batch
        degraded  >= 50   investigate now, plan a retrain
        critical   < 50   do not trust this model's output
    """
    if score >= BAND_HEALTHY:
        return "healthy"
    if score >= BAND_WATCH:
        return "watch"
    if score >= BAND_DEGRADED:
        return "degraded"
    return "critical"


@dataclass
class DriftScore:
    score: float
    band: Band
    mode: Mode
    performance_component: float | None
    max_psi_component: float
    breadth_component: float
    n_features: int
    n_drifted: int
    max_psi: float
    relative_change: float | None

    @property
    def actionable(self) -> bool:
        """True when the score warrants a human looking at it."""
        return self.band in ("degraded", "critical")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["actionable"] = self.actionable
        return d

    def __str__(self) -> str:
        suffix = " (inputs only — no labels yet)" if self.mode == "inputs_only" else ""
        return f"DriftScore {self.score:.0f}/100 — {self.band}{suffix}"


def _psi_component(max_psi: float) -> float:
    """100 when nothing moved, 0 at or beyond PSI_FLOOR, linear between."""
    return 100.0 * (1.0 - min(max(max_psi, 0.0) / PSI_FLOOR, 1.0))


def _breadth_component(n_drifted: int, n_features: int) -> float:
    """100 when no feature drifted, 0 when all of them did."""
    if n_features <= 0:
        return 100.0
    return 100.0 * (1.0 - min(n_drifted / n_features, 1.0))


def _performance_component(relative_change: float) -> float:
    """
    100 when the metric held or improved, 0 at or beyond PERFORMANCE_FLOOR.

    Improvement is not rewarded above 100 — a model scoring better than its
    baseline is not healthier than one matching it, and letting a lucky batch
    push the score up would mask drift in the other components.
    """
    drop = max(0.0, -relative_change)
    return 100.0 * (1.0 - min(drop / PERFORMANCE_FLOOR, 1.0))


def compute_driftscore(
    max_psi: float,
    n_drifted: int,
    n_features: int,
    relative_change: float | None = None,
) -> DriftScore:
    """
    Combine drift and performance evidence into one 0-100 health score.

    `relative_change` is the signed proportional move in the model's metric
    against its baseline, as produced by `compare_performance` — so -0.113 for
    an 11.3% fall. Pass None when labels have not arrived yet; the score then
    uses the input terms alone and says so via `mode`.
    """
    psi_c = _psi_component(max_psi)
    breadth_c = _breadth_component(n_drifted, n_features)

    if relative_change is None:
        # Reweight the two input terms to fill the space performance vacated,
        # keeping their 2:1 ratio, so the number stays on a 0-100 scale.
        total = W_MAX_PSI + W_BREADTH
        score = (W_MAX_PSI / total) * psi_c + (W_BREADTH / total) * breadth_c
        perf_c = None
        mode: Mode = "inputs_only"
    else:
        perf_c = _performance_component(relative_change)
        score = (
            W_PERFORMANCE * perf_c
            + W_MAX_PSI * psi_c
            + W_BREADTH * breadth_c
        )
        mode = "full"

    score = round(min(max(score, 0.0), 100.0), 1)

    return DriftScore(
        score=score,
        band=band_for(score),
        mode=mode,
        performance_component=None if perf_c is None else round(perf_c, 1),
        max_psi_component=round(psi_c, 1),
        breadth_component=round(breadth_c, 1),
        n_features=n_features,
        n_drifted=n_drifted,
        max_psi=float(max_psi),
        relative_change=relative_change,
    )


def score_report(drift_frame, performance: dict | None = None) -> DriftScore:
    """
    Convenience wrapper: compute DriftScore straight from a MonitoringReport's
    drift table and performance dict.
    """
    n_features = len(drift_frame)
    n_drifted = int(drift_frame["drifted"].sum()) if n_features else 0
    max_psi = float(drift_frame["psi"].max()) if n_features else 0.0
    rel = performance["relative_change"] if performance else None
    return compute_driftscore(max_psi, n_drifted, n_features, rel)
