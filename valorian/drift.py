"""
Feature-level data drift detection.

Compares a *reference* distribution (what the model was trained on) against a
*current* distribution (what the model is seeing in production) and reports
whether the input data has shifted.

Numeric features  -> Population Stability Index (PSI) + two-sample KS test
Categorical       -> PSI over category shares + chi-square test of independence

PSI convention (standard in credit risk / model monitoring):
    < 0.10   stable
    0.10-0.25 moderate shift, worth investigating
    > 0.25   significant shift, model likely degraded
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy import stats

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
_EPS = 1e-6  # guards log(0) when a bin is empty in one sample

Severity = Literal["stable", "moderate", "significant"]


def _severity(psi: float) -> Severity:
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


@dataclass
class FeatureDrift:
    """Drift result for a single feature."""

    feature: str
    kind: Literal["numeric", "categorical"]
    psi: float
    severity: Severity
    test_name: str
    statistic: float
    p_value: float
    n_reference: int
    n_current: int

    @property
    def drifted(self) -> bool:
        return self.severity != "stable"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["drifted"] = self.drifted
        return d


def _psi_from_counts(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    """PSI between two count vectors that share the same bin definitions."""
    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    # Empty bins would send the log term to +/-inf. Clipping is the usual
    # remedy; it slightly understates PSI for very sparse bins, which is the
    # conservative direction (we under-alarm rather than over-alarm).
    ref_pct = np.clip(ref_pct, _EPS, None)
    cur_pct = np.clip(cur_pct, _EPS, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def numeric_drift(
    feature: str,
    reference: Iterable[float],
    current: Iterable[float],
    bins: int = 10,
) -> FeatureDrift:
    """
    PSI over quantile bins + two-sample Kolmogorov-Smirnov test.

    Bin edges come from the *reference* sample only. This matters: edges must be
    frozen at training time, otherwise both distributions get re-binned together
    and the drift partly cancels itself out.
    """
    ref = pd.Series(list(reference), dtype="float64").dropna().to_numpy()
    cur = pd.Series(list(current), dtype="float64").dropna().to_numpy()

    if ref.size == 0 or cur.size == 0:
        raise ValueError(f"'{feature}': need non-empty reference and current samples")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))

    if edges.size < 2:
        # Constant reference feature - PSI is undefined, so fall back to
        # "did the value change at all?"
        moved = float(not np.allclose(cur, ref[0]))
        return FeatureDrift(
            feature=feature,
            kind="numeric",
            psi=moved,
            severity="significant" if moved else "stable",
            test_name="constant-reference",
            statistic=moved,
            p_value=float("nan"),
            n_reference=int(ref.size),
            n_current=int(cur.size),
        )

    # Open the outer edges so production values beyond the training range still
    # land in the first/last bin instead of being silently dropped.
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    psi = _psi_from_counts(ref_counts, cur_counts)

    ks = stats.ks_2samp(ref, cur)

    return FeatureDrift(
        feature=feature,
        kind="numeric",
        psi=psi,
        severity=_severity(psi),
        test_name="ks_2samp",
        statistic=float(ks.statistic),
        p_value=float(ks.pvalue),
        n_reference=int(ref.size),
        n_current=int(cur.size),
    )


def categorical_drift(
    feature: str,
    reference: Iterable,
    current: Iterable,
) -> FeatureDrift:
    """PSI over category shares + chi-square test on the contingency table."""
    ref = pd.Series(list(reference)).dropna()
    cur = pd.Series(list(current)).dropna()

    if ref.empty or cur.empty:
        raise ValueError(f"'{feature}': need non-empty reference and current samples")

    # Union of categories so unseen production categories are not dropped.
    categories = sorted(set(ref.unique()) | set(cur.unique()), key=str)
    ref_counts = ref.value_counts().reindex(categories, fill_value=0).to_numpy()
    cur_counts = cur.value_counts().reindex(categories, fill_value=0).to_numpy()

    psi = _psi_from_counts(ref_counts, cur_counts)

    table = np.vstack([ref_counts, cur_counts])
    keep = table.sum(axis=0) > 0
    if keep.sum() >= 2:
        chi2, p_value, _, _ = stats.chi2_contingency(table[:, keep])
    else:
        chi2, p_value = 0.0, 1.0

    return FeatureDrift(
        feature=feature,
        kind="categorical",
        psi=psi,
        severity=_severity(psi),
        test_name="chi2_contingency",
        statistic=float(chi2),
        p_value=float(p_value),
        n_reference=int(ref.size),
        n_current=int(cur.size),
    )


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    categorical: list[str] | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """
    Run drift detection across every shared column of two dataframes.

    Returns one row per feature, sorted by PSI descending so the worst
    offenders are at the top.
    """
    shared = [c for c in reference.columns if c in current.columns]
    if not shared:
        raise ValueError("reference and current share no columns")

    categorical = set(categorical or [])
    results = []

    for col in shared:
        is_cat = col in categorical or not pd.api.types.is_numeric_dtype(reference[col])
        if is_cat:
            results.append(categorical_drift(col, reference[col], current[col]))
        else:
            results.append(numeric_drift(col, reference[col], current[col], bins=bins))

    frame = pd.DataFrame([r.to_dict() for r in results])
    return frame.sort_values("psi", ascending=False).reset_index(drop=True)
