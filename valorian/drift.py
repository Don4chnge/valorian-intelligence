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
    > 0.25   significant shift, check measured performance

Two things the convention does not cover, both found by benchmarking this
module against samples with known answers (see benchmarks/):

Small samples. PSI between two samples of the *same* distribution is not zero;
it is roughly (bins - 1) * (1/n_ref + 1/n_cur). With ten bins and 200 rows a
side that is 0.09, so the 0.10 threshold fires on about half of all unchanged
batches. A PSI therefore also has to clear its own noise floor before it
counts as drift (`psi_noise_floor`).

Values beyond the reference range. Quantile bins with open outer edges keep
such values rather than dropping them, but they land in a bin that already
holds 10% of the data, where a few percent of extreme values barely move the
PSI. The share of values beyond the reference's extremes is therefore measured
directly (`out_of_range`) and can raise the severity on its own. "Extremes"
means the outermost 0.1% of the reference at each end rather than its literal
minimum and maximum, because the literal range keeps widening as the reference
grows and the check would weaken with it.
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

# Share of current values outside anything the reference contained: beyond its
# min/max for a numeric feature, or in a category it never had. Conventions.
RANGE_MODERATE = 0.02
RANGE_SIGNIFICANT = 0.10
# The reference's most extreme 0.1% at each end marks the edge of its range.
EXTREME_TAIL = 0.001
_NOISE_QUANTILE = 0.95

Severity = Literal["stable", "moderate", "significant"]
_ORDER = {"stable": 0, "moderate": 1, "significant": 2}


def psi_noise_floor(
    n_reference: int, n_current: int, n_bins: int, n_comparisons: int = 1
) -> float:
    """
    The PSI that two samples of one distribution stay under 95% of the time.

    `n_comparisons` is how many features are being checked together. Twenty
    features each allowed a 5% false-alarm rate will produce at least one
    false alarm in most batches, so the 5% is shared across them (Bonferroni):
    the floor is the level all of them stay under 95% of the time.

    For samples from the same distribution, PSI / (1/n_ref + 1/n_cur) is
    approximately chi-square with (bins - 1) degrees of freedom — PSI is a
    symmetrised KL divergence, and to second order that is Pearson's
    statistic. The approximation is checked empirically in benchmarks/.
    """
    dof = max(n_bins - 1, 1)
    scale = 1.0 / max(n_reference, 1) + 1.0 / max(n_current, 1)
    quantile = 1.0 - (1.0 - _NOISE_QUANTILE) / max(n_comparisons, 1)
    return float(stats.chi2.ppf(quantile, dof) * scale)


def _severity(psi: float, noise_floor: float = 0.0, out_of_range: float = 0.0) -> Severity:
    """
    Severity from PSI and from the out-of-range share, whichever is worse.

    PSI counts only if it also exceeds its noise floor. At a thousand rows a
    side the floor is about 0.03 and never binds; it matters for small batches.
    """
    by_psi: Severity = "stable"
    if psi > noise_floor:
        if psi >= PSI_SIGNIFICANT:
            by_psi = "significant"
        elif psi >= PSI_MODERATE:
            by_psi = "moderate"

    by_range: Severity = "stable"
    if out_of_range >= RANGE_SIGNIFICANT:
        by_range = "significant"
    elif out_of_range >= RANGE_MODERATE:
        by_range = "moderate"

    return max(by_psi, by_range, key=_ORDER.get)


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
    # Share of values that were null before they were dropped. PSI is computed
    # on what remains, so a null spike is invisible to it; these make it visible.
    missing_reference: float = 0.0
    missing_current: float = 0.0
    # The PSI that sampling noise alone would produce at these sample sizes.
    psi_noise_floor: float = 0.0
    # Share of current values beyond the reference's extremes (its outermost
    # 0.1% at each end, or an unseen category). An end is counted only when
    # it holds more than chance would produce.
    out_of_range: float = 0.0

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


def _side_p_value(m: int, n_current: int, n_reference: int, k: int = 1) -> float:
    """
    Chance that m or more current values lie beyond the reference's k-th most
    extreme value at one end.

    If both samples come from one distribution, every ordering of the pooled
    values is equally likely. "At least m current values exceed the
    reference's k-th largest" is the same event as "the top m + k - 1 pooled
    values contain at most k - 1 reference values", which is a hypergeometric
    probability. Exact, and free of any assumption about the distribution's
    shape. With k = 1 it is the chance that the m largest pooled values are
    all current: C(n_cur, m) / C(n_ref + n_cur, m).
    """
    if m <= 0:
        return 1.0
    total = n_reference + n_current
    draws = min(m + k - 1, total)
    return float(stats.hypergeom.cdf(k - 1, total, n_reference, draws))


def _extreme_rank(n_reference: int) -> int:
    """
    Which reference order statistic marks the edge of "extreme".

    The minimum and maximum are the natural edge for a small reference, but
    they move outward as the reference grows: 20 000 draws from a normal
    reach past four standard deviations, so values that far out stop counting
    as out of range at all. Using the k-th most extreme value, with k a fixed
    0.1% of the reference, keeps the edge at the same place in the
    distribution however large the reference is. Below 1 500 rows k is 1 and
    this is the plain minimum and maximum.
    """
    return max(1, int(round(EXTREME_TAIL * n_reference)))


def _out_of_range(
    ref: np.ndarray, cur: np.ndarray, n_comparisons: int = 1
) -> float:
    """
    Share of current values beyond the reference's extremes, counting only an
    end where there are more of them than chance explains.

    A fresh sample routinely pokes a little past the range of an earlier one:
    with equal sizes, three values beyond the old maximum happens one time in
    eight with nothing wrong. Each end is therefore tested separately (at
    0.005, so 0.01 across both, shared across the features being compared)
    and contributes only if it passes.
    """
    if cur.size == 0:
        return 0.0
    k = _extreme_rank(ref.size)
    ordered = np.sort(ref)
    low, high = ordered[k - 1], ordered[-k]
    counted = sum(
        m for m in (int((cur < low).sum()), int((cur > high).sum()))
        if _side_p_value(m, cur.size, ref.size, k) < 0.005 / max(n_comparisons, 1)
    )
    return counted / cur.size


def numeric_drift(
    feature: str,
    reference: Iterable[float],
    current: Iterable[float],
    bins: int = 10,
    n_comparisons: int = 1,
) -> FeatureDrift:
    """
    PSI over quantile bins + two-sample Kolmogorov-Smirnov test.

    Bin edges come from the *reference* sample only. This matters: edges must be
    frozen at training time, otherwise both distributions get re-binned together
    and the drift partly cancels itself out.
    """
    ref_raw = pd.Series(list(reference), dtype="float64")
    cur_raw = pd.Series(list(current), dtype="float64")
    missing_ref = float(ref_raw.isna().mean()) if len(ref_raw) else 0.0
    missing_cur = float(cur_raw.isna().mean()) if len(cur_raw) else 0.0
    ref = ref_raw.dropna().to_numpy()
    cur = cur_raw.dropna().to_numpy()

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
            missing_reference=missing_ref,
            missing_current=missing_cur,
        )

    # Open the outer edges so production values beyond the training range still
    # land in the first/last bin instead of being silently dropped.
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    psi = _psi_from_counts(ref_counts, cur_counts)
    floor = psi_noise_floor(ref.size, cur.size, edges.size - 1, n_comparisons)
    outside = _out_of_range(ref, cur, n_comparisons)

    ks = stats.ks_2samp(ref, cur)

    return FeatureDrift(
        feature=feature,
        kind="numeric",
        psi=psi,
        severity=_severity(psi, floor, outside),
        test_name="ks_2samp",
        statistic=float(ks.statistic),
        p_value=float(ks.pvalue),
        n_reference=int(ref.size),
        n_current=int(cur.size),
        missing_reference=missing_ref,
        missing_current=missing_cur,
        psi_noise_floor=floor,
        out_of_range=outside,
    )


def categorical_drift(
    feature: str,
    reference: Iterable,
    current: Iterable,
    n_comparisons: int = 1,
) -> FeatureDrift:
    """PSI over category shares + chi-square test on the contingency table."""
    ref_raw = pd.Series(list(reference))
    cur_raw = pd.Series(list(current))
    missing_ref = float(ref_raw.isna().mean()) if len(ref_raw) else 0.0
    missing_cur = float(cur_raw.isna().mean()) if len(cur_raw) else 0.0
    ref = ref_raw.dropna()
    cur = cur_raw.dropna()

    if ref.empty or cur.empty:
        raise ValueError(f"'{feature}': need non-empty reference and current samples")

    # Union of categories so unseen production categories are not dropped.
    categories = sorted(set(ref.unique()) | set(cur.unique()), key=str)
    ref_counts = ref.value_counts().reindex(categories, fill_value=0).to_numpy()
    cur_counts = cur.value_counts().reindex(categories, fill_value=0).to_numpy()

    psi = _psi_from_counts(ref_counts, cur_counts)
    floor = psi_noise_floor(ref.size, cur.size, len(categories), n_comparisons)
    # A category the reference never had cannot be explained by resampling the
    # reference, so the unseen share needs no chance correction.
    unseen = float(cur_counts[ref_counts == 0].sum() / cur.size)

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
        severity=_severity(psi, floor, unseen),
        test_name="chi2_contingency",
        statistic=float(chi2),
        p_value=float(p_value),
        n_reference=int(ref.size),
        n_current=int(cur.size),
        missing_reference=missing_ref,
        missing_current=missing_cur,
        psi_noise_floor=floor,
        out_of_range=unseen,
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

    Only shared columns are compared, and nulls are dropped before the
    statistics are computed. Neither a missing column nor a null spike shows up
    in PSI — `check_quality` reports those, and `ModelMonitor.check` runs both.

    The PSI noise floor is shared across the features compared, so that the
    chance of *any* unchanged feature being flagged stays near 5% however many
    there are. It only binds on small batches.
    """
    shared = [c for c in reference.columns if c in current.columns]
    if not shared:
        raise ValueError("reference and current share no columns")

    categorical = set(categorical or [])
    results = []

    for col in shared:
        is_cat = col in categorical or not pd.api.types.is_numeric_dtype(reference[col])
        if is_cat:
            results.append(
                categorical_drift(col, reference[col], current[col], n_comparisons=len(shared))
            )
        else:
            results.append(
                numeric_drift(
                    col, reference[col], current[col], bins=bins, n_comparisons=len(shared)
                )
            )

    frame = pd.DataFrame([r.to_dict() for r in results])
    return frame.sort_values("psi", ascending=False).reset_index(drop=True)
