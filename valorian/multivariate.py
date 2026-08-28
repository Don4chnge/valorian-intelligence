"""
Multivariate drift detection.

Every other detector in this package looks at one feature at a time. That
misses a whole class of drift: two features can each hold a perfectly stable
marginal distribution while the *relationship* between them changes. Income
and age might both look untouched, but if high earners used to be older and
now they are younger, the joint distribution has moved and a model that learned
the old relationship is in trouble. No univariate test can see this.

The technique here is the domain classifier, sometimes called a two-sample
classifier test. It reframes "are these two samples from the same
distribution?" as a supervised learning problem:

    1. Label every reference row 0 and every current row 1.
    2. Shuffle them together and train a classifier to tell them apart.
    3. Score it by cross-validated ROC-AUC.

If the samples are drawn from the same distribution, no classifier can beat
chance and the AUC lands near 0.50. The further above 0.50 it climbs, the more
separable the two batches are — and separability *is* drift, by definition.

The pleasant side effect is attribution. The trained classifier has feature
importances, and they say which columns it used to tell the batches apart.
That gives a ranked explanation of what moved, including combinations that no
single-feature test would have flagged.

Interpretation, by convention rather than derivation:

    AUC < 0.55    indistinguishable — no meaningful drift
    0.55 - 0.65   mild separability, worth watching
    0.65 - 0.80   clear drift
    > 0.80        the batches are trivially distinguishable

Two cautions worth carrying into any write-up:

The AUC is inflated by sample size in the same way a p-value is. With enough
rows a classifier will find *some* signal, so the permutation p-value below is
the honest check on whether the score exceeds what shuffled labels produce.

And a high AUC does not mean the monitored model is broken. It means the input
distribution moved. Whether that matters is a question for the performance
metrics, not this one.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

AUC_INDISTINGUISHABLE = 0.55
AUC_MILD = 0.65
AUC_CLEAR = 0.80

Separability = Literal["indistinguishable", "mild", "clear", "trivial"]


def separability_for(auc: float) -> Separability:
    if auc >= AUC_CLEAR:
        return "trivial"
    if auc >= AUC_MILD:
        return "clear"
    if auc >= AUC_INDISTINGUISHABLE:
        return "mild"
    return "indistinguishable"


@dataclass
class MultivariateDrift:
    auc: float
    separability: Separability
    p_value: float | None
    n_reference: int
    n_current: int
    n_features: int
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def drifted(self) -> bool:
        return self.separability != "indistinguishable"

    def top_contributors(self, n: int = 5) -> list[tuple[str, float]]:
        """Features the classifier leaned on most, highest first."""
        return sorted(self.contributions.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["drifted"] = self.drifted
        return d

    def __str__(self) -> str:
        head = f"Multivariate AUC {self.auc:.3f} — {self.separability}"
        if self.p_value is not None:
            head += f" (p = {self.p_value:.3f})"
        if not self.contributions:
            return head
        top = ", ".join(f"{k} {v:.2f}" for k, v in self.top_contributors(3))
        return f"{head}\nDriven by: {top}"


def _prepare(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    categorical: list[str] | None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Stack the two frames into one design matrix with a domain label."""
    shared = [c for c in reference.columns if c in current.columns]
    if not shared:
        raise ValueError("reference and current share no columns")

    ref = reference[shared].copy()
    cur = current[shared].copy()

    combined = pd.concat([ref, cur], ignore_index=True)

    # One-hot everything non-numeric. Categories are unioned across both
    # frames by construction here, so a category that appears only in the
    # current batch still gets a column — which is exactly the kind of shift
    # the classifier should be able to exploit.
    cat_cols = set(categorical or [])
    cat_cols |= {c for c in shared if not pd.api.types.is_numeric_dtype(combined[c])}

    if cat_cols:
        combined = pd.get_dummies(combined, columns=sorted(cat_cols), dtype=float)

    combined = combined.fillna(combined.median(numeric_only=True)).fillna(0.0)

    labels = np.concatenate([np.zeros(len(ref)), np.ones(len(cur))])
    return combined, labels, shared


def _fit_auc(X: pd.DataFrame, y: np.ndarray, seed: int, folds: int) -> float:
    """Cross-validated AUC for separating the two domains."""
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,          # shallow on purpose: an unconstrained forest
                              # memorises rows and reports drift that is not there
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=seed,
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def multivariate_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    categorical: list[str] | None = None,
    max_rows: int = 20_000,
    folds: int = 3,
    n_permutations: int = 0,
    random_state: int = 0,
) -> MultivariateDrift:
    """
    Detect drift in the joint distribution by trying to tell the batches apart.

    `max_rows` subsamples each side before fitting. Cross-validating a forest
    on 40 000 rows is slow and buys nothing — the AUC estimate is stable well
    before then.

    `n_permutations` optionally runs a permutation test: refit on shuffled
    domain labels and report the fraction of shuffles that match or beat the
    real AUC. That is the honest answer to "is 0.58 actually meaningful at this
    sample size?", and it costs one extra fit per permutation, so it is off by
    default.
    """
    rng = np.random.default_rng(random_state)

    ref = reference.sample(min(len(reference), max_rows), random_state=random_state) \
        if len(reference) > max_rows else reference
    cur = current.sample(min(len(current), max_rows), random_state=random_state) \
        if len(current) > max_rows else current

    if len(ref) < 2 * folds or len(cur) < 2 * folds:
        raise ValueError(
            f"need at least {2 * folds} rows per sample for {folds}-fold CV"
        )

    X, y, sources = _prepare(ref, cur, categorical)
    auc = _fit_auc(X, y, random_state, folds)

    p_value = None
    if n_permutations > 0:
        null = []
        for i in range(n_permutations):
            shuffled = rng.permutation(y)
            null.append(_fit_auc(X, shuffled, random_state + i + 1, folds))
        # +1 in numerator and denominator: the observed value is itself one
        # draw from the null under the exchangeability assumption, and this
        # keeps the p-value from ever being exactly zero.
        p_value = float((np.sum(np.array(null) >= auc) + 1) / (n_permutations + 1))

    contributions = _attribute(X, y, random_state, sources)

    return MultivariateDrift(
        auc=round(auc, 4),
        separability=separability_for(auc),
        p_value=None if p_value is None else round(p_value, 4),
        n_reference=int(len(ref)),
        n_current=int(len(cur)),
        n_features=int(X.shape[1]),
        contributions=contributions,
    )


def _fold_to_source(column: str, sources: list[str]) -> str:
    """
    Map an encoded column back to the feature it came from.

    get_dummies produces "province_Gauteng" from "province". Splitting on the
    first underscore would be wrong for a feature named "monthly_income", so
    the longest matching source name wins instead.
    """
    if column in sources:
        return column
    candidates = [s for s in sources if column.startswith(f"{s}_")]
    return max(candidates, key=len) if candidates else column


def _attribute(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    sources: list[str],
) -> dict[str, float]:
    """
    Which columns did the classifier use to separate the domains?

    Fitted on the full sample rather than cross-validated, because this is a
    ranking rather than a performance estimate. One-hot columns are folded back
    into their source feature so the output is interpretable at the level the
    user thinks in — "province drifted", not "province_Gauteng drifted".
    """
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=seed,
    ).fit(X, y)

    folded: dict[str, float] = {s: 0.0 for s in sources}
    for column, importance in zip(X.columns, model.feature_importances_):
        folded[_fold_to_source(column, sources)] += float(importance)

    total = sum(folded.values()) or 1.0
    return {k: round(v / total, 4) for k, v in sorted(
        folded.items(), key=lambda kv: kv[1], reverse=True
    )}
