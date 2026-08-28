"""
Tests for multivariate drift detection.

The test that matters most is `test_catches_correlation_flip_that_psi_misses`.
It constructs two samples with near-identical marginal distributions and an
inverted correlation, asserts that the univariate detector calls every feature
stable, and asserts that the domain classifier catches it anyway. That single
case is the entire justification for this module — if it ever fails, the module
has no reason to exist.

The false-positive tests matter almost as much. A domain classifier will
happily overfit its way to a high AUC on identical samples if it is allowed to
memorise rows, which is why the forest is depth-limited and the AUC is
cross-validated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from valorian.drift import detect_drift
from valorian.multivariate import (
    AUC_INDISTINGUISHABLE,
    multivariate_drift,
    separability_for,
)

RNG = np.random.default_rng(11)
N = 3_000


def gaussian(n=N, loc=0.0, scale=1.0, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else RNG
    return pd.DataFrame({
        "a": rng.normal(loc, scale, n),
        "b": rng.normal(loc, scale, n),
        "c": rng.uniform(0, 1, n),
    })


# --------------------------------------------------------------------------
# The case this module exists for
# --------------------------------------------------------------------------

def test_catches_correlation_flip_that_psi_misses():
    """
    Two samples, matching marginals, opposite correlation. Univariate tests
    see nothing; the joint distribution has completely changed.
    """
    x1 = RNG.normal(0, 1, N)
    y1 = 0.9 * x1 + RNG.normal(0, 0.44, N)

    x2 = RNG.normal(0, 1, N)
    y2 = -0.9 * x2 + RNG.normal(0, 0.44, N)

    ref = pd.DataFrame({"x": x1, "y": y1})
    cur = pd.DataFrame({"x": x2, "y": y2})

    univariate = detect_drift(ref, cur)
    assert (univariate["severity"] == "stable").all(), (
        "premise broken: univariate tests were supposed to miss this"
    )

    result = multivariate_drift(ref, cur)
    assert result.drifted
    assert result.auc > 0.75, f"expected clear separability, got {result.auc}"


# --------------------------------------------------------------------------
# False positives — the failure mode that would make this useless
# --------------------------------------------------------------------------

def test_identical_distributions_are_indistinguishable():
    result = multivariate_drift(gaussian(seed=1), gaussian(seed=2))
    assert result.separability == "indistinguishable"
    assert not result.drifted
    assert result.auc < AUC_INDISTINGUISHABLE


def test_same_frame_against_itself_does_not_overfit():
    """
    Splitting one sample in half must not look like drift. An unconstrained
    forest would memorise rows and report a high AUC here.
    """
    frame = gaussian(n=4_000, seed=3)
    left, right = frame.iloc[:2_000], frame.iloc[2_000:]
    assert multivariate_drift(left, right).auc < AUC_INDISTINGUISHABLE


def test_small_samples_do_not_produce_spurious_drift():
    result = multivariate_drift(gaussian(n=200, seed=4), gaussian(n=200, seed=5))
    assert result.auc < 0.60


# --------------------------------------------------------------------------
# Genuine drift of various kinds
# --------------------------------------------------------------------------

def test_mean_shift_is_detected():
    result = multivariate_drift(gaussian(seed=6), gaussian(loc=1.2, seed=7))
    assert result.drifted
    assert result.auc > 0.70


def test_variance_shift_is_detected():
    result = multivariate_drift(gaussian(seed=8), gaussian(scale=3.0, seed=9))
    assert result.drifted


def test_auc_increases_with_shift_magnitude():
    ref = gaussian(seed=10)
    aucs = [multivariate_drift(ref, gaussian(loc=s, seed=20 + i)).auc
            for i, s in enumerate((0.0, 0.5, 1.5))]
    assert aucs == sorted(aucs)


def test_new_category_is_detected():
    ref = pd.DataFrame({
        "n": RNG.normal(0, 1, N),
        "cat": RNG.choice(["a", "b"], N),
    })
    cur = pd.DataFrame({
        "n": RNG.normal(0, 1, N),
        "cat": RNG.choice(["a", "b", "z"], N, p=[0.3, 0.3, 0.4]),
    })
    result = multivariate_drift(ref, cur, categorical=["cat"])
    assert result.drifted
    assert "cat" in dict(result.top_contributors(2))


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------

def test_attribution_points_at_the_drifted_feature():
    ref = gaussian(seed=30)
    cur = gaussian(seed=31)
    cur["a"] = cur["a"] + 2.5          # only 'a' moves

    result = multivariate_drift(ref, cur)
    top_feature, _ = result.top_contributors(1)[0]
    assert top_feature == "a"


def test_contributions_sum_to_one():
    result = multivariate_drift(gaussian(seed=32), gaussian(loc=0.8, seed=33))
    assert abs(sum(result.contributions.values()) - 1.0) < 0.01


def test_contributions_cover_every_feature():
    result = multivariate_drift(gaussian(seed=34), gaussian(seed=35))
    assert set(result.contributions) == {"a", "b", "c"}


# --------------------------------------------------------------------------
# Permutation test
# --------------------------------------------------------------------------

def test_permutation_p_value_is_high_when_nothing_drifted():
    result = multivariate_drift(
        gaussian(n=800, seed=40), gaussian(n=800, seed=41), n_permutations=5
    )
    assert result.p_value is not None
    assert result.p_value > 0.1


def test_permutation_p_value_is_low_for_real_drift():
    result = multivariate_drift(
        gaussian(n=800, seed=42), gaussian(n=800, loc=1.5, seed=43), n_permutations=5
    )
    assert result.p_value <= 0.17  # 1/6 is the floor with 5 permutations


def test_p_value_is_none_when_not_requested():
    assert multivariate_drift(gaussian(seed=44), gaussian(seed=45)).p_value is None


# --------------------------------------------------------------------------
# Mechanics
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "auc,expected",
    [(0.50, "indistinguishable"), (0.54, "indistinguishable"), (0.55, "mild"),
     (0.64, "mild"), (0.65, "clear"), (0.79, "clear"), (0.80, "trivial"), (1.0, "trivial")],
)
def test_separability_bands(auc, expected):
    assert separability_for(auc) == expected


def test_no_shared_columns_raises():
    with pytest.raises(ValueError):
        multivariate_drift(gaussian(), pd.DataFrame({"other": [1, 2, 3] * 100}))


def test_too_few_rows_raises():
    with pytest.raises(ValueError):
        multivariate_drift(gaussian(n=4), gaussian(n=4))


def test_subsampling_caps_the_fit():
    result = multivariate_drift(gaussian(n=5_000, seed=50), gaussian(n=5_000, seed=51),
                                max_rows=1_000)
    assert result.n_reference == 1_000
    assert result.n_current == 1_000


def test_nans_do_not_crash():
    ref = gaussian(seed=60)
    cur = gaussian(seed=61)
    cur.loc[cur.sample(300, random_state=1).index, "a"] = np.nan
    assert 0.0 <= multivariate_drift(ref, cur).auc <= 1.0


def test_result_is_serialisable():
    import json
    json.dumps(multivariate_drift(gaussian(seed=70), gaussian(seed=71)).to_dict())


def test_str_is_readable():
    text = str(multivariate_drift(gaussian(seed=72), gaussian(loc=1.0, seed=73)))
    assert "AUC" in text
    assert "Driven by" in text
