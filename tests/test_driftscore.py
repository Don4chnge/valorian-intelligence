"""
Tests for DriftScore.

The tests that matter here are the ordering ones. A composite score is easy to
make look reasonable and hard to make behave correctly, and the failure mode
is always the same: some combination of inputs produces a number that ranks a
healthy model below a broken one. The demo series is used as a fixture because
its ground truth is known by construction.
"""

from __future__ import annotations

import pytest

from valorian.driftscore import (
    BAND_HEALTHY,
    band_for,
    compute_driftscore,
)

# (label, max_psi, n_drifted, relative_change) from the demo run.
CLEAN_1 = ("2026-03", 0.008002, 0, +0.026)
CLEAN_2 = ("2026-04", 0.008593, 0, -0.014)
CLEAN_3 = ("2026-05", 0.005928, 0, -0.002)
INPUT_AND_PERF = ("2026-06", 0.270367, 1, -0.075)
INPUT_ONLY = ("2026-07", 0.898686, 2, +0.001)
CONCEPT_DRIFT = ("2026-08", 0.011387, 0, -0.113)

N_FEATURES = 5


def s(case) -> float:
    _, psi, n, rel = case
    return compute_driftscore(psi, n, N_FEATURES, rel).score


# --------------------------------------------------------------------------
# Boundary behaviour
# --------------------------------------------------------------------------

def test_perfect_model_scores_100():
    assert compute_driftscore(0.0, 0, 5, 0.0).score == 100.0


def test_total_collapse_scores_0():
    result = compute_driftscore(1.0, 5, 5, -0.90)
    assert result.score == 0.0
    assert result.band == "critical"


def test_score_never_leaves_range():
    for psi, n, rel in [(0, 0, 5.0), (99, 5, -99), (0.3, 2, 0.5), (0, 0, -0.0001)]:
        assert 0.0 <= compute_driftscore(psi, n, 5, rel).score <= 100.0


def test_improvement_is_not_rewarded_above_baseline():
    """A lucky batch must not push the score up and mask input drift."""
    held = compute_driftscore(0.30, 1, 5, 0.0).score
    improved = compute_driftscore(0.30, 1, 5, +0.50).score
    assert improved == held


# --------------------------------------------------------------------------
# Ordering — the tests that actually constrain the design
# --------------------------------------------------------------------------

def test_clean_months_are_healthy():
    for case in (CLEAN_1, CLEAN_2, CLEAN_3):
        assert s(case) >= BAND_HEALTHY, f"{case[0]} should be healthy, got {s(case)}"


def test_broken_model_scores_below_healthy_model_with_worse_inputs():
    """
    The central claim. 2026-07 has the worst input drift in the series and a
    model performing at baseline; 2026-08 has no input drift and a model that
    lost 11% of its power. The broken one must score lower.
    """
    assert s(CONCEPT_DRIFT) < s(INPUT_ONLY)


def test_concept_drift_month_is_actionable():
    result = compute_driftscore(*CONCEPT_DRIFT[1:3], N_FEATURES, CONCEPT_DRIFT[3])
    assert result.actionable
    assert result.band in ("degraded", "critical")


def test_input_only_month_is_not_treated_as_a_crisis():
    """Severe input drift with a healthy model warrants watching, not alarm."""
    result = compute_driftscore(*INPUT_ONLY[1:3], N_FEATURES, INPUT_ONLY[3])
    assert not result.actionable
    assert result.band == "watch"


def test_clean_months_outrank_every_drifted_month():
    worst_clean = min(s(c) for c in (CLEAN_1, CLEAN_2, CLEAN_3))
    best_drifted = max(s(c) for c in (INPUT_AND_PERF, INPUT_ONLY, CONCEPT_DRIFT))
    assert worst_clean > best_drifted


def test_score_is_monotonic_in_performance_drop():
    scores = [compute_driftscore(0.05, 0, 5, -d).score for d in (0.0, 0.05, 0.10, 0.20)]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores), "each additional drop should move the score"


def test_score_is_monotonic_in_psi():
    scores = [compute_driftscore(p, 0, 5, 0.0).score for p in (0.0, 0.1, 0.25, 0.5)]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------
# Saturation
# --------------------------------------------------------------------------

def test_performance_saturates_at_floor():
    """Past a 20% drop the model is unusable; further loss adds no information."""
    assert compute_driftscore(0.0, 0, 5, -0.20).score == compute_driftscore(0.0, 0, 5, -0.80).score


def test_psi_saturates_at_floor():
    assert compute_driftscore(0.50, 0, 5, 0.0).score == compute_driftscore(5.0, 0, 5, 0.0).score


# --------------------------------------------------------------------------
# The labels-arrive-late path
# --------------------------------------------------------------------------

def test_inputs_only_mode_when_no_labels():
    result = compute_driftscore(0.30, 1, 5, None)
    assert result.mode == "inputs_only"
    assert result.performance_component is None
    assert 0.0 <= result.score <= 100.0


def test_inputs_only_still_scores_100_on_clean_data():
    assert compute_driftscore(0.0, 0, 5, None).score == 100.0


def test_inputs_only_is_harsher_than_full_mode_on_same_drift():
    """
    Without labels, input drift carries the whole score rather than 30% of it,
    so the same drift reads worse. That is intended — it is a different
    question, and the mode flag exists so the two are not compared blindly.
    """
    drifted = compute_driftscore(0.40, 3, 5, None).score
    with_healthy_perf = compute_driftscore(0.40, 3, 5, 0.0).score
    assert drifted < with_healthy_perf


# --------------------------------------------------------------------------
# Bands and reporting
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "score,expected",
    [(100, "healthy"), (85, "healthy"), (84.9, "watch"), (70, "watch"),
     (69.9, "degraded"), (50, "degraded"), (49.9, "critical"), (0, "critical")],
)
def test_band_cutoffs(score, expected):
    assert band_for(score) == expected


def test_zero_features_does_not_divide_by_zero():
    result = compute_driftscore(0.0, 0, 0, 0.0)
    assert result.score == 100.0


def test_str_mentions_mode_when_labels_missing():
    assert "inputs only" in str(compute_driftscore(0.1, 0, 5, None))
    assert "inputs only" not in str(compute_driftscore(0.1, 0, 5, 0.0))


def test_to_dict_is_serialisable():
    import json
    json.dumps(compute_driftscore(0.2, 1, 5, -0.05).to_dict())
