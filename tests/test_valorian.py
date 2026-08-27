"""
Tests for Valorian Intelligence.

The important tests here are the ones that plant a known fault and assert we
catch it, and — just as important — the ones that assert we *don't* fire on
clean data. A drift detector that alarms on everything is useless.

Run:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from valorian import (
    ModelMonitor,
    MonitoringStore,
    categorical_drift,
    compare_performance,
    detect_drift,
    numeric_drift,
    score,
)

RNG = np.random.default_rng(7)


@pytest.fixture
def reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "income": RNG.lognormal(10, 0.5, 3_000),
            "age": RNG.normal(40, 10, 3_000),
            "region": RNG.choice(["north", "south", "east"], 3_000, p=[0.5, 0.3, 0.2]),
        }
    )


# --------------------------------------------------------------------------
# Numeric drift
# --------------------------------------------------------------------------

def test_identical_distributions_are_stable():
    sample = RNG.normal(0, 1, 5_000)
    result = numeric_drift("x", sample, sample)
    assert result.psi < 0.01
    assert result.severity == "stable"
    assert not result.drifted


def test_same_distribution_different_draws_stays_stable():
    result = numeric_drift("x", RNG.normal(0, 1, 5_000), RNG.normal(0, 1, 5_000))
    assert result.severity == "stable", f"false positive at PSI {result.psi}"


def test_shifted_mean_is_flagged():
    result = numeric_drift("x", RNG.normal(0, 1, 5_000), RNG.normal(1.5, 1, 5_000))
    assert result.severity == "significant"
    assert result.p_value < 0.01
    assert result.drifted


def test_psi_increases_with_shift_magnitude():
    ref = RNG.normal(0, 1, 5_000)
    psis = [numeric_drift("x", ref, RNG.normal(s, 1, 5_000)).psi for s in (0.2, 0.6, 1.2)]
    assert psis == sorted(psis), "PSI should grow monotonically with the shift"


def test_values_outside_reference_range_are_not_dropped():
    """Production values beyond the training range must land in the edge bins."""
    ref = RNG.uniform(0, 1, 3_000)
    cur = RNG.uniform(5, 6, 3_000)  # entirely outside reference support
    result = numeric_drift("x", ref, cur)
    assert result.severity == "significant"
    assert result.n_current == 3_000


def test_constant_reference_feature_does_not_crash():
    result = numeric_drift("x", np.ones(500), np.ones(500))
    assert result.severity == "stable"

    moved = numeric_drift("x", np.ones(500), np.full(500, 9.0))
    assert moved.severity == "significant"


def test_nans_are_dropped_not_fatal():
    ref = np.concatenate([RNG.normal(0, 1, 1_000), [np.nan] * 50])
    result = numeric_drift("x", ref, RNG.normal(0, 1, 1_000))
    assert result.n_reference == 1_000


def test_empty_sample_raises():
    with pytest.raises(ValueError):
        numeric_drift("x", [], [1, 2, 3])


# --------------------------------------------------------------------------
# Categorical drift
# --------------------------------------------------------------------------

def test_stable_categories():
    ref = RNG.choice(["a", "b", "c"], 3_000, p=[0.5, 0.3, 0.2])
    cur = RNG.choice(["a", "b", "c"], 3_000, p=[0.5, 0.3, 0.2])
    assert categorical_drift("cat", ref, cur).severity == "stable"


def test_category_mix_shift_is_flagged():
    ref = RNG.choice(["a", "b", "c"], 3_000, p=[0.7, 0.2, 0.1])
    cur = RNG.choice(["a", "b", "c"], 3_000, p=[0.2, 0.3, 0.5])
    result = categorical_drift("cat", ref, cur)
    assert result.severity == "significant"
    assert result.p_value < 0.01


def test_unseen_category_is_counted():
    """A brand new category in production is exactly what we must not miss."""
    ref = RNG.choice(["a", "b"], 2_000)
    cur = np.concatenate([RNG.choice(["a", "b"], 1_000), ["z"] * 1_000])
    result = categorical_drift("cat", ref, cur)
    assert result.psi > 0.25
    assert result.n_current == 2_000


# --------------------------------------------------------------------------
# Frame-level detection
# --------------------------------------------------------------------------

def test_detect_drift_returns_row_per_feature(reference):
    result = detect_drift(reference, reference.copy(), categorical=["region"])
    assert len(result) == 3
    assert set(result["feature"]) == {"income", "age", "region"}
    assert not result["drifted"].any()


def test_detect_drift_sorted_by_psi_descending(reference):
    current = reference.copy()
    current["income"] = current["income"] * 3.0
    result = detect_drift(reference, current, categorical=["region"])
    assert result.iloc[0]["feature"] == "income"
    assert list(result["psi"]) == sorted(result["psi"], reverse=True)


def test_object_columns_treated_as_categorical_automatically(reference):
    result = detect_drift(reference, reference.copy())
    assert result.loc[result["feature"] == "region", "kind"].item() == "categorical"


def test_no_shared_columns_raises(reference):
    with pytest.raises(ValueError):
        detect_drift(reference, pd.DataFrame({"unrelated": [1, 2, 3]}))


# --------------------------------------------------------------------------
# Performance / concept drift
# --------------------------------------------------------------------------

def test_stable_performance_not_degraded():
    y = RNG.binomial(1, 0.4, 1_000)
    p = np.where(y == 1, RNG.uniform(0.5, 1, 1_000), RNG.uniform(0, 0.5, 1_000))
    baseline = score(y, p)
    assert not compare_performance(baseline, y, p).degraded


def test_performance_drop_is_flagged():
    y = RNG.binomial(1, 0.4, 1_000)
    good = np.where(y == 1, RNG.uniform(0.5, 1, 1_000), RNG.uniform(0, 0.5, 1_000))
    noise = RNG.uniform(0, 1, 1_000)
    result = compare_performance(score(y, good), y, noise)
    assert result.degraded
    assert result.relative_change < 0


def test_lower_is_better_metric_direction():
    """log_loss going *up* is degradation, not improvement."""
    y = RNG.binomial(1, 0.5, 1_000)
    bad = np.where(y == 1, 0.1, 0.9)  # confidently wrong
    result = compare_performance(0.35, y, bad, metric="log_loss")
    assert result.degraded


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        score([0, 1], [0.2, 0.8], metric="not_a_metric")


# --------------------------------------------------------------------------
# Monitor + store integration
# --------------------------------------------------------------------------

def test_monitor_end_to_end_flags_planted_drift(reference, tmp_path):
    monitor = ModelMonitor(
        "test-model",
        reference,
        categorical=["region"],
        db_path=tmp_path / "t.db",
    )

    clean = monitor.check(reference.sample(1_500, random_state=1), batch_label="clean")
    assert clean.status == "ok"
    assert clean.alerts == []

    drifted = reference.sample(1_500, random_state=2).copy()
    drifted["income"] *= 4.0
    dirty = monitor.check(drifted, batch_label="dirty")
    assert dirty.status == "critical"
    assert "income" in dirty.drifted_features
    assert dirty.alerts


def test_monitor_persists_and_reloads(reference, tmp_path):
    db = tmp_path / "t.db"
    monitor = ModelMonitor("test-model", reference, categorical=["region"], db_path=db)
    monitor.check(reference.copy(), batch_label="b1")
    monitor.check(reference.copy(), batch_label="b2")

    store = MonitoringStore(db)
    history = store.history("test-model")
    assert len(history) == 2
    assert set(history["batch_label"]) == {"b1", "b2"}

    trace = store.feature_history("income", "test-model")
    assert len(trace) == 2


def test_performance_without_baseline_raises(reference, tmp_path):
    monitor = ModelMonitor("m", reference, db_path=tmp_path / "t.db")
    with pytest.raises(ValueError):
        monitor.check(
            reference.copy(),
            y_true=[0, 1],
            y_pred_proba=[0.2, 0.8],
        )


def test_report_summary_is_printable(reference, tmp_path):
    monitor = ModelMonitor("m", reference, categorical=["region"], db_path=tmp_path / "t.db")
    text = monitor.check(reference.copy(), batch_label="x").summary()
    assert "Status:" in text
    assert "m" in text
