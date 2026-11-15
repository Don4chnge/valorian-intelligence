"""
Tests for the two blind spots the benchmark found in threshold-only PSI.

Both were invisible to the original suite because every fixture used thousands
of rows and shifts inside the reference range. The benchmark in benchmarks/
varies sample size and shift shape, and showed that at 200 rows a side the
0.10 threshold fired on about half of unchanged batches, and that 5% of rows
moving far beyond the reference range left PSI under 0.10 at every size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from valorian import ModelMonitor, MonitoringStore, categorical_drift, numeric_drift
from valorian.drift import PSI_MODERATE, psi_noise_floor


def rate(flags) -> float:
    return float(np.mean(list(flags)))


# --------------------------------------------------------------------------
# Small samples: PSI has a noise floor
# --------------------------------------------------------------------------

def test_raw_psi_threshold_misfires_on_small_unchanged_samples():
    """The problem, stated as a test so nobody removes the fix as redundant."""
    psis = [
        numeric_drift("x", r.normal(0, 1, 200), r.normal(0, 1, 200)).psi
        for r in (np.random.default_rng(s) for s in range(100))
    ]
    assert rate(p >= PSI_MODERATE for p in psis) > 0.25


@pytest.mark.parametrize("n", [100, 200, 500, 2_000])
def test_unchanged_samples_are_rarely_flagged_at_any_size(n):
    flagged = [
        numeric_drift("x", r.normal(0, 1, n), r.normal(0, 1, n)).drifted
        for r in (np.random.default_rng(s) for s in range(150))
    ]
    assert rate(flagged) <= 0.10


@pytest.mark.parametrize("n", [100, 300, 2_000])
def test_unchanged_categories_are_rarely_flagged_at_any_size(n):
    levels, p = list("abcde"), [0.4, 0.25, 0.15, 0.12, 0.08]
    flagged = [
        categorical_drift("x", r.choice(levels, n, p=p), r.choice(levels, n, p=p)).drifted
        for r in (np.random.default_rng(s) for s in range(150))
    ]
    assert rate(flagged) <= 0.10


def test_large_shift_is_still_caught_on_a_small_sample():
    flagged = [
        numeric_drift("x", r.normal(0, 1, 200), r.normal(1.0, 1, 200)).drifted
        for r in (np.random.default_rng(s) for s in range(50))
    ]
    assert rate(flagged) >= 0.95


def test_noise_floor_shrinks_with_sample_size_and_grows_with_bins():
    assert psi_noise_floor(200, 200, 10) > psi_noise_floor(2_000, 2_000, 10)
    assert psi_noise_floor(500, 500, 20) > psi_noise_floor(500, 500, 5)


def test_noise_floor_does_not_bind_at_ordinary_batch_sizes():
    """At a thousand rows a side the convention is untouched."""
    assert psi_noise_floor(1_000, 1_000, 10) < PSI_MODERATE
    rng = np.random.default_rng(0)
    result = numeric_drift("x", rng.normal(0, 1, 3_000), rng.normal(0.4, 1, 3_000))
    assert result.psi > PSI_MODERATE > result.psi_noise_floor
    assert result.drifted


def test_noise_floor_is_reported_on_the_result():
    rng = np.random.default_rng(1)
    result = numeric_drift("x", rng.normal(0, 1, 200), rng.normal(0, 1, 200))
    assert result.psi_noise_floor == pytest.approx(psi_noise_floor(200, 200, 10))


# --------------------------------------------------------------------------
# Values beyond the reference range
# --------------------------------------------------------------------------

def tail_shifted(rng, n, share=0.05):
    x = rng.normal(0, 1, n)
    x[: int(share * n)] = rng.normal(6, 0.5, int(share * n))
    return x


def test_values_far_outside_the_range_barely_move_psi_but_are_flagged():
    rng = np.random.default_rng(2)
    result = numeric_drift("x", rng.normal(0, 1, 5_000), tail_shifted(rng, 5_000))
    assert result.psi < PSI_MODERATE            # the blind spot
    assert result.out_of_range == pytest.approx(0.05, abs=0.01)
    assert result.severity == "moderate"


def test_large_out_of_range_share_is_significant():
    rng = np.random.default_rng(3)
    result = numeric_drift("x", rng.normal(0, 1, 5_000), tail_shifted(rng, 5_000, share=0.15))
    assert result.severity == "significant"


def test_chance_excursions_beyond_a_small_reference_are_not_reported():
    """A new draw beats the range of 100 old ones about 2% of the time anyway."""
    shares = [
        numeric_drift("x", r.normal(0, 1, 100), r.normal(0, 1, 100)).out_of_range
        for r in (np.random.default_rng(s) for s in range(200))
    ]
    assert rate(s > 0 for s in shares) <= 0.05


def test_unseen_category_share_is_measured():
    ref = pd.Series(["a"] * 600 + ["b"] * 400)
    cur = pd.Series(["a"] * 570 + ["b"] * 380 + ["new"] * 50)
    result = categorical_drift("x", ref, cur)
    assert result.out_of_range == pytest.approx(0.05)
    assert result.drifted


def test_out_of_range_is_named_in_the_alert_and_persisted(tmp_path):
    rng = np.random.default_rng(4)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 5_000)})
    cur = pd.DataFrame({"x": tail_shifted(rng, 5_000, share=0.15)})
    db = tmp_path / "t.db"
    report = ModelMonitor("m", ref, db_path=db).check(cur)

    assert report.health["input_shift"] == "significant"
    assert any("outside the reference range" in a for a in report.alerts)
    stored = MonitoringStore(db).run_detail(report.run_id)["features"].iloc[0]
    assert stored["out_of_range"] == pytest.approx(0.15, abs=0.02)
    assert stored["psi_noise_floor"] > 0


# --------------------------------------------------------------------------
# Many features at once
# --------------------------------------------------------------------------

def test_twenty_unchanged_features_rarely_raise_any_alarm_on_a_small_batch():
    """Per-feature 5% false alarms compound: 20 features would fire most of the time."""
    from valorian import detect_drift

    any_flagged = []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        ref = pd.DataFrame({f"f{i}": rng.normal(0, 1, 200) for i in range(20)})
        cur = pd.DataFrame({f"f{i}": rng.normal(0, 1, 200) for i in range(20)})
        any_flagged.append(detect_drift(ref, cur)["drifted"].any())
    assert rate(any_flagged) <= 0.15


def test_noise_floor_rises_with_the_number_of_features_compared():
    assert psi_noise_floor(200, 200, 10, n_comparisons=20) > psi_noise_floor(200, 200, 10)


# --------------------------------------------------------------------------
# The out-of-range check on a large reference
# --------------------------------------------------------------------------

def test_out_of_range_check_does_not_weaken_as_the_reference_grows():
    """
    Min/max widen with sample size: 20 000 normal draws reach past 4 sd, so
    values at 4 sd stopped counting and detection fell to 80% in the
    benchmark. The edge is now the reference's outermost 0.1%.
    """
    def tail(rng, n):
        x = rng.normal(0, 1, n)
        x[: int(0.05 * n)] = rng.normal(4, 1, int(0.05 * n))
        return x

    for n in (1_000, 20_000):
        flagged = [
            numeric_drift("x", r.normal(0, 1, n), tail(r, n)).drifted
            for r in (np.random.default_rng(s) for s in range(40))
        ]
        assert rate(flagged) >= 0.95, n


def test_large_unchanged_samples_report_nothing_out_of_range():
    shares = [
        numeric_drift("x", r.normal(0, 1, 20_000), r.normal(0, 1, 20_000)).out_of_range
        for r in (np.random.default_rng(s) for s in range(60))
    ]
    assert rate(s > 0 for s in shares) <= 0.05
    assert max(shares) < 0.02


def test_heavily_tied_data_is_not_flagged():
    """Integer-coded features pile up on their extremes; that is not a shift."""
    flagged = [
        numeric_drift("age", r.integers(15, 65, 5_000), r.integers(15, 65, 5_000)).drifted
        for r in (np.random.default_rng(s) for s in range(40))
    ]
    assert rate(flagged) == 0.0
