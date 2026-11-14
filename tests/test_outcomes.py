"""
Tests for outcomes that arrive after the predictions, and for cohorts.

The rule under test throughout is the same one as in test_health.py: the
monitor must not report a verdict it has not earned. With delayed labels that
has a new form — the outcomes that arrive first are not a fair sample of the
batch, so a metric computed on them is reported but not judged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from valorian import ModelMonitor, MonitoringStore, cohort_performance, score

RNG = np.random.default_rng(21)
N = 2_000


def scored(n=N, strength=0.35, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else RNG
    y = rng.binomial(1, 0.4, n)
    p = np.clip(strength * y + rng.uniform(0, 1 - strength, n), 0, 1)
    return y, p


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.DataFrame({"a": RNG.normal(0, 1, N), "b": RNG.normal(0, 1, N)})


@pytest.fixture
def setup(features, tmp_path):
    y, p = scored(seed=1)
    monitor = ModelMonitor("m", features, baseline_score=score(y, p), db_path=tmp_path / "t.db")
    ids = [f"p{i}" for i in range(N)]
    return monitor, ids, y, p


# --------------------------------------------------------------------------
# Predictions first, outcomes later
# --------------------------------------------------------------------------

def test_predictions_without_outcomes_are_not_observed(setup, features):
    monitor, ids, _, p = setup
    report = monitor.check(features, batch_label="jan", y_pred_proba=p, prediction_ids=ids)
    assert report.health["performance"] == "not_observed"
    assert report.status == "ok"


def test_evaluating_before_any_outcome_is_insufficient_evidence(setup, features):
    monitor, ids, _, p = setup
    monitor.check(features, batch_label="jan", y_pred_proba=p, prediction_ids=ids)
    result = monitor.evaluate("jan")
    assert result.n_labelled == 0 and result.coverage == 0.0
    assert result.performance["evaluable"] is False
    assert "no outcomes" in result.performance["reason"]
    assert result.status == "warning"


def test_partial_outcomes_are_reported_but_not_judged(setup, features):
    monitor, ids, y, p = setup
    monitor.check(features, batch_label="jan", y_pred_proba=p, prediction_ids=ids)
    monitor.record_outcomes(ids[:800], y[:800])            # 40% of the batch

    result = monitor.evaluate("jan")
    assert result.coverage == pytest.approx(0.4)
    assert result.performance["evaluable"] is False
    assert not np.isnan(result.performance["current"])     # the number is still shown
    assert "40%" in result.performance["reason"]
    assert "so far" in result.summary()
    assert result.status == "warning"


def test_full_outcomes_are_judged_and_the_stored_run_is_updated(setup, features):
    monitor, ids, y, p = setup
    report = monitor.check(features, batch_label="jan", y_pred_proba=p, prediction_ids=ids)
    monitor.record_outcomes(ids, y)

    result = monitor.evaluate("jan")
    assert result.coverage == 1.0
    assert result.performance["status"] == "within_tolerance"
    assert result.run_id == report.run_id

    store = MonitoringStore(monitor.store.path)
    run = store.run_detail(report.run_id)
    assert run["run"]["performance_status"] == "within_tolerance"
    assert run["run"]["status"] == "ok"
    assert run["performance"].iloc[0]["label_coverage"] == 1.0
    assert len(run["features"]) == 2                       # drift rows untouched


def test_run_turns_critical_when_outcomes_show_the_model_was_wrong(setup, features):
    monitor, ids, _, _ = setup
    y, noise = scored(seed=2)[0], np.random.default_rng(3).uniform(0, 1, N)
    report = monitor.check(features, batch_label="feb", y_pred_proba=noise, prediction_ids=ids)
    assert report.status == "ok"                           # nothing to see at scoring time

    monitor.record_outcomes(ids, y)
    result = monitor.evaluate("feb")
    assert result.performance["status"] == "degraded"
    assert result.status == "critical"
    assert result.alerts
    assert monitor.store.history("m").iloc[0]["status"] == "critical"


def test_evaluate_can_be_repeated_as_outcomes_accumulate(setup, features):
    monitor, ids, y, p = setup
    report = monitor.check(features, batch_label="jan", y_pred_proba=p, prediction_ids=ids)
    for upto in (500, 1_500, 2_000):
        monitor.record_outcomes(ids[:upto], y[:upto])
        result = monitor.evaluate("jan")
    assert result.performance["evaluable"] is True
    assert len(monitor.store.run_detail(report.run_id)["performance"]) == 1


def test_label_delay_is_measured(features, tmp_path):
    y, p = scored(seed=4)
    monitor = ModelMonitor("m", features, baseline_score=score(y, p), db_path=tmp_path / "t.db")
    ids = [f"p{i}" for i in range(N)]
    monitor.store.log_predictions("m", ids, p, batch_label="jan", predicted_at="2026-01-15")
    monitor.record_outcomes(ids[:1_000], y[:1_000], observed_at="2026-03-16")
    monitor.record_outcomes(ids[1_000:], y[1_000:], observed_at="2026-04-15")

    result = monitor.evaluate("jan")
    assert result.median_delay_days == pytest.approx(75, abs=1)
    assert result.run_id is None and result.status is None   # logged without a run


def test_early_outcomes_are_a_biased_sample_and_are_not_judged(setup, features):
    """
    The reason coverage is enforced. The first outcomes in are the failures the
    model ranked most confidently, plus a few of everything else. The metric on
    them flatters the model by a wide margin.
    """
    monitor, ids, y, p = setup
    monitor.check(features, batch_label="jan", y_pred_proba=p, prediction_ids=ids)

    ids_arr = np.array(ids)
    positives = y == 1
    easy = positives & (p > np.median(p[positives]))
    some_negatives = (~positives) & (np.random.default_rng(5).uniform(size=N) < 0.25)
    early = easy | some_negatives
    monitor.record_outcomes(ids_arr[early], y[early])

    result = monitor.evaluate("jan")
    assert result.performance["current"] > score(y, p) + 0.05   # looks much better than it is
    assert result.performance["evaluable"] is False
    assert result.performance["status"] == "insufficient_evidence"


def test_coverage_floor_is_configurable(features, tmp_path):
    y, p = scored(seed=6)
    monitor = ModelMonitor("m", features, baseline_score=score(y, p),
                           db_path=tmp_path / "t.db", min_label_coverage=0.3)
    ids = [f"p{i}" for i in range(N)]
    monitor.check(features, batch_label="jan", y_pred_proba=p, prediction_ids=ids)
    monitor.record_outcomes(ids[:800], y[:800])
    assert monitor.evaluate("jan").performance["evaluable"] is True


def test_labels_supplied_with_ids_are_stored_as_outcomes(setup, features):
    monitor, ids, y, p = setup
    report = monitor.check(features, batch_label="jan", y_true=y, y_pred_proba=p, prediction_ids=ids)
    assert report.health["performance"] == "within_tolerance"
    assert monitor.evaluate("jan").coverage == 1.0


# --------------------------------------------------------------------------
# Re-sent, conflicting and unknown records
# --------------------------------------------------------------------------

def test_relogging_predictions_does_not_duplicate_or_rewrite(setup):
    monitor, ids, _, p = setup
    first = monitor.store.log_predictions("m", ids, p, batch_label="jan")
    again = monitor.store.log_predictions("m", ids, np.zeros(N), batch_label="jan")
    assert first == {"logged": N, "duplicates": 0}
    assert again == {"logged": 0, "duplicates": N}
    stored = monitor.store.labelled_predictions("m", "jan")
    assert len(stored) == N
    assert stored["score"].to_numpy() == pytest.approx(p)     # first write stands


def test_resending_outcomes_is_harmless(setup):
    monitor, ids, y, p = setup
    monitor.store.log_predictions("m", ids, p, batch_label="jan")
    assert monitor.record_outcomes(ids, y)["recorded"] == N
    again = monitor.record_outcomes(ids, y)
    assert again["recorded"] == 0 and again["duplicates"] == N and again["conflicts"] == 0


def test_conflicting_outcome_keeps_the_original_unless_overwritten(setup):
    monitor, ids, y, p = setup
    monitor.store.log_predictions("m", ids, p, batch_label="jan")
    monitor.record_outcomes(ids, y)

    flipped = 1 - y[:5]
    result = monitor.record_outcomes(ids[:5], flipped)
    assert result["conflicts"] == 5 and result["overwritten"] == 0
    assert result["conflict_ids"] == ids[:5]
    kept = monitor.store.labelled_predictions("m", "jan")["outcome"].to_numpy()[:5]
    assert kept == pytest.approx(y[:5])

    result = monitor.record_outcomes(ids[:5], flipped, overwrite=True)
    assert result["overwritten"] == 5
    changed = monitor.store.labelled_predictions("m", "jan")["outcome"].to_numpy()[:5]
    assert changed == pytest.approx(flipped)


def test_outcomes_for_unknown_predictions_are_reported_not_stored(setup):
    monitor, ids, y, p = setup
    monitor.store.log_predictions("m", ids, p, batch_label="jan")
    result = monitor.record_outcomes(["ghost-1", "ghost-2", ids[0]], [1, 0, y[0]])
    assert result["unknown"] == 2 and result["recorded"] == 1
    assert result["unknown_ids"] == ["ghost-1", "ghost-2"]
    assert monitor.store.labelled_predictions("m", "jan")["outcome"].notna().sum() == 1


def test_outcomes_do_not_leak_between_models(setup, features, tmp_path):
    monitor, ids, y, p = setup
    monitor.store.log_predictions("m", ids, p, batch_label="jan")
    other = ModelMonitor("other", features, baseline_score=0.7, db_path=monitor.store.path)
    assert other.record_outcomes(ids, y)["unknown"] == N


def test_bad_inputs_raise_clear_errors(setup, features):
    monitor, ids, y, p = setup
    with pytest.raises(ValueError, match="unique"):
        monitor.store.log_predictions("m", ["a", "a"], [0.1, 0.2])
    with pytest.raises(ValueError, match="scores"):
        monitor.store.log_predictions("m", ["a", "b"], [0.1])
    monitor.store.log_predictions("m", ids, p, batch_label="jan")
    with pytest.raises(ValueError, match="NaN"):
        monitor.record_outcomes(ids[:2], [1.0, float("nan")])
    with pytest.raises(KeyError):
        monitor.evaluate("no-such-batch")
    with pytest.raises(ValueError, match="y_pred_proba"):
        monitor.check(features, prediction_ids=ids)
    with pytest.raises(ValueError, match="store"):
        ModelMonitor("m", features, db_path=None).check(
            features, y_pred_proba=p, prediction_ids=ids
        )


# --------------------------------------------------------------------------
# Cohorts
# --------------------------------------------------------------------------

def two_cohorts(bad_strength=0.35, seed=7, n_each=1_500):
    """Cohort 'north' behaves; cohort 'south' has its own signal strength."""
    y1, p1 = scored(n_each, 0.35, seed)
    y2, p2 = scored(n_each, bad_strength, seed + 1)
    return (
        np.concatenate([y1, y2]),
        np.concatenate([p1, p2]),
        np.array(["north"] * n_each + ["south"] * n_each),
    )


def test_cohort_without_a_baseline_is_measured_but_not_judged():
    y, p, c = two_cohorts()
    table = cohort_performance(y, p, c)
    assert table["cohort"].tolist() == ["north", "south"]
    assert (table["status"] == "no_baseline").all()
    assert table["current"].notna().all() and table["ci_low"].notna().all()
    assert table["relative_change"].isna().all()


def test_cohort_that_got_worse_is_flagged_against_its_own_baseline():
    y, p, c = two_cohorts(bad_strength=0.05)
    healthy = score(*scored(20_000, 0.35, 99))
    table = cohort_performance(y, p, c, baselines={"north": healthy, "south": healthy})
    status = table.set_index("cohort")["status"]
    assert status["north"] == "within_tolerance"
    assert status["south"] == "degraded"


def test_cohort_that_was_always_weaker_is_not_flagged():
    """Judged against the overall baseline, 'south' would be degraded forever."""
    y, p, c = two_cohorts(bad_strength=0.15)
    strong, weak = score(*scored(20_000, 0.35, 99)), score(*scored(20_000, 0.15, 98))
    assert weak < strong * 0.95                              # it IS >5% below overall

    table = cohort_performance(y, p, c, baselines={"north": strong, "south": weak})
    assert (table["status"] == "within_tolerance").all()


def test_small_cohort_is_insufficient_evidence():
    y, p, c = two_cohorts()
    c = c.astype(object)
    c[:30] = "tiny"
    table = cohort_performance(y, p, c, baselines={"tiny": 0.7}).set_index("cohort")
    assert table.loc["tiny", "status"] == "insufficient_evidence"
    assert np.isnan(table.loc["tiny", "current"])


def test_missing_cohort_labels_get_their_own_row():
    y, p, c = two_cohorts()
    c = c.astype(object)
    c[:200] = None
    assert "(missing)" in cohort_performance(y, p, c)["cohort"].tolist()


def test_overall_holds_while_one_cohort_degrades(tmp_path):
    """The case cohorts exist for: the average hides it."""
    y, p, c = two_cohorts(bad_strength=0.0, n_each=3_000)
    keep = np.r_[0:3_000, 3_000:3_300]                       # 'south' is 9% of rows
    y, p, c = y[keep], p[keep], c[keep]
    features = pd.DataFrame({"a": RNG.normal(0, 1, len(y))})
    healthy = score(*scored(20_000, 0.35, 99))

    monitor = ModelMonitor(
        "m", features, baseline_score=score(y, p) * 1.02,
        cohort_baselines={"north": healthy, "south": healthy}, db_path=tmp_path / "t.db",
    )
    report = monitor.check(features, y_true=y, y_pred_proba=p, cohorts=c)

    assert report.health["performance"] == "within_tolerance"
    assert report.cohorts.set_index("cohort").loc["south", "status"] == "degraded"
    assert any("Overall performance within tolerance, but cohort 'south'" in a
               for a in report.alerts)

    stored = MonitoringStore(tmp_path / "t.db").run_detail(report.run_id)["cohorts"]
    assert stored.set_index("cohort").loc["south", "status"] == "degraded"


def test_cohorts_logged_with_predictions_are_used_when_outcomes_arrive(tmp_path):
    y, p, c = two_cohorts(bad_strength=0.05)
    features = pd.DataFrame({"a": RNG.normal(0, 1, len(y))})
    healthy = score(*scored(20_000, 0.35, 99))
    monitor = ModelMonitor(
        "m", features, baseline_score=healthy,
        cohort_baselines={"north": healthy, "south": healthy}, db_path=tmp_path / "t.db",
    )
    ids = np.arange(len(y))
    monitor.check(features, batch_label="jan", y_pred_proba=p, prediction_ids=ids, cohorts=c)

    monitor.record_outcomes(ids[::2], y[::2])                # half the outcomes
    partial = monitor.evaluate("jan")
    assert set(partial.cohorts["status"]) == {"insufficient_evidence"}

    monitor.record_outcomes(ids, y)
    full = monitor.evaluate("jan").cohorts.set_index("cohort")["status"]
    assert full["north"] == "within_tolerance" and full["south"] == "degraded"


def test_report_without_cohorts_has_none(setup, features):
    monitor, _, y, p = setup
    assert monitor.check(features, y_true=y, y_pred_proba=p).cohorts is None
