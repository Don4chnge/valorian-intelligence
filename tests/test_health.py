"""
Tests for the things drift statistics cannot see, and for what a status means.

Two failure modes drive this file. The first is a monitor that reports health
it has not measured: a batch with too few labels, or labels of one class, must
come back as "insufficient evidence" and never as "within tolerance". The
second is a monitor that conflates questions: a null spike is not drift, and
drift is not a broken model. Each gets its own dimension and its own tests.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from valorian import (
    ModelMonitor,
    MonitoringStore,
    check_quality,
    compare_performance,
    compute_driftscore,
    detect_drift,
    score,
)
from valorian.driftscore import score_report

RNG = np.random.default_rng(11)


@pytest.fixture
def reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "income": RNG.lognormal(10, 0.5, 3_000),
            "age": RNG.normal(40, 10, 3_000),
            "region": RNG.choice(["north", "south", "east"], 3_000, p=[0.5, 0.3, 0.2]),
        }
    )


def labelled(n: int = 2_000):
    """Labels and scores from a model with real but imperfect signal."""
    y = RNG.binomial(1, 0.4, n)
    p = np.clip(0.35 * y + RNG.uniform(0, 0.65, n), 0, 1)
    return y, p


# --------------------------------------------------------------------------
# Insufficient evidence is never healthy
# --------------------------------------------------------------------------

def test_single_class_batch_is_insufficient_evidence_not_healthy():
    """The original bug: ROC-AUC is NaN, NaN < -0.05 is False, so 'not degraded'."""
    result = compare_performance(0.70, np.ones(500), RNG.uniform(0, 1, 500))
    assert not result.evaluable
    assert result.status == "insufficient_evidence"
    assert "single class" in result.reason


def test_tiny_batch_is_insufficient_evidence():
    y, p = labelled(40)
    result = compare_performance(0.70, y, p)
    assert result.status == "insufficient_evidence"
    assert result.n == 40


def test_rare_class_below_floor_is_insufficient_evidence():
    y = np.zeros(1_000, dtype=int)
    y[:4] = 1  # four positives cannot pin down an AUC
    result = compare_performance(0.70, y, RNG.uniform(0, 1, 1_000))
    assert result.status == "insufficient_evidence"
    assert "rarest class" in result.reason


def test_evidence_floors_are_configurable():
    y, p = labelled(40)
    result = compare_performance(score(y, p), y, p, min_samples=20, min_per_class=5)
    assert result.evaluable
    assert result.status == "within_tolerance"


def test_adequate_batch_is_still_evaluated():
    y, p = labelled()
    assert compare_performance(score(y, p), y, p).status == "within_tolerance"
    assert compare_performance(0.99, y, p).status == "degraded"


def test_unevaluable_performance_does_not_inflate_driftscore(reference, tmp_path):
    """NaN performance used to score 100 on the component carrying 70% of the weight."""
    monitor = ModelMonitor("m", reference, baseline_score=0.7, db_path=None)
    drifted = reference.copy()
    drifted["income"] *= 4.0

    report = monitor.check(
        drifted, y_true=np.ones(len(drifted)), y_pred_proba=RNG.uniform(0, 1, len(drifted))
    )
    inputs_only = monitor.check(drifted)

    assert report.driftscore.mode == "inputs_only"
    assert report.driftscore.performance_component is None
    assert report.driftscore.score == inputs_only.driftscore.score


def test_nan_relative_change_is_treated_as_no_labels():
    with_nan = compute_driftscore(0.4, 2, 5, float("nan"))
    without = compute_driftscore(0.4, 2, 5, None)
    assert with_nan.mode == "inputs_only"
    assert with_nan.score == without.score


def test_insufficient_evidence_is_never_status_ok(reference):
    monitor = ModelMonitor("m", reference, baseline_score=0.7, db_path=None)
    y, p = labelled(30)
    report = monitor.check(reference.sample(30, random_state=0), y_true=y, y_pred_proba=p)
    assert report.health["performance"] == "insufficient_evidence"
    assert report.status != "ok"
    assert any("not evaluated" in a for a in report.alerts)
    assert "not evaluated" in report.summary()


def test_mismatched_label_and_prediction_lengths_raise():
    with pytest.raises(ValueError, match="rows"):
        compare_performance(0.7, [0, 1, 0], [0.1, 0.9])


# --------------------------------------------------------------------------
# Regression metrics that were declared but never implemented
# --------------------------------------------------------------------------

def test_rmse_and_mae_are_supported():
    y = RNG.normal(100, 10, 500)
    good = y + RNG.normal(0, 1, 500)
    bad = y + RNG.normal(0, 5, 500)
    for metric in ("rmse", "mae"):
        baseline = score(y, good, metric=metric)
        assert compare_performance(baseline, y, good, metric=metric).status == "within_tolerance"
        assert compare_performance(baseline, y, bad, metric=metric).status == "degraded"


def test_regression_metrics_do_not_need_two_classes():
    y = np.full(500, 3.0)
    assert compare_performance(0.5, y, y + 0.1, metric="mae").evaluable


# --------------------------------------------------------------------------
# Data quality: what PSI cannot see
# --------------------------------------------------------------------------

def test_null_spike_is_invisible_to_psi_but_fails_quality(reference):
    batch = reference.copy()
    batch.loc[batch.sample(frac=0.6, random_state=1).index, "income"] = np.nan

    drift = detect_drift(reference, batch, categorical=["region"])
    income = drift.set_index("feature").loc["income"]
    assert income["severity"] == "stable"           # PSI sees nothing wrong
    assert income["missing_current"] == pytest.approx(0.6)
    assert income["missing_reference"] == 0.0

    quality = check_quality(reference, batch, categorical=["region"])
    assert quality.status == "failed"
    assert any("income" in i and "null rate" in i for i in quality.issues)


def test_moderate_null_rise_is_a_warning(reference):
    batch = reference.copy()
    batch.loc[batch.sample(frac=0.08, random_state=1).index, "age"] = np.nan
    assert check_quality(reference, batch).status == "warning"


def test_stable_null_rate_is_not_flagged(reference):
    """A feature that was always 30% null is not broken for staying 30% null."""
    ref = reference.copy()
    ref.loc[ref.sample(frac=0.3, random_state=1).index, "age"] = np.nan
    batch = reference.copy()
    batch.loc[batch.sample(frac=0.3, random_state=2).index, "age"] = np.nan
    assert check_quality(ref, batch).status == "healthy"


def test_clean_batch_is_healthy(reference):
    quality = check_quality(reference, reference.sample(1_000, random_state=3))
    assert quality.status == "healthy"
    assert quality.issues == []


def test_missing_column_is_reported_not_skipped(reference):
    quality = check_quality(reference, reference.drop(columns=["age"]))
    assert quality.status == "failed"
    assert quality.missing_columns == ["age"]


def test_extra_column_is_a_warning(reference):
    batch = reference.assign(surprise=1.0)
    quality = check_quality(reference, batch)
    assert quality.status == "warning"
    assert quality.extra_columns == ["surprise"]


def test_numeric_column_arriving_as_text_is_reported(reference):
    """The QLFS 2022 Q1 `Status` fault: a numeric column shipped as text."""
    batch = reference.copy()
    batch["income"] = batch["income"].astype(str)
    quality = check_quality(reference, batch, categorical=["region"])
    assert quality.status == "failed"
    assert "income" in quality.type_changed


def test_declared_categorical_stored_as_numbers_is_not_a_type_change():
    ref = pd.DataFrame({"province": RNG.integers(1, 10, 500)})
    cur = pd.DataFrame({"province": RNG.integers(1, 10, 500).astype(float)})
    assert check_quality(ref, cur, categorical=["province"]).status == "healthy"


def test_all_null_column_is_reported_once(reference):
    batch = reference.copy()
    batch["age"] = np.nan
    quality = check_quality(reference, batch)
    assert quality.empty_columns == ["age"]
    assert "age" not in quality.type_changed


def test_monitor_survives_broken_columns_and_reports_them(reference, tmp_path):
    """A wrong-typed or empty column used to crash the whole run."""
    monitor = ModelMonitor("m", reference, categorical=["region"], db_path=tmp_path / "t.db")
    batch = reference.copy()
    batch["income"] = batch["income"].astype(str)
    batch["age"] = np.nan

    report = monitor.check(batch.drop(columns=["region"]).assign(region=reference["region"]))
    assert report.health["data_quality"] == "failed"
    assert report.status == "critical"
    assert report.drift["feature"].tolist() == ["region"]
    assert sum("Data quality" in a for a in report.alerts) == 2


def test_monitor_with_every_shared_column_broken(reference):
    monitor = ModelMonitor("m", reference[["age"]], db_path=None)
    report = monitor.check(pd.DataFrame({"age": [np.nan] * 200}))
    assert report.status == "critical"
    assert len(report.drift) == 0
    assert report.driftscore.n_features == 0


def test_no_shared_columns_still_raises(reference):
    monitor = ModelMonitor("m", reference, db_path=None)
    with pytest.raises(ValueError):
        monitor.check(pd.DataFrame({"unrelated": [1, 2, 3]}))


# --------------------------------------------------------------------------
# Health dimensions and what `status` means
# --------------------------------------------------------------------------

def test_severe_input_drift_with_stable_performance_is_not_critical(reference):
    """The demo's 2026-07 case: inputs moved a long way, the model did not."""
    y, p = labelled(len(reference))
    monitor = ModelMonitor("m", reference, baseline_score=score(y, p), db_path=None)
    drifted = reference.copy()
    drifted["income"] *= 4.0

    report = monitor.check(drifted, y_true=y, y_pred_proba=p)
    assert report.health == {
        "data_quality": "healthy",
        "input_shift": "significant",
        "performance": "within_tolerance",
    }
    assert report.status == "warning"
    assert report.legacy_status == "critical"


def test_degraded_performance_with_stable_inputs_is_critical(reference):
    """The demo's 2026-08 case: nothing to see in the inputs, model is wrong."""
    y, p = labelled(len(reference))
    monitor = ModelMonitor("m", reference, baseline_score=0.99, db_path=None)
    report = monitor.check(reference.copy(), y_true=y, y_pred_proba=p)
    assert report.health["input_shift"] == "stable"
    assert report.health["performance"] == "degraded"
    assert report.status == "critical"


def test_no_labels_is_not_observed_rather_than_within_tolerance(reference):
    report = ModelMonitor("m", reference, db_path=None).check(reference.copy())
    assert report.health["performance"] == "not_observed"
    assert report.driftscore.mode == "inputs_only"
    assert "[inputs only]" in report.summary()


def test_summary_shows_every_dimension(reference):
    text = ModelMonitor("m", reference, db_path=None).check(reference.copy()).summary()
    for word in ("data quality", "input shift", "performance"):
        assert word in text


def test_score_report_ignores_performance_marked_unevaluable(reference):
    drift = detect_drift(reference, reference.copy())
    unevaluable = {"relative_change": 0.0, "degraded": False, "evaluable": False}
    assert score_report(drift, unevaluable).mode == "inputs_only"


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def test_health_and_quality_are_persisted(reference, tmp_path):
    db = tmp_path / "t.db"
    monitor = ModelMonitor("m", reference, categorical=["region"],
                           baseline_score=0.7, db_path=db)
    batch = reference.copy()
    batch.loc[batch.sample(frac=0.5, random_state=1).index, "income"] = np.nan
    y, p = labelled(30)
    report = monitor.check(batch.head(30), y_true=y, y_pred_proba=p)

    detail = MonitoringStore(db).run_detail(report.run_id)
    assert detail["run"]["data_quality"] == "failed"
    assert detail["run"]["performance_status"] == "insufficient_evidence"
    assert "income" in detail["run"]["quality_issues"]
    assert detail["performance"].iloc[0]["evaluable"] == 0
    assert detail["features"].set_index("feature").loc["income", "missing_current"] > 0.2
    MonitoringStore(db).to_json(report.run_id)  # NaN-free, so it serialises


def test_database_from_0_1_0_is_upgraded_in_place(reference, tmp_path):
    """Opening an old database must add the new columns and keep the old rows."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL, batch_label TEXT, created_at TEXT NOT NULL,
            n_features INTEGER, n_drifted INTEGER, max_psi REAL, status TEXT, notes TEXT);
        CREATE TABLE feature_drift (run_id INTEGER NOT NULL, feature TEXT NOT NULL,
            kind TEXT, psi REAL, severity TEXT, test_name TEXT, statistic REAL, p_value REAL);
        CREATE TABLE performance (run_id INTEGER NOT NULL, metric TEXT, baseline REAL,
            current REAL, relative_change REAL, degraded INTEGER);
        INSERT INTO runs (model_name, batch_label, created_at, status)
            VALUES ('m', 'old', '2026-08-28T08:42:25+00:00', 'critical');
        """
    )
    conn.commit()
    conn.close()

    ModelMonitor("m", reference, categorical=["region"], db_path=db).check(
        reference.copy(), batch_label="new"
    )
    history = MonitoringStore(db).history("m")
    assert set(history["batch_label"]) == {"old", "new"}
    old = history.set_index("batch_label").loc["old"]
    assert old["status"] == "critical"          # untouched
    assert old["data_quality"] is None or pd.isna(old["data_quality"])
