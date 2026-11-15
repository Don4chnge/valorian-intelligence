"""
SQLite persistence for monitoring runs.

Deliberately boring: one file, no server, no ORM. A monitoring tool that needs
its own infrastructure to stand up will not get used.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name  TEXT NOT NULL,
    batch_label TEXT,
    created_at  TEXT NOT NULL,
    n_features  INTEGER,
    n_drifted   INTEGER,
    max_psi     REAL,
    status      TEXT,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS feature_drift (
    run_id     INTEGER NOT NULL,
    feature    TEXT NOT NULL,
    kind       TEXT,
    psi        REAL,
    severity   TEXT,
    test_name  TEXT,
    statistic  REAL,
    p_value    REAL,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);

CREATE TABLE IF NOT EXISTS performance (
    run_id          INTEGER NOT NULL,
    metric          TEXT,
    baseline        REAL,
    current         REAL,
    relative_change REAL,
    degraded        INTEGER,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);

CREATE TABLE IF NOT EXISTS multivariate (
    run_id        INTEGER NOT NULL,
    auc           REAL,
    separability  TEXT,
    p_value       REAL,
    attribution   TEXT,
    contributions TEXT,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);

-- One row per prediction the model made, logged when it was made. Outcomes
-- arrive later, sometimes months later, and are joined on prediction_id.
CREATE TABLE IF NOT EXISTS predictions (
    model_name    TEXT NOT NULL,
    prediction_id TEXT NOT NULL,
    run_id        INTEGER,
    batch_label   TEXT,
    predicted_at  TEXT NOT NULL,
    score         REAL NOT NULL,
    cohort        TEXT,
    PRIMARY KEY (model_name, prediction_id)
);

CREATE TABLE IF NOT EXISTS outcomes (
    model_name    TEXT NOT NULL,
    prediction_id TEXT NOT NULL,
    outcome       REAL NOT NULL,
    observed_at   TEXT NOT NULL,
    PRIMARY KEY (model_name, prediction_id)
);

CREATE TABLE IF NOT EXISTS cohort_performance (
    run_id          INTEGER NOT NULL,
    cohort          TEXT NOT NULL,
    n               INTEGER,
    metric          TEXT,
    current         REAL,
    ci_low          REAL,
    ci_high         REAL,
    baseline        REAL,
    relative_change REAL,
    status          TEXT,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_batch ON predictions (model_name, batch_label);
CREATE INDEX IF NOT EXISTS idx_feature_drift_run ON feature_drift (run_id);
CREATE INDEX IF NOT EXISTS idx_runs_model ON runs (model_name, created_at);
"""

# Columns added after 0.1.0. CREATE TABLE IF NOT EXISTS leaves an existing
# table untouched, so a database written by an older version would be missing
# these; they are added in place on open. Additive only — old rows keep NULL,
# and nothing is ever dropped or rewritten.
_ADDED_COLUMNS = {
    "runs": {
        "data_quality": "TEXT",
        "input_shift": "TEXT",
        "performance_status": "TEXT",
        "quality_issues": "TEXT",
    },
    "feature_drift": {
        "missing_reference": "REAL",
        "missing_current": "REAL",
        "psi_noise_floor": "REAL",
        "out_of_range": "REAL",
    },
    "performance": {
        "n": "INTEGER",
        "evaluable": "INTEGER",
        "reason": "TEXT",
        "ci_low": "REAL",
        "ci_high": "REAL",
        "confidence": "REAL",
        "label_coverage": "REAL",
        "median_label_delay_days": "REAL",
    },
}


def _timestamp(value) -> str:
    """Normalise a datetime, ISO string or None (meaning now) to UTC ISO-8601."""
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.isoformat(timespec="seconds")


def _nullable(value) -> float | None:
    """NaN has no meaning in a SQL REAL column; store it as NULL."""
    if value is None:
        return None
    value = float(value)
    return None if value != value else value


class MonitoringStore:
    def __init__(self, path: str | Path = "valorian.db"):
        self.path = str(path)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            for table, columns in _ADDED_COLUMNS.items():
                present = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
                for name, sql_type in columns.items():
                    if name not in present:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save_run(
        self,
        model_name: str,
        drift_frame: pd.DataFrame,
        batch_label: str | None = None,
        performance: dict | None = None,
        status: str = "ok",
        notes: str | None = None,
        health: dict | None = None,
        quality_issues: list[str] | None = None,
        multivariate: dict | None = None,
        cohorts: pd.DataFrame | None = None,
    ) -> int:
        """Persist one monitoring run and return its run_id."""
        n_drifted = int(drift_frame["drifted"].sum()) if len(drift_frame) else 0
        max_psi = float(drift_frame["psi"].max()) if len(drift_frame) else 0.0

        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO runs
                   (model_name, batch_label, created_at, n_features,
                    n_drifted, max_psi, status, notes,
                    data_quality, input_shift, performance_status, quality_issues)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    model_name,
                    batch_label,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    len(drift_frame),
                    n_drifted,
                    max_psi,
                    status,
                    notes,
                    (health or {}).get("data_quality"),
                    (health or {}).get("input_shift"),
                    (health or {}).get("performance"),
                    json.dumps(quality_issues) if quality_issues is not None else None,
                ),
            )
            run_id = int(cur.lastrowid)

            conn.executemany(
                """INSERT INTO feature_drift
                   (run_id, feature, kind, psi, severity, test_name, statistic,
                    p_value, missing_reference, missing_current,
                    psi_noise_floor, out_of_range)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        run_id,
                        row["feature"],
                        row["kind"],
                        float(row["psi"]),
                        row["severity"],
                        row["test_name"],
                        float(row["statistic"]),
                        _nullable(row["p_value"]),
                        _nullable(row.get("missing_reference")),
                        _nullable(row.get("missing_current")),
                        _nullable(row.get("psi_noise_floor")),
                        _nullable(row.get("out_of_range")),
                    )
                    for _, row in drift_frame.iterrows()
                ],
            )

            if performance:
                conn.execute(
                    """INSERT INTO performance
                       (run_id, metric, baseline, current, relative_change,
                        degraded, n, evaluable, reason, ci_low, ci_high, confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        performance["metric"],
                        performance["baseline"],
                        _nullable(performance["current"]),
                        _nullable(performance["relative_change"]),
                        int(performance["degraded"]),
                        performance.get("n"),
                        int(performance.get("evaluable", True)),
                        performance.get("reason"),
                        _nullable(performance.get("ci_low")),
                        _nullable(performance.get("ci_high")),
                        _nullable(performance.get("confidence")),
                    ),
                )

            if cohorts is not None and len(cohorts):
                self._write_cohorts(conn, run_id, cohorts)

            if multivariate:
                conn.execute(
                    """INSERT INTO multivariate
                       (run_id, auc, separability, p_value, attribution, contributions)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        multivariate["auc"],
                        multivariate["separability"],
                        _nullable(multivariate.get("p_value")),
                        multivariate.get("attribution"),
                        json.dumps(multivariate.get("contributions", {})),
                    ),
                )

        return run_id

    @staticmethod
    def _write_cohorts(conn, run_id: int, cohorts: pd.DataFrame) -> None:
        conn.execute("DELETE FROM cohort_performance WHERE run_id = ?", (run_id,))
        conn.executemany(
            """INSERT INTO cohort_performance
               (run_id, cohort, n, metric, current, ci_low, ci_high,
                baseline, relative_change, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    run_id, r["cohort"], int(r["n"]), r["metric"],
                    _nullable(r["current"]), _nullable(r["ci_low"]), _nullable(r["ci_high"]),
                    _nullable(r["baseline"]), _nullable(r["relative_change"]), r["status"],
                )
                for _, r in cohorts.iterrows()
            ],
        )

    # ------------------------------------------------------------------
    # Predictions now, outcomes later
    # ------------------------------------------------------------------

    def log_predictions(
        self,
        model_name: str,
        prediction_ids,
        scores,
        batch_label: str | None = None,
        run_id: int | None = None,
        cohorts=None,
        predicted_at=None,
    ) -> dict:
        """
        Record the scores a model produced, keyed by an ID outcomes can be
        joined to later.

        Safe to call twice with the same batch: an ID already on file is left
        as it was first recorded and counted under `duplicates`, so a retried
        job cannot double-count or rewrite history.
        """
        ids = [str(i) for i in prediction_ids]
        scores = [float(s) for s in scores]
        if len(ids) != len(scores):
            raise ValueError(f"{len(ids)} prediction_ids but {len(scores)} scores")
        if len(set(ids)) != len(ids):
            raise ValueError("prediction_ids must be unique within a batch")
        if cohorts is None:
            cohort_values = [None] * len(ids)
        else:
            cohort_values = [None if pd.isna(c) else str(c) for c in cohorts]
            if len(cohort_values) != len(ids):
                raise ValueError(f"{len(ids)} prediction_ids but {len(cohort_values)} cohorts")

        stamp = _timestamp(predicted_at)
        with self._connect() as conn:
            before = conn.total_changes
            conn.executemany(
                """INSERT OR IGNORE INTO predictions
                   (model_name, prediction_id, run_id, batch_label, predicted_at, score, cohort)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (model_name, i, run_id, batch_label, stamp, s, c)
                    for i, s, c in zip(ids, scores, cohort_values)
                ],
            )
            inserted = conn.total_changes - before
        return {"logged": inserted, "duplicates": len(ids) - inserted}

    def log_outcomes(
        self,
        model_name: str,
        prediction_ids,
        outcomes,
        observed_at=None,
        overwrite: bool = False,
    ) -> dict:
        """
        Record what actually happened for predictions logged earlier.

        Returns counts rather than raising, because a late-arriving outcomes
        file is rarely perfectly clean and one bad row should not lose the
        rest:

            recorded     new outcomes stored
            duplicates   already on file with the same value; ignored
            conflicts    already on file with a DIFFERENT value; the original
                         is kept unless `overwrite=True`
            unknown      no prediction with that ID was ever logged; not stored
        """
        ids = [str(i) for i in prediction_ids]
        values = [float(v) for v in outcomes]
        if len(ids) != len(values):
            raise ValueError(f"{len(ids)} prediction_ids but {len(values)} outcomes")
        if any(v != v for v in values):
            raise ValueError("outcomes contain NaN; leave unlabelled predictions out instead")
        stamp = _timestamp(observed_at)

        with self._connect() as conn:
            known = {
                r["prediction_id"] for r in conn.execute(
                    "SELECT prediction_id FROM predictions WHERE model_name = ?", (model_name,)
                )
            }
            existing = {
                r["prediction_id"]: r["outcome"] for r in conn.execute(
                    "SELECT prediction_id, outcome FROM outcomes WHERE model_name = ?",
                    (model_name,),
                )
            }

            new, changed = {}, {}
            unknown, duplicates, conflicts = [], 0, []
            for i, v in zip(ids, values):
                if i not in known:
                    unknown.append(i)
                elif i in existing or i in new:
                    previous = existing.get(i, new.get(i))
                    if previous == v:
                        duplicates += 1
                    else:
                        conflicts.append(i)
                        if overwrite:
                            changed[i] = v
                else:
                    new[i] = v

            conn.executemany(
                "INSERT INTO outcomes (model_name, prediction_id, outcome, observed_at) "
                "VALUES (?, ?, ?, ?)",
                [(model_name, i, v, stamp) for i, v in new.items() if i not in changed],
            )
            conn.executemany(
                "INSERT OR REPLACE INTO outcomes (model_name, prediction_id, outcome, observed_at) "
                "VALUES (?, ?, ?, ?)",
                [(model_name, i, v, stamp) for i, v in changed.items()],
            )

        return {
            "recorded": len(new),
            "duplicates": duplicates,
            "conflicts": len(conflicts),
            "unknown": len(unknown),
            "overwritten": len(changed),
            "conflict_ids": conflicts[:20],
            "unknown_ids": unknown[:20],
        }

    def labelled_predictions(self, model_name: str, batch_label: str | None) -> pd.DataFrame:
        """Every prediction in a batch, with its outcome where one has arrived."""
        query = """
            SELECT p.prediction_id, p.run_id, p.predicted_at, p.score, p.cohort,
                   o.outcome, o.observed_at
            FROM predictions p
            LEFT JOIN outcomes o
              ON o.model_name = p.model_name AND o.prediction_id = p.prediction_id
            WHERE p.model_name = ? AND p.batch_label IS ?
            ORDER BY p.rowid
        """
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=(model_name, batch_label))

    def update_performance(
        self,
        run_id: int,
        performance: dict,
        status: str,
        performance_status: str,
        coverage: float | None = None,
        median_delay_days: float | None = None,
        cohorts: pd.DataFrame | None = None,
    ) -> None:
        """
        Attach (or refresh) the performance finding on a run that was stored
        before its outcomes existed. Drift and quality rows are not touched.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM performance WHERE run_id = ?", (run_id,))
            conn.execute(
                """INSERT INTO performance
                   (run_id, metric, baseline, current, relative_change, degraded, n,
                    evaluable, reason, ci_low, ci_high, confidence,
                    label_coverage, median_label_delay_days)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, performance["metric"], performance["baseline"],
                    _nullable(performance["current"]), _nullable(performance["relative_change"]),
                    int(performance["degraded"]), performance.get("n"),
                    int(performance.get("evaluable", True)), performance.get("reason"),
                    _nullable(performance.get("ci_low")), _nullable(performance.get("ci_high")),
                    _nullable(performance.get("confidence")),
                    _nullable(coverage), _nullable(median_delay_days),
                ),
            )
            conn.execute(
                "UPDATE runs SET status = ?, performance_status = ? WHERE run_id = ?",
                (status, performance_status, run_id),
            )
            if cohorts is not None and len(cohorts):
                self._write_cohorts(conn, run_id, cohorts)

    def history(self, model_name: str | None = None, limit: int = 100) -> pd.DataFrame:
        """Run-level history, newest first."""
        query = "SELECT * FROM runs"
        params: tuple = ()
        if model_name:
            query += " WHERE model_name = ?"
            params = (model_name,)
        query += " ORDER BY run_id DESC LIMIT ?"
        params = params + (limit,)

        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def performance_history(self, model_name: str | None = None) -> pd.DataFrame:
        """Every run's performance row, oldest first — the decay curve with its intervals."""
        query = """
            SELECT r.run_id, r.batch_label, p.*
            FROM runs r
            JOIN performance p ON p.run_id = r.run_id
        """
        params: tuple = ()
        if model_name:
            query += " WHERE r.model_name = ?"
            params = (model_name,)
        query += " ORDER BY r.run_id"
        with self._connect() as conn:
            frame = pd.read_sql_query(query, conn, params=params)
        return frame.loc[:, ~frame.columns.duplicated()]

    def feature_history(self, feature: str, model_name: str | None = None) -> pd.DataFrame:
        """PSI trajectory for one feature — the view you want when plotting."""
        query = """
            SELECT r.run_id, r.created_at, r.batch_label, f.psi, f.severity
            FROM feature_drift f
            JOIN runs r ON r.run_id = f.run_id
            WHERE f.feature = ?
        """
        params: tuple = (feature,)
        if model_name:
            query += " AND r.model_name = ?"
            params += (model_name,)
        query += " ORDER BY r.run_id"

        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def run_detail(self, run_id: int) -> dict:
        with self._connect() as conn:
            run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if run is None:
                raise KeyError(f"no run with id {run_id}")
            features = pd.read_sql_query(
                "SELECT * FROM feature_drift WHERE run_id = ? ORDER BY psi DESC",
                conn,
                params=(run_id,),
            )
            perf = pd.read_sql_query(
                "SELECT * FROM performance WHERE run_id = ?", conn, params=(run_id,)
            )
            mv = conn.execute(
                "SELECT * FROM multivariate WHERE run_id = ?", (run_id,)
            ).fetchone()
            cohorts = pd.read_sql_query(
                "SELECT * FROM cohort_performance WHERE run_id = ? ORDER BY cohort",
                conn, params=(run_id,),
            )

        multivariate = None
        if mv is not None:
            multivariate = dict(mv)
            multivariate.pop("run_id")
            multivariate["contributions"] = json.loads(multivariate["contributions"] or "{}")

        return {
            "run": dict(run),
            "features": features,
            "performance": perf,
            "multivariate": multivariate,
            "cohorts": cohorts,
        }

    def to_json(self, run_id: int) -> str:
        detail = self.run_detail(run_id)
        return json.dumps(
            {
                "run": detail["run"],
                "features": detail["features"].to_dict(orient="records"),
                "performance": detail["performance"].to_dict(orient="records"),
                "multivariate": detail["multivariate"],
            },
            indent=2,
        )
