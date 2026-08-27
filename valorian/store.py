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

CREATE INDEX IF NOT EXISTS idx_feature_drift_run ON feature_drift (run_id);
CREATE INDEX IF NOT EXISTS idx_runs_model ON runs (model_name, created_at);
"""


class MonitoringStore:
    def __init__(self, path: str | Path = "valorian.db"):
        self.path = str(path)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

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
    ) -> int:
        """Persist one monitoring run and return its run_id."""
        n_drifted = int(drift_frame["drifted"].sum()) if len(drift_frame) else 0
        max_psi = float(drift_frame["psi"].max()) if len(drift_frame) else 0.0

        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO runs
                   (model_name, batch_label, created_at, n_features,
                    n_drifted, max_psi, status, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    model_name,
                    batch_label,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    len(drift_frame),
                    n_drifted,
                    max_psi,
                    status,
                    notes,
                ),
            )
            run_id = int(cur.lastrowid)

            conn.executemany(
                """INSERT INTO feature_drift
                   (run_id, feature, kind, psi, severity, test_name, statistic, p_value)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        run_id,
                        row["feature"],
                        row["kind"],
                        float(row["psi"]),
                        row["severity"],
                        row["test_name"],
                        float(row["statistic"]),
                        float(row["p_value"]),
                    )
                    for _, row in drift_frame.iterrows()
                ],
            )

            if performance:
                conn.execute(
                    """INSERT INTO performance
                       (run_id, metric, baseline, current, relative_change, degraded)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        performance["metric"],
                        performance["baseline"],
                        performance["current"],
                        performance["relative_change"],
                        int(performance["degraded"]),
                    ),
                )

        return run_id

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

        return {"run": dict(run), "features": features, "performance": perf}

    def to_json(self, run_id: int) -> str:
        detail = self.run_detail(run_id)
        return json.dumps(
            {
                "run": detail["run"],
                "features": detail["features"].to_dict(orient="records"),
                "performance": detail["performance"].to_dict(orient="records"),
            },
            indent=2,
        )
