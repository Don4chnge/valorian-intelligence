"""
Validation on real data: South African Quarterly Labour Force Survey.

Everything in demo/run_demo.py is synthetic. The drift was injected on purpose,
which proves the detector catches faults it was designed to catch and nothing
more. DriftScore's weights were then tuned against that same series, which is
circular.

This script breaks the circle. It trains an employment classifier on one
quarter of real QLFS microdata and monitors it across the following five,
with no drift injected anywhere. Whatever Valorian reports is whatever
actually happened in the South African labour market.

There is one thing worth watching for. Stats SA collected the QLFS by
telephone from 2020 Q2 through 2021 Q4 because of COVID, then reverted to
face-to-face interviewing from 2022 Q1. The reference quarter here is the last
telephone quarter and the first monitored quarter is the first face-to-face
one. A change in how a survey is administered can move response distributions
even when the underlying population has not changed. If Valorian flags 2022 Q1,
that is a real methodology break being detected without anyone pointing at it.
If it does not, that is also a finding: the change did not move the features
being monitored.

Data (not in the repo — see data/README.md):
    isibaloweb.statssa.gov.za/pages/surveys/pss/qlfs/qlfsp.php

Run:  python demo/run_qlfs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from valorian import ModelMonitor, score  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"

# Reference quarter first, then the batches to monitor in order.
REFERENCE = ("2021Q4", "QLFS202104.csv")
BATCHES = [
    ("2022Q1", "QLFS202201.csv"),
    ("2022Q2", "QLFS202202.csv"),
    ("2022Q3", "QLFS202203.csv"),
    ("2022Q4", "QLFS202204.csv"),
    ("2023Q1", "QLFS202301.csv"),
]

FEATURES = ["Q13GENDER", "Q14AGE", "Education_Status", "Province", "Geo_type_code"]
CATEGORICAL = ["Q13GENDER", "Education_Status", "Province", "Geo_type_code"]
TARGET = "Status"

# Status: 1 employed, 2 unemployed, 3 discouraged work-seeker,
# 4 other not economically active. Collapsed to employed vs not.
EMPLOYED = 1.0


def load(filename: str) -> pd.DataFrame:
    """
    Load one quarter, restricted to working-age respondents with a known
    labour market status.

    Under-15s carry no status at all, and `sector1` is deliberately excluded
    from FEATURES: it is only recorded for people who already have a job, so
    including it would leak the target into the features.
    """
    path = DATA / filename
    if not path.exists():
        raise SystemExit(
            f"Missing {path}.\nDownload the QLFS CSVs into data/ — see data/README.md"
        )

    frame = pd.read_csv(path, usecols=FEATURES + [TARGET], low_memory=False)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame.loc[frame[TARGET] > 10, TARGET] = np.nan
    frame = frame[
        frame["Q14AGE"].between(15, 64) & frame[TARGET].notna()
    ].reset_index(drop=True)

    frame["employed"] = (frame[TARGET] == EMPLOYED).astype(int)
    return frame


def encode(frame: pd.DataFrame, columns=None) -> pd.DataFrame:
    """One-hot the categorical codes, aligned to the training columns."""
    X = pd.get_dummies(frame[FEATURES], columns=CATEGORICAL, dtype=float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def main() -> None:
    db = Path(__file__).resolve().parents[1] / "qlfs.db"
    if db.exists():
        db.unlink()

    label, filename = REFERENCE
    reference = load(filename)
    X_ref = encode(reference)
    scaler = StandardScaler().fit(X_ref)

    model = LogisticRegression(max_iter=2_000).fit(
        scaler.transform(X_ref), reference["employed"]
    )
    baseline = score(
        reference["employed"],
        model.predict_proba(scaler.transform(X_ref))[:, 1],
    )

    print(f"Reference: {label} — {len(reference):,} respondents aged 15-64")
    print(f"Employment rate: {reference['employed'].mean():.1%}")
    print(f"Baseline ROC-AUC: {baseline:.4f}\n")

    monitor = ModelMonitor(
        model_name="qlfs-employment-v1",
        reference=reference[FEATURES],
        categorical=CATEGORICAL,
        baseline_score=baseline,
        metric="roc_auc",
        performance_threshold=0.05,
        db_path=db,
    )

    rows = []
    for label, filename in BATCHES:
        batch = load(filename)
        proba = model.predict_proba(
            scaler.transform(encode(batch, X_ref.columns))
        )[:, 1]

        report = monitor.check(
            batch[FEATURES],
            batch_label=label,
            y_true=batch["employed"],
            y_pred_proba=proba,
        )

        print("=" * 58)
        print(report.summary())
        print()

        ds = report.driftscore
        rows.append(
            {
                "quarter": label,
                "n": len(batch),
                "employed": f"{batch['employed'].mean():.1%}",
                "max_psi": round(report.drift["psi"].max(), 4),
                "drifted": len(report.drifted_features),
                "roc_auc": round(report.performance["current"], 4),
                "vs_base": f"{report.performance['relative_change']:+.1%}",
                "score": ds.score,
                "band": ds.band,
            }
        )

    print("=" * 58)
    print("\nSummary\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nRuns written to {db.name}. Dashboard: streamlit run app.py")


if __name__ == "__main__":
    main()
