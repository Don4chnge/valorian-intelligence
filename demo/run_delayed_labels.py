"""
Delayed outcomes: the same credit model as run_demo.py, but honest about when
the labels show up.

run_demo.py hands the monitor every outcome at scoring time, which never
happens with loans. Here the August batch is scored with no outcomes at all,
and they arrive the way they really do: a default is known when it happens, a
good loan only when it reaches maturity. The batch is evaluated twice — 75 days
in, when only early defaults are known, and after maturity.

The August batch has the concept drift from run_demo.py planted in it, so the
right final answer is "degraded". The point is what the monitor says before
it can know that.

Run:  python demo/run_delayed_labels.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from valorian import ModelMonitor, score  # noqa: E402
from run_demo import FEATURES, encode, make_batch  # noqa: E402

MATURITY_DAYS = 180


def main() -> None:
    reference = make_batch(n=6_000)
    X_ref = encode(reference)
    scaler = StandardScaler().fit(X_ref)
    model = LogisticRegression(max_iter=1_000).fit(scaler.transform(X_ref), reference["default"])
    ref_scores = model.predict_proba(scaler.transform(X_ref))[:, 1]
    baseline = score(reference["default"], ref_scores)

    # Each province's own baseline, measured the same way as the overall one.
    cohort_baselines = {
        province: score(chunk["default"], ref_scores[chunk.index])
        for province, chunk in reference.groupby("province")
    }

    db = Path(tempfile.mkdtemp()) / "delayed.db"
    monitor = ModelMonitor(
        model_name="credit-default-v1",
        reference=reference[FEATURES],
        categorical=["province"],
        baseline_score=baseline,
        cohort_baselines=cohort_baselines,
        db_path=db,
    )

    batch = make_batch(n=3_000, weaken_income_signal=True).reset_index(drop=True)
    proba = model.predict_proba(scaler.transform(encode(batch, X_ref.columns)))[:, 1]
    ids = [f"app-{i:05d}" for i in range(len(batch))]

    report = monitor.check(
        batch[FEATURES], batch_label="2026-08",
        y_pred_proba=proba, prediction_ids=ids, cohorts=batch["province"],
    )
    print("AT SCORING TIME — no outcomes exist")
    print("=" * 62)
    print(report.summary())

    # A default surfaces some time before maturity; a good loan is only known
    # to be good once it gets there.
    rng = np.random.default_rng(8)
    defaulted = batch["default"].to_numpy() == 1
    known_on_day = np.where(
        defaulted, rng.uniform(20, MATURITY_DAYS, len(batch)), MATURITY_DAYS
    )
    scored_on = pd.Timestamp.now(tz="UTC")

    for day in (75, 200):
        arrived = known_on_day <= day
        counts = monitor.record_outcomes(
            np.array(ids)[arrived],
            batch["default"].to_numpy()[arrived],
            observed_at=scored_on + pd.Timedelta(days=day),
        )
        print(f"\nDAY {day} — outcomes recorded: {counts['recorded']}, "
              f"already on file: {counts['duplicates']}")
        print("=" * 62)
        print(monitor.evaluate("2026-08").summary())


if __name__ == "__main__":
    main()
