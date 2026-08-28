"""
End-to-end demo: a credit default model that quietly rots over six months.

We generate data where we *know* the ground truth — drift is injected on
purpose from month 4 onward — then check whether Valorian catches it. A
monitoring tool that has never been shown to catch a planted fault is not
evidence of anything.

What gets injected:
    months 1-3  clean, model should look healthy
    month 4     applicant income distribution shifts downward (data drift)
    month 5     income shift deepens + province mix changes (data drift)
    month 6     the income -> default relationship weakens (concept drift),
                inputs stay in range so *only* performance catches it

Run:  python demo/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from valorian import ModelMonitor, multivariate_drift, score  # noqa: E402

RNG = np.random.default_rng(42)
PROVINCES = ["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape", "Limpopo"]
FEATURES = ["monthly_income", "age", "months_employed", "debt_ratio", "province"]


def make_batch(
    n: int = 2_000,
    income_shift: float = 0.0,
    province_skew: bool = False,
    weaken_income_signal: bool = False,
) -> pd.DataFrame:
    """Generate one month of loan applications plus their default outcomes."""
    income = RNG.lognormal(mean=10.0 + income_shift, sigma=0.55, size=n)
    age = RNG.normal(37, 11, size=n).clip(18, 75)
    months_employed = RNG.gamma(shape=2.2, scale=22, size=n).clip(0, 400)
    debt_ratio = RNG.beta(2.4, 5.0, size=n)

    if province_skew:
        weights = [0.22, 0.14, 0.16, 0.26, 0.22]  # migration away from Gauteng
    else:
        weights = [0.42, 0.20, 0.18, 0.10, 0.10]
    province = RNG.choice(PROVINCES, size=n, p=weights)

    # True default relationship. Under concept drift the income coefficient
    # collapses — income stops protecting borrowers the way it used to.
    income_coef = -0.15 if weaken_income_signal else -0.85

    z = (
        1.10
        + income_coef * (np.log(income) - 10.0)
        + 2.30 * debt_ratio
        - 0.010 * (age - 37)
        - 0.0045 * months_employed
    )
    p_default = 1 / (1 + np.exp(-z))
    default = RNG.binomial(1, p_default)

    return pd.DataFrame(
        {
            "monthly_income": income,
            "age": age,
            "months_employed": months_employed,
            "debt_ratio": debt_ratio,
            "province": province,
            "default": default,
        }
    )


def encode(frame: pd.DataFrame, columns=None) -> pd.DataFrame:
    """One-hot the province column, aligned to the training columns."""
    X = pd.get_dummies(frame[FEATURES], columns=["province"], dtype=float)
    X["monthly_income"] = np.log(X["monthly_income"])
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def main() -> None:
    db = Path(__file__).resolve().parents[1] / "valorian.db"
    if db.exists():
        db.unlink()  # fresh run every time so the demo is reproducible

    # ---- Train the model on month 0 -------------------------------------
    reference = make_batch(n=6_000)
    X_ref = encode(reference)
    scaler = StandardScaler().fit(X_ref)
    model = LogisticRegression(max_iter=1_000).fit(scaler.transform(X_ref), reference["default"])

    baseline = score(reference["default"], model.predict_proba(scaler.transform(X_ref))[:, 1])
    print(f"Trained on 6,000 reference applications. Baseline ROC-AUC: {baseline:.4f}\n")

    monitor = ModelMonitor(
        model_name="credit-default-v1",
        reference=reference[FEATURES],
        categorical=["province"],
        baseline_score=baseline,
        metric="roc_auc",
        performance_threshold=0.05,
        db_path=db,
    )

    # ---- Six months of production ---------------------------------------
    schedule = [
        ("2026-03", dict()),
        ("2026-04", dict()),
        ("2026-05", dict()),
        ("2026-06", dict(income_shift=-0.30)),
        ("2026-07", dict(income_shift=-0.55, province_skew=True)),
        ("2026-08", dict(weaken_income_signal=True)),
    ]

    for label, kwargs in schedule:
        batch = make_batch(n=3_000, **kwargs)
        proba = model.predict_proba(scaler.transform(encode(batch, X_ref.columns)))[:, 1]

        report = monitor.check(
            batch[FEATURES],
            batch_label=label,
            y_true=batch["default"],
            y_pred_proba=proba,
        )

        mv = multivariate_drift(
            reference[FEATURES], batch[FEATURES], categorical=["province"]
        )

        print("=" * 58)
        print(report.summary())
        print(f"\n{mv}")
        print()

    print("=" * 58)
    print(f"\n{len(schedule)} runs written to {db.name}")
    print("Inspect with: monitor.store.history() or the Streamlit dashboard.\n")

    history = monitor.store.history()
    print(history[["run_id", "batch_label", "n_drifted", "max_psi", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
