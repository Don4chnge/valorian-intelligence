# Valorian Intelligence

**ML observability: drift detection and performance monitoring for models already in production.**

Most machine learning projects stop at the point a model is trained and a test-set score is reported. That is the easy half. The harder problem is that a model which scored well in January can be quietly wrong by August, because the world it was trained on has moved and nothing in the system announces it.

Valorian Intelligence watches a deployed model and answers two questions on every batch of production data:

1. **Have the inputs changed?** (data drift) — Population Stability Index, Kolmogorov–Smirnov, chi-square
2. **Has the model stopped working?** (concept drift) — rolling performance against a frozen baseline

The second question is the one that matters, and the one most monitoring setups miss.

---

## Why this is not just "run a KS test"

A single p-value on a large batch is close to useless. With 3,000 rows, a KS test returns `p < 0.001` for shifts far too small to affect any decision the model makes — statistical significance stops meaning practical significance long before the sample sizes involved in production monitoring.

So Valorian leads with **effect size (PSI)** and reports the p-value alongside it as supporting evidence:

| PSI | Interpretation |
|---|---|
| < 0.10 | Stable |
| 0.10 – 0.25 | Moderate shift — investigate |
| > 0.25 | Significant shift — model likely degraded |

Two design decisions follow from actually thinking about how this fails:

- **Bin edges are frozen from the reference sample.** If you re-bin both distributions together at check time, the drift partially cancels itself out and the monitor under-reports. Edges come from training data only.
- **Outer bins are opened to ±∞.** Production values outside the training range are the most alarming thing a monitor can see. They must land in the edge bins, not be silently dropped.

---

## Does it actually work?

`demo/run_demo.py` builds a credit-default model, then simulates six months of production data with drift injected on purpose — so there is a ground truth to check the detector against. A monitoring tool that has never been shown to catch a planted fault is not evidence of anything.

| Batch | What was injected | Max PSI | Status | ROC-AUC vs baseline |
|---|---|---|---|---|
| 2026-03 | nothing | 0.008 | ok | +2.6% |
| 2026-04 | nothing | 0.009 | ok | −1.4% |
| 2026-05 | nothing | 0.006 | ok | −0.2% |
| 2026-06 | income distribution shifts down | 0.270 | **critical** | −7.5% |
| 2026-07 | deeper income shift + province mix change | 0.899 | **critical** | +0.1% |
| 2026-08 | income→default relationship weakens | 0.011 | **critical** | **−11.3%** |

Three things worth reading off this table:

**The clean months stay quiet.** A detector that fires on everything is worse than no detector, because people learn to ignore it. Months 3–5 are noise-level.

**Month 7 is the false-comfort case.** Two features drifted severely (PSI 0.90 and 0.45) but ROC-AUC barely moved. Input drift does not automatically mean broken model — this is why alerting on drift alone produces alert fatigue.

**Month 8 is the case this project exists for.** Every input distribution looks perfectly normal — max PSI 0.011, nothing to see. Meanwhile the model has lost 11.3% of its discriminative power because the relationship between income and default changed underneath it. Input-only monitoring is blind here. Valorian raises it explicitly:

```
Alerts:
  ! roc_auc degraded -11.3% vs baseline — retraining likely needed
  ! Performance dropped without significant input drift —
    suspect concept drift or a data pipeline/labelling fault
```

---

## Usage

```python
from valorian import ModelMonitor

monitor = ModelMonitor(
    model_name="credit-default-v1",
    reference=training_df[FEATURES],   # frozen at training time
    categorical=["province"],
    baseline_score=0.6697,
    metric="roc_auc",
)

report = monitor.check(
    live_df,
    batch_label="2026-08",
    y_true=outcomes,              # optional — omit for input-only monitoring
    y_pred_proba=predictions,
)

print(report.summary())
report.status            # 'ok' | 'warning' | 'critical'
report.drifted_features  # ['monthly_income', 'province']
```

Every run is written to SQLite, so drift is a trend you can plot rather than a one-off number:

```python
monitor.store.history("credit-default-v1")
monitor.store.feature_history("monthly_income")
```

---

## Quick start

```bash
git clone https://github.com/Don4chnge/valorian-intelligence
cd valorian-intelligence
pip install -r requirements.txt

python demo/run_demo.py     # generates valorian.db and prints six reports
streamlit run app.py        # dashboard over the results
pytest -q                   # 23 tests
```

---

## Project layout

```
valorian/
  drift.py         PSI, KS, chi-square — feature-level data drift
  performance.py   baseline comparison and rolling metrics — concept drift
  monitor.py       ModelMonitor orchestrator, alert logic, status rules
  store.py         SQLite persistence and history queries
demo/run_demo.py   six months of simulated production with injected drift
tests/             23 tests, including "does not fire on clean data"
app.py             Streamlit dashboard
```

---

## Limitations

Stated plainly, because a monitoring tool that oversells itself is a liability:

- **Concept drift detection needs labels, so it runs on a lag.** You learn that August's predictions were wrong only once August's outcomes arrive. For credit or churn that lag can be months. Input drift is the early warning; performance drift is the confirmation.
- **PSI is sensitive to bin count and to small samples.** Below roughly 1,000 rows per batch the estimate gets noisy. Ten quantile bins is a convention, not a derived optimum.
- **Features are checked one at a time.** A shift in the *correlation* between two features — each individually stable — will not be caught. Multivariate drift detection is on the roadmap.
- **The thresholds are conventions, not laws.** 0.10 and 0.25 come from credit risk practice. Any serious deployment should calibrate them against its own history of what did and did not turn out to matter.
- **The demo uses synthetic data.** That is deliberate — it is the only way to have ground truth about drift — but performance on synthetic data is not evidence of performance on real data. Validating against a real dataset is the next milestone.

---

## Roadmap

See [ROADMAP.md](ROADMAP.md).

---

Built by **Thabang Manyama** — BSc Information Technology: Data Science, EDUVOS.
