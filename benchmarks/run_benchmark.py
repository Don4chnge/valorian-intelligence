"""
Benchmark: how often does each detector fire when it should, and when it
should not?

Every scenario draws a reference and a current sample from distributions that
are known by construction, so the right answer is never in doubt. Each is
repeated across many seeds and several sample sizes, and what is reported is
the *rate* at which a detector flags drift, with a 95% Wilson interval.

    no-change scenarios   the rate is a false-positive rate; lower is better
    planted-shift ones    the rate is detection power; higher is better

"raw PSI >= 0.10" is the rule Valorian used up to 0.3.0. "valorian drifted" is
the current rule, which adds a noise floor and an out-of-range check; both were
introduced because of what this benchmark showed, and the raw rule is kept in
the tables so the before and after can be read side by side.

Valorian's detectors are compared with Evidently's on the same samples. Two
Evidently configurations are used: PSI at a 0.1 threshold, which is the
like-for-like comparison, and Evidently's own default method for the column
type and sample size, which is what a user gets without choosing.

Nothing here is tuned to make a detector look good, and every row that is
computed is written out, including the ones where Valorian does worse.

Run:
    python benchmarks/run_benchmark.py            # full run, about 25 minutes
    python benchmarks/run_benchmark.py --quick    # smoke test, about a minute

Writes benchmarks/results.csv and benchmarks/RESULTS.md.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import valorian  # noqa: E402
from valorian import (  # noqa: E402
    categorical_drift,
    compare_performance,
    detect_drift,
    multivariate_drift,
    numeric_drift,
    score,
)

warnings.filterwarnings("ignore")

try:
    import evidently
    from evidently import Report
    from evidently.metrics import ValueDrift

    HAVE_EVIDENTLY = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_EVIDENTLY = False

HERE = Path(__file__).resolve().parent
ALPHA = 0.05


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def evidently_flags(ref: pd.Series, cur: pd.Series) -> dict[str, bool]:
    """Evidently's verdict on one column, under PSI and under its default."""
    if not HAVE_EVIDENTLY:
        return {}
    snapshot = Report(
        [ValueDrift(column="x", method="psi", threshold=0.1), ValueDrift(column="x")]
    ).run(pd.DataFrame({"x": cur}), pd.DataFrame({"x": ref}))
    psi, default = snapshot.dict()["metrics"]
    cfg = default["config"]
    # Evidently's p-value methods flag when the value is BELOW the threshold;
    # its distance methods flag when the value is at or above it.
    is_p_value = "p_value" in cfg["method"] or cfg["method"] in ("ks", "chisquare", "z")
    default_flag = (
        default["value"] < cfg["threshold"] if is_p_value
        else default["value"] >= cfg["threshold"]
    )
    return {
        "evidently PSI >= 0.10": bool(psi["value"] >= 0.1),
        "evidently default": bool(default_flag),
    }


class Tally:
    def __init__(self):
        self.rows: dict[tuple, list[int]] = {}

    def add(self, family: str, scenario: str, kind: str, n: int, flags: dict[str, bool]):
        for detector, flag in flags.items():
            cell = self.rows.setdefault((family, scenario, kind, n, detector), [0, 0])
            cell[0] += int(flag)
            cell[1] += 1

    def frame(self) -> pd.DataFrame:
        out = []
        for (family, scenario, kind, n, detector), (k, trials) in self.rows.items():
            low, high = wilson(k, trials)
            out.append(
                {
                    "family": family, "scenario": scenario, "expected": kind, "n": n,
                    "detector": detector, "flagged": k, "trials": trials,
                    "rate": round(k / trials, 4),
                    "ci_low": round(low, 4), "ci_high": round(high, 4),
                }
            )
        return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Scenario definitions. Each returns (reference, current) for a seed and n.
# --------------------------------------------------------------------------

def _tail(rng, n):
    x = rng.normal(0, 1, n)
    k = int(0.05 * n)
    x[:k] = rng.normal(4, 1, k)     # 5% of rows move far into the upper tail
    return x


NUMERIC = {
    # name: (expected, generator of the current sample)
    "no change": ("quiet", lambda rng, n: rng.normal(0, 1, n)),
    "mean +0.10 sd": ("fire", lambda rng, n: rng.normal(0.10, 1, n)),
    "mean +0.25 sd": ("fire", lambda rng, n: rng.normal(0.25, 1, n)),
    "mean +0.50 sd": ("fire", lambda rng, n: rng.normal(0.50, 1, n)),
    "variance x1.5": ("fire", lambda rng, n: rng.normal(0, 1.5, n)),
    "5% of rows into upper tail": ("fire", _tail),
}

LEVELS = ["a", "b", "c", "d", "e"]
BASE_P = [0.40, 0.25, 0.15, 0.12, 0.08]


def _unseen(rng, n):
    x = rng.choice(LEVELS, n, p=BASE_P).astype(object)
    x[: int(0.02 * n)] = "NEW"      # 2% of rows carry a level never seen before
    return x


CATEGORICAL = {
    "no change": ("quiet", lambda rng, n: rng.choice(LEVELS, n, p=BASE_P)),
    "small mix shift": ("fire", lambda rng, n: rng.choice(LEVELS, n, p=[0.36, 0.27, 0.16, 0.13, 0.08])),
    "large mix shift": ("fire", lambda rng, n: rng.choice(LEVELS, n, p=[0.25, 0.25, 0.20, 0.18, 0.12])),
    "2% unseen category": ("fire", _unseen),
}


def _pair(rng, n, rho):
    xy = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    return pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "z": rng.normal(0, 1, n)})


MULTIVARIATE = {
    "no change": ("quiet", 0.6, 0.6),
    "correlation 0.6 -> 0.0": ("fire", 0.6, 0.0),
    "correlation 0.6 -> -0.6": ("fire", 0.6, -0.6),
}


# --------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------

def run_numeric(tally: Tally, sizes, seeds, evidently_seeds):
    for name, (kind, make) in NUMERIC.items():
        for n in sizes:
            for seed in range(seeds):
                rng = np.random.default_rng([1, n, seed])
                ref, cur = rng.normal(0, 1, n), make(rng, n)
                r = numeric_drift("x", ref, cur)
                flags = {
                    "raw PSI >= 0.10 (valorian <= 0.3)": r.psi >= 0.10,
                    "valorian drifted": r.drifted,
                    "valorian significant": r.severity == "significant",
                    "KS p < 0.05": r.p_value < ALPHA,
                }
                if seed < evidently_seeds:
                    flags |= evidently_flags(pd.Series(ref), pd.Series(cur))
                tally.add("numeric", name, kind, n, flags)


def run_categorical(tally: Tally, sizes, seeds, evidently_seeds):
    for name, (kind, make) in CATEGORICAL.items():
        for n in sizes:
            for seed in range(seeds):
                rng = np.random.default_rng([2, n, seed])
                ref, cur = rng.choice(LEVELS, n, p=BASE_P), make(rng, n)
                r = categorical_drift("x", ref, cur)
                flags = {
                    "raw PSI >= 0.10 (valorian <= 0.3)": r.psi >= 0.10,
                    "valorian drifted": r.drifted,
                    "valorian significant": r.severity == "significant",
                    "chi-square p < 0.05": r.p_value < ALPHA,
                }
                if seed < evidently_seeds:
                    flags |= evidently_flags(pd.Series(ref), pd.Series(cur))
                tally.add("categorical", name, kind, n, flags)


def run_multivariate(tally: Tally, sizes, seeds):
    for name, (kind, rho_ref, rho_cur) in MULTIVARIATE.items():
        for n in sizes:
            for seed in range(seeds):
                rng = np.random.default_rng([3, n, seed])
                ref, cur = _pair(rng, n, rho_ref), _pair(rng, n, rho_cur)
                per_feature = detect_drift(ref, cur)
                mv = multivariate_drift(ref, cur, attribution="impurity", random_state=seed)
                tally.add("multivariate", name, kind, n, {
                    "any feature drifted (valorian)": bool(per_feature["drifted"].any()),
                    "multivariate AUC >= 0.55 (mild)": mv.auc >= 0.55,
                    "multivariate AUC >= 0.65 (clear)": mv.auc >= 0.65,
                })


def run_many_features(tally: Tally, sizes, seeds, n_features=20):
    """With many stable features, how often does at least one get flagged?"""
    for n in sizes:
        for seed in range(seeds):
            rng = np.random.default_rng([4, n, seed])
            ref, cur = {}, {}
            for i in range(n_features):
                ref[f"f{i}"], cur[f"f{i}"] = rng.normal(0, 1, n), rng.normal(0, 1, n)
            results = detect_drift(pd.DataFrame(ref), pd.DataFrame(cur))
            p_values = results["p_value"].to_numpy()
            tally.add("many features", f"no change, {n_features} features", "quiet", n, {
                "any raw PSI >= 0.10 (valorian <= 0.3)": bool((results["psi"] >= 0.10).any()),
                "any feature drifted (valorian)": bool(results["drifted"].any()),
                "any KS p < 0.05": bool((p_values < ALPHA).any()),
                "any KS p < 0.05 / 20 (Bonferroni)": bool((p_values < ALPHA / n_features).any()),
            })


def _scored(rng, n, strength, prevalence=0.3):
    """Labels and a score whose separation from the label is set by `strength`."""
    y = rng.binomial(1, prevalence, n)
    return y, rng.normal(strength * y, 1.0, n)


def run_performance(tally: Tally, sizes, seeds):
    """
    A model with ROC-AUC near 0.76, monitored against a baseline measured on
    200 000 rows so the baseline itself is effectively exact.
    """
    healthy, weakened = 1.0, 0.62          # weakened gives roughly a 10% AUC drop
    rng = np.random.default_rng(99)
    y, s = _scored(rng, 200_000, healthy)
    baseline = score(y, s)
    y, s = _scored(rng, 200_000, weakened)
    true_drop = score(y, s) / baseline - 1

    for name, kind, strength in [
        ("no change", "quiet", healthy),
        (f"AUC {true_drop:+.0%}", "fire", weakened),
    ]:
        for n in sizes:
            for seed in range(seeds):
                rng = np.random.default_rng([5, n, seed])
                y, s = _scored(rng, n, strength)
                r = compare_performance(baseline, y, s)
                tally.add("performance", name, kind, n, {
                    "degraded (point estimate past 5%)": r.degraded,
                    "degraded and confirmed by interval": r.confirmed is True,
                    "insufficient evidence": not r.evaluable,
                })
    return baseline, true_drop


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _table(frame: pd.DataFrame, family: str) -> str:
    sub = frame[frame["family"] == family]
    if sub.empty:
        return "_not run_\n"
    sub = sub.assign(cell=sub["rate"].map("{:.0%}".format))
    wide = sub.pivot_table(
        index=["scenario", "expected", "detector"], columns="n", values="cell",
        aggfunc="first", sort=False,
    ).reset_index()
    wide.columns = [str(c) if not isinstance(c, str) else c for c in wide.columns]
    sizes = [c for c in wide.columns if c not in ("scenario", "expected", "detector")]
    head = "| Scenario | Should | Detector | " + " | ".join(f"n={int(s):,}" for s in sizes) + " |"
    rule = "|---|---|---|" + "---:|" * len(sizes)
    lines, last = [head, rule], None
    for _, row in wide.iterrows():
        label = row["scenario"] if row["scenario"] != last else ""
        should = row["expected"] if row["scenario"] != last else ""
        last = row["scenario"]
        cells = " | ".join("" if pd.isna(row[s]) else row[s] for s in sizes)
        lines.append(f"| {label} | {should} | {row['detector']} | {cells} |")
    return "\n".join(lines) + "\n"


def write_report(frame, meta):
    trials = frame.groupby("family")["trials"].agg(["min", "max"])
    text = f"""# Benchmark results

Generated by `benchmarks/run_benchmark.py`. Do not edit by hand; the reading of
these numbers is in [README.md](README.md).

Each cell is the share of trials in which the detector flagged drift. In rows
marked **quiet** nothing changed, so the cell is a false-positive rate and lower
is better. In rows marked **fire** a shift was planted, so the cell is detection
power and higher is better. `n` is the size of both the reference and the
current sample. 95% Wilson intervals for every cell are in `results.csv`.

## Environment

| | |
|---|---|
| valorian | {valorian.__version__} |
| evidently | {meta['evidently']} |
| numpy / pandas / scipy / scikit-learn | {meta['versions']} |
| Python | {platform.python_version()} on {platform.system()} {platform.machine()}, {meta['cpus']} CPUs |
| Mode | {meta['mode']} |
| Runtime | {meta['runtime']:.0f} s |

Trials per cell: {", ".join(f"{fam} {int(r['min'])}" + (f"-{int(r['max'])}" if r['max'] != r['min'] else "") for fam, r in trials.iterrows())}.
Where a family shows a range, the smaller number is the Evidently rows, which
are run on the first seeds of the same samples.

## Numeric features

Reference is N(0, 1). Evidently's default for a numeric column is a KS test up
to 1 000 rows and normalised Wasserstein distance at a 0.1 threshold above that.

{_table(frame, 'numeric')}
## Categorical features

Five levels with shares 40/25/15/12/8. Evidently's default for a categorical
column is a chi-square test up to 1 000 rows and Jensen-Shannon distance at a
0.1 threshold above that.

{_table(frame, 'categorical')}
## Multivariate: correlation changes, marginals do not

Three standard normal features; two are correlated and the correlation is what
moves. Every marginal distribution is identical before and after.

{_table(frame, 'multivariate')}
## Many features at once

Twenty independent features, none of which changed. The cell is the share of
trials in which at least one was flagged.

{_table(frame, 'many features')}
## Performance

A classifier with baseline ROC-AUC {meta['baseline']:.3f}, 30% positive rate. `n` is the
number of labelled rows in the batch. Defaults: 5% relative tolerance, 100-row
and 10-per-class evidence floors, 200 bootstrap resamples.

{_table(frame, 'performance')}"""
    (HERE / "RESULTS.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="few seeds, small sizes")
    args = parser.parse_args()

    if args.quick:
        cfg = dict(sizes=[200, 1_000], seeds=20, ev=10, mv_sizes=[500], mv_seeds=4,
                   perf_sizes=[100, 1_000], perf_seeds=30)
    else:
        cfg = dict(sizes=[200, 500, 1_000, 5_000, 20_000], seeds=300, ev=100,
                   mv_sizes=[500, 2_000, 5_000], mv_seeds=40,
                   perf_sizes=[100, 300, 1_000, 5_000], perf_seeds=300)

    start = time.time()
    tally = Tally()
    steps = [
        ("numeric", lambda: run_numeric(tally, cfg["sizes"], cfg["seeds"], cfg["ev"])),
        ("categorical", lambda: run_categorical(tally, cfg["sizes"], cfg["seeds"], cfg["ev"])),
        ("many features", lambda: run_many_features(tally, cfg["sizes"], cfg["seeds"])),
        ("multivariate", lambda: run_multivariate(tally, cfg["mv_sizes"], cfg["mv_seeds"])),
    ]
    for name, step in steps:
        t = time.time()
        step()
        print(f"{name:<14} {time.time() - t:6.0f} s", flush=True)

    t = time.time()
    baseline, _ = run_performance(tally, cfg["perf_sizes"], cfg["perf_seeds"])
    print(f"{'performance':<14} {time.time() - t:6.0f} s", flush=True)

    frame = tally.frame()
    frame.to_csv(HERE / "results.csv", index=False)

    import scipy
    import sklearn

    write_report(frame, {
        "evidently": evidently.__version__ if HAVE_EVIDENTLY else "not installed - rows omitted",
        "versions": f"{np.__version__} / {pd.__version__} / {scipy.__version__} / {sklearn.__version__}",
        "cpus": __import__("os").cpu_count(),
        "mode": "quick (smoke test - rates are too noisy to read)" if args.quick else "full",
        "runtime": time.time() - start,
        "baseline": baseline,
    })
    print(f"\nWrote results.csv and RESULTS.md in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
