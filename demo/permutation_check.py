"""
Is the multivariate AUC of ~0.58 real, or is it what 40,000 rows produce
from noise? Refit on shuffled domain labels and see how often the null
reaches the observed score.

The smallest p-value a permutation test can report is 1 / (n + 1). Twenty
shuffles can therefore say no more than p = 0.048 however strong the signal,
which is where this script started. The default is now 500, giving a floor of
0.002.

Run:  python demo/permutation_check.py [n_permutations]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from valorian import multivariate_drift
from run_qlfs import CATEGORICAL, FEATURES, load

n = int(sys.argv[1]) if len(sys.argv) > 1 else 500

ref = load("QLFS202104.csv")
cur = load("QLFS202301.csv")

print(f"Running {n} permutations — roughly {n * 1.7 / 60:.0f} minutes on two cores.\n")
start = time.time()
result = multivariate_drift(
    ref[FEATURES], cur[FEATURES],
    categorical=CATEGORICAL,
    n_permutations=n,
)
print(result)
print(f"\np = {result.p_value}  (smallest reportable: {1 / (n + 1):.4f})")
print(f"{time.time() - start:.0f}s")
