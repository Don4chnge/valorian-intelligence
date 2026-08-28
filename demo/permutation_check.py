"""
Is the multivariate AUC of ~0.57 real, or is it what 40,000 rows produce
from noise? Refit on shuffled domain labels and see how often the null
reaches the observed score.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from valorian import multivariate_drift
from run_qlfs import CATEGORICAL, FEATURES, load

ref = load("QLFS202104.csv")
cur = load("QLFS202301.csv")

print("Running 20 permutations — this takes a few minutes.\n")
result = multivariate_drift(
    ref[FEATURES], cur[FEATURES],
    categorical=CATEGORICAL,
    n_permutations=20,
)
print(result)
print(f"\np = {result.p_value}")