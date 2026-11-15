"""
Is the multivariate AUC of ~0.58 real, or is it what 40,000 rows produce
from noise? Refit on shuffled batch labels and see how often the null
reaches the observed score.

Two things this script has had to learn:

The smallest p-value a permutation test can report is 1 / (n + 1). Twenty
shuffles cannot say less than p = 0.048 however strong the signal, which is
where this started. The default is now 500, giving a floor of 0.002.

The QLFS samples dwellings, not people. Everyone in a household shares a
province and a geography type, so 41,000 respondents are closer to 19,000
independent observations. Shuffling individual rows treats them as 41,000 and
understates the noise. This shuffles whole households instead, and keeps each
household in one cross-validation fold.

Run:  python demo/permutation_check.py [n_permutations]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from valorian import multivariate_drift
from run_qlfs import CATEGORICAL, FEATURES, HOUSEHOLD, load

n = int(sys.argv[1]) if len(sys.argv) > 1 else 500

ref = load("QLFS202104.csv")
cur = load("QLFS202301.csv")

print(f"Reference: {len(ref):,} respondents in {ref[HOUSEHOLD].nunique():,} households")
print(f"Current:   {len(cur):,} respondents in {cur[HOUSEHOLD].nunique():,} households")
print(f"\nRunning {n} household-level permutations.\n")
start = time.time()
result = multivariate_drift(
    ref[FEATURES], cur[FEATURES],
    categorical=CATEGORICAL,
    n_permutations=n,
    groups_reference=ref[HOUSEHOLD],
    groups_current=cur[HOUSEHOLD],
)
print(result)
print(f"\np = {result.p_value}  (smallest reportable: {1 / (n + 1):.4f})")
print(f"{time.time() - start:.0f}s")
