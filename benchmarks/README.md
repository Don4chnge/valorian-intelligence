# Benchmark

`run_benchmark.py` draws reference and current samples from distributions that
are known by construction, repeats each scenario across hundreds of seeds and
several sample sizes, and records how often each detector flags drift. Where
nothing changed that rate is a false-positive rate; where a shift was planted
it is detection power. Evidently is run on the same samples for comparison.

The full tables are in [RESULTS.md](RESULTS.md) and every cell, with its 95%
interval, is in `results.csv`. This page is the reading of them.

```bash
pip install evidently            # optional; its rows are omitted without it
python benchmarks/run_benchmark.py           # about 25 minutes on two cores
python benchmarks/run_benchmark.py --quick   # one-minute smoke test
```

## What it found, and what changed because of it

The benchmark was written to check Valorian, and the first run found two
faults in it. Both are fixed in 0.4.0. The tables keep the old rule as
"raw PSI >= 0.10" so the before and after sit side by side.

**1. On small batches the PSI threshold fired on noise.** With 200 rows a side
and nothing changed, PSI crossed 0.10 in 33% of trials. Evidently's PSI, which
bins differently, did so in 49%. This is arithmetic rather than bad luck: PSI
between two samples of one distribution averages about
(bins − 1) × (1/n_ref + 1/n_cur), which is 0.09 here. Valorian now requires PSI
to clear its own 95% noise level as well as the threshold, which brings the
false-alarm rate to 6% at 200 rows and leaves results at 1 000 rows and above
untouched.

**2. Values far outside the reference range were nearly invisible.** With 5% of
rows moved well beyond the reference maximum, raw PSI flagged 2% of trials at
500 rows and none at 1 000 or more. The README claimed the opposite: that
opening the outer bins was what kept such values from being missed. It keeps
them from being *dropped*, but they land in a bin that already holds a tenth of
the data, where they barely move the statistic. Evidently's PSI caught all of
them. Valorian now measures the out-of-range share directly, with an exact test
for whether it exceeds chance, and catches 90% at 200 rows and 99–100% from
500 rows up.

The first version of that check used the reference's literal minimum and
maximum, and detection fell to 80% at 20 000 rows: a larger reference has a
wider range of its own, so fewer of the shifted values fell outside it. Since
0.6.0 the edge is the reference's outermost 0.1% at each end, which stays at
the same place in the distribution however large the reference is. The
false-positive rows did not move.

## Where Valorian is weaker, by design

**Small shifts on large batches are not flagged.** A mean shift of a quarter of
a standard deviation is caught by a KS test every time from 1 000 rows up, and
by Valorian 17% of the time at 1 000 rows and never at 5 000. The same holds
for a small change in category mix. That is the effect-size rule working as
intended — a 0.25 sd shift has a true PSI near 0.06, under the 0.10 threshold —
but it is a choice, and it means a slow drift of that size is reported as
stable for as long as it stays that size. Anyone who needs it caught should
lower `PSI_MODERATE`, or read the KS p-value, which Valorian reports alongside.

**The "clear" multivariate cutoff misses moderate correlation changes.** A
correlation falling from 0.6 to 0 was separable at the mild level (AUC 0.55) in
every trial, but reached the clear level (0.65), which is what changes a
monitor's status, in only 5% of trials at 500 rows and 75% at 5 000. The
false-positive rate at the mild level was 0–2%, so on this evidence the status
rule could use the lower cutoff. The QLFS data argues the other way: real
batches a quarter apart sit at 0.56–0.58 with nothing wrong with the model.
Synthetic data with no background drift cannot settle that.

## What it shows about performance monitoring

**A hundred labels is not enough for a 5% tolerance.** With the model unchanged,
the point estimate fell more than 5% below baseline in 18% of batches of 100
labelled rows and 13% of batches of 300. At 1 000 it was 1%. The evidence floor
of 100 rows stops verdicts on batches that cannot support any; it does not make
a batch of 100 trustworthy.

**The interval does what it was added for.** Requiring the whole confidence
interval to sit below the baseline cut those false alarms to 2% at both sizes.
The cost is power: a real 12% drop was confirmed in 30% of batches of 100 and
82% of batches of 300, against 82% and 97% for the point estimate alone. From
1 000 labels up, both rules caught every real drop and almost no false ones.

## Many features at once

Twenty unchanged features checked with a KS test at p < 0.05 produced at least
one "significant" result in 55–66% of batches at every sample size; a
Bonferroni correction brought that to 5–7%. Valorian shares its noise allowance
across the features in a batch for the same reason. With the raw
threshold, at least one of twenty unchanged features was flagged in every batch
of 200 rows; with the shared allowance it is 11% at 200 rows, 5% at 500 and
none above.

## About the Evidently comparison

Evidently 0.7.23 was run in two configurations on the first 100 seeds of each
scenario: PSI at a 0.1 threshold, and its default method for the column type
and size.

- On PSI the two libraries agree closely wherever sample size is adequate.
  Evidently's is noisier on small unchanged samples and much better on values
  outside the reference range, as above.
- Evidently's default for numeric columns (KS up to 1 000 rows, Wasserstein
  above) held its false-positive rate at or under 6% throughout.
- Evidently's default for categorical columns up to 1 000 rows flagged 32–34% of
  unchanged samples. It uses a goodness-of-fit chi-square that treats the
  reference shares as exact. This benchmark uses equal-sized samples, where the
  reference is as noisy as the current batch; against a reference many times
  larger than the batch the rate would be much closer to 5%. Valorian's
  chi-square is a two-sample test and stayed at 4–6%.
- Above 1 000 rows Evidently's categorical default switches to Jensen-Shannon
  distance at 0.1, which did not flag the 2% unseen category or the small mix
  shift at all.

This is one version of Evidently under its defaults, on synthetic data. It is
not a ranking of the two libraries, and Evidently does a great deal that
Valorian does not attempt.

## Limits

- **Synthetic, well-behaved data.** Gaussian features and clean categorical
  draws. Real features are skewed, heavy-tailed, rounded and correlated.
- **Equal sample sizes.** Reference and current are the same size in every
  scenario. Production references are usually much larger than a batch, which
  lowers the noise in everything measured here.
- **Independent rows.** No clustering or time dependence, both of which the
  QLFS data has.
- **No concept-shift scenario for the input detectors.** Performance is tested
  separately, with the labels all present; delayed and partial labels are not
  modelled.
- **Thresholds are the defaults.** No detector was tuned, including Valorian's.
