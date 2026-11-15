"""
Why did Province drift? The population, or the sample?

The multivariate test finds that the QLFS batches after 2022 Q1 are separable
from the 2021 Q4 reference, and attributes it to Province. That admits two
readings. Either people moved between provinces, or the mix of people the
survey managed to interview changed. They call for different responses, and
the microdata can tell them apart:

    sample share      each province's share of the respondents actually
                      interviewed — what the model, and the drift detector, see
    population share  the same, weighted by Stats SA's survey weights, which
                      are benchmarked to independent population estimates

If the population moved, both shift. If only the sample share moved, the
country did not change; the survey's coverage of it did.

The script also reports how many of each quarter's households were already in
the reference quarter, because the QLFS is a rotating panel and the batches
are not independent of the reference.

Run:  python demo/qlfs_province_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_qlfs import BATCHES, DATA, HOUSEHOLD, REFERENCE, load  # noqa: E402

PROVINCES = {
    1: "Western Cape", 2: "Eastern Cape", 3: "Northern Cape", 4: "Free State",
    5: "KwaZulu-Natal", 6: "North West", 7: "Gauteng", 8: "Mpumalanga", 9: "Limpopo",
}


def quarter(filename: str) -> pd.DataFrame:
    """The monitored rows of one quarter, with their survey weight."""
    frame = load(filename)
    weights = pd.read_csv(DATA / filename, usecols=["Weight", "Q14AGE", "Status"], low_memory=False)
    status = pd.to_numeric(weights["Status"], errors="coerce")
    keep = weights["Q14AGE"].between(15, 64) & status.notna() & ~(status > 10)
    frame["weight"] = pd.to_numeric(weights.loc[keep, "Weight"], errors="coerce").to_numpy()
    return frame


def main() -> None:
    quarters = {label: quarter(filename) for label, filename in [REFERENCE, *BATCHES]}
    reference_households = set(quarters[REFERENCE[0]][HOUSEHOLD])

    sample, population, summary = {}, {}, []
    for label, frame in quarters.items():
        sample[label] = frame["Province"].value_counts(normalize=True)
        population[label] = frame.groupby("Province")["weight"].sum() / frame["weight"].sum()
        households = set(frame[HOUSEHOLD])
        summary.append(
            {
                "quarter": label,
                "respondents": len(frame),
                "households": len(households),
                "also in reference": f"{len(households & reference_households) / len(households):.0%}",
                "Gauteng respondents": int((frame["Province"] == 7).sum()),
            }
        )

    def table(shares: dict) -> pd.DataFrame:
        out = pd.DataFrame(shares).rename(index=PROVINCES).sort_index()
        return (out * 100).round(1)

    print("Respondents, households and overlap with the reference quarter\n")
    print(pd.DataFrame(summary).to_string(index=False))

    print("\n\nSAMPLE share of respondents by province (%) — what the detector sees\n")
    print(table(sample).to_string())

    print("\n\nPOPULATION share by province (%) — weighted by Stats SA survey weights\n")
    print(table(population).to_string())

    first, last = REFERENCE[0], BATCHES[-1][0]
    moved_sample = (table(sample)[last] - table(sample)[first]).abs().max()
    moved_population = (table(population)[last] - table(population)[first]).abs().max()
    print(
        f"\n\nLargest change in any province's share, {first} to {last}:\n"
        f"  sample      {moved_sample:.1f} points\n"
        f"  population  {moved_population:.1f} points"
    )


if __name__ == "__main__":
    main()
