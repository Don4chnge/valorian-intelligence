"""
Data quality and schema checks.

Drift statistics answer "did the distribution of the values move?". They are
computed on the values that are present, in the columns both frames share,
which means two whole classes of failure pass straight through them:

    - a column that stops arriving, arrives under a new name, or arrives as
      text where it used to be numeric;
    - a column that is still there but is suddenly 60% null.

PSI on the surviving 40% of rows can look perfectly stable. So these checks run
*before* drift detection and report on the batch's fitness to be monitored at
all. They are deliberately separate from drift severity: a null spike is a
pipeline fault, not a population shift, and the response is different.

The thresholds are conventions, not derived constants. A rise of 5 percentage
points in a feature's null rate is flagged as a warning and 20 points as a
failure; both are arguments to `check_quality` and should be set from what a
given pipeline normally looks like.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

import pandas as pd

NULL_RATE_WARNING = 0.05
NULL_RATE_FAILED = 0.20

QualityStatus = Literal["healthy", "warning", "failed"]


def _kind(series: pd.Series, declared_categorical: bool) -> str:
    if declared_categorical:
        return "categorical"
    return "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"


@dataclass
class DataQualityReport:
    """Whether a batch meets the contract implied by the reference data."""

    status: QualityStatus
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    type_changed: dict[str, dict[str, str]] = field(default_factory=dict)
    empty_columns: list[str] = field(default_factory=list)
    missingness: list[dict] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def unusable_columns(self) -> list[str]:
        """Shared columns that cannot be drift-tested in this batch."""
        return sorted(set(self.type_changed) | set(self.empty_columns))

    def to_dict(self) -> dict:
        return asdict(self)


def check_quality(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    categorical: list[str] | None = None,
    null_rate_warning: float = NULL_RATE_WARNING,
    null_rate_failed: float = NULL_RATE_FAILED,
) -> DataQualityReport:
    """
    Compare a batch against the reference frame's columns, types and null rates.

    The reference frame is the contract: every column in it is expected in the
    batch with the same kind (numeric or categorical).

        failed    a reference column is absent, changed kind, is entirely null,
                  or its null rate rose by `null_rate_failed` or more
        warning   the batch carries columns the reference never had, or a null
                  rate rose by `null_rate_warning` or more
        healthy   none of the above

    Null-rate changes are measured in absolute percentage points against the
    reference rate, so a feature that was always 30% null is not flagged for
    staying 30% null.
    """
    declared = set(categorical or [])
    issues: list[str] = []
    worst: QualityStatus = "healthy"

    def escalate(level: QualityStatus) -> None:
        nonlocal worst
        if level == "failed" or (level == "warning" and worst == "healthy"):
            worst = level

    missing = [c for c in reference.columns if c not in current.columns]
    extra = [c for c in current.columns if c not in reference.columns]
    shared = [c for c in reference.columns if c in current.columns]

    for col in missing:
        issues.append(f"{col}: column missing from batch")
        escalate("failed")
    if extra:
        issues.append(f"Unexpected columns in batch: {', '.join(map(str, extra))}")
        escalate("warning")

    type_changed: dict[str, dict[str, str]] = {}
    empty: list[str] = []
    missingness: list[dict] = []

    for col in shared:
        ref_rate = float(reference[col].isna().mean()) if len(reference) else 0.0
        cur_rate = float(current[col].isna().mean()) if len(current) else 1.0
        delta = cur_rate - ref_rate

        if cur_rate >= 1.0:
            level: QualityStatus = "failed"
            empty.append(col)
            issues.append(f"{col}: every value in the batch is null")
        elif delta >= null_rate_failed:
            level = "failed"
            issues.append(
                f"{col}: null rate {ref_rate:.1%} -> {cur_rate:.1%} "
                f"(+{delta * 100:.1f} points)"
            )
        elif delta >= null_rate_warning:
            level = "warning"
            issues.append(
                f"{col}: null rate {ref_rate:.1%} -> {cur_rate:.1%} "
                f"(+{delta * 100:.1f} points)"
            )
        else:
            level = "healthy"
        escalate(level)

        missingness.append(
            {
                "feature": col,
                "missing_reference": ref_rate,
                "missing_current": cur_rate,
                "delta": delta,
                "status": level,
            }
        )

        # An all-null column has no meaningful dtype, so it is reported once,
        # as empty, rather than also as a type change.
        if cur_rate < 1.0:
            ref_kind = _kind(reference[col], col in declared)
            cur_kind = _kind(current[col], col in declared)
            if ref_kind != cur_kind:
                type_changed[col] = {
                    "reference": f"{ref_kind} ({reference[col].dtype})",
                    "current": f"{cur_kind} ({current[col].dtype})",
                }
                issues.append(
                    f"{col}: type changed from {ref_kind} ({reference[col].dtype}) "
                    f"to {cur_kind} ({current[col].dtype})"
                )
                escalate("failed")

    return DataQualityReport(
        status=worst,
        missing_columns=missing,
        extra_columns=extra,
        type_changed=type_changed,
        empty_columns=empty,
        missingness=missingness,
        issues=issues,
    )
