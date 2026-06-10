"""Validity check: schema conformance, type correctness, range/regex guards."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from dqk.checks.base import BaseCheck, CheckResult, CheckSeverity
from dqk.core.schema import ColumnDtype


class ValidityCheck(BaseCheck):
    name = "validity"
    description = "Schema/type conformance, range guards, regex patterns, constant columns"
    weight = 1.2

    def __init__(
        self,
        range_guards: dict[str, tuple[float, float]] | None = None,
        regex_guards: dict[str, str] | None = None,
        fail_threshold: float = 0.01,
        warn_threshold: float = 0.001,
    ) -> None:
        """
        Arguments:
        range_guards: ``{col_name: (min_val, max_val)}`` — rows outside this range are violations.
        regex_guards: ``{col_name: pattern}`` — non-null rows not matching
        the pattern are violations.
        fail_threshold: Violation rate above which a FAIL issue is raised.
        warn_threshold: Violation rate above which a WARN issue is raised.
        """
        self.range_guards = range_guards or {}
        self.regex_guards = regex_guards or {}
        self.fail_threshold = fail_threshold
        self.warn_threshold = warn_threshold

    def run(self, dataset: Any) -> CheckResult:  # noqa: ANN401
        df: pd.DataFrame = dataset.df
        result = self._empty_result()
        n_rows = len(df)
        if n_rows == 0:
            result.score = 0.0
            result.severity = CheckSeverity.FAIL
            result.add_issue("Dataset is empty.", severity=CheckSeverity.FAIL)
            return result

        type_violation_rates: dict[str, float] = {}
        range_violation_rates: dict[str, float] = {}
        regex_violation_rates: dict[str, float] = {}
        constant_cols: list[str] = []
        total_violation_weight = 0.0

        # per-column checks
        for col_meta in dataset.schema.columns:
            col = col_meta.name
            series = df[col]

            # 1. Type conformance
            vrate = self._type_violation_rate(series, col_meta.dtype)
            type_violation_rates[col] = round(vrate, 4)
            if vrate >= self.fail_threshold:
                result.add_issue(
                    f"Column '{col}' has {vrate:.1%} type-invalid values"
                    f" (expected {col_meta.dtype.value}).",
                    column=col,
                    severity=CheckSeverity.FAIL,
                    violation_rate=vrate,
                )
                total_violation_weight += vrate
            elif vrate >= self.warn_threshold:
                result.add_issue(
                    f"Column '{col}' has {vrate:.1%} type-invalid values.",
                    column=col,
                    severity=CheckSeverity.WARN,
                    violation_rate=vrate,
                )
                total_violation_weight += vrate * 0.5

            # 2. Constant columns
            if col_meta.n_unique == 1:
                constant_cols.append(col)
                result.add_issue(
                    f"Column '{col}' has only one unique value — likely uninformative.",
                    column=col,
                    severity=CheckSeverity.WARN,
                )

            # 3. Range guards
            if col in self.range_guards:
                lo, hi = self.range_guards[col]
                numeric = pd.to_numeric(series, errors="coerce")
                out_of_range = ((numeric < lo) | (numeric > hi)).sum()
                rrate = int(out_of_range) / n_rows
                range_violation_rates[col] = round(rrate, 4)
                if rrate >= self.fail_threshold:
                    result.add_issue(
                        f"Column '{col}' has {rrate:.1%} values outside [{lo}, {hi}].",
                        column=col,
                        severity=CheckSeverity.FAIL,
                        violation_rate=rrate,
                        range=(lo, hi),
                    )
                    total_violation_weight += rrate
                elif rrate >= self.warn_threshold:
                    result.add_issue(
                        f"Column '{col}' has {rrate:.1%} values outside [{lo}, {hi}].",
                        column=col,
                        severity=CheckSeverity.WARN,
                        violation_rate=rrate,
                    )

            # 4. Regex guards
            if col in self.regex_guards:
                pattern = self.regex_guards[col]
                compiled = re.compile(pattern)
                non_null = series.dropna().astype(str)
                n_mismatch = int((~non_null.str.match(compiled)).sum())
                rxrate = n_mismatch / n_rows
                regex_violation_rates[col] = round(rxrate, 4)
                if rxrate >= self.fail_threshold:
                    result.add_issue(
                        f"Column '{col}' has {rxrate:.1%} values not matching pattern '{pattern}'.",
                        column=col,
                        severity=CheckSeverity.FAIL,
                        violation_rate=rxrate,
                    )
                    total_violation_weight += rxrate
                elif rxrate >= self.warn_threshold:
                    result.add_issue(
                        f"Column '{col}' has {rxrate:.1%} values not matching pattern '{pattern}'.",
                        column=col,
                        severity=CheckSeverity.WARN,
                    )

        score = max(0.0, 1.0 - min(total_violation_weight, 1.0))
        result.score = round(score, 4)
        result.metrics = {
            "type_violation_rates": type_violation_rates,
            "range_violation_rates": range_violation_rates,
            "regex_violation_rates": regex_violation_rates,
            "constant_cols": constant_cols,
        }
        return result

    @staticmethod
    def _type_violation_rate(series: pd.Series, expected: ColumnDtype) -> float:
        """Return the fraction of non-null values that violate the expected type."""
        non_null = series.dropna()
        n = len(non_null)
        if n == 0:
            return 0.0

        if expected == ColumnDtype.INTEGER or expected == ColumnDtype.FLOAT:
            violations = pd.to_numeric(non_null, errors="coerce").isna().sum()
        elif expected == ColumnDtype.BOOLEAN:
            violations = int(
                (~non_null.astype(str).str.lower().isin({"true", "false", "1", "0"})).sum()
            )
        elif expected == ColumnDtype.DATETIME:
            violations = int(pd.to_datetime(non_null, errors="coerce").isna().sum())
        else:
            return 0.0  # STRING / UNKNOWN / CATEGORY - no type check

        return violations / len(series)  # relative to total rows (incl. nulls)
