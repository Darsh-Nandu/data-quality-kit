from __future__ import annotations

from typing import Any

import pandas as pd

from dqk.checks.base import BaseCheck, CheckIssue, CheckResult, CheckSeverity


class CompletenessCheck(BaseCheck):
    """
    Check for missing values across the dataset.
    """
    name = "completeness"
    description = "Check null rates, missing paterns, and row-level completeness"
    weight = 1.5

    def __init__(
        self,
        warn_threshold: float = 0.05,
        fail_threshold: float = 0.20,
    ) -> None:
        """
        Arguments:
        warn_threshold: Null rate above which a column triggers a WARN issue.
        fail_threshold: Null rate above which a column triggers a FAIL issue,
        """
        self.warn_threshold = warn_threshold
        self.fail_threshold = fail_threshold

    def run(self, dataset: Any) -> CheckResult:
        df: pd.DataFrame = dataset.df
        n_rows, n_cols = df.shape
        result = self._empty_result()

        if n_rows == 0:
            result.score = 0.0
            result.severity = CheckSeverity.FAIL
            result.add_issue("Dataset is empty.", severity=CheckSeverity.FAIL)
            return result

        # Per column null rates
        null_counts = df.isnull().sum()
        null_rates = (null_counts/n_rows).to_dict()

        empty_columns = [col for col, rate in null_rates.items() if rate == 1.0]
        for col in empty_columns:
            result.add_issue(
                f"Column '{col}' is 100% empty.",
                column=col,
                severity=CheckSeverity.FAIL,
                null_rate = 1.0,
            )

        for col, rate in null_rates.items():
            if rate >= self.fail_threshold and col not in empty_columns:
                result.add_issue(
                    f"Column '{col}' has {rate:.1%} missing values (threshold: {self.fail_threshold:.0%}).",
                    column=col,
                    severity=CheckSeverity.FAIL,
                    null_rate=rate,
                )
            elif rate >= self.warn_threshold and col not in empty_columns:
                result.add_issue(
                    f"Column '{col}' has {rate:.1%} missing values (threshold: {self.warn_threshold:.0%}).",
                    column=col,
                    severity=CheckSeverity.WARN,
                    null_rate=rate,
                )

        # row level completeness
        complete_rows = int((df.isnull().sum(axis=1) == 0).sum())
        complete_row_rate = complete_rows/n_rows

        # overall null rates
        total_cells = n_rows * n_cols
        total_nulls = int(null_counts.sum())
        overall_null_rate = total_nulls / total_cells if total_cells > 0 else 0.0

        # missing pattern analysis
        missing_pattern_cols = [c for c, r in null_rates.items() if 0 < r < 1]
        correlated_pairs: list[dict[str, Any]] = []
        if len(missing_pattern_cols) >=2:
            missing_mask = df[missing_pattern_cols].isnull()
            corr = missing_mask.corr()
            for i, c1 in enumerate(missing_pattern_cols):
                for c2 in missing_pattern_cols[i+1:]:
                    r = corr.loc[c1, c2]
                    if abs(r) > 0.7:
                        correlated_pairs.append({"col_a": c1, "col_b": c2, "correlation": round(r, 3)})
                        result.add_issue(
                            f"Columns '{c1}' and '{c2}' have correlated missingness "
                            f"(r={r:.2f}) — possible MNAR pattern.",
                            severity=CheckSeverity.WARN,
                            col_a=c1,
                            col_b=c2,
                            correlation=r,
                        )

        # score: penalise by overall null rate and empty columns
        empty_col_penalty = len(empty_columns) / n_cols if n_cols > 0 else 0.0
        score = max(0.0, 1.0 - overall_null_rate * 2 - empty_col_penalty)

        result.score = round(score, 4)
        result.metrics = {
            "null_rates": {k: round(v, 4) for k, v in null_rates.items()},
            "overall_null_rate": round(overall_null_rate, 4),
            "empty_cols": empty_columns,
            "high_null_cols": [c for c, r in null_rates.items() if r >= self.warn_threshold],
            "n_complete_rows": complete_rows,
            "complete_row_rate": round(complete_row_rate, 4),
            "missing_pattern_correlations": correlated_pairs,
        }

        return result