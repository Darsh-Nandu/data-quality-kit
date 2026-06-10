"""Distribution check: outliers, skewness, cardinality, and statistical anomalies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from dqk.checks.base import BaseCheck, CheckResult, CheckSeverity


class DistributionCheck(BaseCheck):
    """
    Detect statistical distribution anomalies across numeric and categorical columns.

    - Z-score + IQR outlier detection for numeric columns
    - Skewness / heavy-tail detection
    - High-cardinality and rare-category detection for categorical columns
    - Constant / near-constant column detection
    """

    name = "distribution"
    description = (
        "Outlier detection (Z-score + IQR), skewness, kurtosis, "
        "and cardinality checks for numeric and categorical columns"
    )
    weight = 1.0

    def __init__(
        self,
        zscore_threshold: float = 3.5,
        iqr_multiplier: float = 1.5,
        skew_warn: float = 2.0,
        skew_fail: float = 5.0,
        outlier_warn: float = 0.02,
        outlier_fail: float = 0.05,
        rare_category_threshold: float = 0.01,
        high_cardinality_threshold: float = 0.95,
    ) -> None:
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.skew_warn = skew_warn
        self.skew_fail = skew_fail
        self.outlier_warn = outlier_warn
        self.outlier_fail = outlier_fail
        self.rare_category_threshold = rare_category_threshold
        self.high_cardinality_threshold = high_cardinality_threshold

    def run(self, dataset: Any) -> CheckResult:
        df: pd.DataFrame = dataset.df
        n_rows = len(df)
        result = self._empty_result()

        if n_rows == 0:
            result.score = 0.0
            result.severity = CheckSeverity.FAIL
            result.add_issue("Dataset is empty.", severity=CheckSeverity.FAIL)
            return result

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["category"]).columns.tolist()
        categorical_cols += df.select_dtypes(include=["object"]).columns.tolist()

        total_penalty = 0.0
        outlier_summary: dict[str, dict[str, Any]] = {}
        skewness_summary: dict[str, float] = {}
        cardinality_summary: dict[str, dict[str, Any]] = {}

        # Numeric column checks
        for col in numeric_cols:
            series = df[col].dropna()
            n = len(series)
            if n < 4:
                continue

            # Z-score outliers
            z_scores = np.abs(stats.zscore(series, nan_policy="omit"))
            z_outliers = int((z_scores > self.zscore_threshold).sum())

            # IQR outliers
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                iqr_outliers = int(
                    (
                        (series < q1 - self.iqr_multiplier * iqr)
                        | (series > q3 + self.iqr_multiplier * iqr)
                    ).sum()
                )
            else:
                iqr_outliers = 0

            # Use the more conservative (higher) count
            n_outliers = max(z_outliers, iqr_outliers)
            outlier_rate = n_outliers / n_rows

            outlier_summary[col] = {
                "n_outliers": n_outliers,
                "outlier_rate": round(outlier_rate, 4),
                "z_score_outliers": z_outliers,
                "iqr_outliers": iqr_outliers,
                "q1": round(float(q1), 4),
                "q3": round(float(q3), 4),
                "iqr": round(float(iqr), 4),
            }

            if outlier_rate >= self.outlier_fail:
                result.add_issue(
                    f"Column '{col}' has {outlier_rate:.1%} outliers "
                    f"({n_outliers:,} rows exceed"
                    f" Z>{self.zscore_threshold} or IQR×{self.iqr_multiplier}).",
                    column=col,
                    severity=CheckSeverity.FAIL,
                    outlier_rate=outlier_rate,
                )
                total_penalty += outlier_rate
            elif outlier_rate >= self.outlier_warn:
                result.add_issue(
                    f"Column '{col}' has {outlier_rate:.1%} potential outliers.",
                    column=col,
                    severity=CheckSeverity.WARN,
                    outlier_rate=outlier_rate,
                )
                total_penalty += outlier_rate * 0.5

            # Skewness
            skew = float(series.skew())
            skewness_summary[col] = round(skew, 4)
            abs_skew = abs(skew)
            if abs_skew >= self.skew_fail:
                result.add_issue(
                    f"Column '{col}' is severely skewed (skewness={skew:.2f}). "
                    "Consider log/power transform.",
                    column=col,
                    severity=CheckSeverity.FAIL,
                    skewness=skew,
                )
                total_penalty += min(abs_skew / 20, 0.1)
            elif abs_skew >= self.skew_warn:
                result.add_issue(
                    f"Column '{col}' is moderately skewed (skewness={skew:.2f}).",
                    column=col,
                    severity=CheckSeverity.WARN,
                    skewness=skew,
                )
                total_penalty += min(abs_skew / 50, 0.05)

            # Near-constant column (std ≈ 0)
            if series.std() == 0:
                result.add_issue(
                    f"Column '{col}' has zero variance — constant after dropping nulls.",
                    column=col,
                    severity=CheckSeverity.WARN,
                )

        # ── Categorical column checks ──────────────────────────────────────
        for col in categorical_cols:
            series = df[col].dropna()
            n = len(series)
            if n == 0:
                continue

            n_unique = series.nunique()
            cardinality_ratio = n_unique / n
            value_counts = series.value_counts(normalize=True)

            cardinality_summary[col] = {
                "n_unique": int(n_unique),
                "cardinality_ratio": round(cardinality_ratio, 4),
                "top_category_rate": (
                    round(float(value_counts.iloc[0]), 4)
                    if len(value_counts) > 0
                    else 0.0
                ),
            }

            # High cardinality (likely free-text stored as category)
            if cardinality_ratio >= self.high_cardinality_threshold and n_unique > 10:
                result.add_issue(
                    f"Column '{col}' has very high cardinality "
                    f"({n_unique:,} unique / {n:,} non-null = {cardinality_ratio:.1%}). "
                    "Consider treating as text or ID.",
                    column=col,
                    severity=CheckSeverity.WARN,
                    cardinality_ratio=cardinality_ratio,
                )
                total_penalty += 0.03

            # Rare categories
            rare = value_counts[value_counts < self.rare_category_threshold]
            if len(rare) > 0:
                result.add_issue(
                    f"Column '{col}' has {len(rare)} rare categories "
                    f"(each <{self.rare_category_threshold:.0%} of data): "
                    f"{list(rare.index[:5])}{'...' if len(rare) > 5 else ''}.",
                    column=col,
                    severity=CheckSeverity.WARN,
                    rare_category_count=len(rare),
                )
                total_penalty += min(len(rare) * 0.005, 0.05)

        score = max(0.0, 1.0 - min(total_penalty, 1.0))
        result.score = round(score, 4)
        result.metrics = {
            "numeric_columns_checked": len(numeric_cols),
            "categorical_columns_checked": len(categorical_cols),
            "outlier_summary": outlier_summary,
            "skewness": skewness_summary,
            "cardinality": cardinality_summary,
        }
        return result