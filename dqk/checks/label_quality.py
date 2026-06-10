"""Label quality check: class imbalance, rare classes, label entropy, and coverage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dqk.checks.base import BaseCheck, CheckResult, CheckSeverity
from dqk.core.schema import ColumnRole


class LabelQualityCheck(BaseCheck):
    """
    Quality checks for classification / structured-prediction label columns.

    - Class imbalance ratio (majority / minority class count)
    - Rare class detection (classes with < ``rare_threshold`` of samples)
    - Label entropy (low entropy = near-constant label distribution)
    - Label coverage (# of distinct labels)
    - Missing label rate

    Runs on columns with role=LABEL or columns listed via ``label_columns``.
    """

    name = "label_quality"
    description = (
        "Class imbalance detection, rare-class flagging, "
        "label entropy analysis, and missing label rate"
    )
    weight = 1.3  # Higher weight — label quality directly impacts model training

    def __init__(
        self,
        label_columns: list[str] | None = None,
        imbalance_warn: float = 5.0,
        imbalance_fail: float = 20.0,
        rare_threshold: float = 0.01,
        missing_warn: float = 0.001,
        missing_fail: float = 0.05,
    ) -> None:
        """
        Parameters
        ----------
        label_columns:
            Columns to treat as labels. If None, auto-detected via schema role.
        imbalance_warn:
            Majority/minority class ratio above which a WARN is raised.
        imbalance_fail:
            Majority/minority class ratio above which a FAIL is raised.
        rare_threshold:
            Fraction below which a class is considered rare (WARN).
        missing_warn / missing_fail:
            Missing label rate thresholds.
        """
        self.label_columns = label_columns
        self.imbalance_warn = imbalance_warn
        self.imbalance_fail = imbalance_fail
        self.rare_threshold = rare_threshold
        self.missing_warn = missing_warn
        self.missing_fail = missing_fail

    def run(self, dataset: Any) -> CheckResult:
        df: pd.DataFrame = dataset.df
        n_rows = len(df)
        result = self._empty_result()

        if n_rows == 0:
            result.score = 0.0
            result.severity = CheckSeverity.FAIL
            result.add_issue("Dataset is empty.", severity=CheckSeverity.FAIL)
            return result

        label_cols = self.label_columns or [
            c.name for c in dataset.schema.columns if c.role == ColumnRole.LABEL
        ]

        if not label_cols:
            result.description = (
                "No label columns detected — skipping label quality checks. "
                "Name a column 'label', 'target', or 'class' to enable this check."
            )
            result.severity = CheckSeverity.SKIP
            return result

        total_penalty = 0.0
        col_metrics: dict[str, Any] = {}

        for col in label_cols:
            if col not in df.columns:
                continue

            series = df[col]

            # Missing labels
            n_missing = int(series.isna().sum())
            missing_rate = n_missing / n_rows

            if missing_rate >= self.missing_fail:
                result.add_issue(
                    f"Label column '{col}' has {missing_rate:.1%} missing values — "
                    "model training will be severely impacted.",
                    column=col,
                    severity=CheckSeverity.FAIL,
                    missing_rate=missing_rate,
                )
                total_penalty += missing_rate
            elif missing_rate >= self.missing_warn:
                result.add_issue(
                    f"Label column '{col}' has {missing_rate:.1%} missing values.",
                    column=col,
                    severity=CheckSeverity.WARN,
                    missing_rate=missing_rate,
                )
                total_penalty += missing_rate * 0.5

            non_null = series.dropna()
            n_valid = len(non_null)
            if n_valid == 0:
                continue

            value_counts = non_null.value_counts()
            n_classes = len(value_counts)
            class_dist = (value_counts / n_valid).to_dict()

            # Class imbalance
            majority_rate = float(value_counts.iloc[0]) / n_valid
            minority_rate = float(value_counts.iloc[-1]) / n_valid
            imbalance_ratio = majority_rate / minority_rate if minority_rate > 0 else float("inf")

            if imbalance_ratio >= self.imbalance_fail:
                result.add_issue(
                    f"Label column '{col}' has severe class imbalance: "
                    f"majority/minority ratio = {imbalance_ratio:.1f}x. "
                    f"Consider oversampling (SMOTE), class weights, or stratified sampling.",
                    column=col,
                    severity=CheckSeverity.FAIL,
                    imbalance_ratio=round(imbalance_ratio, 2),
                    majority_class=str(value_counts.index[0]),
                    majority_rate=round(majority_rate, 4),
                )
                total_penalty += min(imbalance_ratio / 100, 0.3)
            elif imbalance_ratio >= self.imbalance_warn:
                result.add_issue(
                    f"Label column '{col}' has moderate class imbalance "
                    f"(ratio = {imbalance_ratio:.1f}x).",
                    column=col,
                    severity=CheckSeverity.WARN,
                    imbalance_ratio=round(imbalance_ratio, 2),
                )
                total_penalty += min(imbalance_ratio / 200, 0.1)

            # Rare classes
            rare_classes = [
                str(cls) for cls, rate in class_dist.items()
                if rate < self.rare_threshold
            ]
            if rare_classes:
                result.add_issue(
                    f"Label column '{col}' has {len(rare_classes)} rare classes "
                    f"(each <{self.rare_threshold:.0%}): {rare_classes[:5]}"
                    f"{'...' if len(rare_classes) > 5 else ''}.",
                    column=col,
                    severity=CheckSeverity.WARN,
                    rare_classes=rare_classes,
                )
                total_penalty += min(len(rare_classes) * 0.02, 0.1)

            # Label entropy
            probs = np.array(list(class_dist.values()), dtype=float)
            entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
            max_entropy = np.log2(n_classes) if n_classes > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

            # Near-constant label (entropy near 0) in multi-class → suspicious
            if n_classes > 1 and normalized_entropy < 0.2:
                result.add_issue(
                    f"Label column '{col}' has very low entropy ({normalized_entropy:.2f}) — "
                    "almost all samples share one label. Dataset may not be representative.",
                    column=col,
                    severity=CheckSeverity.WARN,
                    normalized_entropy=round(normalized_entropy, 4),
                )
                total_penalty += 0.1

            col_metrics[col] = {
                "n_classes": n_classes,
                "class_distribution": {str(k): round(float(v), 4) for k, v in class_dist.items()},
                "majority_class": str(value_counts.index[0]),
                "majority_rate": round(majority_rate, 4),
                "minority_class": str(value_counts.index[-1]),
                "minority_rate": round(minority_rate, 4),
                "imbalance_ratio": (
                    round(imbalance_ratio, 2) if not np.isinf(imbalance_ratio) else None
                ),
                "rare_classes": rare_classes,
                "entropy_bits": round(entropy, 4),
                "normalized_entropy": round(normalized_entropy, 4),
                "missing_rate": round(missing_rate, 4),
            }

        score = max(0.0, 1.0 - min(total_penalty, 1.0))
        result.score = round(score, 4)
        result.metrics = {
            "label_columns_checked": label_cols,
            "column_metrics": col_metrics,
        }
        return result