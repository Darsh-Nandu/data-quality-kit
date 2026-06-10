"""Uniqueness check: exact duplicates and fuzzy near-duplicates via MinHash."""

from __future__ import annotations

from typing import Any

import pandas as pd

from dqk.checks.base import BaseCheck, CheckResult, CheckSeverity


class UniquenessCheck(BaseCheck):
    name = "uniqueness"
    description = "Exact row deduplication, key-column uniqueness, fuzzy text near-dedup"
    weight = 1.0

    def __init__(
        self,
        key_columns: list[str] | None = None,
        text_columns: list[str] | None = None,
        fuzzy: bool = False,
        fuzzy_threshold: float = 0.85,
        fail_threshold: float = 0.05,
        warn_threshold: float = 0.01,
    ) -> None:
        """
        Arguments:
        key_columns: Columns that must be unique (e.g. IDs). Inferred from schema if None.
        text_columns: Text columns to run fuzzy dedup on. Inferred from schema if None.
        fuzzy: Enable MinHash fuzzy deduplication (requires ``pip install datasketch``).
        fuzzy_threshold: Jaccard similarity above which two rows are considered near-duplicates.
        fail_threshold: Duplicate rate above which a FAIL issue is raised.
        warn_threshold: Duplicate rate above which a WARN issue is raised.
        """
        self.key_columns = key_columns
        self.text_columns = text_columns
        self.fuzzy = fuzzy
        self.fuzzy_threshold = fuzzy_threshold
        self.fail_threshold = fail_threshold
        self.warn_threshold = warn_threshold

    def run(self, dataset: Any) -> CheckResult:
        df: pd.DataFrame = dataset.df
        result = self._empty_result()
        n_rows = len(df)
        if n_rows == 0:
            result.score = 0.0
            result.severity = CheckSeverity.FAIL
            result.add_issue("Dataset is empty.", severity=CheckSeverity.FAIL)
            return result

        total_penalty = 0.0

        # 1. Exact duplicates
        n_exact_dups = int(df.duplicated().sum())
        exact_rate = n_exact_dups / n_rows
        if exact_rate >= self.fail_threshold:
            result.add_issue(
                f"{exact_rate:.1%} of rows are exact duplicates ({n_exact_dups:,} rows).",
                severity=CheckSeverity.FAIL,
                duplicate_count=n_exact_dups,
            )
            total_penalty += exact_rate
        elif exact_rate >= self.warn_threshold:
            result.add_issue(
                f"{exact_rate:.1%} of rows are exact duplicates ({n_exact_dups:,} rows).",
                severity=CheckSeverity.WARN,
                duplicate_count=n_exact_dups,
            )
            total_penalty += exact_rate * 0.5

        # 2. Key column uniqueness
        from dqk.core.schema import ColumnRole

        key_cols = self.key_columns or [
            c.name for c in dataset.schema.columns if c.role == ColumnRole.ID
        ]
        key_dup_rates: dict[str, float] = {}
        for col in key_cols:
            if col not in df.columns:
                continue
            n_key_dups = int(df[col].duplicated().sum())
            rate = n_key_dups / n_rows
            key_dup_rates[col] = round(rate, 4)
            if rate >= self.fail_threshold:
                result.add_issue(
                    f"Key column '{col}' has {rate:.1%} duplicate values ({n_key_dups:,}).",
                    column=col,
                    severity=CheckSeverity.FAIL,
                    duplicate_count=n_key_dups,
                )
                total_penalty += rate
            elif rate >= self.warn_threshold:
                result.add_issue(
                    f"Key column '{col}' has {rate:.1%} duplicate values.",
                    column=col,
                    severity=CheckSeverity.WARN,
                )
                total_penalty += rate * 0.5

        # 3. Fuzzy text deduplication
        fuzzy_rate = 0.0
        from dqk.core.schema import ColumnRole

        text_cols = self.text_columns or [
            c.name for c in dataset.schema.columns if c.role == ColumnRole.TEXT
        ]

        if self.fuzzy and text_cols:
            fuzzy_rate = self._fuzzy_dedup(df, text_cols, result)
            total_penalty += fuzzy_rate * 0.8

        score = max(0.0, 1.0 - min(total_penalty, 1.0))
        result.score = round(score, 4)
        result.metrics = {
            "exact_duplicate_rate": round(exact_rate, 4),
            "exact_duplicate_count": n_exact_dups,
            "key_duplicate_rates": key_dup_rates,
            "fuzzy_duplicate_rate": round(fuzzy_rate, 4),
        }
        return result

    def _fuzzy_dedup(
        self,
        df: pd.DataFrame,
        text_cols: list[str],
        result: CheckResult,
    ) -> float:
        """Return the fraction of rows that are near-duplicates in any text column."""
        try:
            from datasketch import MinHash, MinHashLSH  # type: ignore[import]
        except ImportError:
            result.add_issue(
                "Fuzzy deduplication skipped — install 'datasketch': pip install datasketch",
                severity=CheckSeverity.SKIP,
            )
            result.severity = CheckSeverity.PASS
            return 0.0

        n_rows = len(df)
        near_dup_indices: set[int] = set()

        for col in text_cols:
            if col not in df.columns:
                continue
            lsh = MinHashLSH(threshold=self.fuzzy_threshold, num_perm=128)
            minhashes: dict[int, Any] = {}

            for idx, val in df[col].dropna().items():
                tokens = set(str(val).lower().split())
                m = MinHash(num_perm=128)
                for t in tokens:
                    m.update(t.encode("utf-8"))
                try:
                    lsh.insert(str(idx), m)
                    minhashes[int(idx)] = m  # type: ignore[arg-type]
                except ValueError:
                    pass

            for idx, m in minhashes.items():
                neighbors = lsh.query(m)
                neighbors_int = [int(n) for n in neighbors if int(n) != idx]
                if neighbors_int:
                    near_dup_indices.add(idx)

        rate = len(near_dup_indices) / n_rows if n_rows > 0 else 0.0
        if rate >= self.fail_threshold:
            result.add_issue(
                f"{rate:.1%} of rows have near-duplicate text content.",
                severity=CheckSeverity.FAIL,
                near_duplicate_count=len(near_dup_indices),
            )
        elif rate >= self.warn_threshold:
            result.add_issue(
                f"{rate:.1%} of rows have near-duplicate text content.",
                severity=CheckSeverity.WARN,
            )
        return rate
