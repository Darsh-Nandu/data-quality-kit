"""Text quality check: length, encoding, language consistency, and content heuristics."""

from __future__ import annotations

from typing import Any

import pandas as pd

from dqk.checks.base import BaseCheck, CheckResult, CheckSeverity
from dqk.core.schema import ColumnRole


class TextQualityCheck(BaseCheck):
    """
    Quality checks specific to natural-language text columns.

    - Empty / whitespace-only string detection
    - Extreme length outliers (too short, too long)
    - All-uppercase ratio (often signals noise)
    - Exact duplicate text detection within the column
    - Optional language consistency check (requires ``pip install langdetect``)

    Runs only on columns with role=TEXT or columns explicitly listed via ``text_columns``.
    """

    name = "text_quality"
    description = (
        "Text-specific checks: empty strings, length anomalies, "
        "all-caps noise, exact text duplicates, language consistency"
    )
    weight = 0.8

    def __init__(
        self,
        text_columns: list[str] | None = None,
        min_length: int = 3,
        max_length: int = 50_000,
        empty_fail_threshold: float = 0.10,
        empty_warn_threshold: float = 0.02,
        short_warn_threshold: float = 0.05,
        allcaps_warn_threshold: float = 0.10,
        dup_warn_threshold: float = 0.05,
        check_language: bool = False,
    ) -> None:
        self.text_columns = text_columns
        self.min_length = min_length
        self.max_length = max_length
        self.empty_fail_threshold = empty_fail_threshold
        self.empty_warn_threshold = empty_warn_threshold
        self.short_warn_threshold = short_warn_threshold
        self.allcaps_warn_threshold = allcaps_warn_threshold
        self.dup_warn_threshold = dup_warn_threshold
        self.check_language = check_language

    def run(self, dataset: Any) -> CheckResult:
        df: pd.DataFrame = dataset.df
        n_rows = len(df)
        result = self._empty_result()

        if n_rows == 0:
            result.score = 0.0
            result.severity = CheckSeverity.FAIL
            result.add_issue("Dataset is empty.", severity=CheckSeverity.FAIL)
            return result

        # Resolve which columns to check
        text_cols = self.text_columns or [
            c.name for c in dataset.schema.columns if c.role == ColumnRole.TEXT
        ]
        if not text_cols:
            result.description = "No text columns detected — skipping text quality checks."
            result.severity = CheckSeverity.SKIP
            return result

        total_penalty = 0.0
        col_metrics: dict[str, Any] = {}

        for col in text_cols:
            if col not in df.columns:
                continue

            series = df[col]
            non_null = series.dropna()
            str_series = non_null.astype(str)

            # Empty / whitespace-only
            empty_mask = str_series.str.strip() == ""
            n_empty = int(empty_mask.sum())
            empty_rate = n_empty / n_rows

            if empty_rate >= self.empty_fail_threshold:
                result.add_issue(
                    f"Column '{col}' has {empty_rate:.1%} empty/whitespace-only strings ({n_empty:,} rows).",
                    column=col,
                    severity=CheckSeverity.FAIL,
                    empty_rate=empty_rate,
                )
                total_penalty += empty_rate
            elif empty_rate >= self.empty_warn_threshold:
                result.add_issue(
                    f"Column '{col}' has {empty_rate:.1%} empty/whitespace-only strings.",
                    column=col,
                    severity=CheckSeverity.WARN,
                    empty_rate=empty_rate,
                )
                total_penalty += empty_rate * 0.5

            # Length analysis
            lengths = str_series.str.len()
            mean_len = float(lengths.mean())
            median_len = float(lengths.median())
            p95_len = float(lengths.quantile(0.95))
            p5_len = float(lengths.quantile(0.05))

            # Too short
            n_short = int((lengths < self.min_length).sum())
            short_rate = n_short / n_rows
            if short_rate >= self.short_warn_threshold:
                result.add_issue(
                    f"Column '{col}' has {short_rate:.1%} texts shorter than {self.min_length} chars.",
                    column=col,
                    severity=CheckSeverity.WARN,
                    short_rate=short_rate,
                )
                total_penalty += short_rate * 0.3

            # Too long
            n_long = int((lengths > self.max_length).sum())
            long_rate = n_long / n_rows
            if long_rate > 0.001:
                result.add_issue(
                    f"Column '{col}' has {long_rate:.1%} texts exceeding {self.max_length:,} chars.",
                    column=col,
                    severity=CheckSeverity.WARN,
                    long_rate=long_rate,
                )

            # All-uppercase noise
            alpha_mask = str_series.str.contains(r"[a-zA-Z]", regex=True, na=False)
            if alpha_mask.sum() > 0:
                allcaps_mask = str_series[alpha_mask].str.upper() == str_series[alpha_mask]
                allcaps_rate = int(allcaps_mask.sum()) / n_rows
            else:
                allcaps_rate = 0.0

            if allcaps_rate >= self.allcaps_warn_threshold:
                result.add_issue(
                    f"Column '{col}' has {allcaps_rate:.1%} all-uppercase texts — possible data noise.",
                    column=col,
                    severity=CheckSeverity.WARN,
                    allcaps_rate=allcaps_rate,
                )
                total_penalty += allcaps_rate * 0.2

            # Exact text duplicates
            n_dup_text = int(str_series.duplicated().sum())
            dup_rate = n_dup_text / n_rows
            if dup_rate >= self.dup_warn_threshold:
                result.add_issue(
                    f"Column '{col}' has {dup_rate:.1%} duplicate text values ({n_dup_text:,} rows).",
                    column=col,
                    severity=CheckSeverity.WARN,
                    dup_rate=dup_rate,
                )
                total_penalty += dup_rate * 0.15

            # Language consistency (optional)
            lang_distribution: dict[str, float] = {}
            if self.check_language:
                lang_distribution = self._detect_languages(str_series, result, col, n_rows)

            col_metrics[col] = {
                "mean_length": round(mean_len, 1),
                "median_length": round(median_len, 1),
                "p5_length": round(p5_len, 1),
                "p95_length": round(p95_len, 1),
                "n_empty": n_empty,
                "empty_rate": round(empty_rate, 4),
                "n_short": n_short,
                "short_rate": round(short_rate, 4),
                "n_long": n_long,
                "allcaps_rate": round(allcaps_rate, 4),
                "dup_rate": round(dup_rate, 4),
                "lang_distribution": lang_distribution,
            }

        score = max(0.0, 1.0 - min(total_penalty, 1.0))
        result.score = round(score, 4)
        result.metrics = {
            "text_columns_checked": text_cols,
            "column_metrics": col_metrics,
        }
        return result

    def _detect_languages(
        self,
        series: pd.Series,
        result: CheckResult,
        col: str,
        n_rows: int,
    ) -> dict[str, float]:
        try:
            from langdetect import detect, LangDetectException  # type: ignore[import]
        except ImportError:
            result.add_issue(
                "Language detection skipped — install 'langdetect': pip install langdetect",
                severity=CheckSeverity.SKIP,
            )
            return {}

        lang_counts: dict[str, int] = {}
        sample = series.dropna().sample(min(200, len(series)), random_state=42)
        for text in sample:
            try:
                lang = detect(str(text))
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            except LangDetectException:
                lang_counts["unknown"] = lang_counts.get("unknown", 0) + 1

        total = sum(lang_counts.values())
        lang_dist = {k: round(v / total, 3) for k, v in lang_counts.items()}

        # Warn if multiple languages detected (top lang < 90%)
        top_lang_rate = max(lang_dist.values()) if lang_dist else 1.0
        if top_lang_rate < 0.90 and len(lang_dist) > 1:
            top_lang = max(lang_dist, key=lang_dist.get)  # type: ignore[arg-type]
            result.add_issue(
                f"Column '{col}' appears multilingual — dominant language "
                f"'{top_lang}' covers only {top_lang_rate:.0%} of sampled texts.",
                column=col,
                severity=CheckSeverity.WARN,
                lang_distribution=lang_dist,
            )
        return lang_dist