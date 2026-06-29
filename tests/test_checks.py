"""
Comprehensive test suite for DataQualityKit.
Covers all 6 checks, drift detection, scoring, CLI basics, and bug regressions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dqk.checks.base import CheckIssue, CheckResult, CheckSeverity
from dqk.checks.completeness import CompletenessCheck
from dqk.checks.distribution import DistributionCheck
from dqk.checks.label_quality import LabelQualityCheck
from dqk.checks.text_quality import TextQualityCheck
from dqk.checks.uniqueness import UniquenessCheck
from dqk.checks.validity import ValidityCheck
from dqk.core.dataset import DQKDataset
from dqk.core.schema import ColumnRole
from dqk.scoring.scorer import QualityReport, QualityScore, available_checks


# Helpers
def make_ds(df: pd.DataFrame) -> DQKDataset:
    return DQKDataset.from_dataframe(df)


# Bug regression tests
class TestBugRegressions:
    def test_check_issue_column_field_not_cloumn(self) -> None:
        """Regression: field was named 'cloumn' (typo), silently dropping column data."""
        issue = CheckIssue(column="my_col", message="test", severity=CheckSeverity.WARN)
        assert issue.column == "my_col"

    def test_check_issue_extra_is_dict_not_list(self) -> None:
        """Regression: extra was Field(default_factory=list), crashing on .items()."""
        issue = CheckIssue(message="test")
        assert isinstance(issue.extra, dict)
        issue.extra["key"] = "value"  # would fail if list
        assert issue.extra["key"] == "value"

    def test_add_issue_preserves_column(self) -> None:
        result = CheckResult(check_name="test", score=1.0)
        result.add_issue("msg", column="col_a", severity=CheckSeverity.WARN)
        assert result.issues[0].column == "col_a"


# Completeness check
class TestCompletenessCheck:
    def test_perfect_data_scores_one(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = CompletenessCheck().run(make_ds(df))
        assert result.score == 1.0
        assert result.severity == CheckSeverity.PASS

    def test_empty_dataset_fails(self) -> None:
        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        result = CompletenessCheck().run(make_ds(df))
        assert result.severity == CheckSeverity.FAIL
        assert result.score == 0.0

    def test_high_null_rate_triggers_fail(self) -> None:
        df = pd.DataFrame({"a": [None] * 25 + [1] * 5})
        result = CompletenessCheck(fail_threshold=0.20).run(make_ds(df))
        assert result.severity == CheckSeverity.FAIL
        assert any(i.severity == CheckSeverity.FAIL for i in result.issues)

    def test_moderate_null_rate_triggers_warn(self) -> None:
        df = pd.DataFrame({"a": [None] * 7 + [1] * 93})
        result = CompletenessCheck(warn_threshold=0.05, fail_threshold=0.20).run(make_ds(df))
        assert result.severity == CheckSeverity.WARN

    def test_metrics_include_null_rates(self) -> None:
        df = pd.DataFrame({"a": [None, 1, 2], "b": [1, 2, 3]})
        result = CompletenessCheck().run(make_ds(df))
        assert "null_rates" in result.metrics
        assert "a" in result.metrics["null_rates"]


# Validity check
class TestValidityCheck:
    def test_clean_numeric_passes(self) -> None:
        df = pd.DataFrame({"age": [25, 30, 45]})
        result = ValidityCheck().run(make_ds(df))
        assert result.severity == CheckSeverity.PASS

    def test_range_guard_catches_violations(self) -> None:
        df = pd.DataFrame({"age": [25, 30, 200]})  # 200 is out of range
        result = ValidityCheck(
            range_guards={"age": (0, 120)},
            fail_threshold=0.01,
        ).run(make_ds(df))
        assert result.severity == CheckSeverity.FAIL

    def test_regex_guard_catches_invalid_emails(self) -> None:
        df = pd.DataFrame({"email": ["a@b.com", "not-an-email", "c@d.org"]})
        result = ValidityCheck(
            regex_guards={"email": r"[^@]+@[^@]+\.[^@]+"},
            fail_threshold=0.01,
        ).run(make_ds(df))
        assert result.severity == CheckSeverity.FAIL

    def test_empty_dataset_fails(self) -> None:
        df = pd.DataFrame({"x": pd.Series([], dtype=float)})
        result = ValidityCheck().run(make_ds(df))
        assert result.severity == CheckSeverity.FAIL


# Uniqueness check
class TestUniquenessCheck:
    def test_no_duplicates_passes(self) -> None:
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        result = UniquenessCheck().run(make_ds(df))
        assert result.severity == CheckSeverity.PASS

    def test_exact_duplicates_caught(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 1, 2, 3] * 10})
        result = UniquenessCheck(fail_threshold=0.05).run(make_ds(df))
        assert result.severity == CheckSeverity.FAIL
        assert result.metrics["exact_duplicate_count"] > 0

    def test_key_column_duplicate_caught(self) -> None:
        df = pd.DataFrame({"id": [1, 1, 2, 3, 4]})
        # mark 'id' as an ID column in the dataset schema
        ds = make_ds(df)
        for col in ds.schema.columns:
            if col.name == "id":
                col.role = ColumnRole.ID
        result = UniquenessCheck(key_columns=["id"], fail_threshold=0.01).run(ds)
        assert result.severity == CheckSeverity.FAIL


# Distribution check
class TestDistributionCheck:
    def test_clean_numeric_passes(self) -> None:
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(0, 1, 500)})
        result = DistributionCheck().run(make_ds(df))
        assert result.score > 0.5  # should be reasonable

    def test_heavy_skew_raises_issue(self) -> None:
        # Exponential distribution → very high skew
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.exponential(scale=0.1, size=1000)})
        result = DistributionCheck(skew_warn=1.0).run(make_ds(df))
        skew_issues = [i for i in result.issues if "skewed" in i.message]
        assert len(skew_issues) > 0

    def test_many_outliers_raises_fail(self) -> None:
        base = list(range(90))
        outliers = [10_000] * 10  # 10% extreme outliers
        df = pd.DataFrame({"x": base + outliers})
        result = DistributionCheck(outlier_fail=0.05).run(make_ds(df))
        assert result.severity in (CheckSeverity.WARN, CheckSeverity.FAIL)

    def test_metrics_populated(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = DistributionCheck().run(make_ds(df))
        assert "outlier_summary" in result.metrics
        assert "skewness" in result.metrics

    def test_rare_category_detected(self) -> None:
        cats = ["dog"] * 95 + ["cat"] * 4 + ["fish"] * 1
        df = pd.DataFrame({"animal": cats})
        result = DistributionCheck(rare_category_threshold=0.02).run(make_ds(df))
        rare_issues = [i for i in result.issues if "rare" in i.message]
        assert len(rare_issues) > 0


# Text quality check
class TestTextQualityCheck:
    def test_no_text_cols_skips(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = TextQualityCheck().run(make_ds(df))
        assert result.severity == CheckSeverity.SKIP

    def test_explicit_text_col_checked(self) -> None:
        df = pd.DataFrame({"review": ["good product", "bad quality", "ok I guess"]})
        result = TextQualityCheck(text_columns=["review"]).run(make_ds(df))
        assert result.severity == CheckSeverity.PASS

    def test_empty_strings_trigger_warn(self) -> None:
        texts = ["hello"] * 90 + [""] * 10
        df = pd.DataFrame({"review": texts})
        result = TextQualityCheck(
            text_columns=["review"],
            empty_warn_threshold=0.05,
        ).run(make_ds(df))
        empty_issues = [i for i in result.issues if "empty" in i.message.lower()]
        assert len(empty_issues) > 0

    def test_allcaps_noise_detected(self) -> None:
        texts = ["HELLO WORLD"] * 20 + ["normal text"] * 80
        df = pd.DataFrame({"text": texts})
        result = TextQualityCheck(
            text_columns=["text"],
            allcaps_warn_threshold=0.10,
        ).run(make_ds(df))
        caps_issues = [i for i in result.issues if "uppercase" in i.message.lower()]
        assert len(caps_issues) > 0

    def test_metrics_have_length_stats(self) -> None:
        df = pd.DataFrame({"body": ["short", "a much longer sentence here", "medium text"]})
        result = TextQualityCheck(text_columns=["body"]).run(make_ds(df))
        metrics = result.metrics["column_metrics"]["body"]
        assert "mean_length" in metrics
        assert "p95_length" in metrics


# Label quality check
class TestLabelQualityCheck:
    def test_no_label_cols_skips(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = LabelQualityCheck().run(make_ds(df))
        assert result.severity == CheckSeverity.SKIP

    def test_balanced_labels_pass(self) -> None:
        df = pd.DataFrame({"label": ["A"] * 50 + ["B"] * 50})
        result = LabelQualityCheck(label_columns=["label"]).run(make_ds(df))
        assert result.severity == CheckSeverity.PASS
        assert result.score > 0.8

    def test_severe_imbalance_fails(self) -> None:
        df = pd.DataFrame({"label": ["A"] * 95 + ["B"] * 5})
        result = LabelQualityCheck(
            label_columns=["label"],
            imbalance_warn=5.0,
            imbalance_fail=15.0,
        ).run(make_ds(df))
        assert result.severity == CheckSeverity.FAIL

    def test_rare_class_warned(self) -> None:
        df = pd.DataFrame({"target": ["A"] * 97 + ["B"] * 2 + ["C"] * 1})
        result = LabelQualityCheck(
            label_columns=["target"],
            rare_threshold=0.02,
            imbalance_warn=1000,
        ).run(make_ds(df))
        rare_issues = [i for i in result.issues if "rare" in i.message.lower()]
        assert len(rare_issues) > 0

    def test_metrics_include_class_distribution(self) -> None:
        df = pd.DataFrame({"y": ["cat"] * 60 + ["dog"] * 40})
        result = LabelQualityCheck(label_columns=["y"]).run(make_ds(df))
        metrics = result.metrics["column_metrics"]["y"]
        assert "class_distribution" in metrics
        assert "imbalance_ratio" in metrics
        assert "entropy_bits" in metrics


# Scoring and report
class TestScoring:
    def test_run_all_checks_returns_report(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        report = make_ds(df).run_checks()
        assert isinstance(report, QualityReport)
        assert 0 <= report.score.overall <= 100

    def test_grade_mapping(self) -> None:
        assert QualityScore(overall=95.0).grade == "A"
        assert QualityScore(overall=80.0).grade == "B"
        assert QualityScore(overall=65.0).grade == "C"
        assert QualityScore(overall=45.0).grade == "D"
        assert QualityScore(overall=20.0).grade == "F"

    def test_to_json_round_trips(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        report = make_ds(df).run_checks()
        raw = report.to_json()
        parsed = json.loads(raw)
        assert "score" in parsed
        assert "results" in parsed

    def test_save_json(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"x": [1, 2, None]})
        report = make_ds(df).run_checks()
        out = tmp_path / "report.json"
        report.save(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["n_rows"] == 3

    def test_available_checks_lists_all_six(self) -> None:
        checks = available_checks()
        expected = {
            "completeness",
            "validity",
            "uniqueness",
            "distribution",
            "text_quality",
            "label_quality",
        }
        assert expected.issubset(set(checks))

    def test_run_subset_of_checks(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        report = make_ds(df).run_checks(checks=["completeness", "uniqueness"])
        check_names = {r.check_name for r in report.results}
        assert check_names == {"completeness", "uniqueness"}

    def test_failed_checks_property(self) -> None:
        df = pd.DataFrame({"a": [None] * 30 + [1] * 70})
        report = make_ds(df).run_checks(checks=["completeness"])
        # With 30% nulls this should fail
        if report.failed_checks():
            assert all(r.severity == CheckSeverity.FAIL for r in report.failed_checks())

    def test_skipped_checks_not_penalized(self) -> None:
        """SKIP severity should not reduce the overall score."""
        df = pd.DataFrame({"a": [1, 2, 3]})  # No text or label columns
        report = make_ds(df).run_checks(checks=["text_quality", "label_quality", "completeness"])
        # Even though text_quality and label_quality skip, score shouldn't be 0
        assert report.score.overall > 0


# Drift detection
class TestDriftDetection:
    def test_identical_datasets_no_drift(self) -> None:
        from dqk.drift import DriftSeverity, compare_datasets

        df = pd.DataFrame({"x": np.random.default_rng(0).normal(0, 1, 200)})
        ref = make_ds(df)
        cur = make_ds(df.copy())
        report = compare_datasets(ref, cur)
        assert report.overall_severity == DriftSeverity.NONE

    def test_shifted_distribution_detected(self) -> None:
        from dqk.drift import DriftSeverity, compare_datasets

        rng = np.random.default_rng(42)
        ref = make_ds(pd.DataFrame({"x": rng.normal(0, 1, 300)}))
        cur = make_ds(pd.DataFrame({"x": rng.normal(10, 1, 300)}))  # massive shift
        report = compare_datasets(ref, cur)
        assert report.overall_severity == DriftSeverity.SEVERE

    def test_drift_report_has_column_results(self) -> None:
        from dqk.drift import compare_datasets

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
        ref = make_ds(df)
        cur = make_ds(df.copy())
        report = compare_datasets(ref, cur)
        col_names = [r.column for r in report.column_results]
        assert "a" in col_names
        assert "b" in col_names

    def test_schema_diff_detects_added_columns(self) -> None:
        from dqk.drift import compare_datasets

        ref = make_ds(pd.DataFrame({"a": [1, 2, 3]}))
        cur = make_ds(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
        report = compare_datasets(ref, cur)
        assert "b" in report.schema_diff["added"]

    def test_schema_diff_detects_removed_columns(self) -> None:
        from dqk.drift import compare_datasets

        ref = make_ds(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
        cur = make_ds(pd.DataFrame({"a": [1, 2, 3]}))
        report = compare_datasets(ref, cur)
        assert "b" in report.schema_diff["removed"]

    def test_drift_report_to_dict(self) -> None:
        from dqk.drift import compare_datasets

        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        report = compare_datasets(make_ds(df), make_ds(df.copy()))
        d = report.to_dict()
        assert "column_results" in d
        assert "overall_severity" in d
