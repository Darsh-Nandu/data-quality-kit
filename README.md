# 🧪 DataQualityKit

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](http://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
[![Tests](https://img.shields.io/badge/tests-pytest-success)](https://docs.pytest.org/)

> **Dataset quality testing for ML teams** — profile, score, and gate your data in one command.

DataQualityKit (`dqk`) is a lightweight, extensible Python library and CLI for running structured quality checks on tabular and text datasets before they reach your ML pipelines. Load from CSV, JSON, Parquet, HuggingFace Hub, SQL, or an existing DataFrame; run completeness, validity, and uniqueness checks; and get back a scored, graded report — in seconds.

---

## ✨ Features

- **Multi-source ingestion** — CSV, JSON/JSONL, Parquet, HuggingFace Hub, SQLAlchemy (any DB), pandas & polars DataFrames, all via a single unified API.
- **Auto schema inference** — column dtypes, semantic roles (feature / label / text / ID / timestamp), missing rates, and unique counts are inferred automatically.
- **Built-in quality checks:**
  - **Completeness** — per-column null rates, full-row completeness, MNAR (missing-not-at-random) pattern detection via missingness correlation.
  - **Validity** — type conformance, constant-column detection, configurable range guards and regex pattern guards.
  - **Uniqueness** — exact row deduplication, key-column uniqueness, and optional fuzzy near-dedup via MinHash LSH.
- **Weighted quality score** — an overall 0–100 score (graded A–F) aggregated from all checks.
- **Rich CLI** — colour-coded terminal output with per-check tables, issue lists, and a `--fail-under` CI gate.
- **Multiple report formats** — save results as JSON or a standalone HTML report.
- **Jupyter-friendly** — `DQKDataset` and `QualityReport` both expose `_repr_html_` for inline notebook display.
- **CI-ready** — exit code 1 when score drops below a configurable threshold; pairs with pre-commit hooks out of the box.

---

## 📦 Installation

```bash
pip install data-quality-kit
```

**Optional extras:**

```bash
# HuggingFace Hub support
pip install datasets

# SQL support (PostgreSQL, MySQL, SQLite, …)
pip install sqlalchemy

# Fuzzy deduplication via MinHash
pip install datasketch

# Polars interop
pip install polars
```

---

## 🚀 Quickstart

### Python API

```python
from dqk.core.dataset import DQKDataset

# Load from any source
ds = DQKDataset.from_csv("data/train.csv")
ds = DQKDataset.from_parquet("data/train.parquet")
ds = DQKDataset.from_huggingface("imdb", split="train")
ds = DQKDataset.from_sql("postgresql://user:pass@localhost/db", query="SELECT * FROM events")
ds = DQKDataset.from_dataframe(my_pandas_df)

# Inspect schema
print(ds)
# DQKDataset(rows=25_000, cols=8, source='data/train.csv', format='csv')

print(ds.summary())

# Run all checks
report = ds.run_checks()

print(f"Score: {report.score.overall:.1f}/100 ({report.score.grade})")
print(f"Issues found: {report.n_issues}")

# Run a subset of checks
report = ds.run_checks(checks=["completeness", "uniqueness"])

# Save the report
report.save("report.html")   # standalone HTML
report.save("report.json")   # machine-readable JSON
```

### CLI

```bash
# Run all checks on a local CSV
dqk check data/train.csv

# Force format, select checks, save report
dqk check data/train.parquet --format parquet --checks completeness,validity --output report.html

# Gate a CI pipeline — exit 1 if score < 80
dqk check data/train.csv --fail-under 80

# Load from HuggingFace Hub
dqk check imdb --split train

# Inspect inferred schema
dqk schema data/train.csv
```

**Example CLI output:**

```
Loaded: 25,000 rows × 8 cols (csv)

Quality Score: 84.3/100  (B)  WARN

┌─────────────┬────────┬──────────┬────────┐
│ Check       │  Score │ Severity │ Issues │
├─────────────┼────────┼──────────┼────────┤
│ completeness│  0.921 │  warn    │      3 │
│ validity    │  0.990 │  pass    │      0 │
│ uniqueness  │  0.800 │  fail    │      1 │
└─────────────┴────────┴──────────┴────────┘

Issues (4 total):
  ⚠ (age) Column 'age' has 8.4% missing values (threshold: 5%)
  ⚠ (income) Column 'income' has 6.1% missing values (threshold: 5%)
  ⚠ Columns 'age' and 'income' have correlated missingness (r=0.81) — possible MNAR pattern.
  ✗ 7.2% of rows are exact duplicates (1,800 rows).
```

---

## 📐 Checks Reference

### Completeness

Detects missing data at column and row level, and flags correlated missingness patterns (MNAR).

| Parameter | Default | Description |
|---|---|---|
| `warn_threshold` | `0.05` | Null rate above which a WARN is raised |
| `fail_threshold` | `0.20` | Null rate above which a FAIL is raised |

**Metrics returned:** `null_rates`, `overall_null_rate`, `complete_row_rate`, `missing_pattern_correlations`

### Validity

Checks that column values conform to their inferred types, respect optional range bounds, and match optional regex patterns. Also flags constant (zero-variance) columns.

| Parameter | Default | Description |
|---|---|---|
| `range_guards` | `{}` | `{col: (min, max)}` — values outside the range are violations |
| `regex_guards` | `{}` | `{col: pattern}` — non-matching non-null values are violations |
| `warn_threshold` | `0.001` | Violation rate above which a WARN is raised |
| `fail_threshold` | `0.01` | Violation rate above which a FAIL is raised |

```python
from dqk.checks.validity import ValidityCheck

check = ValidityCheck(
    range_guards={"age": (0, 120), "score": (0.0, 1.0)},
    regex_guards={"email": r"^[\w.+-]+@[\w-]+\.\w+$"},
)
```

### Uniqueness

Detects exact duplicate rows, key-column uniqueness violations, and (optionally) near-duplicate text via MinHash LSH.

| Parameter | Default | Description |
|---|---|---|
| `key_columns` | `None` | Columns that must be unique; inferred from schema (ID role) if `None` |
| `text_columns` | `None` | Text columns for fuzzy dedup; inferred from schema if `None` |
| `fuzzy` | `False` | Enable MinHash near-dedup (requires `datasketch`) |
| `fuzzy_threshold` | `0.85` | Jaccard similarity above which two rows are near-duplicates |
| `fail_threshold` | `0.05` | Duplicate rate above which a FAIL is raised |

---

## 🗂️ Project Structure

```
data-quality-kit/
├── dqk/
│   ├── checks/
│   │   ├── base.py           # BaseCheck, CheckResult, CheckIssue, CheckSeverity
│   │   ├── completeness.py   # CompletenessCheck
│   │   ├── validity.py       # ValidityCheck
│   │   └── uniqueness.py     # UniquenessCheck
│   ├── core/
│   │   ├── dataset.py        # DQKDataset — central user-facing object
│   │   ├── loader.py         # Multi-source ingestion engine + schema inference
│   │   └── schema.py         # DatasetSchema, ColumnMeta, ColumnDtype, ColumnRole
│   └── scoring/
│       └── scorer.py         # QualityScore, QualityReport, run_all_checks()
├── cli.py                    # Typer CLI (dqk check / dqk schema)
├── tests/
│   ├── conftest.py
│   └── test_dataset.py
├── examples/
│   ├── sample.csv
│   └── report.json
├── requirements.txt
├── pyproject.toml
└── .pre-commit-config.yaml
```

---

## 🔌 Extending DQK

Custom checks subclass `BaseCheck` and implement a single `run` method:

```python
from dqk.checks.base import BaseCheck, CheckResult, CheckSeverity

class LabelBalanceCheck(BaseCheck):
    name = "label_balance"
    description = "Check class distribution in the label column"
    weight = 1.0

    def run(self, dataset) -> CheckResult:
        result = self._empty_result()
        label_cols = dataset.schema.label_columns
        if not label_cols:
            result.severity = CheckSeverity.SKIP
            return result

        col = label_cols[0].name
        counts = dataset.df[col].value_counts(normalize=True)
        imbalance = counts.max() - counts.min()

        if imbalance > 0.5:
            result.add_issue(
                f"Label imbalance detected (max skew: {imbalance:.1%}).",
                severity=CheckSeverity.WARN,
            )

        result.score = round(1.0 - imbalance, 4)
        return result
```

---

## 🔗 Scoring

Each check returns a `score` in `[0, 1]`. Checks are combined into an overall score using a weighted average:

| Check | Weight |
|---|---|
| Completeness | 1.5 |
| Validity | 1.2 |
| Uniqueness | 1.0 |

The overall score is scaled to `[0, 100]` and assigned a letter grade:

| Grade | Score range |
|---|---|
| **A** | ≥ 90 |
| **B** | ≥ 75 |
| **C** | ≥ 60 |
| **D** | ≥ 40 |
| **F** | < 40 |

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository and create a feature branch.
2. Install pre-commit hooks: `pre-commit install`
3. Make your changes — the hooks enforce `ruff` formatting and `mypy` type checking.
4. Add or update tests under `tests/`.
5. Open a pull request with a clear description of the change.

---

## 📄 License

MIT © [Darsh-Nandu](https://github.com/Darsh-Nandu)