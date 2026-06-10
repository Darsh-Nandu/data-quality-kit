"""
Dataset drift detection for DataQualityKit.

Compares a reference (training) dataset against a current (production/new) dataset
to detect statistical distribution shift.

Supported tests:
- Numeric columns: Population Stability Index (PSI), Kolmogorov–Smirnov test
- Categorical columns: Chi-squared test, Jensen–Shannon divergence
- Schema: column addition / removal / type change
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from dqk.core.dataset import DQKDataset


class DriftSeverity(str, Enum):
    NONE = "none"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class ColumnDriftResult:
    column: str
    dtype: str
    severity: DriftSeverity = DriftSeverity.NONE
    psi: float | None = None
    ks_statistic: float | None = None
    ks_pvalue: float | None = None
    js_divergence: float | None = None
    chi2_statistic: float | None = None
    chi2_pvalue: float | None = None
    note: str = ""


@dataclass
class DriftReport:
    """Full drift comparison report between reference and current datasets."""

    reference_source: str
    current_source: str
    n_ref_rows: int
    n_cur_rows: int
    column_results: list[ColumnDriftResult] = field(default_factory=list)
    schema_diff: dict[str, Any] = field(default_factory=dict)

    @property
    def drifted_columns(self) -> list[ColumnDriftResult]:
        return [r for r in self.column_results if r.severity != DriftSeverity.NONE]

    @property
    def overall_severity(self) -> DriftSeverity:
        if any(r.severity == DriftSeverity.SEVERE for r in self.column_results):
            return DriftSeverity.SEVERE
        if any(r.severity == DriftSeverity.MODERATE for r in self.column_results):
            return DriftSeverity.MODERATE
        return DriftSeverity.NONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_source": self.reference_source,
            "current_source": self.current_source,
            "n_ref_rows": self.n_ref_rows,
            "n_cur_rows": self.n_cur_rows,
            "overall_severity": self.overall_severity.value,
            "n_drifted_columns": len(self.drifted_columns),
            "schema_diff": self.schema_diff,
            "column_results": [
                {
                    "column": r.column,
                    "dtype": r.dtype,
                    "severity": r.severity.value,
                    "psi": r.psi,
                    "ks_statistic": r.ks_statistic,
                    "ks_pvalue": r.ks_pvalue,
                    "js_divergence": r.js_divergence,
                    "note": r.note,
                }
                for r in self.column_results
            ],
        }


# Numeric drift
def _psi(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """Population Stability Index (PSI). PSI < 0.1 = stable, 0.1-0.25 = moderate, >0.25 = severe."""
    ref_clean = reference.dropna()
    cur_clean = current.dropna()

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    # Build bins from reference distribution
    min_val = min(ref_clean.min(), cur_clean.min())
    max_val = max(ref_clean.max(), cur_clean.max())
    if min_val == max_val:
        return 0.0

    bins = np.linspace(min_val, max_val, n_bins + 1)
    ref_hist, _ = np.histogram(ref_clean, bins=bins)
    cur_hist, _ = np.histogram(cur_clean, bins=bins)

    # Add small epsilon to avoid log(0)
    eps = 1e-6
    ref_pct = (ref_hist + eps) / (len(ref_clean) + eps * n_bins)
    cur_pct = (cur_hist + eps) / (len(cur_clean) + eps * n_bins)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 6)


def _numeric_drift(col: str, ref: pd.Series, cur: pd.Series) -> ColumnDriftResult:
    psi_val = _psi(ref, cur)
    ks_stat, ks_p = stats.ks_2samp(ref.dropna(), cur.dropna())

    severity = DriftSeverity.NONE
    note = f"PSI={psi_val:.3f}, KS={ks_stat:.3f} (p={ks_p:.4f})"

    if psi_val >= 0.25 or ks_p < 0.001:
        severity = DriftSeverity.SEVERE
    elif psi_val >= 0.10 or ks_p < 0.05:
        severity = DriftSeverity.MODERATE

    return ColumnDriftResult(
        column=col,
        dtype="numeric",
        severity=severity,
        psi=psi_val,
        ks_statistic=round(float(ks_stat), 6),
        ks_pvalue=round(float(ks_p), 6),
        note=note,
    )


# Categorical drift
def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon divergence (0 = identical, 1 = maximally different)."""
    eps = 1e-10
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _categorical_drift(col: str, ref: pd.Series, cur: pd.Series) -> ColumnDriftResult:
    ref_clean = ref.dropna().astype(str)
    cur_clean = cur.dropna().astype(str)

    all_cats = sorted(set(ref_clean.unique()) | set(cur_clean.unique()))
    ref_counts = ref_clean.value_counts()
    cur_counts = cur_clean.value_counts()

    ref_vec = np.array([ref_counts.get(c, 0) for c in all_cats], dtype=float)
    cur_vec = np.array([cur_counts.get(c, 0) for c in all_cats], dtype=float)

    # JS divergence
    ref_pct = ref_vec / ref_vec.sum() if ref_vec.sum() > 0 else ref_vec
    cur_pct = cur_vec / cur_vec.sum() if cur_vec.sum() > 0 else cur_vec
    jsd = _js_divergence(ref_pct.copy(), cur_pct.copy())

    # Chi-squared test
    try:
        chi2, chi2_p = stats.chi2_contingency(np.array([ref_vec, cur_vec]))[:2]
    except Exception:
        chi2, chi2_p = 0.0, 1.0

    severity = DriftSeverity.NONE
    if jsd >= 0.30 or chi2_p < 0.001:
        severity = DriftSeverity.SEVERE
    elif jsd >= 0.10 or chi2_p < 0.05:
        severity = DriftSeverity.MODERATE

    note = f"JS-div={jsd:.3f}, χ²={chi2:.2f} (p={chi2_p:.4f})"
    return ColumnDriftResult(
        column=col,
        dtype="categorical",
        severity=severity,
        js_divergence=round(jsd, 6),
        chi2_statistic=round(float(chi2), 4),
        chi2_pvalue=round(float(chi2_p), 6),
        note=note,
    )


# Schema diff
def _schema_diff(ref_ds: DQKDataset, cur_ds: DQKDataset) -> dict[str, Any]:
    ref_cols = {c.name: c.dtype.value for c in ref_ds.schema.columns}
    cur_cols = {c.name: c.dtype.value for c in cur_ds.schema.columns}

    added = [c for c in cur_cols if c not in ref_cols]
    removed = [c for c in ref_cols if c not in cur_cols]
    type_changed = {
        c: {"from": ref_cols[c], "to": cur_cols[c]}
        for c in ref_cols
        if c in cur_cols and ref_cols[c] != cur_cols[c]
    }
    return {"added": added, "removed": removed, "type_changed": type_changed}


# Public API
def compare_datasets(
    reference: DQKDataset,
    current: DQKDataset,
    columns: list[str] | None = None,
) -> DriftReport:
    """
    Compare ``current`` against ``reference`` and return a :class:`DriftReport`.

    Parameters
    ----------
    reference:
        The baseline dataset (e.g. training split).
    current:
        The dataset to compare against the baseline (e.g. production / new data).
    columns:
        Subset of columns to compare. If None, all shared columns are checked.
    """
    ref_df = reference.df
    cur_df = current.df

    schema_diff = _schema_diff(reference, current)
    shared_cols = [c for c in ref_df.columns if c in cur_df.columns]
    if columns:
        shared_cols = [c for c in columns if c in shared_cols]

    col_results: list[ColumnDriftResult] = []
    for col in shared_cols:
        ref_s = ref_df[col]
        cur_s = cur_df[col]

        if pd.api.types.is_numeric_dtype(ref_s) and pd.api.types.is_numeric_dtype(cur_s):
            col_results.append(_numeric_drift(col, ref_s, cur_s))
        else:
            col_results.append(_categorical_drift(col, ref_s, cur_s))

    return DriftReport(
        reference_source=reference.schema.source,
        current_source=current.schema.source,
        n_ref_rows=len(ref_df),
        n_cur_rows=len(cur_df),
        column_results=col_results,
        schema_diff=schema_diff,
    )
