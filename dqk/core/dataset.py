"""
DQKDataset — the central object in DataQualityKit.

All quality checks, scoring, and reporting flow through this class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dqk.core.loader import infer_schema, load
from dqk.core.schema import DatasetSchema


class DQKDataset:
    """
    A wrapped dataset ready for quality analysis.

    Create via one of the class-method constructors::

        ds = DQKDataset.from_csv("data.csv")
        ds = DQKDataset.from_parquet("data.parquet")
        ds = DQKDataset.from_huggingface("imdb", split="train")
        ds = DQKDataset.from_dataframe(my_df)

    Then run checks::

        report = ds.run_checks()
        report.save("report.html")
    """

    def __init__(self, df: pd.DataFrame, schema: DatasetSchema) -> None:
        self._df = df
        self.schema = schema

    # Constructors
    @classmethod
    def from_csv(cls, path: str | Path, **kwargs: Any) -> DQKDataset:
        """Load a CSV file."""
        df, source, fmt = load(path, format="csv", **kwargs)
        return cls(df, infer_schema(df, source=source, fmt=fmt))

    @classmethod
    def from_json(cls, path: str | Path, **kwargs: Any) -> DQKDataset:
        """Load a JSON or JSONL file."""
        df, source, fmt = load(path, format="json", **kwargs)
        return cls(df, infer_schema(df, source=source, fmt=fmt))

    @classmethod
    def from_parquet(cls, path: str | Path, **kwargs: Any) -> DQKDataset:
        """Load a Parquet file."""
        df, source, fmt = load(path, format="parquet", **kwargs)
        return cls(df, infer_schema(df, source=source, fmt=fmt))

    @classmethod
    def from_huggingface(
        cls,
        dataset_id: str,
        split: str = "train",
        **kwargs: Any,
    ) -> DQKDataset:
        """Load a dataset from HuggingFace Hub."""
        df, source, fmt = load(dataset_id, format="hf", split=split, **kwargs)
        return cls(df, infer_schema(df, source=source, fmt=fmt))

    @classmethod
    def from_sql(
        cls,
        connection_string: str,
        query: str = "SELECT * FROM data",
        **kwargs: Any,
    ) -> DQKDataset:
        """Load from a SQL database via SQLAlchemy connection string."""
        df, source, fmt = load(
            connection_string, format="sql", sql_query=query, **kwargs
        )
        return cls(df, infer_schema(df, source=source, fmt=fmt))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, source: str = "dataframe") -> DQKDataset:
        """Wrap an existing pandas or polars DataFrame."""
        actual_df, src, fmt = load(df)
        return cls(actual_df, infer_schema(actual_df, source=source, fmt=fmt))

    # Core properties
    @property
    def df(self) -> pd.DataFrame:
        """The underlying pandas DataFrame (read-only view)."""
        return self._df

    @property
    def shape(self) -> tuple[int, int]:
        return self._df.shape

    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return (
            f"DQKDataset("
            f"rows={self.shape[0]}, "
            f"cols={self.shape[1]}, "
            f"source='{self.schema.source}', "
            f"format='{self.schema.format}')"
        )

    # Slicing helpers
    def sample(self, n: int = 5, random_state: int = 42) -> DQKDataset:
        """Return a DQKDataset with a random sample of rows."""
        sampled = self._df.sample(min(n, len(self._df)), random_state=random_state)
        return DQKDataset(sampled.reset_index(drop=True), self.schema)

    def select_columns(self, cols: list[str]) -> DQKDataset:
        """Return a DQKDataset restricted to the given columns."""
        sub = self._df[cols].copy()
        sub_schema_cols = [c for c in self.schema.columns if c.name in cols]
        from dqk.core.schema import DatasetSchema
        new_schema = DatasetSchema(
            n_rows=len(sub),
            n_cols=len(cols),
            columns=sub_schema_cols,
            source=self.schema.source,
            format=self.schema.format,
        )
        return DQKDataset(sub, new_schema)

    # Quality checks entry-point (wired in later phases)
    def run_checks(
        self,
        checks: list[str] | None = None,
        label_col: str | None = None,
    ) -> QualityReport:  # type: ignore[name-defined]
        """
        Run all (or a subset of) quality checks and return a QualityReport.

        Parameters
        ----------
        checks:
            List of check names to run. If None, runs all available checks.
        label_col:
            Column name to treat as the label, overriding schema inference.
        """
        from dqk.scoring.scorer import run_all_checks
        return run_all_checks(self, checks=checks, label_col=label_col)

    # Summary / display
    def summary(self) -> dict[str, Any]:
        """Return a human-readable summary dict of the dataset."""
        return {
            "source": self.schema.source,
            "format": self.schema.format,
            "n_rows": self.schema.n_rows,
            "n_cols": self.schema.n_cols,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype.value,
                    "role": c.role.value,
                    "missing_rate": c.missing_rate,
                    "n_unique": c.n_unique,
                }
                for c in self.schema.columns
            ],
        }

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        rows = "".join(
            f"<tr>"
            f"<td><code>{c.name}</code></td>"
            f"<td>{c.dtype.value}</td>"
            f"<td>{c.role.value}</td>"
            f"<td>{c.missing_rate:.1%}" if c.missing_rate is not None else "<td>—"
            f"</td>"
            f"<td>{c.n_unique}</td>"
            f"</tr>"
            for c in self.schema.columns
        )
        return f"""
        <div style="font-family: monospace; font-size: 13px;">
          <b>DQKDataset</b> — {self.schema.n_rows:,} rows × {self.schema.n_cols} cols
          &nbsp;|&nbsp; source: <code>{self.schema.source}</code>
          &nbsp;|&nbsp; format: <code>{self.schema.format}</code>
          <table border="1" cellpadding="4" cellspacing="0"
            style="border-collapse: collapse; margin-top: 6px;">
            <thead>
              <tr style="background: #f5f5f5;">
                <th>column</th><th>dtype</th><th>role</th>
                <th>missing%</th><th>n_unique</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """