from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ColumnDtype(str, Enum):
    """Canonical column data types used across DQK."""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORY = "category"
    EMBEDDING = "embedding"  # list[float] columns
    UNKNOWN = "unknown"


class ColumnRole(str, Enum):
    """Semantic role of a column in an ML context."""

    FEATURE = "feature"
    LABEL = "label"
    ID = "id"
    TEXT = "text"
    TIMESTAMP = "timestamp"
    UNKNOWN = "unknown"


class ColumnMeta(BaseModel):
    """Metadata for a single column inferred during ingestion."""

    name: str
    dtype: ColumnDtype = ColumnDtype.UNKNOWN
    role: ColumnRole = ColumnRole.UNKNOWN
    nullable: bool = True
    n_unique: int | None = None
    n_missing: int | None = None
    sample_values: list[Any] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def missing_rate(self) -> float | None:
        """Fraction of missing values (requires n_missing and total to be set)."""
        total = self.extra.get("n_rows")
        if self.n_missing is None or total is None or total == 0:
            return None
        return self.n_missing / total


class DatasetSchema(BaseModel):
    """Full schema of a DQKDataset."""

    n_rows: int
    n_cols: int
    columns: list[ColumnMeta]
    source: str = "unknown"
    format: str = "unknown"
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _propagate_n_rows(self) -> DatasetSchema:
        for col in self.columns:
            col.extra.setdefault("n_rows", self.n_rows)
        return self

    def column(self, name: str) -> ColumnMeta:
        """Look up a column by name."""
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"Column '{name}' not found in schema.")

    @property
    def label_columns(self) -> list[ColumnMeta]:
        return [c for c in self.columns if c.role == ColumnRole.LABEL]

    @property
    def feature_columns(self) -> list[ColumnMeta]:
        return [c for c in self.columns if c.role == ColumnRole.FEATURE]

    @property
    def text_columns(self) -> list[ColumnMeta]:
        return [c for c in self.columns if c.role == ColumnRole.TEXT]
