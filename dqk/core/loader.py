"""
Ingestion engine for DataQualityKit.

Supports: CSV, JSON, JSONL, Parquet, HuggingFace Hub, SQLAlchemy, pandas/polars DataFrames.
All sources resolve to a canonical pandas DataFrame + an inferred DatasetSchema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from dqk.core.schema import ColumnDtype, ColumnMeta, ColumnRole, DatasetSchema

if TYPE_CHECKING:
    pass

# Dtype inference

_DTYPE_MAP: dict[str, ColumnDtype] = {
    "int8": ColumnDtype.INTEGER,
    "int16": ColumnDtype.INTEGER,
    "int32": ColumnDtype.INTEGER,
    "int64": ColumnDtype.INTEGER,
    "uint8": ColumnDtype.INTEGER,
    "uint16": ColumnDtype.INTEGER,
    "uint32": ColumnDtype.INTEGER,
    "uint64": ColumnDtype.INTEGER,
    "float16": ColumnDtype.FLOAT,
    "float32": ColumnDtype.FLOAT,
    "float64": ColumnDtype.FLOAT,
    "bool": ColumnDtype.BOOLEAN,
    "object": ColumnDtype.STRING,
    "string": ColumnDtype.STRING,
    "category": ColumnDtype.CATEGORY,
}

_TIMESTAMP_NAMES = re.compile(
    r"(date|time|timestamp|created_at|updated_at|dt|day|month|year)",
    re.IGNORECASE,
)
_LABEL_NAMES = re.compile(
    r"^(label|target|class|y|output|ground_truth|answer)$",
    re.IGNORECASE,
)
_ID_NAMES = re.compile(
    r"^(id|uuid|key|index|idx|row_id|record_id)$",
    re.IGNORECASE,
)
_TEXT_NAMES = re.compile(
    r"(text|sentence|utterance|content|body|description|review|comment|message|prompt|response)",
    re.IGNORECASE,
)


def _infer_dtype(series: pd.Series) -> ColumnDtype:
    """Map a pandas dtype + heuristics to a ColumnDtype."""
    dtype_str = str(series.dtype)
    if dtype_str in _DTYPE_MAP:
        return _DTYPE_MAP[dtype_str]
    if "datetime" in dtype_str:
        return ColumnDtype.DATETIME
    # Check if object column holds lists (embedding-like)
    if dtype_str == "object":
        sample = series.dropna().head(10)
        if len(sample) > 0 and all(isinstance(v, (list, np.ndarray)) for v in sample):
            return ColumnDtype.EMBEDDING
    return ColumnDtype.UNKNOWN


def _infer_role(col: str, dtype: ColumnDtype) -> ColumnRole:
    """Guess the semantic role of a column from its name and dtype."""
    if _ID_NAMES.match(col):
        return ColumnRole.ID
    if _LABEL_NAMES.match(col):
        return ColumnRole.LABEL
    if _TIMESTAMP_NAMES.search(col) or dtype == ColumnDtype.DATETIME:
        return ColumnRole.TIMESTAMP
    if _TEXT_NAMES.search(col) and dtype in (ColumnDtype.STRING, ColumnDtype.UNKNOWN):
        return ColumnRole.TEXT
    if dtype in (ColumnDtype.INTEGER, ColumnDtype.FLOAT, ColumnDtype.CATEGORY):
        return ColumnRole.FEATURE
    return ColumnRole.UNKNOWN


def _build_column_meta(series: pd.Series, n_rows: int) -> ColumnMeta:
    dtype = _infer_dtype(series)
    role = _infer_role(series.name, dtype)
    n_missing = int(series.isna().sum())
    try:
        n_unique = int(series.nunique(dropna=True))
    except TypeError:
        n_unique = None

    sample = series.dropna().head(5).tolist()
    # Keep samples JSON-serialisable
    safe_sample: list[Any] = []
    for v in sample:
        try:
            json.dumps(v)
            safe_sample.append(v)
        except (TypeError, ValueError):
            safe_sample.append(str(v))

    return ColumnMeta(
        name=str(series.name),
        dtype=dtype,
        role=role,
        nullable=n_missing > 0,
        n_unique=n_unique,
        n_missing=n_missing,
        sample_values=safe_sample,
        extra={"n_rows": n_rows},
    )


def infer_schema(df: pd.DataFrame, source: str = "unknown", fmt: str = "unknown") -> DatasetSchema:
    """Build a DatasetSchema from a pandas DataFrame."""
    n_rows, n_cols = df.shape
    columns = [_build_column_meta(df[col], n_rows) for col in df.columns]
    return DatasetSchema(
        n_rows=n_rows,
        n_cols=n_cols,
        columns=columns,
        source=source,
        format=fmt,
    )


# Format loaders


def _load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def _load_json(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    # JSONL vs JSON array
    with path.open() as f:
        first = f.read(1)
    if first == "[":
        return pd.read_json(path, **kwargs)
    return pd.read_json(path, lines=True, **kwargs)


def _load_parquet(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_parquet(path, **kwargs)


def _load_huggingface(dataset_id: str, split: str = "train", **kwargs: Any) -> pd.DataFrame:
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "Install the 'datasets' package to load from HuggingFace Hub: pip install datasets"
        ) from e
    ds = load_dataset(dataset_id, split=split, **kwargs)
    return ds.to_pandas()


def _load_sql(connection_string: str, query: str, **kwargs: Any) -> pd.DataFrame:
    try:
        from sqlalchemy import create_engine, text  # type: ignore[import]
    except ImportError as e:
        raise ImportError("Install sqlalchemy to load from SQL: pip install sqlalchemy") from e
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, **kwargs)


def _load_polars(path_or_df: Any, **kwargs: Any) -> pd.DataFrame:
    """Accept a polars DataFrame or path and convert to pandas."""
    try:
        import polars as pl  # type: ignore[import]
    except ImportError as e:
        raise ImportError("Install polars: pip install polars") from e

    if isinstance(path_or_df, pl.DataFrame):
        return path_or_df.to_pandas()
    # polars can read CSV/Parquet too
    suffix = Path(str(path_or_df)).suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path_or_df, **kwargs).to_pandas()
    if suffix in (".parquet", ".pq"):
        return pl.read_parquet(path_or_df, **kwargs).to_pandas()
    raise ValueError(f"Unsupported polars path type: {suffix}")


# Public loader entry-point


def load(
    source: str | Path | pd.DataFrame | Any,
    *,
    format: str | None = None,
    split: str = "train",
    sql_query: str = "SELECT * FROM data",
    **kwargs: Any,
) -> tuple[pd.DataFrame, str, str]:

    # pandas DataFrame passthrough
    if isinstance(source, pd.DataFrame):
        return source, "dataframe", "dataframe"

    # polars DataFrame passthrough
    try:
        import polars as pl  # type: ignore[import]

        if isinstance(source, pl.DataFrame):
            return _load_polars(source), "polars_dataframe", "polars"
    except ImportError:
        pass

    source_str = str(source)

    # infer format
    if format is None:
        path = Path(source_str)
        if source_str.startswith("sql://") or source_str.startswith("postgresql://"):
            format = "sql"
        elif path.suffix.lower() == ".csv":
            format = "csv"
        elif path.suffix.lower() in (".json", ".jsonl"):
            format = "json"
        elif path.suffix.lower() in (".parquet", ".pq"):
            format = "parquet"
        elif not path.exists():
            # Assume HuggingFace dataset ID (e.g. "imdb", "squad")
            format = "hf"
        else:
            raise ValueError(
                f"Cannot infer format from source '{source_str}'. Pass format= explicitly."
            )

    fmt = format.lower()

    if fmt == "csv":
        df = _load_csv(source_str, **kwargs)
    elif fmt in ("json", "jsonl"):
        df = _load_json(source_str, **kwargs)
    elif fmt in ("parquet", "pq"):
        df = _load_parquet(source_str, **kwargs)
    elif fmt == "hf":
        df = _load_huggingface(source_str, split=split, **kwargs)
    elif fmt == "sql":
        df = _load_sql(source_str, sql_query, **kwargs)
    else:
        raise ValueError(f"Unsupported format: '{fmt}'")

    label = urlparse(source_str).path if fmt == "sql" else source_str
    return df, label, fmt
