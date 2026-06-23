"""
Manifest loading and cleaning utilities for InSAR-Sci.

The manifest is the scientific control table for the project. It contains
interferometric pair IDs, processing metadata, manual labels, candidate SSE IDs,
quality flags, correction flags, and comments.

This module should not plot and should not read raster values. Its job is to
produce a clean, analysis-ready pandas DataFrame while preserving the original
CSV information.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from insar_sci.paths import add_pair_dates


TRUE_STRINGS = {"true", "t", "yes", "y", "1"}
FALSE_STRINGS = {"false", "f", "no", "n", "0"}


# =============================================================================
# Loading and parsing
# =============================================================================

def load_manifest(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the pair manifest CSV.

    Parameters
    ----------
    csv_path
        Path to the manifest CSV.

    Returns
    -------
    pandas.DataFrame
        Raw manifest table.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {csv_path}")

    return pd.read_csv(csv_path)


def _parse_bool_value(value: object) -> bool | pd.NA:
    """
    Parse a single boolean-like value.

    Handles real booleans, numeric 0/1, and strings such as TRUE/FALSE.
    Missing or ambiguous values return pandas.NA.
    """
    if pd.isna(value):
        return pd.NA

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float, np.integer, np.floating)):
        if value == 1:
            return True
        if value == 0:
            return False
        return pd.NA

    text = str(value).strip().lower()

    if text in TRUE_STRINGS:
        return True
    if text in FALSE_STRINGS:
        return False

    return pd.NA


def parse_bool_series(series: pd.Series) -> pd.Series:
    """
    Convert a manifest column to pandas nullable BooleanDtype.
    """
    return series.apply(_parse_bool_value).astype("boolean")


def parse_sse_id_value(value: object) -> int | pd.NA:
    """
    Parse one SSE ID value.

    The manifest may store IDs as floats because of NaNs, e.g. 1.0.
    This function converts valid values to integers and preserves missing values.
    """
    if pd.isna(value):
        return pd.NA

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return pd.NA


def combine_sse_ids(
    row: pd.Series,
    columns: Iterable[str] = ("SSE ID", "SSE2 ID"),
) -> list[int]:
    """
    Combine one or more SSE ID columns into a clean list of integer IDs.

    Duplicate IDs are removed while preserving order.
    """
    ids: list[int] = []

    for col in columns:
        if col not in row.index:
            continue

        value = parse_sse_id_value(row[col])

        if pd.isna(value):
            continue

        value = int(value)

        if value not in ids:
            ids.append(value)

    return ids


# =============================================================================
# Config-aware helpers
# =============================================================================

def manifest_boolean_columns_from_config(config: dict | None) -> list[str]:
    """
    Return boolean manifest columns from CONFIG.

    Missing columns are tolerated later in ``prepare_manifest``.
    """
    if config is None:
        return [
            "SSE",
            "Noise",
            "Dirty",
            "processing issues",
            "Interesting Feature",
            "Plane Removal",
            "GACOS",
            "Use GACOS",
            "export_ok",
            "plot_ok",
        ]

    manifest_cfg = config.get("manifest", {})
    cols = list(manifest_cfg.get("boolean_columns", []))
    cols += list(manifest_cfg.get("future_boolean_columns", []))

    # Preserve order while removing duplicates.
    seen = set()
    clean = []

    for col in cols:
        if col not in seen:
            clean.append(col)
            seen.add(col)

    return clean


def sse_id_columns_from_config(config: dict | None) -> list[str]:
    """
    Return SSE ID columns from CONFIG.
    """
    if config is None:
        return ["SSE ID", "SSE2 ID"]

    return list(config.get("manifest", {}).get("sse_id_columns", ["SSE ID", "SSE2 ID"]))


def pair_col_from_config(config: dict | None, default: str = "pair_id") -> str:
    """
    Return pair ID column name from CONFIG.
    """
    if config is None:
        return default

    return config.get("manifest", {}).get("pair_id_column", default)


# =============================================================================
# Manifest preparation
# =============================================================================

def prepare_manifest(
    manifest: pd.DataFrame,
    config: dict | None = None,
    pair_col: str | None = None,
    boolean_columns: Iterable[str] | None = None,
    sse_id_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Prepare the raw manifest for analysis.

    This function preserves original columns and adds derived columns:

    - master
    - slave
    - master_date
    - slave_date
    - dt_days_from_pair_id
    - *_bool columns for selected manual/processing/correction flags
    - sse_id_primary
    - sse_id_secondary
    - sse_ids
    - has_sse_id
    - sse_candidate

    Parameters
    ----------
    manifest
        Raw manifest DataFrame.
    config
        Optional CONFIG dictionary. If provided, column names are read from it.
    pair_col
        Optional override for pair ID column.
    boolean_columns
        Optional override for boolean-like columns.
    sse_id_columns
        Optional override for SSE ID columns.
    """
    if pair_col is None:
        pair_col = pair_col_from_config(config)

    if boolean_columns is None:
        boolean_columns = manifest_boolean_columns_from_config(config)

    if sse_id_columns is None:
        sse_id_columns = sse_id_columns_from_config(config)

    out = manifest.copy()

    if pair_col not in out.columns:
        raise KeyError(f"Manifest does not contain required pair column {pair_col!r}.")

    out[pair_col] = out[pair_col].astype(str).str.strip()
    out = add_pair_dates(out, pair_col=pair_col)

    # Parse all known boolean/correction flags that actually exist.
    for col in boolean_columns:
        if col in out.columns:
            out[f"{col}_bool"] = parse_bool_series(out[col])

    # Ensure future columns do not crash downstream code if absent.
    # We do NOT invent raw columns, only bool columns with NA.
    for col in boolean_columns:
        bool_col = f"{col}_bool"
        if bool_col not in out.columns:
            out[bool_col] = pd.Series(pd.NA, index=out.index, dtype="boolean")

    # Primary and secondary SSE IDs are kept for convenience.
    if len(sse_id_columns) >= 1 and sse_id_columns[0] in out.columns:
        out["sse_id_primary"] = (
            out[sse_id_columns[0]].apply(parse_sse_id_value).astype("Int64")
        )
    else:
        out["sse_id_primary"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    if len(sse_id_columns) >= 2 and sse_id_columns[1] in out.columns:
        out["sse_id_secondary"] = (
            out[sse_id_columns[1]].apply(parse_sse_id_value).astype("Int64")
        )
    else:
        out["sse_id_secondary"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    out["sse_ids"] = out.apply(
        lambda row: combine_sse_ids(row, columns=sse_id_columns),
        axis=1,
    )

    out["has_sse_id"] = out["sse_ids"].apply(lambda ids: len(ids) > 0)

    if "SSE_bool" in out.columns:
        out["sse_candidate"] = out["SSE_bool"].fillna(False) | out["has_sse_id"]
    else:
        out["sse_candidate"] = out["has_sse_id"]

    return out


def subset_for_development(
    manifest: pd.DataFrame,
    first_n: int | None = None,
) -> pd.DataFrame:
    """
    Return a development subset of the manifest.

    If first_n is None, the full manifest is returned.
    """
    if first_n is None:
        return manifest.copy()

    return manifest.head(first_n).copy()


# =============================================================================
# Summaries and grouping
# =============================================================================

def summarize_manifest(manifest: pd.DataFrame) -> dict[str, object]:
    """
    Return high-level manifest summary values.
    """
    summary: dict[str, object] = {
        "n_rows": int(len(manifest)),
        "n_unique_pairs": (
            int(manifest["pair_id"].nunique()) if "pair_id" in manifest else None
        ),
    }

    if "master" in manifest.columns:
        summary["n_unique_masters"] = int(manifest["master"].nunique())

    if "slave" in manifest.columns:
        summary["n_unique_slaves"] = int(manifest["slave"].nunique())

    if "sse_candidate" in manifest.columns:
        summary["n_sse_candidates"] = int(manifest["sse_candidate"].sum())

    if "has_sse_id" in manifest.columns:
        summary["n_rows_with_sse_id"] = int(manifest["has_sse_id"].sum())

    if "dt_days_from_pair_id" in manifest.columns:
        summary["dt_days_counts"] = (
            manifest["dt_days_from_pair_id"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

    # Useful correction-flag summaries.
    for col in [
        "Plane Removal_bool",
        "Use GACOS_bool",
        "Use atm correction_bool",
        "Noise_bool",
        "Dirty_bool",
    ]:
        if col in manifest.columns:
            summary[f"n_{col.replace(' ', '_').replace('_bool', '').lower()}_true"] = (
                int(manifest[col].fillna(False).sum())
            )

    return summary


def summary_as_dataframe(summary: dict[str, object]) -> pd.DataFrame:
    """
    Convert a manifest summary dictionary into a two-column DataFrame.

    Nested dictionaries are stringified so the table can be saved to CSV.
    """
    rows = []

    for key, value in summary.items():
        rows.append(
            {
                "metric": key,
                "value": value if not isinstance(value, dict) else str(value),
            }
        )

    return pd.DataFrame(rows)


def group_by_master(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Group an analysis-ready manifest by master acquisition date.
    """
    if "master" not in manifest.columns:
        manifest = prepare_manifest(manifest)

    work = manifest.sort_values(["master_date", "slave_date", "pair_id"])

    return {
        master: group.copy()
        for master, group in work.groupby("master", sort=True)
    }


def rows_for_sse_id(
    manifest: pd.DataFrame,
    sse_id: int,
) -> pd.DataFrame:
    """
    Return rows associated with a specific candidate SSE ID.
    """
    if "sse_ids" not in manifest.columns:
        manifest = prepare_manifest(manifest)

    mask = manifest["sse_ids"].apply(lambda ids: int(sse_id) in ids)

    return manifest.loc[mask].copy()


def rows_with_sse_ids(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Return candidate rows that have at least one SSE ID.
    """
    if "sse_ids" not in manifest.columns:
        manifest = prepare_manifest(manifest)

    return manifest.loc[manifest["has_sse_id"]].copy()


def candidate_rows_without_sse_id(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows marked as SSE candidates but lacking an SSE ID.

    These rows are useful globally but cannot be linked to the point shapefile
    for local AOI extraction.
    """
    if "sse_candidate" not in manifest.columns:
        manifest = prepare_manifest(manifest)

    mask = manifest["sse_candidate"] & ~manifest["has_sse_id"]

    return manifest.loc[mask].copy()
