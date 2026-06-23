"""
Path, product-discovery, and product-selection utilities for InSAR-Sci.

This module defines the relationship between the pair manifest and exported
GeoTIFF product folders. It does not read raster values and it does not plot.

Expected current folder structure
---------------------------------
<geotiff_root>/
    <pair_id>/
        geo/
            <pair_id>_disp.tif
            <pair_id>_disp_pr.tif
            <pair_id>_coh.tif
            <pair_id>_coh_raw.tif
            <pair_id>_phase_raw.tif
            <pair_id>_phase_filt.tif
            <pair_id>_unw.tif
            <pair_id>_unw_pr.tif
        plots/
            ...

Important distinction
---------------------
product_key:
    Actual file product, e.g. ``disp``, ``disp_pr``, ``coh``, ``unw_pr``.

product_role:
    Scientific role requested by analysis, e.g. ``disp``, ``coh``, ``phase``,
    or ``unw``.

variant:
    Correction state, e.g. ``raw``, ``plane_removed``, ``gacos``,
    ``atm_corrected``.

In auto mode, a manifest row decides whether product_role="disp" resolves to
``disp`` or ``disp_pr`` using the row-level ``Plane Removal`` flag.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


PAIR_ID_RE = re.compile(r"^(?P<master>\d{8})_(?P<slave>\d{8})$")

TRUE_STRINGS = {"true", "t", "yes", "y", "1"}
FALSE_STRINGS = {"false", "f", "no", "n", "0"}


# =============================================================================
# Pair/path basics
# =============================================================================

def parse_pair_id(pair_id: str) -> dict[str, str]:
    """
    Parse an interferometric pair identifier.

    Parameters
    ----------
    pair_id
        Pair identifier formatted as YYYYMMDD_YYYYMMDD.

    Returns
    -------
    dict
        Dictionary with ``pair_id``, ``master``, and ``slave`` as strings.

    Raises
    ------
    ValueError
        If the pair ID does not match YYYYMMDD_YYYYMMDD.
    """
    pair_id = str(pair_id).strip()
    match = PAIR_ID_RE.match(pair_id)

    if match is None:
        raise ValueError(
            f"Invalid pair_id: {pair_id!r}. Expected format YYYYMMDD_YYYYMMDD."
        )

    return {
        "pair_id": pair_id,
        "master": match.group("master"),
        "slave": match.group("slave"),
    }


def pair_dir(geotiff_root: str | Path, pair_id: str) -> Path:
    """
    Return the root directory for one interferometric pair.
    """
    parsed = parse_pair_id(pair_id)
    return Path(geotiff_root) / parsed["pair_id"]


def pair_geo_dir(
    geotiff_root: str | Path,
    pair_id: str,
    geo_subdir: str = "geo",
) -> Path:
    """
    Return the GeoTIFF directory for one interferometric pair.
    """
    return pair_dir(geotiff_root, pair_id) / geo_subdir


def product_path(
    geotiff_root: str | Path,
    pair_id: str,
    product_key: str,
    suffixes: Mapping[str, str],
    geo_subdir: str = "geo",
) -> Path:
    """
    Build the expected GeoTIFF path for one pair and one product.

    Parameters
    ----------
    geotiff_root
        Root directory containing one folder per interferometric pair.
    pair_id
        Pair identifier formatted as YYYYMMDD_YYYYMMDD.
    product_key
        Actual product key, for example ``disp``, ``disp_pr``, or ``phase_filt``.
    suffixes
        Mapping from product key to filename suffix.
    geo_subdir
        Subfolder containing GeoTIFFs inside each pair folder.

    Returns
    -------
    pathlib.Path
        Expected GeoTIFF path.

    Raises
    ------
    KeyError
        If ``product_key`` is not defined in ``suffixes``.
    """
    parsed = parse_pair_id(pair_id)

    if product_key not in suffixes:
        valid = ", ".join(sorted(suffixes))
        raise KeyError(f"Unknown product_key {product_key!r}. Valid keys: {valid}")

    return (
        Path(geotiff_root)
        / parsed["pair_id"]
        / geo_subdir
        / f"{parsed['pair_id']}{suffixes[product_key]}"
    )


# =============================================================================
# Manifest flag parsing for product selection
# =============================================================================

def _parse_bool_value(value: object) -> bool | None:
    """
    Parse a boolean-like value into True, False, or None.

    This intentionally mirrors manifest parsing but avoids importing
    ``manifest.py`` here, preventing circular imports.
    """
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None

    text = str(value).strip().lower()

    if text in TRUE_STRINGS:
        return True
    if text in FALSE_STRINGS:
        return False

    return None


def _row_flag(
    row: pd.Series | Mapping[str, object],
    column_name: str,
) -> bool | None:
    """
    Read a boolean flag from a manifest row.

    Prefer the prepared ``<column>_bool`` column if it exists; otherwise parse
    the raw manifest column.
    """
    bool_col = f"{column_name}_bool"

    if bool_col in row:
        return _parse_bool_value(row[bool_col])

    if column_name in row:
        return _parse_bool_value(row[column_name])

    return None


# =============================================================================
# Variant/product-role resolution
# =============================================================================

def resolve_variant_from_row(
    row: pd.Series | Mapping[str, object] | None,
    config: dict,
    variant: str = "auto",
) -> str:
    """
    Resolve which correction variant should be used.

    Parameters
    ----------
    row
        Manifest row. Required when ``variant`` is ``auto`` or
        ``auto_from_manifest`` and product_selection.mode is auto.
    config
        Project CONFIG dictionary.
    variant
        Requested variant. Use:
        - ``"auto"`` or ``"auto_from_manifest"`` to use manifest flags
        - explicit variants such as ``"raw"`` or ``"plane_removed"``

    Returns
    -------
    str
        Resolved variant name.

    Notes
    -----
    Current important rule:

    ``Plane Removal=True``  -> ``plane_removed``

    ``Plane Removal=False`` -> ``raw``

    Missing flag -> ``raw``
    """
    if variant not in {"auto", "auto_from_manifest", None}:
        return str(variant)

    selection = config.get("product_selection", {})
    mode = selection.get("mode", "auto_from_manifest")

    if mode == "explicit_variant":
        return selection.get("selected_variant", "raw")

    if mode != "auto_from_manifest":
        raise ValueError(
            f"Unknown product_selection mode {mode!r}. "
            "Use 'auto_from_manifest' or 'explicit_variant'."
        )

    if row is None:
        # Conservative fallback. Useful for functions that are not row-aware.
        return selection.get("missing_flag_policy", "raw")

    cols = selection.get("manifest_columns", {})

    plane_col = cols.get("plane_removed", "Plane Removal")
    gacos_col = cols.get("gacos", "Use GACOS")
    atm_col = cols.get("atm_corrected", "Use atm correction")

    use_plane = _row_flag(row, plane_col)
    use_gacos = _row_flag(row, gacos_col)
    use_atm = _row_flag(row, atm_col)

    use_plane = bool(use_plane) if use_plane is not None else False
    use_gacos = bool(use_gacos) if use_gacos is not None else False
    use_atm = bool(use_atm) if use_atm is not None else False

    # Precedence is explicit and conservative.
    # Future variants are inactive in CONFIG until files actually exist.
    if use_plane and use_gacos and use_atm:
        return "plane_removed_gacos_atm"

    if use_plane and use_gacos:
        return "plane_removed_gacos"

    if use_atm:
        return "atm_corrected"

    if use_gacos:
        return "gacos"

    if use_plane:
        return "plane_removed"

    return "raw"


def product_key_for_role(
    product_role: str,
    config: dict,
    row: pd.Series | Mapping[str, object] | None = None,
    variant: str = "auto",
) -> str:
    """
    Resolve a scientific product role to an actual product key.

    Examples
    --------
    In auto mode:

    ``product_role="disp"`` + row Plane Removal=True -> ``disp_pr``

    ``product_role="disp"`` + row Plane Removal=False -> ``disp``

    ``product_role="coh"`` always -> ``coh`` for raw/plane_removed variants.
    """
    resolved_variant = resolve_variant_from_row(
        row=row,
        config=config,
        variant=variant,
    )

    role_maps = config["product_selection"]["role_product_maps"]

    if resolved_variant not in role_maps:
        valid = ", ".join(sorted(role_maps))
        raise KeyError(
            f"Variant {resolved_variant!r} is not defined in role_product_maps. "
            f"Valid variants: {valid}"
        )

    role_map = role_maps[resolved_variant]

    if product_role not in role_map:
        valid = ", ".join(sorted(role_map))
        raise KeyError(
            f"Product role {product_role!r} is not defined for variant "
            f"{resolved_variant!r}. Valid roles: {valid}"
        )

    return role_map[product_role]


def product_path_for_role(
    pair_id: str,
    product_role: str,
    config: dict,
    row: pd.Series | Mapping[str, object] | None = None,
    variant: str = "auto",
) -> Path:
    """
    Build the expected GeoTIFF path from a product role and variant.

    This is the preferred path builder for analysis code because it can respect
    the row-level manifest flags.
    """
    product_key = product_key_for_role(
        product_role=product_role,
        config=config,
        row=row,
        variant=variant,
    )

    return product_path(
        geotiff_root=config["paths"]["geotiff_root"],
        pair_id=pair_id,
        product_key=product_key,
        suffixes=config["products"]["suffixes"],
        geo_subdir=config["products"]["geo_subdir"],
    )


# =============================================================================
# Inventory utilities
# =============================================================================

def inventory_pair_products(
    geotiff_root: str | Path,
    pair_id: str,
    suffixes: Mapping[str, str],
    geo_subdir: str = "geo",
    include_paths: bool = True,
    product_keys: Sequence[str] | None = None,
) -> dict[str, object]:
    """
    Build a product-existence inventory for one interferometric pair.

    Parameters
    ----------
    geotiff_root
        Root directory containing pair folders.
    pair_id
        Pair identifier.
    suffixes
        Mapping from product key to filename suffix.
    geo_subdir
        GeoTIFF subdirectory.
    include_paths
        If True, include ``<product>_path`` columns.
    product_keys
        Optional subset of product keys to inventory. If None, all suffix keys
        are inspected.
    """
    parsed = parse_pair_id(pair_id)
    this_pair_dir = pair_dir(geotiff_root, pair_id)
    this_geo_dir = pair_geo_dir(geotiff_root, pair_id, geo_subdir=geo_subdir)

    if product_keys is None:
        product_keys = list(suffixes.keys())

    record: dict[str, object] = {
        "pair_id": parsed["pair_id"],
        "master": parsed["master"],
        "slave": parsed["slave"],
        "pair_dir": this_pair_dir,
        "geo_dir": this_geo_dir,
        "pair_dir_exists": this_pair_dir.exists(),
        "geo_dir_exists": this_geo_dir.exists(),
    }

    for product_key in product_keys:
        path = product_path(
            geotiff_root=geotiff_root,
            pair_id=pair_id,
            product_key=product_key,
            suffixes=suffixes,
            geo_subdir=geo_subdir,
        )

        record[f"{product_key}_exists"] = path.exists()

        if include_paths:
            record[f"{product_key}_path"] = path

    return record


def inventory_from_manifest(
    manifest: pd.DataFrame,
    geotiff_root: str | Path,
    suffixes: Mapping[str, str],
    pair_col: str = "pair_id",
    geo_subdir: str = "geo",
    first_n: int | None = None,
    include_paths: bool = True,
    product_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Build a GeoTIFF product inventory for pairs listed in a manifest.

    Returns one row per unique pair ID.
    """
    if pair_col not in manifest.columns:
        raise KeyError(f"Manifest does not contain required column {pair_col!r}.")

    work = manifest.copy()

    if first_n is not None:
        work = work.head(first_n)

    pair_ids = (
        work[pair_col]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    records = [
        inventory_pair_products(
            geotiff_root=geotiff_root,
            pair_id=pair_id,
            suffixes=suffixes,
            geo_subdir=geo_subdir,
            include_paths=include_paths,
            product_keys=product_keys,
        )
        for pair_id in pair_ids
    ]

    return pd.DataFrame.from_records(records)


def inventory_selected_products_from_manifest(
    manifest: pd.DataFrame,
    config: dict,
    product_roles: Sequence[str] = ("disp", "coh"),
    variant: str = "auto",
    pair_col: str | None = None,
    first_n: int | None = None,
) -> pd.DataFrame:
    """
    Inventory products selected by role and manifest flags.

    This answers: for each row, what product key/path will the analysis actually
    use under the current product-selection policy?
    """
    if pair_col is None:
        pair_col = config["manifest"]["pair_id_column"]

    if pair_col not in manifest.columns:
        raise KeyError(f"Manifest does not contain required column {pair_col!r}.")

    work = manifest.copy()

    if first_n is not None:
        work = work.head(first_n)

    records: list[dict[str, object]] = []

    for _, row in work.iterrows():
        pair_id = str(row[pair_col]).strip()
        resolved_variant = resolve_variant_from_row(
            row=row,
            config=config,
            variant=variant,
        )

        rec: dict[str, object] = {
            "pair_id": pair_id,
            "resolved_variant": resolved_variant,
        }

        for role in product_roles:
            product_key = product_key_for_role(
                product_role=role,
                config=config,
                row=row,
                variant=variant,
            )

            path = product_path_for_role(
                pair_id=pair_id,
                product_role=role,
                config=config,
                row=row,
                variant=variant,
            )

            rec[f"{role}_product_key"] = product_key
            rec[f"{role}_path"] = path
            rec[f"{role}_exists"] = path.exists()

        records.append(rec)

    return pd.DataFrame.from_records(records)


def add_pair_dates(
    manifest: pd.DataFrame,
    pair_col: str = "pair_id",
) -> pd.DataFrame:
    """
    Add parsed master/slave date columns to a manifest table.

    Adds:
    - ``master``
    - ``slave``
    - ``master_date``
    - ``slave_date``
    - ``dt_days_from_pair_id``

    Existing columns are preserved.
    """
    if pair_col not in manifest.columns:
        raise KeyError(f"Manifest does not contain required column {pair_col!r}.")

    out = manifest.copy()

    parsed = out[pair_col].apply(parse_pair_id).apply(pd.Series)
    out["master"] = parsed["master"]
    out["slave"] = parsed["slave"]

    out["master_date"] = pd.to_datetime(out["master"], format="%Y%m%d")
    out["slave_date"] = pd.to_datetime(out["slave"], format="%Y%m%d")
    out["dt_days_from_pair_id"] = (
        out["slave_date"] - out["master_date"]
    ).dt.days

    return out


def group_pairs_by_master(
    manifest: pd.DataFrame,
    pair_col: str = "pair_id",
) -> dict[str, pd.DataFrame]:
    """
    Group manifest rows by master acquisition date.
    """
    dated = add_pair_dates(manifest, pair_col=pair_col)
    dated = dated.sort_values(["master_date", "slave_date", pair_col])

    return {
        master: group.copy()
        for master, group in dated.groupby("master", sort=True)
    }


def product_exists_columns(product_keys: Sequence[str]) -> list[str]:
    """
    Return existence column names for a sequence of product keys.
    """
    return [f"{key}_exists" for key in product_keys]


def summarize_inventory(
    inventory: pd.DataFrame,
    product_keys: Sequence[str],
) -> pd.DataFrame:
    """
    Summarize product completeness across an inventory table.
    """
    rows = []

    for product_key in product_keys:
        col = f"{product_key}_exists"

        if col not in inventory.columns:
            raise KeyError(f"Inventory does not contain expected column {col!r}.")

        available = int(inventory[col].sum())
        total = int(len(inventory))

        rows.append(
            {
                "product": product_key,
                "available": available,
                "missing": total - available,
                "fraction_available": available / total if total else float("nan"),
            }
        )

    return pd.DataFrame(rows)
