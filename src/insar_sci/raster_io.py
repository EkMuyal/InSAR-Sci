"""
Raster I/O and masking utilities for InSAR-Sci.

This module reads GeoTIFF products and prepares valid-data masks. It does not
plot and it does not know about the manifest.

Important masking policy
------------------------
The exported GeoTIFFs currently report nodata = 0.0. This is useful for removing
zero-padded areas outside the valid footprint, but it is also dangerous because
zero can be a physically meaningful value in displacement, phase, and coherence
products.

Therefore, all masking is explicit and configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.plot import plotting_extent


@dataclass(frozen=True)
class RasterMetadata:
    """
    Small container for essential raster metadata.
    """

    path: Path
    crs: Any
    transform: Any
    bounds: Any
    shape: tuple[int, int]
    resolution: tuple[float, float]
    nodata: float | int | None
    dtype: str
    extent: tuple[float, float, float, float]


def open_raster(path: str | Path) -> DatasetReader:
    """
    Open a rasterio dataset after checking that the file exists.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    return rasterio.open(path)


def read_metadata(path: str | Path) -> RasterMetadata:
    """
    Read essential GeoTIFF metadata without loading the full raster array.
    """
    path = Path(path)

    with open_raster(path) as src:
        metadata = RasterMetadata(
            path=path,
            crs=src.crs,
            transform=src.transform,
            bounds=src.bounds,
            shape=(src.height, src.width),
            resolution=src.res,
            nodata=src.nodata,
            dtype=src.dtypes[0],
            extent=plotting_extent(src),
        )

    return metadata


def read_array(
    path: str | Path,
    band: int = 1,
    dtype: str = "float32",
) -> tuple[np.ndarray, RasterMetadata]:
    """
    Read one raster band as a NumPy array, without applying an automatic mask.

    Masking is intentionally separated from reading, because automatic masking
    can silently treat zero as nodata when the GeoTIFF has nodata=0.
    """
    path = Path(path)

    with open_raster(path) as src:
        array = src.read(band).astype(dtype, copy=False)
        metadata = RasterMetadata(
            path=path,
            crs=src.crs,
            transform=src.transform,
            bounds=src.bounds,
            shape=(src.height, src.width),
            resolution=src.res,
            nodata=src.nodata,
            dtype=src.dtypes[0],
            extent=plotting_extent(src),
        )

    return array, metadata


def valid_data_mask(
    array: np.ndarray,
    nodata: float | int | None = None,
    respect_file_nodata: bool = True,
    zero_is_nodata: bool = False,
) -> np.ndarray:
    """
    Build a valid-data mask for a raster array.

    Parameters
    ----------
    array
        Raster array.
    nodata
        Nodata value from raster metadata.
    respect_file_nodata
        If True, pixels equal to the file nodata value are invalid.
    zero_is_nodata
        If True, exact zeros are invalid even if file nodata is not zero.
        Use only when you know the product has zero padding.

    Returns
    -------
    numpy.ndarray
        Boolean mask where True means valid.
    """
    mask = np.isfinite(array)

    if respect_file_nodata and nodata is not None:
        if np.isnan(nodata):
            mask &= ~np.isnan(array)
        else:
            mask &= array != nodata

    if zero_is_nodata:
        mask &= array != 0

    return mask


def masked_array(
    array: np.ndarray,
    nodata: float | int | None = None,
    respect_file_nodata: bool = True,
    zero_is_nodata: bool = False,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Return an array with invalid pixels replaced by NaN.
    """
    mask = valid_data_mask(
        array=array,
        nodata=nodata,
        respect_file_nodata=respect_file_nodata,
        zero_is_nodata=zero_is_nodata,
    )

    return np.where(mask, array, fill_value).astype("float32", copy=False)


def read_masked_array(
    path: str | Path,
    band: int = 1,
    dtype: str = "float32",
    respect_file_nodata: bool = True,
    zero_is_nodata: bool = False,
) -> tuple[np.ndarray, RasterMetadata, np.ndarray]:
    """
    Read a raster and return masked data, metadata, and the valid-data mask.

    Returns
    -------
    masked
        Array with invalid pixels set to NaN.
    metadata
        RasterMetadata object.
    mask
        Boolean valid-data mask.
    """
    array, metadata = read_array(path, band=band, dtype=dtype)

    mask = valid_data_mask(
        array=array,
        nodata=metadata.nodata,
        respect_file_nodata=respect_file_nodata,
        zero_is_nodata=zero_is_nodata,
    )

    masked = np.where(mask, array, np.nan).astype("float32", copy=False)

    return masked, metadata, mask


def raster_stats(
    array: np.ndarray,
    percentiles: tuple[float, ...] = (1, 2, 5, 50, 95, 98, 99),
) -> dict[str, float | int]:
    """
    Compute basic statistics for a masked raster array.

    The input should contain NaNs where pixels are invalid.
    """
    values = array[np.isfinite(array)]

    if values.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    stats: dict[str, float | int] = {
        "count": int(values.size),
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "std": float(np.nanstd(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
    }

    for p in percentiles:
        key = f"p{str(p).replace('.', 'p')}"
        stats[key] = float(np.nanpercentile(values, p))

    return stats


def robust_limits(
    array: np.ndarray,
    percentile: float = 98.0,
    symmetric: bool = False,
    fixed_limits: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """
    Compute robust plotting limits from finite raster values.

    Parameters
    ----------
    array
        Masked raster array with NaNs for invalid pixels.
    percentile
        Upper percentile for robust limit. The lower percentile is 100 - percentile.
    symmetric
        If True, return limits symmetric around zero.
    fixed_limits
        If provided, return these limits directly.
    """
    if fixed_limits is not None:
        return fixed_limits

    values = array[np.isfinite(array)]

    if values.size == 0:
        return (np.nan, np.nan)

    lower_q = 100.0 - percentile
    upper_q = percentile

    vmin = float(np.nanpercentile(values, lower_q))
    vmax = float(np.nanpercentile(values, upper_q))

    if symmetric:
        vmax_abs = max(abs(vmin), abs(vmax))
        return (-vmax_abs, vmax_abs)

    return (vmin, vmax)


def product_mask_settings(
    product_key: str,
    config: dict,
) -> dict[str, bool]:
    """
    Get mask settings for a product from CONFIG.

    Defaults are conservative:
    - respect_file_nodata=True
    - zero_is_nodata=False
    """
    masking = config.get("masking", {})

    zero_by_product = masking.get("zero_is_nodata_by_product", {})
    respect_by_product = masking.get("respect_file_nodata_by_product", {})

    return {
        "respect_file_nodata": bool(
            respect_by_product.get(product_key, True)
        ),
        "zero_is_nodata": bool(
            zero_by_product.get(product_key, False)
        ),
    }


def read_product_array(
    path: str | Path,
    product_key: str,
    config: dict,
) -> tuple[np.ndarray, RasterMetadata, np.ndarray, dict[str, float | int]]:
    """
    Read one InSAR product using product-specific mask settings from CONFIG.
    """
    mask_settings = product_mask_settings(product_key, config)

    array, metadata, mask = read_masked_array(
        path=path,
        respect_file_nodata=mask_settings["respect_file_nodata"],
        zero_is_nodata=mask_settings["zero_is_nodata"],
    )

    stats = raster_stats(array)

    return array, metadata, mask, stats
