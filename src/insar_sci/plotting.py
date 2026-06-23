"""
Plotting utilities for InSAR-Sci.

Main idea:
- product_role = scientific request: disp, coh, phase, unw
- product_key  = actual file: disp, disp_pr, coh, phase_filt, unw_pr

In auto mode:
    Plane Removal=True  -> disp role uses disp_pr
    Plane Removal=False -> disp role uses disp

Display normalization is only for plotting. Raw raster values are not modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from insar_sci.paths import (
    product_key_for_role,
    product_path,
    product_path_for_role,
    resolve_variant_from_row,
)
from insar_sci.raster_io import read_product_array, raster_stats, robust_limits


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def decimate_for_plot(array: np.ndarray, decimation: int = 1) -> np.ndarray:
    if decimation is None or decimation <= 1:
        return array
    return array[::decimation, ::decimation]


def product_class(product_key: str, config: dict) -> str:
    classes = config["products"].get("classes", {})

    for class_name, product_keys in classes.items():
        if product_key in product_keys:
            return class_name

    return "other"


def infer_product_role_from_key(product_key: str, config: dict) -> str | None:
    pclass = product_class(product_key, config)

    if pclass == "displacement":
        return "disp"
    if pclass == "coherence":
        return "coh"
    if pclass == "wrapped_phase":
        return "phase"
    if pclass == "unwrapped_phase":
        return "unw"

    return None


def same_grid(metadata_a, metadata_b) -> bool:
    return (
        metadata_a.shape == metadata_b.shape
        and metadata_a.crs == metadata_b.crs
        and metadata_a.transform == metadata_b.transform
    )


def format_axis(ax) -> None:
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.ticklabel_format(useOffset=False)


def default_footprint_config(config: dict) -> dict:
    """
    Return footprint-mask config.

    If CONFIG does not yet define masking.plot_footprint_mask, this gives
    the desired default behavior: use raw disp as the footprint for derived
    displacement/phase/unwrapped products.
    """
    default = {
        "enabled": True,
        "reference_product_by_role": {
            "disp": "disp",
            "unw": "disp",
            "phase": "disp",
            "coh": "coh",
        },
        "on_grid_mismatch": "warn",
    }

    user_cfg = config.get("masking", {}).get("plot_footprint_mask", {})

    out = default.copy()
    out.update(user_cfg)

    if "reference_product_by_role" in user_cfg:
        ref = default["reference_product_by_role"].copy()
        ref.update(user_cfg["reference_product_by_role"])
        out["reference_product_by_role"] = ref

    return out


def apply_display_normalization(
    array: np.ndarray,
    product_key: str,
    config: dict,
) -> tuple[np.ndarray, str, float | None]:
    normalization = config["plotting"].get("display_normalization", {})

    if not normalization.get("enabled", False):
        return array, "raw", None

    method = normalization.get("method_by_product", {}).get(product_key, "none")

    if method == "none":
        return array, "raw", None

    values = array[np.isfinite(array)]

    if values.size == 0:
        return array, "raw", None

    if method == "mean":
        offset = float(np.nanmean(values))
        return array - offset, f"mean removed ({offset:.5g})", offset

    if method == "median":
        offset = float(np.nanmedian(values))
        return array - offset, f"median removed ({offset:.5g})", offset

    raise ValueError(
        f"Unknown display normalization method {method!r}. "
        "Use 'none', 'mean', or 'median'."
    )


def product_plot_spec(
    product_key: str,
    array: np.ndarray,
    config: dict,
    shared_limits: tuple[float, float] | None = None,
) -> dict[str, object]:
    plotting = config["plotting"]
    pclass = product_class(product_key, config)

    cmap = plotting["colormaps"].get(product_key, "viridis")
    label = plotting["units"].get(product_key, product_key)

    if shared_limits is not None:
        vmin, vmax = shared_limits

    elif pclass == "coherence":
        vmin, vmax = plotting.get("coherence_limits", (0.0, 1.0))

    elif pclass == "wrapped_phase":
        vmin, vmax = plotting.get("wrapped_phase_limits", (-np.pi, np.pi))

    elif pclass == "displacement":
        vmin, vmax = robust_limits(
            array,
            percentile=plotting.get("robust_percentile", 98.0),
            symmetric=plotting.get("symmetric_displacement_limits", True),
        )

    else:
        vmin, vmax = robust_limits(
            array,
            percentile=plotting.get("robust_percentile", 98.0),
            symmetric=False,
        )

    return {
        "cmap": cmap,
        "label": label,
        "vmin": vmin,
        "vmax": vmax,
    }


def shared_limits_for_items(
    items: list[dict[str, object]],
    config: dict,
) -> tuple[float, float]:
    product_keys = [item["product_key"] for item in items]
    product_classes = {product_class(key, config) for key in product_keys}

    if product_classes == {"coherence"}:
        return config["plotting"].get("coherence_limits", (0.0, 1.0))

    if product_classes == {"wrapped_phase"}:
        return config["plotting"].get("wrapped_phase_limits", (-np.pi, np.pi))

    symmetric = product_classes == {"displacement"}
    limits = []

    for item in items:
        array = item["display_array"]
        vmin, vmax = robust_limits(
            array,
            percentile=config["plotting"].get("robust_percentile", 98.0),
            symmetric=symmetric,
        )

        if np.isfinite(vmin) and np.isfinite(vmax):
            limits.append((vmin, vmax))

    if not limits:
        return float("nan"), float("nan")

    if symmetric:
        vmax_abs = max(max(abs(vmin), abs(vmax)) for vmin, vmax in limits)
        return -vmax_abs, vmax_abs

    return min(vmin for vmin, _ in limits), max(vmax for _, vmax in limits)


def read_product_for_plot(
    pair_id: str,
    config: dict,
    product_key: str | None = None,
    product_role: str | None = None,
    row: pd.Series | Mapping[str, object] | None = None,
    variant: str = "auto",
) -> dict[str, object]:
    if product_key is None and product_role is None:
        raise ValueError("Provide either product_key or product_role.")

    if product_key is not None and product_role is not None:
        raise ValueError("Provide only one of product_key or product_role.")

    if product_role is not None:
        resolved_variant = resolve_variant_from_row(
            row=row,
            config=config,
            variant=variant,
        )

        product_key = product_key_for_role(
            product_role=product_role,
            config=config,
            row=row,
            variant=variant,
        )

        path = product_path_for_role(
            pair_id=pair_id,
            product_role=product_role,
            config=config,
            row=row,
            variant=variant,
        )

        footprint_role = product_role

    else:
        resolved_variant = "explicit_product_key"

        path = product_path(
            geotiff_root=config["paths"]["geotiff_root"],
            pair_id=pair_id,
            product_key=product_key,
            suffixes=config["products"]["suffixes"],
            geo_subdir=config["products"]["geo_subdir"],
        )

        footprint_role = infer_product_role_from_key(product_key, config)

    raw_array, metadata, selected_mask, raw_stats_before = read_product_array(
        path=path,
        product_key=product_key,
        config=config,
    )

    array_after_footprint = raw_array.copy()

    footprint_cfg = default_footprint_config(config)
    footprint_applied = False
    footprint_product_key = None
    footprint_warning = None

    if footprint_cfg.get("enabled", False) and footprint_role is not None:
        footprint_product_key = footprint_cfg["reference_product_by_role"].get(
            footprint_role,
            None,
        )

        if footprint_product_key is not None:
            footprint_path = product_path(
                geotiff_root=config["paths"]["geotiff_root"],
                pair_id=pair_id,
                product_key=footprint_product_key,
                suffixes=config["products"]["suffixes"],
                geo_subdir=config["products"]["geo_subdir"],
            )

            _, footprint_metadata, footprint_mask, _ = read_product_array(
                path=footprint_path,
                product_key=footprint_product_key,
                config=config,
            )

            if same_grid(metadata, footprint_metadata):
                array_after_footprint[~footprint_mask] = np.nan
                footprint_applied = True
            else:
                footprint_warning = (
                    f"{product_key} grid differs from {footprint_product_key} grid."
                )

                if footprint_cfg.get("on_grid_mismatch", "warn") == "warn":
                    print(f"WARNING {pair_id}: footprint skipped; {footprint_warning}")

    raw_stats_after = raster_stats(array_after_footprint)

    display_array, norm_label, norm_offset = apply_display_normalization(
        array=array_after_footprint,
        product_key=product_key,
        config=config,
    )

    display_stats = {
        "display_mean": float(np.nanmean(display_array)),
        "display_median": float(np.nanmedian(display_array)),
        "display_std": float(np.nanstd(display_array)),
    }

    return {
        "pair_id": pair_id,
        "product_key": product_key,
        "product_role": product_role if product_role is not None else footprint_role,
        "variant": resolved_variant,
        "path": path,
        "display_array": display_array,
        "metadata": metadata,
        "raw_stats_before_footprint": raw_stats_before,
        "raw_stats_after_footprint": raw_stats_after,
        "display_stats": display_stats,
        "norm_label": norm_label,
        "norm_offset": norm_offset,
        "footprint_applied": footprint_applied,
        "footprint_product_key": footprint_product_key,
        "footprint_warning": footprint_warning,
    }


def item_to_summary_row(
    item: dict[str, object],
    dt_days=None,
) -> dict[str, object]:
    before = item["raw_stats_before_footprint"]
    after = item["raw_stats_after_footprint"]
    display = item["display_stats"]

    return {
        "pair_id": item["pair_id"],
        "dt_days": dt_days,
        "product_role": item["product_role"],
        "product_key": item["product_key"],
        "variant": item["variant"],
        "path": str(item["path"]),
        "footprint_applied": item["footprint_applied"],
        "footprint_product_key": item["footprint_product_key"],
        "footprint_warning": item["footprint_warning"],
        "count_before_footprint": int(before["count"]),
        "count_after_footprint": int(after["count"]),
        "count_removed_by_footprint": int(before["count"]) - int(after["count"]),
        "norm_label": item["norm_label"],
        "norm_offset": item["norm_offset"],
        "mean": after["mean"],
        "median": after["median"],
        "std": after["std"],
        "min": after["min"],
        "max": after["max"],
        "p02": after["p2"],
        "p05": after["p5"],
        "p50": after["p50"],
        "p95": after["p95"],
        "p98": after["p98"],
        "display_mean": display["display_mean"],
        "display_median": display["display_median"],
        "display_std": display["display_std"],
    }


def summary_table_from_items(
    items: list[dict[str, object]],
    dt_days_by_pair: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    rows = []

    for item in items:
        pair_id = item["pair_id"]
        dt_days = None

        if dt_days_by_pair is not None:
            dt_days = dt_days_by_pair.get(pair_id, None)

        rows.append(item_to_summary_row(item, dt_days=dt_days))

    return pd.DataFrame(rows)


def panel_title(item: dict[str, object], dt_days=None) -> str:
    if dt_days is None or pd.isna(dt_days):
        return f"{item['pair_id']}\n{item['variant']} | {item['product_key']}"

    return (
        f"{item['pair_id']}\n"
        f"Δt = {dt_days} d | {item['variant']} | {item['product_key']}"
    )


def footnote_from_summary(
    summary_table: pd.DataFrame,
    shared_limits: tuple[float, float] | None,
    context: str,
    config: dict,
) -> str:
    lines = [context]

    if shared_limits is not None:
        lines.append(
            f"Shared display limits: {shared_limits[0]:.4g} to {shared_limits[1]:.4g}."
        )

    lines.append(
        "Display normalization and footprint masking are for visualization only; raw rasters are not modified."
    )

    for _, row in summary_table.iterrows():
        lines.append(
            f"{row['pair_id']}: {row['norm_label']}; "
            f"mean={row['mean']:.5g}, median={row['median']:.5g}; "
            f"footprint={row['footprint_applied']} ({row['footprint_product_key']}); "
            f"n={int(row['count_after_footprint'])}."
        )

    return "\n".join(lines)


def apply_layout(fig, footnote: str | None, config: dict, top: float = 0.91):
    if footnote:
        fig.text(
            0.01,
            0.01,
            footnote,
            ha="left",
            va="bottom",
            fontsize=config["plotting"].get("footnote_fontsize", 8),
        )
        fig.tight_layout(rect=[0, 0.18, 1, top])
    else:
        fig.tight_layout(rect=[0, 0, 1, top])


def save_table(summary_table: pd.DataFrame, path: Path):
    ensure_dir(path.parent)
    summary_table.to_csv(path, index=False)


def plot_master_group_role(
    master_group: pd.DataFrame,
    product_role: str,
    config: dict,
    variant: str = "auto",
    show: bool | None = None,
    save: bool | None = None,
    decimation: int | None = None,
):
    plotting = config["plotting"]

    if show is None:
        show = plotting.get("show_plots", True)

    if save is None:
        save = plotting.get("save_plots", False)

    if decimation is None:
        decimation = plotting.get("plot_decimation", 1)

    group = master_group.sort_values(["slave_date", "pair_id"]).copy()
    master = str(group["master"].iloc[0])

    items = []
    dt_days_by_pair = {}

    for _, row in group.iterrows():
        pair_id = row["pair_id"]
        dt_days_by_pair[pair_id] = row.get("dt_days_from_pair_id", None)

        item = read_product_for_plot(
            pair_id=pair_id,
            config=config,
            product_role=product_role,
            row=row,
            variant=variant,
        )

        items.append(item)

    summary_table = summary_table_from_items(
        items,
        dt_days_by_pair=dt_days_by_pair,
    )

    shared_limits = shared_limits_for_items(items, config=config)

    n = len(items)
    figsize_per_panel = plotting.get("figsize_per_panel", (7, 5.5))

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, item in zip(axes, items):
        dt_days = dt_days_by_pair.get(item["pair_id"], None)

        spec = product_plot_spec(
            product_key=item["product_key"],
            array=item["display_array"],
            config=config,
            shared_limits=shared_limits,
        )

        plot_array = decimate_for_plot(item["display_array"], decimation=decimation)
        plot_array = np.ma.masked_invalid(plot_array)

        im = ax.imshow(
            plot_array,
            extent=item["metadata"].extent,
            cmap=spec["cmap"],
            vmin=spec["vmin"],
            vmax=spec["vmax"],
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cbar.set_label(spec["label"])

        ax.set_title(panel_title(item, dt_days=dt_days))
        format_axis(ax)

    footnote = None
    if plotting.get("title_detail_mode", "footnote") == "footnote":
        footnote = footnote_from_summary(
            summary_table=summary_table,
            shared_limits=shared_limits,
            context=f"Master {master}; role={product_role}; variant mode={variant}.",
            config=config,
        )

    fig.suptitle(f"Master {master} — {product_role} comparison", y=0.98)
    apply_layout(fig, footnote, config)

    if save:
        out_dir = ensure_dir(
            Path(config["paths"]["output_root"])
            / config["outputs"]["master_groups_subdir"]
            / master
        )
        out_png = out_dir / f"{master}_role_{product_role}_{variant}.png"
        out_csv = out_dir / f"{master}_role_{product_role}_{variant}_summary.csv"

        fig.savefig(out_png, dpi=plotting.get("dpi", 300), bbox_inches="tight")
        save_table(summary_table, out_csv)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes, summary_table


def plot_pair_roles(
    pair_id: str,
    product_roles: Iterable[str],
    config: dict,
    row: pd.Series | Mapping[str, object] | None = None,
    variant: str = "auto",
    show: bool | None = None,
    save: bool | None = None,
    decimation: int | None = None,
):
    plotting = config["plotting"]

    if show is None:
        show = plotting.get("show_plots", True)

    if save is None:
        save = plotting.get("save_plots", False)

    if decimation is None:
        decimation = plotting.get("plot_decimation", 1)

    product_roles = list(product_roles)

    items = [
        read_product_for_plot(
            pair_id=pair_id,
            config=config,
            product_role=role,
            row=row,
            variant=variant,
        )
        for role in product_roles
    ]

    dt_days = None
    if row is not None and "dt_days_from_pair_id" in row:
        dt_days = row["dt_days_from_pair_id"]

    summary_table = summary_table_from_items(
        items,
        dt_days_by_pair={pair_id: dt_days},
    )

    n = len(items)
    figsize_per_panel = plotting.get("figsize_per_panel", (7, 5.5))

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, item in zip(axes, items):
        spec = product_plot_spec(
            product_key=item["product_key"],
            array=item["display_array"],
            config=config,
        )

        plot_array = decimate_for_plot(item["display_array"], decimation=decimation)
        plot_array = np.ma.masked_invalid(plot_array)

        im = ax.imshow(
            plot_array,
            extent=item["metadata"].extent,
            cmap=spec["cmap"],
            vmin=spec["vmin"],
            vmax=spec["vmax"],
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cbar.set_label(spec["label"])

        ax.set_title(panel_title(item, dt_days=dt_days))
        format_axis(ax)

    footnote = None
    if plotting.get("title_detail_mode", "footnote") == "footnote":
        footnote = footnote_from_summary(
            summary_table=summary_table,
            shared_limits=None,
            context=f"Pair {pair_id}; roles={', '.join(product_roles)}.",
            config=config,
        )

    fig.suptitle(f"{pair_id} — role quicklook", y=0.98)
    apply_layout(fig, footnote, config)

    if save:
        out_dir = ensure_dir(
            Path(config["paths"]["output_root"])
            / config["outputs"]["quicklooks_subdir"]
            / pair_id
        )
        role_text = "_".join(product_roles)
        out_png = out_dir / f"{pair_id}_roles_{role_text}_{variant}.png"
        out_csv = out_dir / f"{pair_id}_roles_{role_text}_{variant}_summary.csv"

        fig.savefig(out_png, dpi=plotting.get("dpi", 300), bbox_inches="tight")
        save_table(summary_table, out_csv)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes, summary_table


def plot_master_group_product(
    master_group: pd.DataFrame,
    product_key: str,
    config: dict,
    show: bool | None = None,
    save: bool | None = None,
    decimation: int | None = None,
):
    plotting = config["plotting"]

    if show is None:
        show = plotting.get("show_plots", True)

    if save is None:
        save = plotting.get("save_plots", False)

    if decimation is None:
        decimation = plotting.get("plot_decimation", 1)

    group = master_group.sort_values(["slave_date", "pair_id"]).copy()
    master = str(group["master"].iloc[0])

    items = []
    dt_days_by_pair = {}

    for _, row in group.iterrows():
        pair_id = row["pair_id"]
        dt_days_by_pair[pair_id] = row.get("dt_days_from_pair_id", None)

        item = read_product_for_plot(
            pair_id=pair_id,
            config=config,
            product_key=product_key,
            variant="explicit_product_key",
        )

        items.append(item)

    summary_table = summary_table_from_items(
        items,
        dt_days_by_pair=dt_days_by_pair,
    )

    shared_limits = shared_limits_for_items(items, config=config)

    n = len(items)
    figsize_per_panel = plotting.get("figsize_per_panel", (7, 5.5))

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, item in zip(axes, items):
        dt_days = dt_days_by_pair.get(item["pair_id"], None)

        spec = product_plot_spec(
            product_key=item["product_key"],
            array=item["display_array"],
            config=config,
            shared_limits=shared_limits,
        )

        plot_array = decimate_for_plot(item["display_array"], decimation=decimation)
        plot_array = np.ma.masked_invalid(plot_array)

        im = ax.imshow(
            plot_array,
            extent=item["metadata"].extent,
            cmap=spec["cmap"],
            vmin=spec["vmin"],
            vmax=spec["vmax"],
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cbar.set_label(spec["label"])

        ax.set_title(panel_title(item, dt_days=dt_days))
        format_axis(ax)

    footnote = None
    if plotting.get("title_detail_mode", "footnote") == "footnote":
        footnote = footnote_from_summary(
            summary_table=summary_table,
            shared_limits=shared_limits,
            context=f"Master {master}; explicit product={product_key}.",
            config=config,
        )

    fig.suptitle(f"Master {master} — {product_key} comparison", y=0.98)
    apply_layout(fig, footnote, config)

    if save:
        out_dir = ensure_dir(
            Path(config["paths"]["output_root"])
            / config["outputs"]["master_groups_subdir"]
            / master
        )
        out_png = out_dir / f"{master}_product_{product_key}.png"
        out_csv = out_dir / f"{master}_product_{product_key}_summary.csv"

        fig.savefig(out_png, dpi=plotting.get("dpi", 300), bbox_inches="tight")
        save_table(summary_table, out_csv)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes, summary_table
