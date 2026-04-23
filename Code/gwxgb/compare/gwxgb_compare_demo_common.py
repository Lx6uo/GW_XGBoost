from __future__ import annotations

import argparse
import logging
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

_THIS_DIR = Path(__file__).resolve().parent
_GWXGB_DIR = _THIS_DIR.parent
_CODE_DIR = _GWXGB_DIR.parent
_XGB_DIR = _CODE_DIR / "xgb"
for _path in (_GWXGB_DIR, _XGB_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from gwxgb_shap import (
    _as_bool,
    _get_city_name,
    _iter_local_models,
    _patch_geoxgboost_parallelism,
    _resolve_city_col,
    _select_local_positions,
    _spatial_weights_for_local_data,
    compute_gw_distance_matrix,
    gw_distance_metric_label,
    gw_distance_unit,
    load_dataset,
    optimize_bandwidth,
    optimize_global_model,
    run_gxgb,
)
from xgb_shap import (
    _resolve_path,
    build_and_train_model,
    compute_shap_and_interactions,
    ensure_run_output_dir,
    load_config,
    plot_shap_dependence,
    plot_fixed_base_interactions,
    plot_top_interactions,
    setup_logging,
    summarize_and_save_interactions,
)


@dataclass
class CompareArtifacts:
    config: Dict[str, Any]
    output_dir: Path
    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    coords: pd.DataFrame
    global_shap_values: np.ndarray
    global_interaction_values: np.ndarray | None
    global_importance_df: pd.DataFrame
    local_pooled_df: pd.DataFrame
    local_center_df: pd.DataFrame
    local_importance_df: pd.DataFrame
    local_signed_df: pd.DataFrame
    local_positive_df: pd.DataFrame
    local_negative_df: pd.DataFrame


def _pick_installed_font(candidates: List[str]) -> str | None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


LATIN_FONT_NAME = _pick_installed_font(
    ["Times New Roman", "Cambria", "Georgia", "DejaVu Serif"]
) or "DejaVu Serif"
CJK_FONT_NAME = _pick_installed_font(
    [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "NSimSun",
        "KaiTi",
        "FangSong",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
)


def _configure_plot_fonts() -> None:
    family: List[str] = []
    if CJK_FONT_NAME:
        family.append(CJK_FONT_NAME)
    family.extend([LATIN_FONT_NAME, "DejaVu Sans", "DejaVu Serif"])
    plt.rcParams["font.family"] = family
    plt.rcParams["font.sans-serif"] = (
        [CJK_FONT_NAME, "DejaVu Sans"] if CJK_FONT_NAME else ["DejaVu Sans"]
    )
    plt.rcParams["font.serif"] = [LATIN_FONT_NAME, "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default=None,
        help="YAML 配置文件路径；若不指定则默认使用脚本同目录下的默认 demo 配置。",
    )
    return parser


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并配置字典；override 优先。"""
    merged: Dict[str, Any] = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_known_config_paths(config: Dict[str, Any], *, config_dir: Path) -> Dict[str, Any]:
    for section, key in (("data", "path"), ("output", "output_dir")):
        section_cfg = config.get(section)
        if not isinstance(section_cfg, dict):
            continue
        raw = section_cfg.get(key)
        if raw in (None, ""):
            continue
        section_cfg[key] = str(_resolve_path(raw, config_dir))
    return config


def _load_demo_config_recursive(
    config_path: Path,
    *,
    _seen: set[Path] | None = None,
) -> Dict[str, Any]:
    resolved_path = config_path.resolve()
    seen = _seen or set()
    if resolved_path in seen:
        raise ValueError(f"检测到循环 base_config 引用：{resolved_path}")
    seen.add(resolved_path)

    config = load_config(resolved_path)
    config = _resolve_known_config_paths(config, config_dir=resolved_path.parent)
    base_ref = str(config.get("base_config") or "").strip()
    if base_ref:
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = (resolved_path.parent / base_path).resolve()
        base_config = _load_demo_config_recursive(base_path, _seen=seen)
        overlay = {k: v for k, v in config.items() if k != "base_config"}
        config = _merge_dicts(base_config, overlay)
        config["_base_config_path"] = str(base_path)
        config["_config_path"] = str(resolved_path)
        config["_config_dir"] = str(resolved_path.parent)

    return config


def load_demo_config(config_path: Path) -> Dict[str, Any]:
    config = _load_demo_config_recursive(config_path)
    compare_cfg = config.get("compare")
    if not isinstance(compare_cfg, dict):
        compare_cfg = {}
        config["compare"] = compare_cfg
    return config


def prepare_run(
    config_path: Path,
    *,
    run_prefix: str,
) -> tuple[Dict[str, Any], Path]:
    config = load_demo_config(config_path)
    _patch_geoxgboost_parallelism()
    output_dir = ensure_run_output_dir(config, prefix=run_prefix)
    setup_logging(config, output_dir)
    logging.info(f"输出目录: {output_dir.resolve()}")
    base_config_path = config.get("_base_config_path")
    if base_config_path:
        logging.info(f"继承的基础配置: {base_config_path}")
    return config, output_dir


def _subset_demo_rows(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    coords: pd.DataFrame,
    compare_cfg: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    row_limit = int(compare_cfg.get("row_limit", 0))
    if row_limit <= 0 or row_limit >= len(df):
        return df, X, y, coords

    logging.info(f"compare.row_limit={row_limit}，将只使用前 {row_limit} 行做 demo。")
    df_out = df.iloc[:row_limit].copy()
    X_out = X.iloc[:row_limit].copy()
    y_out = y.iloc[:row_limit].copy()
    coords_out = coords.iloc[:row_limit].copy()
    return df_out, X_out, y_out, coords_out


def build_importance_table(
    feature_names: Iterable[str],
    mean_abs_values: np.ndarray,
    *,
    prefix: str,
) -> pd.DataFrame:
    values = np.asarray(mean_abs_values, dtype=float).reshape(-1)
    df_imp = pd.DataFrame(
        {
            "feature": list(feature_names),
            f"{prefix}_mean_abs_shap": values,
        }
    )
    total = float(np.nansum(values))
    if np.isfinite(total) and total > 0.0:
        shares = values / total
    else:
        shares = np.full_like(values, fill_value=np.nan, dtype=float)
    df_imp[f"{prefix}_importance_share"] = shares
    df_imp.sort_values(f"{prefix}_importance_share", ascending=False, inplace=True)
    df_imp.reset_index(drop=True, inplace=True)
    df_imp[f"{prefix}_rank"] = np.arange(1, len(df_imp) + 1, dtype=int)
    return df_imp


def local_compare_meta_cols() -> List[str]:
    return [
        "model_variant",
        "center_in_train",
        "train_use_spatial_weights",
        "center_label",
        "center_pos",
        "center_index",
        "center_city",
        "bw",
        "kernel",
        "n_neighbors",
    ]


def local_compare_feature_names(
    X: pd.DataFrame,
    value_df: pd.DataFrame,
) -> List[str]:
    return [feature for feature in X.columns if feature in value_df.columns]


def _sign_with_tolerance(values: pd.Series | np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.zeros(arr.shape, dtype=int)
    out[arr > tol] = 1
    out[arr < -tol] = -1
    return out


def compare_plot_max_display(compare_cfg: Dict[str, Any], n_features: int) -> int:
    raw = int(compare_cfg.get("plot_top_n", 10))
    return max(1, min(int(n_features), raw))


def extract_shap_matrix_from_frame(
    shap_df: pd.DataFrame,
    feature_names: Iterable[str],
    *,
    prefix: str = "shap_",
) -> np.ndarray:
    cols = [f"{prefix}{feature}" for feature in feature_names]
    missing = [col for col in cols if col not in shap_df.columns]
    if missing:
        raise KeyError(f"缺少 SHAP 列: {missing}")
    return shap_df.loc[:, cols].to_numpy(dtype=float, copy=False)


def extract_feature_rows_by_position(
    X: pd.DataFrame,
    positions: Iterable[Any],
    *,
    label: str,
) -> pd.DataFrame:
    pos_series = pd.to_numeric(pd.Series(list(positions)), errors="coerce")
    valid_mask = pos_series.notna() & (pos_series >= 0) & (pos_series < len(X))
    if not bool(valid_mask.all()):
        invalid_count = int((~valid_mask).sum())
        logging.warning(f"{label} 中有 {invalid_count} 个无效 sample_pos，已跳过。")
    valid_positions = pos_series.loc[valid_mask].astype(int).to_numpy()
    return X.iloc[valid_positions].reset_index(drop=True)


def save_shap_native_plots(
    shap_values: np.ndarray,
    *,
    path_summary: Path | None,
    path_bar: Path | None,
    features: pd.DataFrame | None,
    feature_names: Iterable[str],
    max_display: int,
    summary_title: str,
    bar_title: str,
) -> None:
    _configure_plot_fonts()
    shap_array = np.asarray(shap_values, dtype=float)
    if shap_array.ndim != 2 or shap_array.shape[0] == 0 or shap_array.shape[1] == 0:
        logging.warning("SHAP 原生图未生成：shap_values 为空或维度不合法。")
        return

    feat_names = list(feature_names)
    max_display = max(1, min(int(max_display), shap_array.shape[1]))

    if path_summary is not None:
        plt.figure()
        shap.summary_plot(
            shap_array,
            features=features,
            feature_names=feat_names,
            max_display=max_display,
            show=False,
        )
        plt.title(summary_title)
        plt.tight_layout()
        plt.savefig(path_summary, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"SHAP 原生 summary 图已保存: {path_summary}")

    if path_bar is not None:
        plt.figure()
        shap.summary_plot(
            shap_array,
            features=features,
            feature_names=feat_names,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.title(bar_title)
        plt.tight_layout()
        plt.savefig(path_bar, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"SHAP 原生 bar 图已保存: {path_bar}")


def save_global_native_plots(
    artifacts: CompareArtifacts,
    *,
    output_dir: Path,
    compare_cfg: Dict[str, Any],
) -> None:
    max_display = compare_plot_max_display(compare_cfg, artifacts.X.shape[1])
    save_shap_native_plots(
        artifacts.global_shap_values,
        path_summary=output_dir / "global_shap_summary.png",
        path_bar=output_dir / "global_shap_bar.png",
        features=artifacts.X.reset_index(drop=True),
        feature_names=artifacts.X.columns,
        max_display=max_display,
        summary_title="Global SHAP Summary",
        bar_title="Global SHAP Mean |SHAP|",
    )


def save_center_local_native_plots(
    artifacts: CompareArtifacts,
    *,
    output_dir: Path,
    compare_cfg: Dict[str, Any],
) -> None:
    if artifacts.local_center_df.empty:
        logging.warning("中心样本局部 SHAP 为空，跳过 SHAP 原生中心图。")
        return
    feature_names = list(artifacts.X.columns)
    center_shap = extract_shap_matrix_from_frame(artifacts.local_center_df, feature_names)
    center_X = extract_feature_rows_by_position(
        artifacts.X,
        artifacts.local_center_df["center_pos"],
        label="center_local",
    )
    n_rows = min(len(center_X), center_shap.shape[0])
    if n_rows <= 0:
        logging.warning("中心样本局部 SHAP 原生图未生成：缺少可对齐的中心特征行。")
        return
    max_display = compare_plot_max_display(compare_cfg, len(feature_names))
    save_shap_native_plots(
        center_shap[:n_rows],
        path_summary=output_dir / "center_local_shap_summary.png",
        path_bar=output_dir / "center_local_shap_bar.png",
        features=center_X.iloc[:n_rows].reset_index(drop=True),
        feature_names=feature_names,
        max_display=max_display,
        summary_title="Center-local SHAP Summary",
        bar_title="Center-local SHAP Mean |SHAP|",
    )


def save_local_importance_native_plots(
    artifacts: CompareArtifacts,
    *,
    output_dir: Path,
    compare_cfg: Dict[str, Any],
) -> None:
    local_imp_df = artifacts.local_importance_df.copy()
    if local_imp_df.empty:
        logging.warning("局部模型重要性为空，跳过 SHAP 原生局部重要性图。")
        return
    feature_names = local_compare_feature_names(artifacts.X, local_imp_df)
    if not feature_names:
        logging.warning("局部模型重要性原生图未生成：未找到可用特征列。")
        return
    matrix = local_imp_df.loc[:, feature_names].to_numpy(dtype=float, copy=False)
    center_X = extract_feature_rows_by_position(
        artifacts.X,
        local_imp_df["center_pos"],
        label="local_importance_center",
    )
    n_rows = min(len(center_X), matrix.shape[0])
    if n_rows <= 0:
        logging.warning("局部模型重要性原生图未生成：缺少可对齐的中心特征行。")
        return
    max_display = compare_plot_max_display(compare_cfg, len(feature_names))
    save_shap_native_plots(
        matrix[:n_rows],
        path_summary=output_dir / "local_model_importance_summary.png",
        path_bar=output_dir / "local_model_importance_bar.png",
        features=center_X.iloc[:n_rows].reset_index(drop=True),
        feature_names=feature_names,
        max_display=max_display,
        summary_title="Per-local-model mean(|SHAP|) Summary",
        bar_title="Per-local-model mean(|SHAP|) Bar",
    )
    save_local_signed_native_plots(
        artifacts,
        output_dir=output_dir,
        compare_cfg=compare_cfg,
    )


def save_pooled_local_native_plots(
    artifacts: CompareArtifacts,
    *,
    output_dir: Path,
    compare_cfg: Dict[str, Any],
) -> None:
    pooled_df = artifacts.local_pooled_df.copy()
    if pooled_df.empty:
        logging.warning("pooled 局部 SHAP 为空，跳过 SHAP 原生 pooled 图。")
        return
    feature_names = list(artifacts.X.columns)
    pooled_shap = extract_shap_matrix_from_frame(pooled_df, feature_names)
    pooled_X = extract_feature_rows_by_position(
        artifacts.X,
        pooled_df["sample_pos"],
        label="pooled_local",
    )
    n_rows = min(len(pooled_X), pooled_shap.shape[0])
    if n_rows <= 0:
        logging.warning("pooled 局部 SHAP 原生图未生成：缺少可对齐的样本特征行。")
        return
    max_display = compare_plot_max_display(compare_cfg, len(feature_names))
    save_shap_native_plots(
        pooled_shap[:n_rows],
        path_summary=output_dir / "pooled_local_shap_summary.png",
        path_bar=output_dir / "pooled_local_shap_bar.png",
        features=pooled_X.iloc[:n_rows].reset_index(drop=True),
        feature_names=feature_names,
        max_display=max_display,
        summary_title="Pooled Local SHAP Summary",
        bar_title="Pooled Local SHAP Mean |SHAP|",
    )


def build_sample_aggregated_local_df(artifacts: CompareArtifacts) -> pd.DataFrame:
    pooled_df = artifacts.local_pooled_df.copy()
    if pooled_df.empty:
        raise ValueError("pooled 局部 SHAP 为空，无法按 sample 聚合。")

    feature_names = list(artifacts.X.columns)
    shap_cols = [f"shap_{feature}" for feature in feature_names]
    missing = [col for col in shap_cols if col not in pooled_df.columns]
    if missing:
        raise KeyError(f"pooled 局部 SHAP 缺少列: {missing}")

    grouped = pooled_df.groupby("sample_pos", sort=True, dropna=False)
    meta_df = grouped.agg(
        sample_index=("sample_index", "first"),
        y=("y", "first"),
        reuse_count=("sample_pos", "size"),
        n_center_models=("center_label", "nunique"),
        mean_distance=("distance", "mean"),
        min_distance=("distance", "min"),
        max_distance=("distance", "max"),
    )
    shap_mean_df = grouped[shap_cols].mean()
    aggregated_df = meta_df.join(shap_mean_df, how="inner").reset_index()
    aggregated_df["sample_pos"] = pd.to_numeric(
        aggregated_df["sample_pos"], errors="coerce"
    ).astype("Int64")
    return aggregated_df


def save_sample_aggregated_local_native_plots(
    artifacts: CompareArtifacts,
    sample_aggregated_df: pd.DataFrame,
    *,
    output_dir: Path,
    compare_cfg: Dict[str, Any],
) -> None:
    if sample_aggregated_df.empty:
        logging.warning("sample 聚合后的局部 SHAP 为空，跳过原生图。")
        return

    feature_names = list(artifacts.X.columns)
    sample_shap = extract_shap_matrix_from_frame(sample_aggregated_df, feature_names)
    sample_X = extract_feature_rows_by_position(
        artifacts.X,
        sample_aggregated_df["sample_pos"],
        label="sample_aggregated_local",
    )
    n_rows = min(len(sample_X), sample_shap.shape[0])
    if n_rows <= 0:
        logging.warning("sample 聚合后的局部 SHAP 原生图未生成：缺少可对齐的样本特征行。")
        return

    sample_shap = sample_shap[:n_rows]
    sample_X = sample_X.iloc[:n_rows].reset_index(drop=True)
    max_display = compare_plot_max_display(compare_cfg, len(feature_names))
    summary_path = output_dir / "sample_aggregated_local_summary.png"
    bar_path = output_dir / "sample_aggregated_local_bar.png"
    save_shap_native_plots(
        sample_shap,
        path_summary=summary_path,
        path_bar=bar_path,
        features=sample_X,
        feature_names=feature_names,
        max_display=max_display,
        summary_title="Sample-aggregated Local SHAP Summary",
        bar_title="Sample-aggregated Local SHAP Mean |SHAP|",
    )

    mean_value_df = build_importance_table(
        feature_names,
        np.mean(np.abs(sample_shap), axis=0),
        prefix="sample_aggregated_local",
    )
    mean_value_path = output_dir / "sample_aggregated_local_shap_mean_value.csv"
    mean_value_df.to_csv(mean_value_path, index=False, encoding="utf-8-sig")
    logging.info("sample 聚合 SHAP mean value 表已保存: %s", mean_value_path.resolve())

    combined_path = output_dir / "sample_aggregated_local_shap_sum.png"
    save_combined_mean_value_summary_plot(
        shap_values=sample_shap,
        features=sample_X,
        feature_names=feature_names,
        max_display=max_display,
        output_path=combined_path,
        title="Sample-aggregated Local SHAP",
    )
    logging.info("sample 聚合 SHAP 综合图已保存: %s", combined_path.resolve())


def save_combined_mean_value_summary_plot(
    *,
    shap_values: np.ndarray,
    features: pd.DataFrame,
    feature_names: Iterable[str],
    max_display: int,
    output_path: Path,
    title: str,
) -> None:
    _configure_plot_fonts()
    shap_array = np.asarray(shap_values, dtype=float)
    if shap_array.ndim != 2 or shap_array.shape[0] == 0 or shap_array.shape[1] == 0:
        raise ValueError("shap_values 为空或维度不合法，无法生成综合图。")

    feat_names = list(feature_names)
    max_display = max(1, min(int(max_display), shap_array.shape[1]))
    explanation = shap.Explanation(
        values=shap_array,
        data=features.reset_index(drop=True),
        feature_names=feat_names,
    )

    fig_width = max(9.6, 0.58 * max_display + 3.8)
    fig_height = max(4.8, 0.56 * max_display + 1.3)
    fig, ax_sum = plt.subplots(figsize=(fig_width, fig_height))
    ax_mean = ax_sum.twiny()
    ax_mean.set_zorder(0)
    ax_sum.set_zorder(1)
    ax_sum.patch.set_alpha(0.0)
    ax_mean.patch.set_alpha(0.0)

    shap.plots.beeswarm(
        explanation,
        max_display=max_display,
        ax=ax_sum,
        show=False,
        group_remaining_features=False,
        color_bar=True,
        plot_size=None,
    )

    y_ticks = ax_sum.get_yticks()
    displayed_features = [tick.get_text() for tick in ax_sum.get_yticklabels()]
    mean_abs = np.mean(np.abs(shap_array), axis=0)
    mean_map = {name: float(value) for name, value in zip(feat_names, mean_abs)}
    bar_values = np.asarray(
        [mean_map.get(feature, np.nan) for feature in displayed_features],
        dtype=float,
    )

    finite_bar = bar_values[np.isfinite(bar_values)]
    if finite_bar.size > 0:
        vmax = float(finite_bar.max())
        vmin = float(finite_bar.min())
    else:
        vmax = 1.0
        vmin = 0.0
    if np.isclose(vmin, vmax):
        vmin = 0.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    bar_cmap = shap.plots.colors.red_blue
    bar_colors = [bar_cmap(norm(value)) if np.isfinite(value) else "#cccccc" for value in bar_values]

    ax_mean.barh(
        y_ticks,
        bar_values,
        height=0.72,
        color=bar_colors,
        edgecolor="#666666",
        linewidth=0.8,
        alpha=0.42,
        zorder=0,
    )
    ax_mean.set_ylim(ax_sum.get_ylim())
    ax_mean.set_yticks(y_ticks)
    ax_mean.tick_params(
        axis="y",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )
    ax_mean.grid(False)
    ax_mean.xaxis.set_ticks_position("top")
    ax_mean.xaxis.set_label_position("top")
    ax_mean.set_xlabel("Mean |SHAP value|", fontsize=12, labelpad=6)
    ax_mean.tick_params(axis="x", pad=3)
    if finite_bar.size > 0:
        ax_mean.set_xlim(0.0, float(finite_bar.max()) * 1.1)
    else:
        ax_mean.set_xlim(0.0, 1.0)

    ax_sum.set_title(title, fontsize=16, pad=16)
    ax_sum.set_xlabel("SHAP value (impact on model output)", fontsize=12)
    ax_sum.set_ylabel("Features", fontsize=12)
    ax_sum.set_yticks(y_ticks)
    ax_sum.set_yticklabels(displayed_features)
    ax_sum.tick_params(axis="x", pad=3)
    ax_sum.tick_params(axis="y", pad=8)

    max_label_chars = max((len(str(label)) for label in displayed_features), default=0)
    left_margin = min(0.44, max(0.28, 0.18 + 0.011 * max_label_chars))
    fig.subplots_adjust(left=left_margin, right=0.84, top=0.88, bottom=0.12)
    ax_mean.set_position(ax_sum.get_position())

    extra_axes = [axis for axis in fig.axes if axis not in (ax_sum, ax_mean)]
    if extra_axes:
        colorbar_ax = extra_axes[0]
        colorbar_ax.set_position([0.87, 0.14, 0.022, 0.72])
        colorbar_ax.tick_params(pad=2)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_sample_aggregated_local_dependence_plots(
    artifacts: CompareArtifacts,
    sample_aggregated_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> None:
    _configure_plot_fonts()
    if sample_aggregated_df.empty:
        logging.warning("sample 聚合后的局部 SHAP 为空，跳过 dependence 图。")
        return

    feature_names = list(artifacts.X.columns)
    sample_shap = extract_shap_matrix_from_frame(sample_aggregated_df, feature_names)
    sample_X = extract_feature_rows_by_position(
        artifacts.X,
        sample_aggregated_df["sample_pos"],
        label="sample_aggregated_local_dependence",
    )
    n_rows = min(len(sample_X), sample_shap.shape[0])
    if n_rows <= 0:
        logging.warning("sample 聚合后的局部 SHAP dependence 图未生成：缺少可对齐的样本特征行。")
        return

    dep_config = deepcopy(artifacts.config)
    output_cfg = dep_config.get("output")
    if not isinstance(output_cfg, dict):
        output_cfg = {}
        dep_config["output"] = output_cfg
    output_cfg["dependence_prefix"] = "sample_aggregated_local_dependence_"

    plot_shap_dependence(
        sample_shap[:n_rows],
        sample_X.iloc[:n_rows].reset_index(drop=True),
        dep_config,
        output_dir,
    )


def save_local_signed_native_plots(
    artifacts: CompareArtifacts,
    *,
    output_dir: Path,
    compare_cfg: Dict[str, Any],
) -> None:
    local_signed_df = artifacts.local_signed_df.copy()
    if local_signed_df.empty:
        logging.warning("局部模型 signed SHAP 为空，跳过方向 summary 图。")
        return
    feature_names = local_compare_feature_names(artifacts.X, local_signed_df)
    if not feature_names:
        logging.warning("局部模型 signed SHAP 图未生成：未找到可用特征列。")
        return
    matrix = local_signed_df.loc[:, feature_names].to_numpy(dtype=float, copy=False)
    center_X = extract_feature_rows_by_position(
        artifacts.X,
        local_signed_df["center_pos"],
        label="local_signed_center",
    )
    n_rows = min(len(center_X), matrix.shape[0])
    if n_rows <= 0:
        logging.warning("局部模型 signed SHAP 图未生成：缺少可对齐的中心特征行。")
        return
    max_display = compare_plot_max_display(compare_cfg, len(feature_names))
    save_shap_native_plots(
        matrix[:n_rows],
        path_summary=output_dir / "local_model_signed_summary.png",
        path_bar=None,
        features=center_X.iloc[:n_rows].reset_index(drop=True),
        feature_names=feature_names,
        max_display=max_display,
        summary_title="Per-local-model mean(SHAP) Summary",
        bar_title="",
    )


def build_local_importance_outputs(artifacts: CompareArtifacts) -> Dict[str, pd.DataFrame]:
    local_imp_df = artifacts.local_importance_df.copy()
    if local_imp_df.empty:
        raise ValueError("未采集到局部模型重要性表。")

    meta_cols = [col for col in local_compare_meta_cols() if col in local_imp_df.columns]
    feature_names = local_compare_feature_names(artifacts.X, local_imp_df)

    long_df = local_imp_df.melt(
        id_vars=meta_cols,
        value_vars=feature_names,
        var_name="feature",
        value_name="local_mean_abs_shap",
    )
    per_model_total = (
        long_df.groupby("center_label", dropna=False)["local_mean_abs_shap"]
        .sum()
        .rename("local_model_total")
        .reset_index()
    )
    long_df = long_df.merge(per_model_total, on="center_label", how="left")
    long_df["local_importance_share"] = np.where(
        long_df["local_model_total"] > 0,
        long_df["local_mean_abs_shap"] / long_df["local_model_total"],
        np.nan,
    )

    summary_df = (
        long_df.groupby("feature", dropna=False)
        .agg(
            local_model_mean_abs_shap=("local_mean_abs_shap", "mean"),
            local_model_median_abs_shap=("local_mean_abs_shap", "median"),
            local_model_std_abs_shap=("local_mean_abs_shap", "std"),
            local_model_mean_share=("local_importance_share", "mean"),
            local_model_median_share=("local_importance_share", "median"),
            local_model_std_share=("local_importance_share", "std"),
            local_model_q25_share=("local_importance_share", lambda s: float(s.quantile(0.25))),
            local_model_q75_share=("local_importance_share", lambda s: float(s.quantile(0.75))),
            n_models=("center_label", "nunique"),
        )
        .reset_index()
    )
    summary_df["local_model_iqr_share"] = (
        summary_df["local_model_q75_share"] - summary_df["local_model_q25_share"]
    )
    summary_df["local_model_cv_abs_shap"] = np.where(
        summary_df["local_model_mean_abs_shap"] > 0,
        summary_df["local_model_std_abs_shap"] / summary_df["local_model_mean_abs_shap"],
        np.nan,
    )

    compare_df = artifacts.global_importance_df.merge(
        summary_df,
        on="feature",
        how="left",
    )
    compare_df["share_diff_local_median_minus_global"] = (
        compare_df["local_model_median_share"] - compare_df["global_importance_share"]
    )
    compare_df.sort_values("global_importance_share", ascending=False, inplace=True)
    compare_df.reset_index(drop=True, inplace=True)

    local_signed_df = artifacts.local_signed_df.copy()
    signed_long_df = local_signed_df.melt(
        id_vars=[col for col in meta_cols if col in local_signed_df.columns],
        value_vars=feature_names,
        var_name="feature",
        value_name="local_mean_shap",
    )
    signed_summary_df = (
        signed_long_df.groupby("feature", dropna=False)
        .agg(
            local_model_mean_signed_shap=("local_mean_shap", "mean"),
            local_model_median_signed_shap=("local_mean_shap", "median"),
            local_model_std_signed_shap=("local_mean_shap", "std"),
            local_model_q25_signed_shap=("local_mean_shap", lambda s: float(s.quantile(0.25))),
            local_model_q75_signed_shap=("local_mean_shap", lambda s: float(s.quantile(0.75))),
            n_models=("center_label", "nunique"),
        )
        .reset_index()
    )
    signed_summary_df["local_model_iqr_signed_shap"] = (
        signed_summary_df["local_model_q75_signed_shap"]
        - signed_summary_df["local_model_q25_signed_shap"]
    )

    local_positive_df = artifacts.local_positive_df.copy()
    local_negative_df = artifacts.local_negative_df.copy()
    positive_long_df = local_positive_df.melt(
        id_vars=[col for col in meta_cols if col in local_positive_df.columns],
        value_vars=feature_names,
        var_name="feature",
        value_name="local_positive_mean_shap",
    )
    negative_long_df = local_negative_df.melt(
        id_vars=[col for col in meta_cols if col in local_negative_df.columns],
        value_vars=feature_names,
        var_name="feature",
        value_name="local_negative_mean_abs_shap",
    )
    pos_neg_long_df = positive_long_df.merge(
        negative_long_df,
        on=meta_cols + ["feature"],
        how="outer",
    )
    pos_neg_summary_df = (
        pos_neg_long_df.groupby("feature", dropna=False)
        .agg(
            local_model_mean_positive_shap=("local_positive_mean_shap", "mean"),
            local_model_median_positive_shap=("local_positive_mean_shap", "median"),
            local_model_mean_negative_abs_shap=("local_negative_mean_abs_shap", "mean"),
            local_model_median_negative_abs_shap=("local_negative_mean_abs_shap", "median"),
            n_models=("center_label", "nunique"),
        )
        .reset_index()
    )
    pos_neg_summary_df["direction_gap_mean_shap"] = (
        pos_neg_summary_df["local_model_mean_positive_shap"]
        - pos_neg_summary_df["local_model_mean_negative_abs_shap"]
    )
    pos_neg_summary_df["dominant_direction"] = np.where(
        pos_neg_summary_df["direction_gap_mean_shap"] > 1e-12,
        "positive",
        np.where(
            pos_neg_summary_df["direction_gap_mean_shap"] < -1e-12,
            "negative",
            "mixed",
        ),
    )

    global_signed_df = pd.DataFrame(
        {
            "feature": feature_names,
            "global_mean_shap": np.mean(artifacts.global_shap_values, axis=0),
            "global_median_shap": np.median(artifacts.global_shap_values, axis=0),
        }
    )
    global_signed_df = global_signed_df.merge(
        artifacts.global_importance_df[["feature", "global_mean_abs_shap", "global_importance_share"]],
        on="feature",
        how="left",
    )

    signed_compare_df = global_signed_df.merge(
        signed_summary_df,
        on="feature",
        how="left",
    ).merge(
        pos_neg_summary_df,
        on="feature",
        how="left",
    )
    signed_compare_df["abs_gap_mean_signed_shap"] = np.abs(
        signed_compare_df["local_model_mean_signed_shap"] - signed_compare_df["global_mean_shap"]
    )
    signed_compare_df["sign_global_mean_shap"] = _sign_with_tolerance(
        signed_compare_df["global_mean_shap"]
    )
    signed_compare_df["sign_local_mean_signed_shap"] = _sign_with_tolerance(
        signed_compare_df["local_model_mean_signed_shap"]
    )
    signed_compare_df["sign_agreement_mean"] = np.where(
        (signed_compare_df["sign_global_mean_shap"] != 0)
        & (signed_compare_df["sign_local_mean_signed_shap"] != 0),
        (
            signed_compare_df["sign_global_mean_shap"]
            == signed_compare_df["sign_local_mean_signed_shap"]
        ).astype(int),
        np.nan,
    )
    signed_compare_df.sort_values("global_mean_abs_shap", ascending=False, inplace=True)
    signed_compare_df.reset_index(drop=True, inplace=True)

    return {
        "local_importance_long_df": long_df,
        "local_importance_summary_df": summary_df,
        "local_importance_compare_df": compare_df,
        "local_signed_long_df": signed_long_df,
        "local_signed_summary_df": signed_summary_df,
        "local_pos_neg_summary_df": pos_neg_summary_df,
        "global_vs_local_signed_df": signed_compare_df,
    }


def _local_shap_bundle(
    result: Dict[str, Any],
    *,
    config: Dict[str, Any],
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    coords: pd.DataFrame,
    bw: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    local_cfg = config.get("local_shap") or {}
    max_models = int(local_cfg.get("max_models", 0))
    log_every = int(local_cfg.get("log_every", 10))
    model_variant = str(local_cfg.get("model_variant", "oob")).strip().lower() or "oob"
    if model_variant not in {"oob", "in_sample"}:
        logging.warning(
            f"local_shap.model_variant={model_variant} 无效，将回退为 oob。支持: oob / in_sample"
        )
        model_variant = "oob"

    n = len(X)
    if n == 0:
        raise ValueError("数据为空，无法构造局部 SHAP 对比结果。")

    models = result.get("bestLocalModel")
    if not isinstance(models, (list, tuple)) or len(models) == 0:
        keys = sorted(str(k) for k in result.keys())
        raise ValueError(
            "result 中未找到可用的 `bestLocalModel`。"
            f"可用 keys: {keys}"
        )

    feature_names = [str(c) for c in X.columns]
    shap_col_names = [f"shap_{name}" for name in feature_names]
    dist_mat = compute_gw_distance_matrix(config, coords, coords)

    gw_cfg = config.get("gw") or {}
    kernel = str(gw_cfg.get("kernel", "Adaptive"))
    spatial_weights = bool(gw_cfg.get("spatial_weights", False))
    feat_importance = gw_cfg.get("feat_importance", "gain")

    in_sample_use_weights = local_cfg.get(
        "in_sample_use_spatial_weights", spatial_weights
    )
    in_sample_use_weights = _as_bool(in_sample_use_weights, default=spatial_weights)

    if model_variant == "in_sample":
        logging.warning(
            "局部对比将使用包含中心点的局部模型进行 SHAP（in-sample）。"
            "该解释更偏拟合描述。"
        )
    else:
        logging.info("局部对比将使用 geoxgboost 输出的 oob 局部模型。")

    shap_cfg = dict(config.get("shap") or {})
    shap_cfg["engine"] = "xgboost"
    shap_cfg["compute_interactions"] = 0
    local_config = dict(config)
    local_config["shap"] = shap_cfg

    city_col = _resolve_city_col(config, df)
    pooled_frames: List[pd.DataFrame] = []
    center_rows: List[Dict[str, Any]] = []
    importance_rows: List[Dict[str, Any]] = []
    signed_rows: List[Dict[str, Any]] = []
    positive_rows: List[Dict[str, Any]] = []
    negative_rows: List[Dict[str, Any]] = []
    n_processed = 0

    for pos, model in _iter_local_models(result):
        if max_models > 0 and n_processed >= max_models:
            break
        if pos >= n:
            continue

        dist_col = np.asarray(dist_mat[:, pos], dtype=float)
        local_positions = _select_local_positions(dist_col, bw=bw, kernel=kernel)
        local_X = X.iloc[local_positions]

        center_in_train = 0
        train_use_spatial_weights = int(spatial_weights)
        model_for_shap = model
        if model_variant == "in_sample":
            center_in_train = 1
            train_use_spatial_weights = int(in_sample_use_weights)
            model_cfg = config.get("model") or {}
            base_params: Dict[str, Any] = dict(model_cfg.get("params") or {})
            random_state = int(model_cfg.get("random_state", 42))
            base_params.setdefault("random_state", random_state)
            base_params.setdefault("n_jobs", 1)
            model_for_shap = xgb.XGBRegressor(
                **base_params, importance_type=feat_importance
            )
            local_y_full = y.iloc[local_positions]
            if in_sample_use_weights:
                weights = _spatial_weights_for_local_data(
                    dist_col[local_positions], bw=bw, kernel=kernel
                )
                model_for_shap.fit(local_X, local_y_full, sample_weight=weights)
            else:
                model_for_shap.fit(local_X, local_y_full)

        shap_values, _ = compute_shap_and_interactions(
            model_for_shap, local_X, local_config
        )
        shap_array = np.asarray(shap_values, dtype=float)

        center_index = X.index[pos]
        center_city = (
            _get_city_name(
                df, city_col=city_col, center_index=center_index, center_pos=pos
            )
            if city_col is not None
            else None
        )
        center_label = (
            f"{pos + 1:04d}_{center_city}" if center_city else f"{pos + 1:04d}"
        )

        mean_abs = np.mean(np.abs(shap_array), axis=0)
        mean_signed = np.mean(shap_array, axis=0)
        mean_positive = np.mean(np.clip(shap_array, a_min=0.0, a_max=None), axis=0)
        mean_negative = np.mean(np.abs(np.clip(shap_array, a_min=None, a_max=0.0)), axis=0)
        rec_meta = {
            "model_variant": model_variant,
            "center_in_train": int(center_in_train),
            "train_use_spatial_weights": int(train_use_spatial_weights),
            "center_label": center_label,
            "center_pos": int(pos),
            "center_index": center_index,
            "center_city": center_city,
            "bw": bw,
            "kernel": kernel,
            "n_neighbors": int(len(local_positions)),
        }
        rec = dict(rec_meta)
        rec_signed = dict(rec_meta)
        rec_positive = dict(rec_meta)
        rec_negative = dict(rec_meta)
        for idx, feature in enumerate(feature_names):
            rec[feature] = float(mean_abs[idx])
            rec_signed[feature] = float(mean_signed[idx])
            rec_positive[feature] = float(mean_positive[idx])
            rec_negative[feature] = float(mean_negative[idx])
        importance_rows.append(rec)
        signed_rows.append(rec_signed)
        positive_rows.append(rec_positive)
        negative_rows.append(rec_negative)

        pooled_frame = pd.DataFrame(shap_array, columns=shap_col_names)
        pooled_frame.insert(0, "model_variant", model_variant)
        pooled_frame.insert(0, "center_in_train", int(center_in_train))
        pooled_frame.insert(0, "train_use_spatial_weights", int(train_use_spatial_weights))
        pooled_frame.insert(0, "distance", dist_col[local_positions])
        pooled_frame.insert(0, "distance_unit", gw_distance_unit(config))
        pooled_frame.insert(0, "distance_metric", gw_distance_metric_label(config))
        pooled_frame.insert(0, "sample_pos", local_positions)
        pooled_frame.insert(0, "sample_index", local_X.index.to_numpy())
        pooled_frame.insert(0, "y", y.iloc[local_positions].to_numpy())
        pooled_frame.insert(0, "center_pos", pos)
        pooled_frame.insert(0, "center_index", center_index)
        pooled_frame.insert(0, "center_city", center_city)
        pooled_frame.insert(0, "center_label", center_label)
        pooled_frame.insert(0, "kernel", kernel)
        pooled_frame.insert(0, "bw", bw)
        pooled_frames.append(pooled_frame)

        center_mask = np.asarray(local_positions, dtype=int) == int(pos)
        center_local_idx = np.flatnonzero(center_mask)
        if center_local_idx.size == 1:
            center_row = {
                "model_variant": model_variant,
                "center_in_train": int(center_in_train),
                "train_use_spatial_weights": int(train_use_spatial_weights),
                "center_label": center_label,
                "center_pos": int(pos),
                "center_index": center_index,
                "center_city": center_city,
                "bw": bw,
                "kernel": kernel,
                "n_neighbors": int(len(local_positions)),
                "y": float(y.iloc[pos]),
            }
            center_values = shap_array[int(center_local_idx[0])]
            for idx, feature in enumerate(feature_names):
                center_row[f"shap_{feature}"] = float(center_values[idx])
            center_rows.append(center_row)
        else:
            logging.warning(
                f"center_pos={pos} 未能唯一定位中心样本在局部邻域中的 SHAP 行，已跳过中心对比行。"
            )

        n_processed += 1
        if log_every > 0 and n_processed % log_every == 0:
            logging.info(f"局部 SHAP 对比采集进度: {n_processed} 个局部模型。")

    if not pooled_frames:
        raise ValueError("未采集到任何局部 SHAP 结果。")

    pooled_df = pd.concat(pooled_frames, ignore_index=True)
    center_df = pd.DataFrame(center_rows)
    importance_df = pd.DataFrame(importance_rows)
    signed_df = pd.DataFrame(signed_rows)
    positive_df = pd.DataFrame(positive_rows)
    negative_df = pd.DataFrame(negative_rows)
    logging.info(
        "局部 SHAP 采集完成："
        f"局部模型 {n_processed} 个，"
        f"pooled 行数 {len(pooled_df)}，"
        f"center 行数 {len(center_df)}。"
    )
    return pooled_df, center_df, importance_df, signed_df, positive_df, negative_df


def run_compare_pipeline(config: Dict[str, Any], output_dir: Path) -> CompareArtifacts:
    df, X, y, coords = load_dataset(config)
    compare_cfg = config.get("compare") or {}
    df, X, y, coords = _subset_demo_rows(df, X, y, coords, compare_cfg)
    logging.info(
        f"已加载对比数据，共 {df.shape[0]} 行，{df.shape[1]} 列（含坐标列）。"
    )

    optimize_global_model(config, X, y)

    logging.info("开始训练全局 XGBoost 基线模型...")
    global_model = build_and_train_model(config, X, y)
    global_shap_values, global_interaction_values = compute_shap_and_interactions(
        global_model, X, config
    )
    global_importance_df = build_importance_table(
        X.columns,
        np.mean(np.abs(np.asarray(global_shap_values, dtype=float)), axis=0),
        prefix="global",
    )
    global_importance_path = output_dir / "global_feature_importance.csv"
    global_importance_df.to_csv(global_importance_path, index=False, encoding="utf-8-sig")
    logging.info(f"全局 SHAP 重要性表已保存: {global_importance_path}")
    summarize_and_save_interactions(global_interaction_values, X, config, output_dir)
    plot_top_interactions(global_interaction_values, X, config, output_dir)
    plot_fixed_base_interactions(global_interaction_values, X, config, output_dir)

    bw_opt = optimize_bandwidth(config, X, y, coords, output_dir=output_dir)
    logging.info(f"对比脚本使用的带宽 bw = {bw_opt}")
    result_local = run_gxgb(config, X, y, coords, bw=bw_opt, output_dir=output_dir)
    (
        local_pooled_df,
        local_center_df,
        local_importance_df,
        local_signed_df,
        local_positive_df,
        local_negative_df,
    ) = _local_shap_bundle(
        result_local,
        config=config,
        df=df,
        X=X,
        y=y,
        coords=coords,
        bw=bw_opt,
    )

    return CompareArtifacts(
        config=config,
        output_dir=output_dir,
        df=df,
        X=X,
        y=y,
        coords=coords,
        global_shap_values=np.asarray(global_shap_values, dtype=float),
        global_interaction_values=(
            np.asarray(global_interaction_values, dtype=float)
            if global_interaction_values is not None
            else None
        ),
        global_importance_df=global_importance_df,
        local_pooled_df=local_pooled_df,
        local_center_df=local_center_df,
        local_importance_df=local_importance_df,
        local_signed_df=local_signed_df,
        local_positive_df=local_positive_df,
        local_negative_df=local_negative_df,
    )


def compare_metric_dict(
    compare_df: pd.DataFrame,
    *,
    left_share_col: str,
    right_share_col: str,
    top_k: int,
    left_name: str,
    right_name: str,
) -> Dict[str, Any]:
    df_use = compare_df.copy()
    df_use = df_use[np.isfinite(df_use[left_share_col]) & np.isfinite(df_use[right_share_col])]
    if df_use.empty:
        return {
            "spearman_share_corr": float("nan"),
            "pearson_share_corr": float("nan"),
            "top_k_overlap_ratio": float("nan"),
            "top_k": int(top_k),
            "left_name": left_name,
            "right_name": right_name,
        }

    spearman = float(df_use[left_share_col].corr(df_use[right_share_col], method="spearman"))
    pearson = float(df_use[left_share_col].corr(df_use[right_share_col], method="pearson"))
    k = max(1, min(int(top_k), len(df_use)))
    left_top = set(
        df_use.sort_values(left_share_col, ascending=False)["feature"].head(k).tolist()
    )
    right_top = set(
        df_use.sort_values(right_share_col, ascending=False)["feature"].head(k).tolist()
    )
    overlap_ratio = len(left_top & right_top) / float(k)
    return {
        "spearman_share_corr": spearman,
        "pearson_share_corr": pearson,
        "top_k_overlap_ratio": overlap_ratio,
        "top_k": int(k),
        "left_name": left_name,
        "right_name": right_name,
    }


def compare_signed_metric_dict(
    compare_df: pd.DataFrame,
    *,
    left_value_col: str,
    right_value_col: str,
    left_name: str,
    right_name: str,
) -> Dict[str, Any]:
    df_use = compare_df.copy()
    df_use = df_use[np.isfinite(df_use[left_value_col]) & np.isfinite(df_use[right_value_col])]
    if df_use.empty:
        return {
            "signed_spearman_corr": float("nan"),
            "signed_pearson_corr": float("nan"),
            "signed_sign_agreement_ratio": float("nan"),
            "signed_n_nonzero_pairs": 0,
            "signed_left_name": left_name,
            "signed_right_name": right_name,
        }

    spearman = float(df_use[left_value_col].corr(df_use[right_value_col], method="spearman"))
    pearson = float(df_use[left_value_col].corr(df_use[right_value_col], method="pearson"))
    left_sign = _sign_with_tolerance(df_use[left_value_col])
    right_sign = _sign_with_tolerance(df_use[right_value_col])
    valid_sign_mask = (left_sign != 0) & (right_sign != 0)
    if bool(np.any(valid_sign_mask)):
        sign_agreement_ratio = float(np.mean(left_sign[valid_sign_mask] == right_sign[valid_sign_mask]))
        n_nonzero_pairs = int(np.sum(valid_sign_mask))
    else:
        sign_agreement_ratio = float("nan")
        n_nonzero_pairs = 0
    return {
        "signed_spearman_corr": spearman,
        "signed_pearson_corr": pearson,
        "signed_sign_agreement_ratio": sign_agreement_ratio,
        "signed_n_nonzero_pairs": n_nonzero_pairs,
        "signed_left_name": left_name,
        "signed_right_name": right_name,
    }


def save_metrics_report(
    path: Path,
    *,
    title: str,
    metrics: Dict[str, Any],
    extra_lines: Iterable[str] = (),
) -> None:
    lines = [title, ""]
    for key, value in metrics.items():
        lines.append(f"{key}: {value}")
    extra = list(extra_lines)
    if extra:
        lines.append("")
        lines.extend(extra)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info(f"指标报告已保存: {path}")


def plot_share_comparison(
    compare_df: pd.DataFrame,
    *,
    left_share_col: str,
    right_share_col: str,
    left_label: str,
    right_label: str,
    path: Path,
    top_n: int,
    title: str,
) -> None:
    _configure_plot_fonts()
    plot_df = compare_df.copy()
    plot_df["plot_score"] = plot_df[[left_share_col, right_share_col]].max(axis=1)
    plot_df = plot_df.sort_values("plot_score", ascending=False).head(max(1, int(top_n)))
    plot_df = plot_df.iloc[::-1]

    y_pos = np.arange(len(plot_df))
    height = 0.38
    plt.figure(figsize=(8, max(4, 0.45 * len(plot_df))))
    plt.barh(y_pos - height / 2, plot_df[left_share_col], height=height, label=left_label)
    plt.barh(y_pos + height / 2, plot_df[right_share_col], height=height, label=right_label)
    plt.yticks(y_pos, plot_df["feature"])
    plt.xlabel("Importance share")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"对比图已保存: {path}")


def plot_local_share_boxplot(
    long_df: pd.DataFrame,
    *,
    feature_order: List[str],
    value_col: str,
    path: Path,
    title: str,
) -> None:
    _configure_plot_fonts()
    data = [
        long_df.loc[long_df["feature"] == feature, value_col].to_numpy(dtype=float)
        for feature in feature_order
    ]
    data = [arr[np.isfinite(arr)] for arr in data]
    labels = [feature for feature, arr in zip(feature_order, data) if arr.size > 0]
    data = [arr for arr in data if arr.size > 0]
    if not data:
        logging.warning("局部重要性箱线图未生成：没有有效数据。")
        return

    plt.figure(figsize=(max(8, 0.75 * len(labels)), 4.8))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"局部重要性箱线图已保存: {path}")


def plot_sample_reuse_histogram(reuse_df: pd.DataFrame, path: Path, *, title: str) -> None:
    _configure_plot_fonts()
    values = reuse_df["reuse_count"].to_numpy(dtype=float, copy=False)
    values = values[np.isfinite(values)]
    if values.size == 0:
        logging.warning("样本复用直方图未生成：没有有效 reuse_count。")
        return

    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=min(20, max(5, len(np.unique(values)))), color="#4C72B0", edgecolor="white")
    plt.xlabel("Reuse count")
    plt.ylabel("Number of samples")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"样本复用直方图已保存: {path}")
