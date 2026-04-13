from __future__ import annotations

import argparse
import copy
import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb

from xgb_shap import (
    compute_shap_and_interactions,
    ensure_run_output_dir,
    load_config,
    load_dataset,
    setup_logging,
)


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False


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

plt.rcParams["font.family"] = LATIN_FONT_NAME
plt.rcParams["font.serif"] = [LATIN_FONT_NAME, "DejaVu Serif"]
if CJK_FONT_NAME:
    plt.rcParams["font.sans-serif"] = [CJK_FONT_NAME, "DejaVu Sans"]


def _contains_cjk(text: Any) -> bool:
    for ch in str(text):
        code = ord(ch)
        if (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
            or 0xFF00 <= code <= 0xFFEF
        ):
            return True
    return False


def _font_properties(
    text: Any,
    *,
    size: float | int | None = None,
    bold: bool = False,
) -> FontProperties:
    family = CJK_FONT_NAME if (_contains_cjk(text) and CJK_FONT_NAME) else LATIN_FONT_NAME
    return FontProperties(
        family=family,
        size=size,
        weight="bold" if bold else "normal",
    )


COLOR_SCHEMES: Dict[int, str] = {
    1: "plasma",
}

STYLE_SCHEMES: Dict[int, str] = {
    1: "o",
    11: "o",
    12: "s",
    13: "D",
    14: "^",
    15: "v",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "基于 train/test 划分、GridSearchCV 与 SHAP interaction，"
            "生成上三角热块 + 下三角蜂群的交互矩阵图。"
        )
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default=None,
        help="YAML 配置文件路径；若不指定则默认使用脚本同目录下的 config.yaml",
    )
    return parser.parse_args()


def _interaction_matrix_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = config.get("interaction_matrix")
    if isinstance(cfg, dict):
        return cfg
    cfg = {}
    config["interaction_matrix"] = cfg
    return cfg


def _build_param_grid(matrix_cfg: Dict[str, Any]) -> Dict[str, List[Any]]:
    raw = matrix_cfg.get("param_grid")
    if not isinstance(raw, dict) or not raw:
        return {"n_estimators": [100, 200, 300]}

    out: Dict[str, List[Any]] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)) and len(value) > 0:
            out[str(key)] = list(value)
        else:
            raise TypeError(
                f"interaction_matrix.param_grid.{key} 必须是非空列表。"
            )
    if not out:
        raise ValueError("interaction_matrix.param_grid 为空。")
    return out


def _expand_range(min_value: float, max_value: float) -> Tuple[float, float]:
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return -1.0, 1.0
    if np.isclose(min_value, max_value):
        pad = 0.1 if np.isclose(min_value, 0.0) else abs(min_value) * 0.1
        return min_value - pad, max_value + pad
    span = max_value - min_value
    pad = span * 0.08
    return min_value - pad, max_value + pad


def _column_axis_specs(
    shap_interaction_values: np.ndarray,
) -> tuple[List[Tuple[float, float]], List[np.ndarray]]:
    n_features = int(shap_interaction_values.shape[1])
    col_xlims: List[Tuple[float, float]] = []
    col_xticks: List[np.ndarray] = []
    locator = MaxNLocator(nbins=6, min_n_ticks=4)

    for col_idx in range(n_features):
        col_values = np.concatenate(
            [
                np.asarray(shap_interaction_values[:, row_idx, col_idx], dtype=float)
                for row_idx in range(col_idx, n_features)
            ]
        )
        finite_values = col_values[np.isfinite(col_values)]
        if finite_values.size == 0:
            xlim = (-1.0, 1.0)
            xticks = np.array([-1.0, 0.0, 1.0], dtype=float)
        else:
            xlim = _expand_range(
                float(finite_values.min()),
                float(finite_values.max()),
            )
            xticks = locator.tick_values(*xlim)
            xticks = xticks[(xticks >= xlim[0]) & (xticks <= xlim[1])]
            if xticks.size < 3:
                xticks = np.linspace(xlim[0], xlim[1], 3)
        col_xlims.append(xlim)
        col_xticks.append(np.asarray(xticks, dtype=float))

    return col_xlims, col_xticks


def _selected_cmap_name(matrix_cfg: Dict[str, Any]) -> tuple[int, str]:
    scheme_index = int(matrix_cfg.get("scheme_index", 1))
    return scheme_index, COLOR_SCHEMES.get(scheme_index, COLOR_SCHEMES[1])


def _selected_marker(matrix_cfg: Dict[str, Any]) -> tuple[int, str]:
    style_index = int(matrix_cfg.get("style_index", 11))
    return style_index, STYLE_SCHEMES.get(style_index, "o")


def _feature_alias_labels(feature_names: List[str]) -> List[str]:
    return [f"X_{idx + 1}" for idx in range(len(feature_names))]


def _mapping_table_rows(
    feature_names: List[str],
    alias_labels: List[str],
    *,
    pairs_per_row: int = 2,
) -> List[List[str]]:
    rows: List[List[str]] = []
    group_size = max(1, int(pairs_per_row))
    for start in range(0, len(feature_names), group_size):
        row: List[str] = []
        for offset in range(group_size):
            idx = start + offset
            if idx < len(feature_names):
                row.extend([alias_labels[idx], feature_names[idx]])
            else:
                row.extend(["", ""])
        rows.append(row)
    return rows


def _add_feature_mapping_table(
    fig: plt.Figure,
    *,
    feature_names: List[str],
    alias_labels: List[str],
    matrix_cfg: Dict[str, Any],
) -> None:
    pairs_per_row = int(matrix_cfg.get("mapping_pairs_per_row", 2))
    col_labels: List[str] = []
    for _ in range(max(1, pairs_per_row)):
        col_labels.extend(["Code", "Feature"])

    rows = _mapping_table_rows(
        feature_names,
        alias_labels,
        pairs_per_row=pairs_per_row,
    )
    table_fontsize = int(matrix_cfg.get("mapping_fontsize", 10))
    table_title = str(matrix_cfg.get("mapping_title", "Feature Mapping")).strip()

    table_ax = fig.add_axes([0.08, 0.015, 0.80, 0.16])
    table_ax.axis("off")
    table_ax.text(
        0.0,
        1.08,
        table_title,
        ha="left",
        va="bottom",
        fontproperties=_font_properties(
            table_title,
            size=table_fontsize + 1,
            bold=True,
        ),
        color="#333333",
        transform=table_ax.transAxes,
    )

    col_widths: List[float] = []
    for _ in range(max(1, pairs_per_row)):
        col_widths.extend([0.10, 0.40])

    table = table_ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        colLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(table_fontsize)
    table.scale(1.0, 1.35)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("#c8c8c8")
        text = cell.get_text().get_text()
        if row_idx == 0:
            cell.set_facecolor("#e9eef4")
            cell.get_text().set_fontproperties(
                _font_properties(text, size=table_fontsize, bold=True)
            )
            continue

        cell.set_facecolor("#ffffff" if row_idx % 2 == 0 else "#f8f8f8")
        is_alias_col = col_idx % 2 == 0
        cell.get_text().set_fontproperties(
            _font_properties(
                text,
                size=table_fontsize,
                bold=is_alias_col and bool(text),
            )
        )
        if is_alias_col:
            cell.get_text().set_ha("center")


def _interaction_output_names(
    matrix_cfg: Dict[str, Any],
    *,
    scheme_index: int,
    style_index: int,
) -> str:
    stem = str(matrix_cfg.get("output_stem", "shap_int")).strip() or "shap_int"
    base_name = f"{stem}_{scheme_index}_{style_index}"
    return f"{base_name}.png"


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true_arr - y_pred_arr) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
    ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return rmse, r2


def simple_beeswarm(
    x_values: np.ndarray,
    *,
    nbins: int = 40,
    width: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    hist_range = (float(np.min(x_values)), float(np.max(x_values)))
    if hist_range[0] == hist_range[1]:
        hist_range = (hist_range[0] - 0.1, hist_range[1] + 0.1)

    counts, edges = np.histogram(x_values, bins=nbins, range=hist_range)
    bin_indices = np.digitize(x_values, edges) - 1
    bin_indices = np.clip(bin_indices, 0, nbins - 1)
    y_values = np.zeros_like(x_values, dtype=float)
    max_count = int(counts.max()) if counts.size > 0 else 0
    rng = np.random.default_rng(random_state)
    if max_count == 0:
        return rng.uniform(-0.1, 0.1, len(x_values))

    for bin_idx in range(len(counts)):
        point_indices = np.where(bin_indices == bin_idx)[0]
        if len(point_indices) == 0:
            continue
        current_width = (counts[bin_idx] / max_count) * width
        ys = np.linspace(-current_width, current_width, len(point_indices))
        rng.shuffle(ys)
        y_values[point_indices] = ys
    return y_values


def _fit_best_model(
    config: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    matrix_cfg: Dict[str, Any],
) -> tuple[xgb.XGBRegressor, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    model_cfg = config.get("model") or {}
    base_params: Dict[str, Any] = dict(model_cfg.get("params") or {})
    random_state = int(
        matrix_cfg.get("random_state", model_cfg.get("random_state", 42))
    )
    test_size = float(matrix_cfg.get("test_size", model_cfg.get("test_size", 0.2)))
    if not (0.0 < test_size < 1.0):
        raise ValueError(
            f"interaction_matrix.test_size 必须在 (0, 1) 之间，当前 {test_size}。"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    if len(X_train) < 2 or len(X_test) < 1:
        raise ValueError("训练/测试划分后的样本数不足。")

    base_params.setdefault("objective", "reg:squarederror")
    base_params.setdefault("random_state", random_state)
    base_params.setdefault("n_jobs", int(matrix_cfg.get("model_n_jobs", 1)))
    estimator = xgb.XGBRegressor(**base_params)

    use_grid_search = int(matrix_cfg.get("use_grid_search", 1)) == 1
    if not use_grid_search:
        estimator.fit(X_train, y_train)
        logging.info("interaction_matrix.use_grid_search=0，将直接使用 model.params 训练。")
        return estimator, X_train, X_test, y_train, y_test

    param_grid = _build_param_grid(matrix_cfg)
    scoring = str(matrix_cfg.get("scoring", "neg_mean_squared_error")).strip()
    cv_splits = int(matrix_cfg.get("cv_splits", 3))
    verbose = int(matrix_cfg.get("grid_verbose", 1))
    grid_n_jobs = int(matrix_cfg.get("grid_n_jobs", 1))
    if cv_splits < 2 or cv_splits > len(X_train):
        raise ValueError(
            f"interaction_matrix.cv_splits 配置无效：cv_splits={cv_splits}, train_samples={len(X_train)}。"
        )

    logging.info(
        "开始 interaction matrix 专用 GridSearchCV："
        f"param_grid={param_grid}, scoring={scoring}, cv={cv_splits}, verbose={verbose}, n_jobs={grid_n_jobs}"
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_splits,
        verbose=verbose,
        n_jobs=grid_n_jobs,
    )
    search.fit(X_train, y_train)
    logging.info(f"最佳参数: {search.best_params_}")
    logging.info(f"最佳分数 ({scoring}): {float(search.best_score_):.6g}")
    return search.best_estimator_, X_train, X_test, y_train, y_test


def plot_shap_interaction_matrix(
    shap_interaction_values: np.ndarray,
    mean_abs_interaction_matrix: np.ndarray,
    mean_signed_interaction_matrix: np.ndarray,
    X_test: pd.DataFrame,
    feature_names: List[str],
    *,
    cmap_name: str,
    marker_symbol: str,
    results_dir: Path,
    matrix_cfg: Dict[str, Any],
    scheme_index: int,
    style_index: int,
) -> Path | None:
    alias_labels = _feature_alias_labels(feature_names)
    n = len(alias_labels)
    if n < 2:
        raise ValueError("至少需要 2 个特征才能绘制交互矩阵图。")

    cmap = plt.get_cmap(cmap_name)
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag_values = mean_abs_interaction_matrix[off_diag_mask]
    finite_off_diag = off_diag_values[np.isfinite(off_diag_values)]
    max_off_diag = float(finite_off_diag.max()) if finite_off_diag.size > 0 else 1.0
    if max_off_diag == 0.0:
        max_off_diag = 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=max_off_diag)

    col_xlims, col_xticks = _column_axis_specs(shap_interaction_values)

    cell_size = float(matrix_cfg.get("figure_cell_size", 1.45))
    fig = plt.figure(figsize=(cell_size * n + 2.4, cell_size * n + 4.2))
    gs = gridspec.GridSpec(n, n, figure=fig, wspace=0.03, hspace=0.03)

    title_fontsize = int(matrix_cfg.get("title_fontsize", 18))
    tick_fontsize = int(matrix_cfg.get("tick_fontsize", 11))
    value_fontsize = int(matrix_cfg.get("value_fontsize", 12))
    beeswarm_bins = int(matrix_cfg.get("beeswarm_bins", 35))
    beeswarm_width = float(matrix_cfg.get("beeswarm_width", 0.15))
    beeswarm_seed = int(matrix_cfg.get("random_state", 42))
    lower_bg = str(matrix_cfg.get("lower_facecolor", "#f2f2f2"))
    upper_bg = str(matrix_cfg.get("upper_facecolor", "#f9f9f9"))

    for row_idx in range(n):
        for col_idx in range(n):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.set_box_aspect(1)
            for spine in ax.spines.values():
                spine.set_edgecolor("white")
                spine.set_linewidth(1.0)

            if row_idx < col_idx:
                ax.set_facecolor(upper_bg)
                abs_val = float(mean_abs_interaction_matrix[row_idx, col_idx])
                signed_val = float(mean_signed_interaction_matrix[row_idx, col_idx])
                ax.imshow(
                    np.array([[abs_val]]),
                    cmap=cmap,
                    norm=norm,
                    interpolation="nearest",
                    extent=(0.0, 1.0, 0.0, 1.0),
                    aspect="auto",
                )
                ax.text(
                    0.5,
                    0.5,
                    f"{signed_val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=value_fontsize,
                    color="#333333",
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "white",
                        "edgecolor": "#bbbbbb",
                        "linewidth": 0.7,
                        "alpha": 0.95,
                    },
                )
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_facecolor(lower_bg)
                xlim = col_xlims[col_idx]
                xticks = col_xticks[col_idx]
                x_vals = np.asarray(
                    shap_interaction_values[:, row_idx, col_idx],
                    dtype=float,
                )
                finite_mask = np.isfinite(x_vals)
                if not bool(np.any(finite_mask)):
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlim(xlim)
                    ax.set_ylim(-0.5, 0.5)
                else:
                    x_vals = x_vals[finite_mask]
                    y_vals = simple_beeswarm(
                        x_vals,
                        nbins=beeswarm_bins,
                        width=beeswarm_width,
                        random_state=beeswarm_seed + row_idx * max(1, n) + col_idx,
                    )
                    feature_values = pd.to_numeric(
                        X_test.iloc[:, col_idx],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    feature_values = feature_values[finite_mask]
                    finite_feature_values = feature_values[np.isfinite(feature_values)]
                    if finite_feature_values.size == 0:
                        feature_values = np.zeros_like(x_vals, dtype=float)
                        vmin, vmax = -1.0, 1.0
                    else:
                        feature_values = np.where(
                            np.isfinite(feature_values),
                            feature_values,
                            float(np.nanmean(finite_feature_values)),
                        )
                        vmin, vmax = _expand_range(
                            float(finite_feature_values.min()),
                            float(finite_feature_values.max()),
                        )
                    c_norm = plt.Normalize(vmin=vmin, vmax=vmax)
                    ax.scatter(
                        x_vals,
                        y_vals,
                        c=feature_values,
                        cmap=cmap,
                        norm=c_norm,
                        s=25,
                        alpha=0.9,
                        edgecolors="none",
                        zorder=2,
                        marker=marker_symbol,
                    )
                    ax.set_yticks([])
                    ax.set_ylim(-0.5, 0.5)
                    ax.set_xlim(xlim)
                    for spine in ax.spines.values():
                        spine.set_edgecolor("black")
                        spine.set_linewidth(1.0)
                    if row_idx == n - 1:
                        ax.set_xticks(xticks)
                        ax.tick_params(
                            axis="x",
                            direction="out",
                            length=3,
                            width=1,
                            colors="black",
                            labelsize=tick_fontsize,
                        )
                        plt.setp(ax.get_xticklabels(), ha="center")
                        if len(xticks) > 5:
                            for label_idx, label in enumerate(ax.xaxis.get_ticklabels()):
                                if label_idx % 2 == 1:
                                    label.set_visible(False)
                    else:
                        ax.set_xticks([])

            if row_idx == 0:
                ax.set_title(
                    alias_labels[col_idx],
                    pad=8,
                    color="#333333",
                    fontproperties=_font_properties(
                        alias_labels[col_idx],
                        size=title_fontsize,
                    ),
                )
            if col_idx == 0:
                ax.set_ylabel(
                    alias_labels[row_idx],
                    labelpad=10,
                    color="#333333",
                    fontproperties=_font_properties(
                        alias_labels[row_idx],
                        size=title_fontsize,
                    ),
                )

    xlabel = str(matrix_cfg.get("xlabel", "SHAP interaction value")).strip()
    colorbar_label = str(matrix_cfg.get("colorbar_label", "Raw feature value")).strip()
    fig.text(
        0.5,
        0.05,
        xlabel,
        ha="center",
        va="center",
        fontproperties=_font_properties(xlabel, size=title_fontsize),
    )

    cbar_ax = fig.add_axes([0.92, 0.28, 0.02, 0.45])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label(
        colorbar_label,
        labelpad=10,
        fontproperties=_font_properties(colorbar_label, size=title_fontsize),
    )
    cbar.set_ticks([])
    cbar.ax.text(
        1.5,
        0,
        "Low",
        ha="left",
        va="center",
        fontproperties=_font_properties("Low", size=title_fontsize),
    )
    cbar.ax.text(
        1.5,
        1,
        "High",
        ha="left",
        va="center",
        fontproperties=_font_properties("High", size=title_fontsize),
    )
    _add_feature_mapping_table(
        fig,
        feature_names=feature_names,
        alias_labels=alias_labels,
        matrix_cfg=matrix_cfg,
    )
    plt.subplots_adjust(left=0.08, right=0.9, top=0.92, bottom=0.23)

    png_name = _interaction_output_names(
        matrix_cfg,
        scheme_index=scheme_index,
        style_index=style_index,
    )
    png_path = results_dir / png_name

    save_png = int(matrix_cfg.get("save_png", 1)) == 1
    png_dpi = int(matrix_cfg.get("png_dpi", 300))
    saved_png: Path | None = None

    if save_png:
        fig.savefig(png_path, format="png", dpi=png_dpi, bbox_inches="tight")
        saved_png = png_path
        logging.info(f"SHAP interaction matrix PNG 已保存: {png_path.resolve()}")

    plt.close(fig)
    return saved_png


def main() -> None:
    run_start = datetime.datetime.now()
    args = parse_args()
    config_path = (
        Path(args.config) if args.config else Path(__file__).with_name("config.yaml")
    )
    config = load_config(config_path)

    output_dir = ensure_run_output_dir(config, prefix="xgb_intmat_")
    setup_logging(config, output_dir)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.info(
        "字体选择：latin=%s, cjk=%s",
        LATIN_FONT_NAME,
        CJK_FONT_NAME or "<not-found>",
    )

    matrix_cfg = _interaction_matrix_cfg(config)
    results_subdir = (
        str(matrix_cfg.get("output_subdir", "interaction_matrix")).strip()
        or "interaction_matrix"
    )
    results_dir = output_dir / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"使用配置文件: {config_path}")
    logging.info(f"交互矩阵输出目录: {results_dir.resolve()}")

    df, X, y = load_dataset(config)
    logging.info(
        f"已加载数据 `{config['data']['path']}`，共 {df.shape[0]} 行，{df.shape[1]} 列。"
    )
    if X.shape[1] < 2:
        raise ValueError("至少需要 2 个特征才能绘制交互矩阵图。")

    model, X_train, X_test, y_train, y_test = _fit_best_model(config, X, y, matrix_cfg)
    y_pred = np.asarray(model.predict(X_test), dtype=float)
    rmse, r2 = _regression_metrics(y_test.to_numpy(dtype=float), y_pred)
    logging.info(
        "interaction matrix 模型已完成训练/选参："
        f"train_samples={len(X_train)}, test_samples={len(X_test)}, "
        f"test_RMSE={rmse:.4f}, test_R2={r2:.4f}"
    )

    shap_config = copy.deepcopy(config)
    shap_cfg = dict(shap_config.get("shap") or {})
    shap_cfg["compute_interactions"] = 1
    shap_config["shap"] = shap_cfg

    _, interaction_values = compute_shap_and_interactions(model, X_test, shap_config)
    if interaction_values is None:
        raise RuntimeError("SHAP interaction values 计算失败。")

    interaction_array = np.asarray(interaction_values, dtype=float)
    mean_signed_interaction_matrix = interaction_array.mean(axis=0)
    mean_abs_interaction_matrix = np.abs(mean_signed_interaction_matrix)

    scheme_index, selected_cmap_name = _selected_cmap_name(matrix_cfg)
    style_index, selected_marker = _selected_marker(matrix_cfg)
    logging.info(
        "绘图样式：scheme_index=%s, cmap=%s, style_index=%s, marker=%s, features=%s",
        scheme_index,
        selected_cmap_name,
        style_index,
        selected_marker,
        list(X_test.columns),
    )

    plot_shap_interaction_matrix(
        interaction_array,
        mean_abs_interaction_matrix,
        mean_signed_interaction_matrix,
        X_test.reset_index(drop=True),
        [str(name) for name in X_test.columns],
        cmap_name=selected_cmap_name,
        marker_symbol=selected_marker,
        results_dir=results_dir,
        matrix_cfg=matrix_cfg,
        scheme_index=scheme_index,
        style_index=style_index,
    )

    run_end = datetime.datetime.now()
    logging.info(
        f"本次运行结束: {run_end}（耗时 {(run_end - run_start).total_seconds():.2f} 秒）"
    )


if __name__ == "__main__":
    main()
