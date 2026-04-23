from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import datetime as dt
import logging
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_CODE_DIR = _THIS_DIR.parent
_XGB_DIR = _CODE_DIR / "xgb"
for _path in (_THIS_DIR, _XGB_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from run_curated_output_batch import (
    OUTPUT_ROOT,
    _configure_cjk_plot_style,
    _configure_gwxgb,
    _dataset_keys,
    _date_prefix,
    _next_run_root,
    _predict_holdout_gwxgb_local,
    _regression_summary,
    _resolve_datasets,
    load_gwxgb_dataset,
)
from gwxgb_shap import gw_distance_metric_label, gw_distance_unit
from xgb_shap import build_and_train_model, setup_logging


DATASET_RESULTS_FILE = "cv_bw_sweep_results.csv"
DATASET_BEST_FILE = "cv_bw_sweep_best_summary.csv"
DATASET_DIAGNOSTICS_FILE = "cv_bw_sweep_local_diagnostics.csv"
DATASET_FOLD_FILE = "cv_bw_sweep_fold_details.csv"
DATASET_PLOT_FILE = "cv_bw_sweep_metrics.png"
RUN_ROOT_RESULTS_FILE = "batch_cv_bw_sweep_results.csv"
RUN_ROOT_BEST_FILE = "batch_cv_bw_sweep_best_summary.csv"
RUN_ROOT_FOLD_FILE = "batch_cv_bw_sweep_fold_details.csv"
RUN_ROOT_PLOT_FILE = "batch_cv_bw_sweep_overview.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在统一 5 折 CV 口径下，对 GW-XGBoost-local 的带宽 bw 做扫描寻优，"
            "并与同折 XGBoost-global baseline 做性能对比。"
        )
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default=None,
        help="输出根目录；若不指定则自动在 Output/ 下创建新的 YYMMDD_n 目录。",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="可选的数据根目录；若不指定则使用仓库默认数据路径。",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="可选的数据集键列表，支持 2005 / 2010 / 2015 / 2020 / full。",
    )
    parser.add_argument("--year-bw-min", type=int, default=190)
    parser.add_argument("--year-bw-max", type=int, default=280)
    parser.add_argument("--year-bw-step", type=int, default=5)
    parser.add_argument("--full-bw-min", type=int, default=3300)
    parser.add_argument("--full-bw-max", type=int, default=3900)
    parser.add_argument("--full-bw-step", type=int, default=30)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-random-state", type=int, default=42)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="并行 worker 数；默认 1，建议从 2-4 开始。",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭终端进度条，仅保留日志和最终输出。",
    )
    return parser.parse_args()


def _setup_sweep_logging(run_root: Path) -> None:
    log_config = {
        "output": {
            "log_file": "cv_bw_sweep_run_log.txt",
            "capture_prints": 0,
        }
    }
    setup_logging(log_config, run_root)


def _bw_values(start: int, stop: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError(f"bw step 必须为正数，当前 step={step}")
    if stop < start:
        raise ValueError(f"bw 范围无效：start={start}, stop={stop}")
    values = list(range(int(start), int(stop) + 1, int(step)))
    if not values or values[-1] != int(stop):
        values.append(int(stop))
    return values


def _dataset_bw_values(dataset_key: str, args: argparse.Namespace) -> list[int]:
    if dataset_key == "full":
        return _bw_values(args.full_bw_min, args.full_bw_max, args.full_bw_step)
    return _bw_values(args.year_bw_min, args.year_bw_max, args.year_bw_step)


def _criterion_best_row(results_df: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "r2":
        idx = results_df[metric].astype(float).idxmax()
    else:
        idx = results_df[metric].astype(float).idxmin()
    return results_df.loc[idx]


def _metric_mean_std(fold_metrics: list[dict[str, float]], key: str) -> tuple[float, float]:
    values = np.asarray([row[key] for row in fold_metrics], dtype=float)
    return float(np.mean(values)), float(np.std(values))


def _overview_hyperparam_lines(config: dict[str, Any], *, cv_splits: int, cv_random_state: int) -> list[str]:
    model_cfg = dict(config.get("model") or {})
    params = dict(model_cfg.get("params") or {})
    gw_cfg = dict(config.get("gw") or {})
    xgb_parts = [
        f"n_estimators={params.get('n_estimators')}",
        f"learning_rate={params.get('learning_rate')}",
        f"max_depth={params.get('max_depth')}",
        f"min_child_weight={params.get('min_child_weight')}",
        f"subsample={params.get('subsample')}",
        f"colsample_bytree={params.get('colsample_bytree')}",
        f"reg_alpha={params.get('reg_alpha')}",
        f"reg_lambda={params.get('reg_lambda')}",
    ]
    gw_parts = [
        f"kernel={gw_cfg.get('kernel')}",
        f"spatial_weights={bool(gw_cfg.get('spatial_weights', False))}",
        f"distance_metric={gw_distance_metric_label(config)}",
        f"distance_unit={gw_distance_unit(config)}",
    ]
    return [
        f"CV 设置: KFold(n_splits={cv_splits}, shuffle=True, random_state={cv_random_state}); 主图指标为 5 折验证集指标均值。",
        "固定超参数(XGBoost): " + ", ".join(xgb_parts) + ".",
        "固定超参数(GW): " + ", ".join(gw_parts) + "; bw 为扫描变量，见横轴与最优点标注。",
    ]


def _bw_task_payload(
    *,
    dataset_key: str,
    folder_name: str,
    data_path: Path,
    dataset_root: Path,
    bw: int,
    cv_splits: int,
    cv_random_state: int,
) -> dict[str, Any]:
    return {
        "dataset_key": dataset_key,
        "folder_name": folder_name,
        "data_path": str(data_path.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "bw": int(bw),
        "cv_splits": int(cv_splits),
        "cv_random_state": int(cv_random_state),
    }


def _cv_splits(n_samples: int, *, cv_splits: int, cv_random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_samples < 3:
        raise ValueError("CV benchmark 至少需要 3 个样本。")
    n_splits = max(2, min(int(cv_splits), n_samples))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(cv_random_state))
    return [
        (np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int))
        for train_idx, val_idx in splitter.split(np.arange(n_samples))
    ]


def _evaluate_cv_bw_task(task: dict[str, Any]) -> dict[str, Any]:
    dataset_key = str(task["dataset_key"])
    folder_name = str(task["folder_name"])
    data_path = Path(task["data_path"]).resolve()
    dataset_root = Path(task["dataset_root"]).resolve()
    bw = int(task["bw"])
    cv_splits = int(task["cv_splits"])
    cv_random_state = int(task["cv_random_state"])

    config = _configure_gwxgb(data_path=data_path, output_dir=dataset_root)
    df, X, y, coords = load_gwxgb_dataset(config)
    n_samples = int(len(X))
    splits = _cv_splits(n_samples, cv_splits=cv_splits, cv_random_state=cv_random_state)

    y_arr = y.to_numpy(dtype=float)
    xgb_oof_pred = np.full(n_samples, np.nan, dtype=float)
    gw_oof_pred = np.full(n_samples, np.nan, dtype=float)
    xgb_fold_metrics: list[dict[str, float]] = []
    gw_fold_metrics: list[dict[str, float]] = []
    fold_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []

    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)
        coords_train = coords.iloc[train_idx].reset_index(drop=True)
        coords_val = coords.iloc[val_idx].reset_index(drop=True)

        global_model = build_and_train_model(config, X_train, y_train)
        xgb_pred = np.asarray(global_model.predict(X_val), dtype=float)
        xgb_oof_pred[val_idx] = xgb_pred
        xgb_metrics = _regression_summary(y_val.to_numpy(dtype=float), xgb_pred)
        xgb_fold_metrics.append(xgb_metrics)
        fold_rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_label": folder_name,
                "bw": int(bw),
                "fold": int(fold_id),
                "model": "XGBoost-global",
                "train_samples": int(len(train_idx)),
                "validation_samples": int(len(val_idx)),
                "r2": xgb_metrics["r2"],
                "rmse": xgb_metrics["rmse"],
                "mae": xgb_metrics["mae"],
            }
        )

        gw_pred, local_diag_rows, _ = _predict_holdout_gwxgb_local(
            config=config,
            dataset_key=dataset_key,
            dataset_label=folder_name,
            X_train=X_train,
            y_train=y_train,
            coords_train=coords_train,
            X_test=X_val,
            y_test=y_val,
            coords_test=coords_val,
            test_source_indices=val_idx,
            bw=bw,
            log_progress=False,
        )
        gw_oof_pred[val_idx] = gw_pred
        gw_metrics = _regression_summary(y_val.to_numpy(dtype=float), gw_pred)
        gw_fold_metrics.append(gw_metrics)
        fold_rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_label": folder_name,
                "bw": int(bw),
                "fold": int(fold_id),
                "model": "GW-XGBoost-local",
                "train_samples": int(len(train_idx)),
                "validation_samples": int(len(val_idx)),
                "r2": gw_metrics["r2"],
                "rmse": gw_metrics["rmse"],
                "mae": gw_metrics["mae"],
            }
        )
        for detail_row in local_diag_rows:
            detail = dict(detail_row)
            detail["bw"] = int(bw)
            detail["fold"] = int(fold_id)
            diagnostics_rows.append(detail)

    gw_oof_metrics = _regression_summary(y_arr, gw_oof_pred)
    xgb_oof_metrics = _regression_summary(y_arr, xgb_oof_pred)
    gw_r2_mean, gw_r2_std = _metric_mean_std(gw_fold_metrics, "r2")
    gw_rmse_mean, gw_rmse_std = _metric_mean_std(gw_fold_metrics, "rmse")
    gw_mae_mean, gw_mae_std = _metric_mean_std(gw_fold_metrics, "mae")
    xgb_r2_mean, xgb_r2_std = _metric_mean_std(xgb_fold_metrics, "r2")
    xgb_rmse_mean, xgb_rmse_std = _metric_mean_std(xgb_fold_metrics, "rmse")
    xgb_mae_mean, xgb_mae_std = _metric_mean_std(xgb_fold_metrics, "mae")

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    result_row = {
        "dataset_key": dataset_key,
        "dataset_label": folder_name,
        "dataset_path": str(data_path.resolve()),
        "bw": int(bw),
        "samples": int(n_samples),
        "cv_splits": int(len(splits)),
        "cv_random_state": int(cv_random_state),
        "spatial_weights": int(bool((config.get("gw") or {}).get("spatial_weights", False))),
        "kernel": str((config.get("gw") or {}).get("kernel", "Adaptive")),
        "distance_metric": gw_distance_metric_label(config),
        "distance_unit": gw_distance_unit(config),
        "r2": gw_r2_mean,
        "rmse": gw_rmse_mean,
        "mae": gw_mae_mean,
        "r2_std": gw_r2_std,
        "rmse_std": gw_rmse_std,
        "mae_std": gw_mae_std,
        "gw_oof_r2": gw_oof_metrics["r2"],
        "gw_oof_rmse": gw_oof_metrics["rmse"],
        "gw_oof_mae": gw_oof_metrics["mae"],
        "xgb_r2": xgb_r2_mean,
        "xgb_rmse": xgb_rmse_mean,
        "xgb_mae": xgb_mae_mean,
        "xgb_r2_std": xgb_r2_std,
        "xgb_rmse_std": xgb_rmse_std,
        "xgb_mae_std": xgb_mae_std,
        "xgb_oof_r2": xgb_oof_metrics["r2"],
        "xgb_oof_rmse": xgb_oof_metrics["rmse"],
        "xgb_oof_mae": xgb_oof_metrics["mae"],
        "r2_diff_gw_minus_xgb": gw_r2_mean - xgb_r2_mean,
        "rmse_diff_gw_minus_xgb": gw_rmse_mean - xgb_rmse_mean,
        "mae_diff_gw_minus_xgb": gw_mae_mean - xgb_mae_mean,
        "local_train_samples_mean": float(diagnostics_df["local_train_samples"].mean()),
        "local_train_samples_min": int(diagnostics_df["local_train_samples"].min()),
        "local_train_samples_max": int(diagnostics_df["local_train_samples"].max()),
        "distance_mean_mean": float(diagnostics_df["distance_mean"].mean()),
        "distance_max_mean": float(diagnostics_df["distance_max"].mean()),
        "abs_error_mean": float(diagnostics_df["abs_error"].mean()),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    if "weight_mean" in diagnostics_df.columns and diagnostics_df["weight_mean"].notna().any():
        result_row["weight_mean_mean"] = float(diagnostics_df["weight_mean"].dropna().mean())
        result_row["weight_sum_mean"] = float(diagnostics_df["weight_sum"].dropna().mean())

    return {
        "dataset_key": dataset_key,
        "dataset_label": folder_name,
        "bw": int(bw),
        "result_row": result_row,
        "diagnostics_rows": diagnostics_rows,
        "fold_rows": fold_rows,
    }


def _assemble_dataset_outputs(
    *,
    dataset_key: str,
    folder_name: str,
    task_outputs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows = [dict(output["result_row"]) for output in task_outputs]
    diagnostics_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for output in task_outputs:
        diagnostics_rows.extend(output.get("diagnostics_rows") or [])
        fold_rows.extend(output.get("fold_rows") or [])

    results_df = pd.DataFrame(result_rows).sort_values("bw").reset_index(drop=True)
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    fold_df = pd.DataFrame(fold_rows).sort_values(["bw", "model", "fold"]).reset_index(drop=True)

    best_rows: list[dict[str, Any]] = []
    for criterion in ("r2", "rmse", "mae"):
        best_row = _criterion_best_row(results_df, criterion)
        best_rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_label": folder_name,
                "criterion": criterion,
                "best_bw": int(best_row["bw"]),
                "distance_metric": str(best_row.get("distance_metric", "")),
                "distance_unit": str(best_row.get("distance_unit", "")),
                "gw_cv_mean_r2": float(best_row["r2"]),
                "gw_cv_mean_rmse": float(best_row["rmse"]),
                "gw_cv_mean_mae": float(best_row["mae"]),
                "gw_cv_std_r2": float(best_row["r2_std"]),
                "gw_cv_std_rmse": float(best_row["rmse_std"]),
                "gw_cv_std_mae": float(best_row["mae_std"]),
                "gw_oof_r2": float(best_row["gw_oof_r2"]),
                "gw_oof_rmse": float(best_row["gw_oof_rmse"]),
                "gw_oof_mae": float(best_row["gw_oof_mae"]),
                "xgb_cv_mean_r2": float(best_row["xgb_r2"]),
                "xgb_cv_mean_rmse": float(best_row["xgb_rmse"]),
                "xgb_cv_mean_mae": float(best_row["xgb_mae"]),
                "xgb_oof_r2": float(best_row["xgb_oof_r2"]),
                "xgb_oof_rmse": float(best_row["xgb_oof_rmse"]),
                "xgb_oof_mae": float(best_row["xgb_oof_mae"]),
                "r2_diff_gw_minus_xgb": float(best_row["r2_diff_gw_minus_xgb"]),
                "rmse_diff_gw_minus_xgb": float(best_row["rmse_diff_gw_minus_xgb"]),
                "mae_diff_gw_minus_xgb": float(best_row["mae_diff_gw_minus_xgb"]),
            }
        )
    return results_df, pd.DataFrame(best_rows), diagnostics_df, fold_df


def _metric_axis_limits(values: np.ndarray, baseline_values: list[float]) -> tuple[float, float]:
    finite_parts = [np.asarray(values, dtype=float)]
    finite_parts.extend(np.asarray([item], dtype=float) for item in baseline_values)
    combined = np.concatenate([part[np.isfinite(part)] for part in finite_parts])
    if combined.size == 0:
        return 0.0, 1.0
    low = float(np.min(combined))
    high = float(np.max(combined))
    span = high - low
    if span <= 0.0:
        span = max(1.0, abs(high) * 0.1)
    return low - span * 0.12, high + span * 0.18


def _draw_optional_baseline(
    *,
    ax: Any,
    bws: np.ndarray,
    value: float | None,
    y_limits: tuple[float, float],
    label: str,
    color: str,
    linestyle: str,
    linewidth: float,
    text_offset: tuple[int, int],
) -> None:
    if value is None or not np.isfinite(value):
        return
    y_min, y_max = y_limits
    if y_min <= value <= y_max:
        ax.axhline(
            value,
            linestyle=linestyle,
            linewidth=linewidth,
            color=color,
            label=label,
        )
        ax.annotate(
            f"{label.split()[0]}={value:.4f}",
            xy=(bws[-1], value),
            xytext=text_offset,
            textcoords="offset points",
            fontsize=8,
            color=color,
            ha="right",
            va="bottom" if text_offset[1] >= 0 else "top",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "0.7",
                "alpha": 0.9,
            },
        )
        return

    edge_y = y_min if value < y_min else y_max
    ax.annotate(
        f"{label.split()[0]}={value:.4f}\noutside scale",
        xy=(bws[-1], edge_y),
        xytext=text_offset,
        textcoords="offset points",
        fontsize=8,
        color=color,
        ha="right",
        va="bottom" if value < y_min else "top",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.9,
        },
    )


def _plot_cv_bw_sweep(
    *,
    dataset_label: str,
    results_df: pd.DataFrame,
    out_path: Path,
) -> None:
    _configure_cjk_plot_style()
    metric_specs = [
        ("r2", "R2", "max", "tab:blue"),
        ("rmse", "RMSE", "min", "tab:orange"),
        ("mae", "MAE", "min", "tab:green"),
    ]
    bws = results_df["bw"].to_numpy(dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(f"{dataset_label} 5-Fold CV BW Sweep", fontsize=14)

    for ax, (metric_key, metric_label, direction, color) in zip(axes, metric_specs):
        values = results_df[metric_key].to_numpy(dtype=float)
        xgb_baseline = float(results_df[f"xgb_{metric_key}"].iloc[0])
        ols_baseline = None
        if f"ols_{metric_key}" in results_df.columns:
            ols_baseline = float(results_df[f"ols_{metric_key}"].iloc[0])
        y_limits = _metric_axis_limits(values, [xgb_baseline])
        best_row = _criterion_best_row(results_df, metric_key)
        best_bw = int(best_row["bw"])
        best_value = float(best_row[metric_key])

        ax.plot(bws, values, marker="o", linewidth=2, color=color, label="GW-XGBoost-local CV mean")
        _draw_optional_baseline(
            ax=ax,
            bws=bws,
            value=xgb_baseline,
            y_limits=y_limits,
            label="XGB CV mean",
            color="0.35",
            linestyle="--",
            linewidth=1.5,
            text_offset=(-6, 6),
        )
        _draw_optional_baseline(
            ax=ax,
            bws=bws,
            value=ols_baseline,
            y_limits=y_limits,
            label="OLS CV mean",
            color="tab:red",
            linestyle=":",
            linewidth=1.7,
            text_offset=(-6, -12),
        )
        ax.scatter([best_bw], [best_value], color="crimson", s=50, zorder=3)
        ax.annotate(
            f"best bw={best_bw}\n{metric_label}={best_value:.4f}",
            xy=(best_bw, best_value),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color="crimson",
        )
        ax.set_ylabel(metric_label)
        ax.set_ylim(*y_limits)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best")
        ax.set_title(f"{metric_label} {'越高越好' if direction == 'max' else '越低越好'}")

    axes[-1].set_xlabel("Bandwidth (bw)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_run_root_cv_overview(
    *,
    dataset_results: list[tuple[str, pd.DataFrame]],
    out_path: Path,
    footer_lines_extra: list[str] | None = None,
) -> None:
    if not dataset_results:
        return

    _configure_cjk_plot_style()
    metric_specs = [
        ("r2", "R2", "max", "tab:blue"),
        ("rmse", "RMSE", "min", "tab:orange"),
        ("mae", "MAE", "min", "tab:green"),
    ]
    n_rows = len(dataset_results)
    n_cols = len(metric_specs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.4 * n_cols, 3.1 * n_rows),
        squeeze=False,
        sharex=False,
    )
    fig.suptitle("5-Fold CV BW Sweep Overview", fontsize=16)

    for row_idx, (dataset_label, results_df) in enumerate(dataset_results):
        bws = results_df["bw"].to_numpy(dtype=float)
        for col_idx, (metric_key, metric_label, direction, color) in enumerate(metric_specs):
            ax = axes[row_idx][col_idx]
            values = results_df[metric_key].to_numpy(dtype=float)
            xgb_baseline = float(results_df[f"xgb_{metric_key}"].iloc[0])
            ols_baseline = None
            if f"ols_{metric_key}" in results_df.columns:
                ols_baseline = float(results_df[f"ols_{metric_key}"].iloc[0])
            y_limits = _metric_axis_limits(values, [xgb_baseline])
            best_row = _criterion_best_row(results_df, metric_key)
            best_bw = int(best_row["bw"])
            best_value = float(best_row[metric_key])

            ax.plot(
                bws,
                values,
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                color=color,
                label="GW-XGBoost-local CV mean",
            )
            _draw_optional_baseline(
                ax=ax,
                bws=bws,
                value=xgb_baseline,
                y_limits=y_limits,
                label="XGB CV mean",
                color="0.35",
                linestyle="--",
                linewidth=1.3,
                text_offset=(-6, 6),
            )
            _draw_optional_baseline(
                ax=ax,
                bws=bws,
                value=ols_baseline,
                y_limits=y_limits,
                label="OLS CV mean",
                color="tab:red",
                linestyle=":",
                linewidth=1.4,
                text_offset=(-6, -12),
            )
            ax.scatter([best_bw], [best_value], color="crimson", s=28, zorder=3)
            ax.annotate(
                f"bw={best_bw}\n{best_value:.4f}",
                xy=(best_bw, best_value),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color="crimson",
            )
            ax.set_ylim(*y_limits)
            ax.grid(True, linestyle=":", alpha=0.45)
            if row_idx == 0:
                subtitle = "越高越好" if direction == "max" else "越低越好"
                ax.set_title(f"{metric_label} ({subtitle})")
            if col_idx == 0:
                ax.set_ylabel(f"{dataset_label}\n{metric_label}")
            else:
                ax.set_ylabel(metric_label)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Bandwidth (bw)")
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="best", fontsize=8)

    footer_lines = [
        "评估口径: 各模型使用完全相同的 5 折 KFold 划分；主图和主表 R2 / RMSE / MAE 为各折验证集指标的平均值。",
        "补充口径: 结果表同时输出合并所有 out-of-fold 预测后的 OOF 整体 R2 / RMSE / MAE，用于复核整体泛化误差。",
        "评估方法: XGBoost-global 和 OLS 每折训练一个全局模型；GW-XGBoost-local 每折对验证样本逐点训练局部加权模型，局部距离和权重由 GW 配置控制。",
    ]
    if footer_lines_extra:
        footer_lines.extend(str(line) for line in footer_lines_extra if str(line).strip())
    fig.text(
        0.5,
        0.024,
        "\n".join(footer_lines),
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="0.2",
    )
    fig.tight_layout(rect=(0, 0.18, 1, 0.97))
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _write_dataset_outputs(
    *,
    dataset_root: Path,
    dataset_label: str,
    results_df: pd.DataFrame,
    best_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    fold_df: pd.DataFrame,
) -> None:
    results_path = dataset_root / DATASET_RESULTS_FILE
    best_path = dataset_root / DATASET_BEST_FILE
    diagnostics_path = dataset_root / DATASET_DIAGNOSTICS_FILE
    fold_path = dataset_root / DATASET_FOLD_FILE
    plot_path = dataset_root / DATASET_PLOT_FILE

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    best_df.to_csv(best_path, index=False, encoding="utf-8-sig")
    diagnostics_df.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    _plot_cv_bw_sweep(dataset_label=dataset_label, results_df=results_df, out_path=plot_path)

    logging.info("已保存 CV BW 扫描结果表：%s", results_path.resolve())
    logging.info("已保存 CV BW 扫描最佳摘要：%s", best_path.resolve())
    logging.info("已保存 CV BW 扫描局部诊断：%s", diagnostics_path.resolve())
    logging.info("已保存 CV BW 扫描折明细：%s", fold_path.resolve())
    logging.info("已保存 CV BW 扫描曲线图：%s", plot_path.resolve())


def main() -> None:
    args = parse_args()
    datasets = _resolve_datasets(args.data_root)
    selected_keys = _dataset_keys(args.datasets)

    if args.run_root:
        run_root = Path(args.run_root).expanduser().resolve()
        run_root.mkdir(parents=True, exist_ok=True)
    else:
        run_root = _next_run_root(OUTPUT_ROOT, _date_prefix())

    print(f"Run root: {run_root}")
    _setup_sweep_logging(run_root)

    dataset_specs: dict[str, dict[str, Any]] = {}
    task_specs: list[dict[str, Any]] = []
    for dataset_key in selected_keys:
        folder_name, data_path = datasets[dataset_key]
        dataset_root = run_root / folder_name
        dataset_root.mkdir(parents=True, exist_ok=True)
        bw_values = _dataset_bw_values(dataset_key, args)
        dataset_specs[dataset_key] = {
            "folder_name": folder_name,
            "data_path": data_path,
            "dataset_root": dataset_root,
            "bw_values": bw_values,
        }
        task_specs.extend(
            _bw_task_payload(
                dataset_key=dataset_key,
                folder_name=folder_name,
                data_path=data_path,
                dataset_root=dataset_root,
                bw=bw,
                cv_splits=args.cv_splits,
                cv_random_state=args.cv_random_state,
            )
            for bw in bw_values
        )

    print(
        f"[INFO] evaluation=5-fold-cv, dataset_count={len(selected_keys)}, "
        f"bw_task_count={len(task_specs)}, jobs={max(1, int(args.jobs))}"
    )
    footer_lines_extra: list[str] = []
    if selected_keys:
        first_spec = dataset_specs[selected_keys[0]]
        footer_lines_extra = _overview_hyperparam_lines(
            _configure_gwxgb(
                data_path=Path(first_spec["data_path"]),
                output_dir=run_root,
            ),
            cv_splits=int(args.cv_splits),
            cv_random_state=int(args.cv_random_state),
        )

    show_progress = not args.no_progress
    dataset_expected_counts = {
        dataset_key: len(spec["bw_values"]) for dataset_key, spec in dataset_specs.items()
    }
    dataset_completed_counts = {dataset_key: 0 for dataset_key in dataset_specs}
    dataset_outputs_map: dict[str, list[dict[str, Any]]] = {
        dataset_key: [] for dataset_key in dataset_specs
    }
    dataset_done: set[str] = set()

    tasks_bar = tqdm(
        total=len(task_specs),
        desc="CV BW tasks",
        unit="task",
        position=0,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    datasets_bar = tqdm(
        total=len(selected_keys),
        desc="Datasets",
        unit="dataset",
        position=1,
        dynamic_ncols=True,
        disable=not show_progress,
    )

    def _record_task_output(output: dict[str, Any]) -> None:
        dataset_key = str(output["dataset_key"])
        dataset_outputs_map[dataset_key].append(output)
        dataset_completed_counts[dataset_key] += 1
        tasks_bar.update(1)
        tasks_bar.set_postfix_str(
            f"{dataset_key}: bw={int(output['bw'])}",
            refresh=False,
        )
        if (
            dataset_completed_counts[dataset_key] >= dataset_expected_counts[dataset_key]
            and dataset_key not in dataset_done
        ):
            dataset_done.add(dataset_key)
            datasets_bar.update(1)
            datasets_bar.set_postfix_str(f"{dataset_key} done", refresh=False)
            print(f"[DONE] {dataset_key} CV bw sweep -> {dataset_specs[dataset_key]['dataset_root']}")

    def _run_parallel_tasks(max_workers: int) -> None:
        backend_name = "process"
        try:
            logging.info("CV BW sweep 启用并行执行：backend=%s, max_workers=%s", backend_name, max_workers)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(_evaluate_cv_bw_task, task): task for task in task_specs
                }
                for future in as_completed(future_to_task):
                    _record_task_output(future.result())
            return
        except PermissionError as exc:
            backend_name = "thread"
            logging.warning("进程并行初始化失败，将回退到线程并行：%s", exc)

        logging.info("CV BW sweep 启用并行执行：backend=%s, max_workers=%s", backend_name, max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(_evaluate_cv_bw_task, task): task for task in task_specs
            }
            for future in as_completed(future_to_task):
                _record_task_output(future.result())

    try:
        if task_specs and max(1, int(args.jobs)) > 1:
            max_workers = min(max(1, int(args.jobs)), len(task_specs))
            _run_parallel_tasks(max_workers)
        else:
            for task in task_specs:
                _record_task_output(_evaluate_cv_bw_task(task))
    finally:
        tasks_bar.close()
        datasets_bar.close()

    all_results: list[pd.DataFrame] = []
    all_best: list[pd.DataFrame] = []
    all_folds: list[pd.DataFrame] = []
    batch_plot_inputs: list[tuple[str, pd.DataFrame]] = []
    for dataset_key in selected_keys:
        spec = dataset_specs[dataset_key]
        results_df, best_df, diagnostics_df, fold_df = _assemble_dataset_outputs(
            dataset_key=dataset_key,
            folder_name=str(spec["folder_name"]),
            task_outputs=dataset_outputs_map[dataset_key],
        )
        _write_dataset_outputs(
            dataset_root=Path(spec["dataset_root"]),
            dataset_label=str(spec["folder_name"]),
            results_df=results_df,
            best_df=best_df,
            diagnostics_df=diagnostics_df,
            fold_df=fold_df,
        )
        all_results.append(results_df)
        all_best.append(best_df)
        all_folds.append(fold_df)
        batch_plot_inputs.append((str(spec["folder_name"]), results_df))

    if all_results:
        batch_results_path = run_root / RUN_ROOT_RESULTS_FILE
        pd.concat(all_results, ignore_index=True, sort=False).to_csv(
            batch_results_path,
            index=False,
            encoding="utf-8-sig",
        )
        logging.info("已保存批量 CV BW 扫描结果总表：%s", batch_results_path.resolve())
        batch_plot_path = run_root / RUN_ROOT_PLOT_FILE
        _plot_run_root_cv_overview(
            dataset_results=batch_plot_inputs,
            out_path=batch_plot_path,
            footer_lines_extra=footer_lines_extra,
        )
        logging.info("已保存批量 CV BW 扫描总览图：%s", batch_plot_path.resolve())

    if all_best:
        batch_best_path = run_root / RUN_ROOT_BEST_FILE
        pd.concat(all_best, ignore_index=True, sort=False).to_csv(
            batch_best_path,
            index=False,
            encoding="utf-8-sig",
        )
        logging.info("已保存批量 CV BW 扫描最佳摘要总表：%s", batch_best_path.resolve())

    if all_folds:
        batch_fold_path = run_root / RUN_ROOT_FOLD_FILE
        pd.concat(all_folds, ignore_index=True, sort=False).to_csv(
            batch_fold_path,
            index=False,
            encoding="utf-8-sig",
        )
        logging.info("已保存批量 CV BW 扫描折明细总表：%s", batch_fold_path.resolve())


if __name__ == "__main__":
    main()
