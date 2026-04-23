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
    _benchmark_split_settings,
    _configure_cjk_plot_style,
    _configure_gwxgb,
    _dataset_keys,
    _date_prefix,
    _next_run_root,
    _predict_holdout_gwxgb_local,
    _regression_summary,
    _resolve_datasets,
    _split_holdout_benchmark_data,
    load_gwxgb_dataset,
)
from gwxgb_shap import gw_distance_metric_label, gw_distance_unit
from xgb_shap import build_and_train_model, setup_logging


DATASET_RESULTS_FILE = "bw_sweep_results.csv"
DATASET_BEST_FILE = "bw_sweep_best_summary.csv"
DATASET_DIAGNOSTICS_FILE = "bw_sweep_local_diagnostics.csv"
DATASET_PLOT_FILE = "bw_sweep_metrics.png"
RUN_ROOT_RESULTS_FILE = "batch_bw_sweep_results.csv"
RUN_ROOT_BEST_FILE = "batch_bw_sweep_best_summary.csv"
RUN_ROOT_PLOT_FILE = "batch_bw_sweep_overview.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在统一 holdout 划分下，对 GeoXGBoost 的带宽 bw 做扫描寻优，"
            "输出逐数据集结果表、最佳摘要和三指标曲线图。"
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
    parser.add_argument("--year-bw-min", type=int, default=140)
    parser.add_argument("--year-bw-max", type=int, default=220)
    parser.add_argument("--year-bw-step", type=int, default=10)
    parser.add_argument("--full-bw-min", type=int, default=2500)
    parser.add_argument("--full-bw-max", type=int, default=4000)
    parser.add_argument("--full-bw-step", type=int, default=150)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="并行 worker 数；默认 1，建议从 2-4 开始尝试。",
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
            "log_file": "bw_sweep_run_log.txt",
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


def _holdout_train_sample_count(*, data_path: Path, dataset_root: Path) -> int:
    config = _configure_gwxgb(data_path=data_path, output_dir=dataset_root)
    df, X, y, coords = load_gwxgb_dataset(config)
    test_size, split_random_state = _benchmark_split_settings(config)
    split = _split_holdout_benchmark_data(
        df=df,
        X=X,
        y=y,
        coords=coords,
        test_size=test_size,
        random_state=split_random_state,
    )
    return int(len(split["X_train"]))


def _dataset_bw_values(
    dataset_key: str,
    args: argparse.Namespace,
    *,
    train_sample_cap: int | None = None,
) -> list[int]:
    if dataset_key == "full":
        return _bw_values(args.full_bw_min, args.full_bw_max, args.full_bw_step)

    year_bw_min = int(args.year_bw_min)
    requested_year_bw_max = int(args.year_bw_max)
    year_bw_max = requested_year_bw_max
    if train_sample_cap is not None:
        if requested_year_bw_max <= 0 or requested_year_bw_max > train_sample_cap:
            year_bw_max = int(train_sample_cap)
    return _bw_values(year_bw_min, year_bw_max, args.year_bw_step)


def _criterion_best_row(results_df: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "r2":
        idx = results_df[metric].astype(float).idxmax()
    else:
        idx = results_df[metric].astype(float).idxmin()
    return results_df.loc[idx]


def _overview_hyperparam_lines(config: dict[str, Any]) -> list[str]:
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
) -> dict[str, Any]:
    return {
        "dataset_key": dataset_key,
        "folder_name": folder_name,
        "data_path": str(data_path.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "bw": int(bw),
    }


def _plot_bw_sweep(
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
    fig.suptitle(f"{dataset_label} Holdout BW Sweep", fontsize=14)

    for ax, (metric_key, metric_label, direction, color) in zip(axes, metric_specs):
        values = results_df[metric_key].to_numpy(dtype=float)
        baseline = float(results_df[f"xgb_{metric_key}"].iloc[0])
        best_row = _criterion_best_row(results_df, metric_key)
        best_bw = int(best_row["bw"])
        best_value = float(best_row[metric_key])

        ax.plot(bws, values, marker="o", linewidth=2, color=color, label="GW-XGBoost-local")
        ax.axhline(
            baseline,
            linestyle="--",
            linewidth=1.5,
            color="0.35",
            label="XGBoost-global",
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
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best")
        if direction == "max":
            ax.set_title(f"{metric_label} 越高越好")
        else:
            ax.set_title(f"{metric_label} 越低越好")

    axes[-1].set_xlabel("Bandwidth (bw)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_run_root_bw_overview(
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
    fig.suptitle("Holdout BW Sweep Overview", fontsize=16)

    for row_idx, (dataset_label, results_df) in enumerate(dataset_results):
        bws = results_df["bw"].to_numpy(dtype=float)
        for col_idx, (metric_key, metric_label, direction, color) in enumerate(metric_specs):
            ax = axes[row_idx][col_idx]
            values = results_df[metric_key].to_numpy(dtype=float)
            baseline = float(results_df[f"xgb_{metric_key}"].iloc[0])
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
                label="GW-XGBoost-local",
            )
            ax.axhline(
                baseline,
                linestyle="--",
                linewidth=1.3,
                color="0.35",
                label="XGBoost-global baseline",
            )
            ax.annotate(
                f"baseline={baseline:.4f}",
                xy=(bws[-1], baseline),
                xytext=(-6, 6),
                textcoords="offset points",
                fontsize=8,
                color="0.25",
                ha="right",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "0.7",
                    "alpha": 0.9,
                },
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
        "评估口径: 两个模型均使用同一 holdout 划分评估（test_size=0.2, random_state=42），并在相同测试集上统一计算 R2 / RMSE / MAE。",
        "评估方法: XGBoost-global 为单个全局模型；GW-XGBoost-local 对每个测试样本按空间距离与带宽训练局部加权模型，合并全部测试点预测后再整体计算指标。",
    ]
    if footer_lines_extra:
        footer_lines.extend(str(line) for line in footer_lines_extra if str(line).strip())
    fig.text(
        0.5,
        0.028,
        "\n".join(footer_lines),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.2",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.97))
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _evaluate_bw_task(task: dict[str, Any]) -> dict[str, Any]:
    dataset_key = str(task["dataset_key"])
    folder_name = str(task["folder_name"])
    data_path = Path(task["data_path"]).resolve()
    dataset_root = Path(task["dataset_root"]).resolve()
    bw = int(task["bw"])

    config = _configure_gwxgb(data_path=data_path, output_dir=dataset_root)
    df, X, y, coords = load_gwxgb_dataset(config)

    test_size, split_random_state = _benchmark_split_settings(config)
    split = _split_holdout_benchmark_data(
        df=df,
        X=X,
        y=y,
        coords=coords,
        test_size=test_size,
        random_state=split_random_state,
    )
    X_train = split["X_train"]
    X_test = split["X_test"]
    y_train = split["y_train"]
    y_test = split["y_test"]
    coords_train = split["coords_train"]
    coords_test = split["coords_test"]

    global_model = build_and_train_model(config, X_train, y_train)
    global_pred = np.asarray(global_model.predict(X_test), dtype=float)
    global_metrics = _regression_summary(y_test.to_numpy(dtype=float), global_pred)
    local_pred, local_diag_rows, local_summary = _predict_holdout_gwxgb_local(
        config=config,
        dataset_key=dataset_key,
        dataset_label=folder_name,
        X_train=X_train,
        y_train=y_train,
        coords_train=coords_train,
        X_test=X_test,
        y_test=y_test,
        coords_test=coords_test,
        test_source_indices=np.asarray(split["test_idx"], dtype=int),
        bw=bw,
        log_progress=False,
    )
    gw_metrics = _regression_summary(y_test.to_numpy(dtype=float), local_pred)
    result_row = {
        "dataset_key": dataset_key,
        "dataset_label": folder_name,
        "dataset_path": str(data_path.resolve()),
        "bw": int(bw),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_size": float(test_size),
        "random_state": int(split_random_state),
        "spatial_weights": int(bool((config.get("gw") or {}).get("spatial_weights", False))),
        "kernel": str((config.get("gw") or {}).get("kernel", "Adaptive")),
        "distance_metric": gw_distance_metric_label(config),
        "distance_unit": gw_distance_unit(config),
        "r2": gw_metrics["r2"],
        "rmse": gw_metrics["rmse"],
        "mae": gw_metrics["mae"],
        "xgb_r2": global_metrics["r2"],
        "xgb_rmse": global_metrics["rmse"],
        "xgb_mae": global_metrics["mae"],
        "r2_diff_gw_minus_xgb": gw_metrics["r2"] - global_metrics["r2"],
        "rmse_diff_gw_minus_xgb": gw_metrics["rmse"] - global_metrics["rmse"],
        "mae_diff_gw_minus_xgb": gw_metrics["mae"] - global_metrics["mae"],
        "local_train_samples_mean": local_summary.get("local_train_samples_mean"),
        "local_train_samples_min": local_summary.get("local_train_samples_min"),
        "local_train_samples_max": local_summary.get("local_train_samples_max"),
        "weight_mean_mean": local_summary.get("weight_mean_mean"),
        "weight_sum_mean": local_summary.get("weight_sum_mean"),
        "distance_mean_mean": local_summary.get("distance_mean_mean"),
        "distance_max_mean": local_summary.get("distance_max_mean"),
        "abs_error_mean": local_summary.get("abs_error_mean"),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    diagnostics_rows: list[dict[str, Any]] = []
    for detail_row in local_diag_rows:
        detail = dict(detail_row)
        detail["bw"] = int(bw)
        diagnostics_rows.append(detail)
    return {
        "dataset_key": dataset_key,
        "dataset_label": folder_name,
        "bw": int(bw),
        "result_row": result_row,
        "diagnostics_rows": diagnostics_rows,
    }


def _assemble_dataset_outputs(
    *,
    dataset_key: str,
    folder_name: str,
    task_outputs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows = [dict(output["result_row"]) for output in task_outputs]
    diagnostics_rows: list[dict[str, Any]] = []
    for output in task_outputs:
        diagnostics_rows.extend(output.get("diagnostics_rows") or [])

    results_df = pd.DataFrame(result_rows).sort_values("bw").reset_index(drop=True)
    diagnostics_df = pd.DataFrame(diagnostics_rows)

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
                "gw_r2": float(best_row["r2"]),
                "gw_rmse": float(best_row["rmse"]),
                "gw_mae": float(best_row["mae"]),
                "xgb_r2": float(best_row["xgb_r2"]),
                "xgb_rmse": float(best_row["xgb_rmse"]),
                "xgb_mae": float(best_row["xgb_mae"]),
                "r2_diff_gw_minus_xgb": float(best_row["r2_diff_gw_minus_xgb"]),
                "rmse_diff_gw_minus_xgb": float(best_row["rmse_diff_gw_minus_xgb"]),
                "mae_diff_gw_minus_xgb": float(best_row["mae_diff_gw_minus_xgb"]),
            }
        )
    best_df = pd.DataFrame(best_rows)
    return results_df, best_df, diagnostics_df


def _evaluate_dataset(
    *,
    dataset_key: str,
    folder_name: str,
    data_path: Path,
    dataset_root: Path,
    bw_values: list[int],
    task_progress: tqdm[Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_outputs: list[dict[str, Any]] = []
    for bw in bw_values:
        output = _evaluate_bw_task(
            _bw_task_payload(
                dataset_key=dataset_key,
                folder_name=folder_name,
                data_path=data_path,
                dataset_root=dataset_root,
                bw=bw,
            )
        )
        task_outputs.append(output)
        if task_progress is not None:
            task_progress.update(1)
            task_progress.set_postfix_str(f"{dataset_key}: bw={bw}", refresh=False)
    return _assemble_dataset_outputs(
        dataset_key=dataset_key,
        folder_name=folder_name,
        task_outputs=task_outputs,
    )


def _write_dataset_outputs(
    *,
    dataset_root: Path,
    dataset_label: str,
    results_df: pd.DataFrame,
    best_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
) -> None:
    results_path = dataset_root / DATASET_RESULTS_FILE
    best_path = dataset_root / DATASET_BEST_FILE
    diagnostics_path = dataset_root / DATASET_DIAGNOSTICS_FILE
    plot_path = dataset_root / DATASET_PLOT_FILE

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    best_df.to_csv(best_path, index=False, encoding="utf-8-sig")
    diagnostics_df.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    _plot_bw_sweep(dataset_label=dataset_label, results_df=results_df, out_path=plot_path)

    logging.info("已保存 BW 扫描结果表：%s", results_path.resolve())
    logging.info("已保存 BW 扫描最佳摘要：%s", best_path.resolve())
    logging.info("已保存 BW 扫描局部诊断：%s", diagnostics_path.resolve())
    logging.info("已保存 BW 扫描曲线图：%s", plot_path.resolve())


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
        train_sample_cap = None
        if dataset_key != "full":
            train_sample_cap = _holdout_train_sample_count(
                data_path=data_path,
                dataset_root=dataset_root,
            )
        bw_values = _dataset_bw_values(
            dataset_key,
            args,
            train_sample_cap=train_sample_cap,
        )
        dataset_specs[dataset_key] = {
            "folder_name": folder_name,
            "data_path": data_path,
            "dataset_root": dataset_root,
            "bw_values": bw_values,
            "train_sample_cap": train_sample_cap,
        }
        task_specs.extend(
            _bw_task_payload(
                dataset_key=dataset_key,
                folder_name=folder_name,
                data_path=data_path,
                dataset_root=dataset_root,
                bw=bw,
            )
            for bw in bw_values
        )

    print(
        f"[INFO] dataset_count={len(selected_keys)}, "
        f"bw_task_count={len(task_specs)}, jobs={max(1, int(args.jobs))}"
    )
    footer_lines_extra: list[str] = []
    if selected_keys:
        first_spec = dataset_specs[selected_keys[0]]
        footer_lines_extra = _overview_hyperparam_lines(
            _configure_gwxgb(
                data_path=Path(first_spec["data_path"]),
                output_dir=run_root,
            )
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
        desc="BW tasks",
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
            print(f"[DONE] {dataset_key} bw sweep -> {dataset_specs[dataset_key]['dataset_root']}")

    def _run_parallel_tasks(max_workers: int) -> None:
        backend_name = "process"
        try:
            logging.info("BW sweep 启用并行执行：backend=%s, max_workers=%s", backend_name, max_workers)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(_evaluate_bw_task, task): task for task in task_specs
                }
                for future in as_completed(future_to_task):
                    output = future.result()
                    _record_task_output(output)
            return
        except PermissionError as exc:
            backend_name = "thread"
            logging.warning(
                "进程并行初始化失败，将回退到线程并行：%s",
                exc,
            )

        logging.info("BW sweep 启用并行执行：backend=%s, max_workers=%s", backend_name, max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(_evaluate_bw_task, task): task for task in task_specs
            }
            for future in as_completed(future_to_task):
                output = future.result()
                _record_task_output(output)

    try:
        if task_specs and max(1, int(args.jobs)) > 1:
            max_workers = min(max(1, int(args.jobs)), len(task_specs))
            _run_parallel_tasks(max_workers)
        else:
            for task in task_specs:
                output = _evaluate_bw_task(task)
                _record_task_output(output)
    finally:
        tasks_bar.close()
        datasets_bar.close()

    all_results: list[pd.DataFrame] = []
    all_best: list[pd.DataFrame] = []
    batch_plot_inputs: list[tuple[str, pd.DataFrame]] = []
    for dataset_key in selected_keys:
        spec = dataset_specs[dataset_key]
        results_df, best_df, diagnostics_df = _assemble_dataset_outputs(
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
        )
        all_results.append(results_df)
        all_best.append(best_df)
        batch_plot_inputs.append((str(spec["folder_name"]), results_df))

    if all_results:
        batch_results_path = run_root / RUN_ROOT_RESULTS_FILE
        pd.concat(all_results, ignore_index=True, sort=False).to_csv(
            batch_results_path,
            index=False,
            encoding="utf-8-sig",
        )
        logging.info("已保存批量 BW 扫描结果总表：%s", batch_results_path.resolve())
        batch_plot_path = run_root / RUN_ROOT_PLOT_FILE
        _plot_run_root_bw_overview(
            dataset_results=batch_plot_inputs,
            out_path=batch_plot_path,
            footer_lines_extra=footer_lines_extra,
        )
        logging.info("已保存批量 BW 扫描总览图：%s", batch_plot_path.resolve())

    if all_best:
        batch_best_path = run_root / RUN_ROOT_BEST_FILE
        pd.concat(all_best, ignore_index=True, sort=False).to_csv(
            batch_best_path,
            index=False,
            encoding="utf-8-sig",
        )
        logging.info("已保存批量 BW 扫描最佳摘要总表：%s", batch_best_path.resolve())


if __name__ == "__main__":
    main()
