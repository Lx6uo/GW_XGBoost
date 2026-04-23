from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

_THIS_DIR = Path(__file__).resolve().parent
_CODE_DIR = _THIS_DIR.parent
_XGB_DIR = _CODE_DIR / "xgb"
for _path in (_THIS_DIR, _XGB_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from run_curated_output_batch import (
    _configure_gwxgb,
    _dataset_keys,
    _regression_summary,
    _resolve_datasets,
    load_gwxgb_dataset,
)
from run_cv_bw_sweep import (
    DATASET_BEST_FILE,
    DATASET_PLOT_FILE,
    DATASET_RESULTS_FILE,
    RUN_ROOT_BEST_FILE,
    RUN_ROOT_PLOT_FILE,
    RUN_ROOT_RESULTS_FILE,
    _overview_hyperparam_lines,
    _plot_cv_bw_sweep,
    _plot_run_root_cv_overview,
)


DATASET_OLS_FILE = "cv_ols_baseline.csv"
DATASET_OLS_FOLD_FILE = "cv_ols_fold_details.csv"
RUN_ROOT_OLS_FILE = "batch_cv_ols_baseline.csv"
RUN_ROOT_OLS_FOLD_FILE = "batch_cv_ols_fold_details.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "对既有 run_cv_bw_sweep 输出增量追加 OLS baseline。"
            "只训练 OLS，不重跑 GW-XGBoost 或 XGBoost。"
        )
    )
    parser.add_argument(
        "--run-root",
        required=True,
        help="既有 CV BW sweep 输出根目录，例如 Output/260422_cv_euclidean_all。",
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
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-random-state", type=int, default=42)
    return parser.parse_args()


def _metric_mean_std(fold_metrics: list[dict[str, float]], key: str) -> tuple[float, float]:
    values = np.asarray([row[key] for row in fold_metrics], dtype=float)
    return float(np.mean(values)), float(np.std(values))


def _compute_ols_cv(
    *,
    dataset_key: str,
    dataset_label: str,
    data_path: Path,
    dataset_root: Path,
    cv_splits: int,
    cv_random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = _configure_gwxgb(data_path=data_path, output_dir=dataset_root)
    _, X, y, _ = load_gwxgb_dataset(config)
    n_samples = int(len(X))
    if n_samples < 3:
        raise ValueError(f"OLS CV 至少需要 3 个样本：{dataset_key}")
    n_splits = max(2, min(int(cv_splits), n_samples))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(cv_random_state))

    y_arr = y.to_numpy(dtype=float)
    oof_pred = np.full(n_samples, np.nan, dtype=float)
    fold_metrics: list[dict[str, float]] = []
    fold_rows: list[dict[str, Any]] = []

    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)

        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train)
        pred = np.asarray(model.predict(X_val), dtype=float)
        oof_pred[val_idx] = pred
        metrics = _regression_summary(y_val.to_numpy(dtype=float), pred)
        fold_metrics.append(metrics)
        fold_rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_label": dataset_label,
                "fold": int(fold_id),
                "model": "OLS",
                "train_samples": int(len(train_idx)),
                "validation_samples": int(len(val_idx)),
                "r2": metrics["r2"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
            }
        )

    oof_metrics = _regression_summary(y_arr, oof_pred)
    r2_mean, r2_std = _metric_mean_std(fold_metrics, "r2")
    rmse_mean, rmse_std = _metric_mean_std(fold_metrics, "rmse")
    mae_mean, mae_std = _metric_mean_std(fold_metrics, "mae")
    baseline_row = {
        "dataset_key": dataset_key,
        "dataset_label": dataset_label,
        "dataset_path": str(data_path.resolve()),
        "model": "OLS",
        "samples": int(n_samples),
        "cv_splits": int(n_splits),
        "cv_random_state": int(cv_random_state),
        "ols_r2": r2_mean,
        "ols_rmse": rmse_mean,
        "ols_mae": mae_mean,
        "ols_r2_std": r2_std,
        "ols_rmse_std": rmse_std,
        "ols_mae_std": mae_std,
        "ols_oof_r2": oof_metrics["r2"],
        "ols_oof_rmse": oof_metrics["rmse"],
        "ols_oof_mae": oof_metrics["mae"],
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    return pd.DataFrame([baseline_row]), pd.DataFrame(fold_rows)


def _add_ols_columns(results_df: pd.DataFrame, ols_row: pd.Series) -> pd.DataFrame:
    df = results_df.copy()
    for metric in ("r2", "rmse", "mae"):
        df[f"ols_{metric}"] = float(ols_row[f"ols_{metric}"])
        df[f"ols_{metric}_std"] = float(ols_row[f"ols_{metric}_std"])
        df[f"ols_oof_{metric}"] = float(ols_row[f"ols_oof_{metric}"])
        df[f"{metric}_diff_gw_minus_ols"] = df[metric].astype(float) - float(
            ols_row[f"ols_{metric}"]
        )
        if f"xgb_{metric}" in df.columns:
            df[f"{metric}_diff_xgb_minus_ols"] = df[f"xgb_{metric}"].astype(float) - float(
                ols_row[f"ols_{metric}"]
            )
    return df


def _add_ols_best_columns(best_df: pd.DataFrame, ols_row: pd.Series) -> pd.DataFrame:
    df = best_df.copy()
    for metric in ("r2", "rmse", "mae"):
        df[f"ols_cv_mean_{metric}"] = float(ols_row[f"ols_{metric}"])
        df[f"ols_cv_std_{metric}"] = float(ols_row[f"ols_{metric}_std"])
        df[f"ols_oof_{metric}"] = float(ols_row[f"ols_oof_{metric}"])
        df[f"{metric}_diff_gw_minus_ols"] = df[f"gw_cv_mean_{metric}"].astype(float) - float(
            ols_row[f"ols_{metric}"]
        )
        if f"xgb_cv_mean_{metric}" in df.columns:
            df[f"{metric}_diff_xgb_minus_ols"] = df[
                f"xgb_cv_mean_{metric}"
            ].astype(float) - float(ols_row[f"ols_{metric}"])
    return df


def _dataset_root_for_label(run_root: Path, dataset_label: str) -> Path:
    return run_root / dataset_label


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"run_root 不存在：{run_root}")

    datasets = _resolve_datasets(args.data_root)
    selected_keys = _dataset_keys(args.datasets)

    all_results: list[pd.DataFrame] = []
    all_best: list[pd.DataFrame] = []
    all_ols: list[pd.DataFrame] = []
    all_ols_folds: list[pd.DataFrame] = []
    batch_plot_inputs: list[tuple[str, pd.DataFrame]] = []
    footer_lines_extra: list[str] = []

    for dataset_key in selected_keys:
        dataset_label, data_path = datasets[dataset_key]
        dataset_root = _dataset_root_for_label(run_root, dataset_label)
        results_path = dataset_root / DATASET_RESULTS_FILE
        best_path = dataset_root / DATASET_BEST_FILE
        if not results_path.exists() or not best_path.exists():
            raise FileNotFoundError(
                f"{dataset_key} 缺少 CV BW 输出，请先运行 run_cv_bw_sweep.py：{dataset_root}"
            )

        ols_df, ols_fold_df = _compute_ols_cv(
            dataset_key=dataset_key,
            dataset_label=dataset_label,
            data_path=data_path,
            dataset_root=dataset_root,
            cv_splits=int(args.cv_splits),
            cv_random_state=int(args.cv_random_state),
        )
        ols_row = ols_df.iloc[0]
        results_df = _add_ols_columns(
            pd.read_csv(results_path, encoding="utf-8-sig"), ols_row
        )
        best_df = _add_ols_best_columns(
            pd.read_csv(best_path, encoding="utf-8-sig"), ols_row
        )

        results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
        best_df.to_csv(best_path, index=False, encoding="utf-8-sig")
        ols_df.to_csv(dataset_root / DATASET_OLS_FILE, index=False, encoding="utf-8-sig")
        ols_fold_df.to_csv(
            dataset_root / DATASET_OLS_FOLD_FILE,
            index=False,
            encoding="utf-8-sig",
        )
        _plot_cv_bw_sweep(
            dataset_label=dataset_label,
            results_df=results_df,
            out_path=dataset_root / DATASET_PLOT_FILE,
        )

        all_results.append(results_df)
        all_best.append(best_df)
        all_ols.append(ols_df)
        all_ols_folds.append(ols_fold_df)
        batch_plot_inputs.append((dataset_label, results_df))

        if not footer_lines_extra:
            cfg = _configure_gwxgb(data_path=data_path, output_dir=run_root)
            footer_lines_extra = _overview_hyperparam_lines(
                cfg,
                cv_splits=int(args.cv_splits),
                cv_random_state=int(args.cv_random_state),
            )
            footer_lines_extra.append("OLS baseline: sklearn LinearRegression(fit_intercept=True)，使用相同 5 折划分。")

    pd.concat(all_results, ignore_index=True, sort=False).to_csv(
        run_root / RUN_ROOT_RESULTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(all_best, ignore_index=True, sort=False).to_csv(
        run_root / RUN_ROOT_BEST_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(all_ols, ignore_index=True, sort=False).to_csv(
        run_root / RUN_ROOT_OLS_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(all_ols_folds, ignore_index=True, sort=False).to_csv(
        run_root / RUN_ROOT_OLS_FOLD_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    _plot_run_root_cv_overview(
        dataset_results=batch_plot_inputs,
        out_path=run_root / RUN_ROOT_PLOT_FILE,
        footer_lines_extra=footer_lines_extra,
    )

    print(f"OLS baseline 已追加到：{run_root}")
    print(f"总览图已更新：{run_root / RUN_ROOT_PLOT_FILE}")
    print(f"OLS 汇总表：{run_root / RUN_ROOT_OLS_FILE}")


if __name__ == "__main__":
    main()
