from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import logging
import math
import posixpath
import re
import shutil
import sys
import warnings
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import font_manager
from matplotlib.patches import Polygon, Rectangle
from sklearn.model_selection import KFold

warnings.filterwarnings(
    "ignore",
    message=r"Glyph .* missing from font\(s\) Times New Roman\.",
)

_THIS_DIR = Path(__file__).resolve().parent
_COMPARE_DIR = _THIS_DIR / "compare"
_CODE_DIR = _THIS_DIR.parent
_XGB_DIR = _CODE_DIR / "xgb"
for _path in (_THIS_DIR, _COMPARE_DIR, _XGB_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from gwxgb_compare_demo_common import (
    CompareArtifacts,
    build_importance_table,
    build_sample_aggregated_local_df,
    compare_plot_max_display,
    load_demo_config,
    save_sample_aggregated_local_dependence_plots,
    save_sample_aggregated_local_native_plots,
)
from gwxgb_shap import (
    _patch_geoxgboost_parallelism,
    _spatial_weights_for_local_data,
    compute_gw_distance_matrix,
    export_local_models_shap,
    gw_distance_metric_label,
    gw_distance_unit,
    load_dataset as load_gwxgb_dataset,
    optimize_bandwidth,
    optimize_global_model,
    run_gxgb,
)
from xgb_shap import (
    build_and_train_model,
    compute_shap_and_interactions,
    load_config as load_xgb_config,
    load_dataset as load_xgb_dataset,
    plot_fixed_base_interactions,
    plot_top_interactions,
    setup_logging,
    summarize_and_save_interactions,
)
from xgb_shap_interaction_matrix import (
    _fit_best_model,
    _interaction_matrix_cfg,
    _regression_metrics,
    _selected_cmap_name,
    _selected_marker,
    plot_shap_interaction_matrix,
)
from spearman_corr_heatmap_batch import (
    compute_feature_corr,
    format_correlation_label,
    plot_corr_heatmap,
)


REPO_ROOT = _CODE_DIR.parent
OUTPUT_ROOT = REPO_ROOT / "Output"
COMPARE_BASE_CONFIG = _COMPARE_DIR / "gwxgb_compare_all_config.yaml"
GWXGB_BASE_CONFIG = _THIS_DIR / "gwxgb_config.yaml"
XGB_BASE_CONFIG = _XGB_DIR / "config.yaml"
FIVE_YEAR_DIR = REPO_ROOT / "Data" / "年份终市级指标数据（5年截面）"
FULL_DATA_PATH = REPO_ROOT / "Data" / "终市级指标数据_with_latlon.csv"

DEFAULT_DATASETS: dict[str, tuple[str, Path]] = {
    "2005": ("2005", FIVE_YEAR_DIR / "终市级指标数据_with_latlon_2005.csv"),
    "2010": ("2010", FIVE_YEAR_DIR / "终市级指标数据_with_latlon_2010.csv"),
    "2015": ("2015", FIVE_YEAR_DIR / "终市级指标数据_with_latlon_2015.csv"),
    "2020": ("2020", FIVE_YEAR_DIR / "终市级指标数据_with_latlon_2020.csv"),
    "full": ("2002_2020_full", FULL_DATA_PATH),
}

CURATED_STAGE_DIRS: dict[str, str] = {
    "gwxgb": "gwxgb_results_and_global_interactions",
    "interaction_matrix": "shap_interaction_matrix",
    "correlation_heatmap": "correlation_heatmap",
    "holdout_benchmark": "holdout_model_benchmark",
    "local_shap": "local_shap_tables",
    "sample_aggregated_local": "sample_aggregated_local_shap",
    "reuse_logs": "reused_source_logs",
}

CURATED_RUN_ROOT_DIRS: dict[str, str] = {
    "sankey": "shap_mean_value_sankey",
}
CURATED_CORRELATION_METHODS: tuple[str, ...] = ("spearman", "pearson")
MODEL_SUMMARY_FILE = "model_metrics_and_hyperparams.csv"
RUN_ROOT_MODEL_SUMMARY_FILE = "batch_model_metrics_and_hyperparams.csv"
MODEL_OVERVIEW_FILE = "model_summary_overview.csv"
MODEL_DETAILS_FILE = "model_summary_details.csv"
RUN_ROOT_MODEL_OVERVIEW_FILE = "batch_model_summary_overview.csv"
RUN_ROOT_MODEL_DETAILS_FILE = "batch_model_summary_details.csv"
MODEL_PERFORMANCE_MATRIX_FILE = "model_performance_matrix.csv"
MODEL_PERFORMANCE_DETAILS_FILE = "model_performance_details.csv"
GWXGB_LOCAL_DIAGNOSTICS_FILE = "gwxgb_local_diagnostics.csv"
RUN_ROOT_MODEL_PERFORMANCE_MATRIX_FILE = "batch_model_performance_matrix.csv"
RUN_ROOT_MODEL_PERFORMANCE_DETAILS_FILE = "batch_model_performance_details.csv"
RUN_ROOT_GWXGB_LOCAL_DIAGNOSTICS_FILE = "batch_gwxgb_local_diagnostics.csv"

GWXGB_STATS_METRIC_MAP: dict[str, str] = {
    "R2_Pred": "r2_pred",
    "MAE_Pred": "mae_pred",
    "RMS_Pred": "rmse_pred",
    "R2_oob": "r2_oob",
    "MAE_oob": "mae_oob",
    "RMS_oob": "rmse_oob",
    "R2oobGl": "r2_oob_global",
    "MAEoobGl": "mae_oob_global",
    "RMSEoobGl": "rmse_oob_global",
    "R2ens": "r2_ensemble",
    "MAEens": "mae_ensemble",
    "RMSEens": "rmse_ensemble",
}

GWXGB_STATS_PARAM_MAP: dict[str, str] = {
    "n_estimators": "xgb_n_estimators",
    "learning_rate": "xgb_learning_rate",
    "max_depth": "xgb_max_depth",
    "min_child_weight": "xgb_min_child_weight",
    "subsample": "xgb_subsample",
    "colsample_bytree": "xgb_colsample_bytree",
    "reg_alpha": "xgb_reg_alpha",
    "reg_lambda": "xgb_reg_lambda",
    "gamma": "xgb_gamma",
    "Spatial Units": "n_spatial_units",
    "Features": "n_features",
    "Kernel": "gw_kernel",
    "Bandwidth": "gw_bandwidth",
    "Spatial Weights": "gw_spatial_weights",
    "Alpha Weight type": "gw_alpha_weight_type",
    "Alpha Weight value": "gw_alpha_weight_value",
    "Feature Importance": "gw_feature_importance",
    "Test Size": "gw_test_size",
    "Seed": "gw_seed",
}

_XLSX_NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
_INTERACTION_BEST_PARAMS_RE = re.compile(r"最佳参数:\s*(\{[^\r\n]+\})")
_INTERACTION_BEST_SCORE_RE = re.compile(
    r"最佳分数\s*\(([^)]+)\):\s*([-+0-9.eE]+)"
)
_INTERACTION_METRICS_RE = re.compile(
    r"interaction matrix 模型已完成训练/选参："
    r"train_samples=(\d+),\s*test_samples=(\d+),\s*"
    r"test_RMSE=([-+0-9.eE]+),\s*test_R2=([-+0-9.eE]+)"
)

SANKEY_REQUIRED_DATASET_KEYS = ["2005", "2010", "2015", "2020", "full"]
SANKEY_DATASET_LABELS: dict[str, str] = {
    "2005": "2005",
    "2010": "2010",
    "2015": "2015",
    "2020": "2020",
    "full": "2002-2020",
}
SANKEY_FEATURE_LABELS: dict[str, str] = {
    "施工面积(万平方米)": "Const. area",
    "总人口（万人）": "Population",
    "建筑面积(万平方米)": "Bldg. area",
    "人均地区生产总值(元）": "GDP per cap.",
    "第三产业比重": "Tertiary share",
    "单位面积建材碳排放（kgco2/m2)": "Embodied carbon",
    "城镇化率": "Urbanization",
    "电力消费量（亿千瓦小时）": "Electricity",
    "电网因子（kgco2/kwh)": "Grid factor",
    "能源结构": "Energy mix",
}
SANKEY_COLORS = [
    "#FFD54F",
    "#E6DC53",
    "#BFD74A",
    "#8ECC59",
    "#59BD62",
    "#2EAD77",
    "#139A8B",
    "#157F92",
    "#206C9E",
    "#3658A5",
]


def _slug_summary_name(value: Any) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip()).strip("_").lower()
    return slug or "value"


def _coerce_summary_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return value

    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered == "true":
        return 1
    if lowered == "false":
        return 0
    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except ValueError:
            return text
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text):
        try:
            return float(text)
        except ValueError:
            return text
    return text


def _prefixed_summary(prefix: str, values: Dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def _summary_column_order(columns: Iterable[str]) -> list[str]:
    all_columns = list(columns)
    ordered: list[str] = []

    preferred = [
        "dataset_key",
        "dataset_label",
        "dataset_path",
        "summary_generated_at",
        "gwxgb_stats_file",
        "gwxgb_bw_results_file",
        "interaction_log_file",
        "gwxgb_stats_found",
        "interaction_training_logged",
    ]
    for name in preferred:
        if name in all_columns and name not in ordered:
            ordered.append(name)

    prefixes = [
        "gwxgb_metric_",
        "gwxgb_bw_search_",
        "gwxgb_xgb_",
        "gwxgb_gw_",
        "gwxgb_model_",
        "gwxgb_grid_search_",
        "xgb_baseline_metric_",
        "xgb_baseline_cv_",
        "xgb_baseline_oof_",
        "xgb_baseline_xgb_",
        "xgb_baseline_",
        "interaction_train_",
        "interaction_test_",
        "interaction_metric_",
        "interaction_grid_search_",
        "interaction_xgb_",
        "interaction_",
    ]
    for prefix in prefixes:
        for name in sorted(col for col in all_columns if col.startswith(prefix)):
            if name not in ordered:
                ordered.append(name)

    for name in sorted(all_columns):
        if name not in ordered:
            ordered.append(name)
    return ordered


def _is_missing_summary_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _summary_value(summary_row: Dict[str, Any], key: str) -> Any:
    value = summary_row.get(key)
    return None if _is_missing_summary_value(value) else value


def _combine_sources(*values: Any) -> str:
    ordered: list[str] = []
    for value in values:
        if _is_missing_summary_value(value):
            continue
        text = str(value).strip()
        if text and text not in ordered:
            ordered.append(text)
    return " | ".join(ordered)


def _append_detail_rows(
    rows: list[dict[str, Any]],
    *,
    summary_row: Dict[str, Any],
    stage_code: str,
    stage_name: str,
    source_file: str,
    fields: list[tuple[str, str, str, str, str]],
) -> None:
    base = {
        "数据集键": summary_row.get("dataset_key"),
        "数据集目录": summary_row.get("dataset_label"),
        "阶段代码": stage_code,
        "阶段": stage_name,
        "来源文件": source_file,
    }
    for group_name, item_code, item_name, key, note in fields:
        value = _summary_value(summary_row, key)
        if value is None:
            continue
        row = dict(base)
        row.update(
            {
                "分组": group_name,
                "条目代码": item_code,
                "条目": item_name,
                "值": value,
                "备注": note,
            }
        )
        rows.append(row)


def _metric_diff(left: Any, right: Any) -> Any:
    if _is_missing_summary_value(left) or _is_missing_summary_value(right):
        return None
    return float(left) - float(right)


def _gwxgb_metric_basis(summary_row: Dict[str, Any]) -> str:
    if _summary_value(summary_row, "gwxgb_bw_search_best_cv_r2") is not None:
        return "OOB 主指标（最终 CV 搜索结果见明细）"
    return "OOB（当前目录无 GeoXGBoost 最终CV均值）"


def _regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true_arr - y_pred_arr) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    y_min = float(np.min(y_true_arr))
    y_max = float(np.max(y_true_arr))
    value_range = y_max - y_min
    nrmse = float("nan") if value_range == 0.0 else rmse / value_range
    ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
    ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse": nrmse,
        "r2": r2,
    }


def _xgb_params_from_summary_row(summary_row: Dict[str, Any], *, prefix: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in summary_row.items():
        if not str(key).startswith(prefix):
            continue
        if _is_missing_summary_value(value):
            continue
        param_name = str(key)[len(prefix) :]
        params[param_name] = _coerce_summary_value(value)
    return params


def _configure_xgb_baseline(
    *,
    data_path: Path,
    gwxgb_config: Dict[str, Any],
    summary_row: Dict[str, Any],
) -> Dict[str, Any]:
    config = load_xgb_config(XGB_BASE_CONFIG)
    config = copy.deepcopy(config)

    data_cfg = dict(gwxgb_config.get("data") or {})
    data_cfg["path"] = str(data_path.resolve())
    config["data"] = data_cfg

    model_cfg = dict(config.get("model") or {})
    params = _xgb_params_from_summary_row(summary_row, prefix="gwxgb_xgb_")
    if not params:
        params = dict((gwxgb_config.get("model") or {}).get("params") or {})
    params["n_jobs"] = 1
    model_cfg["params"] = params
    model_cfg["random_state"] = int(
        _summary_value(summary_row, "gwxgb_model_random_state")
        or (gwxgb_config.get("model") or {}).get("random_state")
        or 42
    )
    config["model"] = model_cfg

    cv_cfg = dict(config.get("cv") or {})
    cv_cfg["use_cv"] = 1
    cv_cfg["n_splits"] = int(
        _summary_value(summary_row, "gwxgb_grid_search_cv")
        or cv_cfg.get("n_splits")
        or 5
    )
    cv_cfg["random_state"] = int(cv_cfg.get("random_state", 233))
    config["cv"] = cv_cfg
    return config


def _compute_xgb_baseline_summary(
    *,
    dataset_key: str,
    data_path: Path,
    gwxgb_config: Dict[str, Any],
    summary_row: Dict[str, Any],
) -> dict[str, Any]:
    config = _configure_xgb_baseline(
        data_path=data_path,
        gwxgb_config=gwxgb_config,
        summary_row=summary_row,
    )
    _, X, y = load_xgb_dataset(config)
    n_samples = int(len(X))
    n_features = int(X.shape[1])

    cv_cfg = config.get("cv") or {}
    n_splits = int(cv_cfg.get("n_splits", 5))
    random_state = int(cv_cfg.get("random_state", 233))
    if n_samples == 0:
        raise ValueError(f"XGBoost baseline 数据为空：{dataset_key}")
    if n_splits < 2 or n_splits > n_samples:
        raise ValueError(
            f"XGBoost baseline CV 配置无效：dataset={dataset_key}, n_splits={n_splits}, samples={n_samples}"
        )

    logging.info(
        "开始计算 XGBoost baseline 交叉验证指标：dataset=%s, samples=%s, features=%s, cv=%s",
        dataset_key,
        n_samples,
        n_features,
        n_splits,
    )

    fold_rmses: list[float] = []
    fold_maes: list[float] = []
    fold_nrmse: list[float] = []
    fold_r2s: list[float] = []
    oof_pred = np.full(n_samples, np.nan, dtype=float)

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, val_idx in splitter.split(X):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)

        model = build_and_train_model(config, X_train, y_train)
        y_pred = np.asarray(model.predict(X_val), dtype=float)
        fold_metrics = _regression_summary(y_val.to_numpy(dtype=float), y_pred)
        fold_rmses.append(fold_metrics["rmse"])
        fold_maes.append(fold_metrics["mae"])
        fold_nrmse.append(fold_metrics["nrmse"])
        fold_r2s.append(fold_metrics["r2"])
        oof_pred[np.asarray(val_idx, dtype=int)] = y_pred

    valid_mask = np.isfinite(oof_pred)
    oof_metrics = _regression_summary(
        y.iloc[valid_mask].to_numpy(dtype=float),
        oof_pred[valid_mask],
    )
    logging.info(
        "XGBoost baseline 交叉验证完成：dataset=%s, cv_mean_rmse=%.4f, cv_mean_mae=%.4f, cv_mean_r2=%.4f, oof_rmse=%.4f, oof_r2=%.4f",
        dataset_key,
        float(np.mean(fold_rmses)),
        float(np.mean(fold_maes)),
        float(np.mean(fold_r2s)),
        oof_metrics["rmse"],
        oof_metrics["r2"],
    )

    summary: dict[str, Any] = {
        "xgb_baseline_metric_source": "kfold_cv_same_params_same_dataset",
        "xgb_baseline_samples": n_samples,
        "xgb_baseline_features": n_features,
        "xgb_baseline_cv_n_splits": n_splits,
        "xgb_baseline_cv_random_state": random_state,
        "xgb_baseline_cv_mean_rmse": float(np.mean(fold_rmses)),
        "xgb_baseline_cv_std_rmse": float(np.std(fold_rmses)),
        "xgb_baseline_cv_mean_mae": float(np.mean(fold_maes)),
        "xgb_baseline_cv_std_mae": float(np.std(fold_maes)),
        "xgb_baseline_cv_mean_nrmse": float(np.mean(fold_nrmse)),
        "xgb_baseline_cv_std_nrmse": float(np.std(fold_nrmse)),
        "xgb_baseline_cv_mean_r2": float(np.mean(fold_r2s)),
        "xgb_baseline_cv_std_r2": float(np.std(fold_r2s)),
        "xgb_baseline_oof_rmse": oof_metrics["rmse"],
        "xgb_baseline_oof_mae": oof_metrics["mae"],
        "xgb_baseline_oof_nrmse": oof_metrics["nrmse"],
        "xgb_baseline_oof_r2": oof_metrics["r2"],
    }
    for key, value in _xgb_params_from_summary_row(summary_row, prefix="gwxgb_xgb_").items():
        summary[f"xgb_baseline_xgb_{key}"] = value
    return summary


def _benchmark_split_settings(config: Dict[str, Any]) -> tuple[float, int]:
    gw_cfg = config.get("gw") or {}
    model_cfg = config.get("model") or {}
    test_size = float(gw_cfg.get("test_size", model_cfg.get("test_size", 0.2)))
    if not (0.0 < test_size < 1.0):
        test_size = 0.2
    random_state = int(model_cfg.get("random_state", 42))
    return test_size, random_state


def _split_holdout_benchmark_data(
    *,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    coords: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    n_samples = int(len(X))
    if n_samples < 3:
        raise ValueError("Holdout benchmark 至少需要 3 个样本。")

    indices = np.random.RandomState(random_state).permutation(n_samples)
    n_test = int(n_samples * test_size)
    n_test = max(1, min(n_samples - 1, n_test))
    test_idx = np.asarray(indices[:n_test], dtype=int)
    train_idx = np.asarray(indices[n_test:], dtype=int)

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_df": df.iloc[train_idx].copy(),
        "test_df": df.iloc[test_idx].copy(),
        "X_train": X.iloc[train_idx].reset_index(drop=True),
        "X_test": X.iloc[test_idx].reset_index(drop=True),
        "y_train": y.iloc[train_idx].reset_index(drop=True),
        "y_test": y.iloc[test_idx].reset_index(drop=True),
        "coords_train": coords.iloc[train_idx].reset_index(drop=True),
        "coords_test": coords.iloc[test_idx].reset_index(drop=True),
    }


def _xgb_regressor_from_config(config: Dict[str, Any]) -> xgb.XGBRegressor:
    model_cfg = config.get("model") or {}
    params: dict[str, Any] = dict(model_cfg.get("params") or {})
    params.setdefault("objective", "reg:squarederror")
    params.setdefault("random_state", int(model_cfg.get("random_state", 42)))
    params.setdefault("n_jobs", 1)
    return xgb.XGBRegressor(**params)


def _select_holdout_local_positions(
    distances: np.ndarray,
    *,
    bw: Any,
    kernel: str,
) -> np.ndarray:
    dist_arr = np.asarray(distances, dtype=float).reshape(-1)
    if dist_arr.size < 2:
        raise ValueError("局部建模至少需要 2 个训练样本。")

    order = np.argsort(dist_arr, kind="mergesort")
    if str(kernel).strip().lower() == "adaptive":
        k = max(2, int(bw))
        k = min(k, len(order))
        return order[:k]

    threshold = float(bw)
    selected = np.flatnonzero(dist_arr <= threshold)
    if len(selected) < 2:
        return order[:2]
    return selected[np.argsort(dist_arr[selected], kind="mergesort")]


def _sanitize_local_weights(weights: np.ndarray) -> np.ndarray:
    weight_arr = np.asarray(weights, dtype=float).reshape(-1)
    weight_arr[~np.isfinite(weight_arr)] = 0.0
    weight_arr = np.clip(weight_arr, 0.0, None)
    if float(np.sum(weight_arr)) <= 0.0:
        return np.ones_like(weight_arr, dtype=float)
    return weight_arr


def _top_feature_from_model(
    model: xgb.XGBRegressor,
    feature_names: list[str],
) -> tuple[str, float | None]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return "", None
    importance_arr = np.asarray(importances, dtype=float).reshape(-1)
    if importance_arr.size != len(feature_names):
        return "", None
    if importance_arr.size == 0 or not np.any(np.isfinite(importance_arr)):
        return "", None
    top_idx = int(np.nanargmax(importance_arr))
    top_value = float(importance_arr[top_idx])
    if not np.isfinite(top_value):
        return "", None
    return str(feature_names[top_idx]), top_value


def _predict_holdout_gwxgb_local(
    *,
    config: Dict[str, Any],
    dataset_key: str,
    dataset_label: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    coords_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    coords_test: pd.DataFrame,
    test_source_indices: np.ndarray,
    bw: Any,
    log_progress: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    gw_cfg = config.get("gw") or {}
    kernel = str(gw_cfg.get("kernel", "Adaptive"))
    spatial_weights = bool(gw_cfg.get("spatial_weights", False))
    feature_names = [str(name) for name in X_train.columns]
    coord_names = [str(name) for name in coords_train.columns]

    distance_metric = gw_distance_metric_label(config)
    distance_unit = gw_distance_unit(config)
    dist_mat = compute_gw_distance_matrix(config, coords_train, coords_test)
    preds: list[float] = []
    diagnostics_rows: list[dict[str, Any]] = []
    n_test = int(len(X_test))
    log_every = max(1, min(100, n_test // 10 or 1)) if log_progress else 0

    for row_pos in range(n_test):
        distances = np.asarray(dist_mat[:, row_pos], dtype=float)
        local_positions = _select_holdout_local_positions(distances, bw=bw, kernel=kernel)
        local_distances = distances[local_positions]
        local_X = X_train.iloc[local_positions].reset_index(drop=True)
        local_y = y_train.iloc[local_positions].reset_index(drop=True)

        model = _xgb_regressor_from_config(config)
        weights_used = None
        if spatial_weights:
            weights_used = _sanitize_local_weights(
                _spatial_weights_for_local_data(local_distances, bw=bw, kernel=kernel)
            )
            model.fit(local_X, local_y, sample_weight=weights_used)
        else:
            model.fit(local_X, local_y)

        pred = float(np.asarray(model.predict(X_test.iloc[[row_pos]]), dtype=float)[0])
        preds.append(pred)

        top_feature, top_importance = _top_feature_from_model(model, feature_names)
        diag_row: dict[str, Any] = {
            "dataset_key": dataset_key,
            "dataset_label": dataset_label,
            "test_row_order": row_pos,
            "source_index": int(test_source_indices[row_pos]),
            "y_true": float(y_test.iloc[row_pos]),
            "y_pred": pred,
            "abs_error": abs(float(y_test.iloc[row_pos]) - pred),
            "local_train_samples": int(len(local_positions)),
            "distance_mean": float(np.mean(local_distances)),
            "distance_max": float(np.max(local_distances)),
            "distance_metric": distance_metric,
            "distance_unit": distance_unit,
            "bandwidth": _coerce_summary_value(bw),
            "kernel": kernel,
            "spatial_weights": int(spatial_weights),
            "top_feature": top_feature,
            "top_importance": top_importance,
        }
        if len(coord_names) >= 2:
            test_coord = coords_test.iloc[row_pos].to_numpy(dtype=float, copy=False)
            diag_row[coord_names[0]] = float(test_coord[0])
            diag_row[coord_names[1]] = float(test_coord[1])
        if weights_used is not None:
            diag_row["weight_mean"] = float(np.mean(weights_used))
            diag_row["weight_min"] = float(np.min(weights_used))
            diag_row["weight_max"] = float(np.max(weights_used))
            diag_row["weight_sum"] = float(np.sum(weights_used))
        else:
            diag_row["weight_mean"] = None
            diag_row["weight_min"] = None
            diag_row["weight_max"] = None
            diag_row["weight_sum"] = None
        diagnostics_rows.append(diag_row)

        if log_every and ((row_pos + 1) % log_every == 0 or row_pos + 1 == n_test):
            logging.info(
                "GW-XGBoost-local holdout 进度：dataset=%s, %s/%s",
                dataset_key,
                row_pos + 1,
                n_test,
            )

    diag_df = pd.DataFrame(diagnostics_rows)
    summary = {
        "local_model_count": int(len(diag_df)),
        "local_train_samples_mean": float(diag_df["local_train_samples"].mean()),
        "local_train_samples_min": int(diag_df["local_train_samples"].min()),
        "local_train_samples_max": int(diag_df["local_train_samples"].max()),
        "distance_mean_mean": float(diag_df["distance_mean"].mean()),
        "distance_max_mean": float(diag_df["distance_max"].mean()),
        "distance_metric": distance_metric,
        "distance_unit": distance_unit,
        "abs_error_mean": float(diag_df["abs_error"].mean()),
    }
    if "weight_mean" in diag_df.columns and diag_df["weight_mean"].notna().any():
        summary["weight_mean_mean"] = float(diag_df["weight_mean"].dropna().mean())
        summary["weight_max_mean"] = float(diag_df["weight_max"].dropna().mean())
        summary["weight_sum_mean"] = float(diag_df["weight_sum"].dropna().mean())
    return np.asarray(preds, dtype=float), diagnostics_rows, summary


def _build_holdout_benchmark_payload(
    *,
    dataset_key: str,
    dataset_root: Path,
    data_path: Path,
) -> dict[str, Any]:
    benchmark_dir = _curated_stage_output_dir(dataset_root, "holdout_benchmark")
    config = _configure_gwxgb(data_path=data_path, output_dir=benchmark_dir)
    _patch_geoxgboost_parallelism()
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

    benchmark_config = copy.deepcopy(config)
    optimize_global_model(benchmark_config, X_train, y_train)

    global_model = build_and_train_model(benchmark_config, X_train, y_train)
    global_pred = np.asarray(global_model.predict(X_test), dtype=float)
    global_metrics = _regression_summary(y_test.to_numpy(dtype=float), global_pred)

    bw_opt = optimize_bandwidth(
        benchmark_config,
        X_train,
        y_train,
        coords_train,
        output_dir=benchmark_dir,
    )
    local_pred, diagnostics_rows, local_summary = _predict_holdout_gwxgb_local(
        config=benchmark_config,
        dataset_key=dataset_key,
        dataset_label=dataset_root.name,
        X_train=X_train,
        y_train=y_train,
        coords_train=coords_train,
        X_test=X_test,
        y_test=y_test,
        coords_test=coords_test,
        test_source_indices=np.asarray(split["test_idx"], dtype=int),
        bw=bw_opt,
    )
    local_metrics = _regression_summary(y_test.to_numpy(dtype=float), local_pred)

    gw_cfg = benchmark_config.get("gw") or {}
    model_cfg = benchmark_config.get("model") or {}
    return {
        "dataset_key": dataset_key,
        "dataset_label": dataset_root.name,
        "dataset_path": str(data_path.resolve()),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_size": test_size,
        "split_random_state": split_random_state,
        "feature_count": int(X.shape[1]),
        "feature_names": [str(name) for name in X.columns],
        "coord_columns": [str(name) for name in coords.columns],
        "xgb_params": dict(model_cfg.get("params") or {}),
        "gw_params": {
            "bandwidth": _coerce_summary_value(bw_opt),
            "bandwidth_source": "optimized_on_train" if int(gw_cfg.get("optimize_bw", 1)) == 1 else "config_fixed",
            "kernel": gw_cfg.get("kernel"),
            "spatial_weights": int(bool(gw_cfg.get("spatial_weights", False))),
            "distance_metric": gw_distance_metric_label(benchmark_config),
            "distance_unit": gw_distance_unit(benchmark_config),
            "coord_order": gw_cfg.get("coord_order", "auto"),
            "alpha_weight_type": gw_cfg.get("alpha_wt_type"),
            "alpha_weight_value": _coerce_summary_value(gw_cfg.get("alpha_wt")),
            "optimize_bw": int(gw_cfg.get("optimize_bw", 0)),
            "bw_min": _coerce_summary_value(gw_cfg.get("bw_min")),
            "bw_max": _coerce_summary_value(gw_cfg.get("bw_max")),
            "bw_step": _coerce_summary_value(gw_cfg.get("bw_step")),
            "gw_seed": _coerce_summary_value(gw_cfg.get("seed")),
            "gw_n_splits": _coerce_summary_value(gw_cfg.get("n_splits")),
        },
        "performance_rows": [
            {
                "Model": "XGBoost-global",
                "R2": global_metrics["r2"],
                "RMSE": global_metrics["rmse"],
                "MAE": global_metrics["mae"],
                "TrainSamples": int(len(X_train)),
                "TestSamples": int(len(X_test)),
                "TestSize": test_size,
                "RandomState": split_random_state,
            },
            {
                "Model": "GW-XGBoost-local",
                "R2": local_metrics["r2"],
                "RMSE": local_metrics["rmse"],
                "MAE": local_metrics["mae"],
                "TrainSamples": int(len(X_train)),
                "TestSamples": int(len(X_test)),
                "TestSize": test_size,
                "RandomState": split_random_state,
            },
        ],
        "local_summary": local_summary,
        "local_diagnostics_rows": diagnostics_rows,
        "benchmark_dir": str(benchmark_dir.resolve()),
        "bw_results_file": str((benchmark_dir / "BW_results.csv").resolve()) if (benchmark_dir / "BW_results.csv").exists() else "",
    }


def _performance_details_rows_from_payload(payload: Dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = {
        "数据集键": payload.get("dataset_key"),
        "数据集目录": payload.get("dataset_label"),
    }

    def add_row(model: str, group_name: str, item_name: str, value: Any, note: str = "") -> None:
        rows.append(
            {
                **base,
                "模型": model,
                "分组": group_name,
                "条目": item_name,
                "值": value,
                "备注": note,
            }
        )

    add_row("总体", "数据集", "dataset_path", payload.get("dataset_path"))
    add_row("总体", "数据集", "feature_count", payload.get("feature_count"))
    add_row("总体", "数据集", "feature_names", " | ".join(payload.get("feature_names") or []))
    add_row("总体", "数据集", "coord_columns", " | ".join(payload.get("coord_columns") or []))
    add_row("总体", "划分", "train_samples", payload.get("train_samples"))
    add_row("总体", "划分", "test_samples", payload.get("test_samples"))
    add_row("总体", "划分", "test_size", payload.get("test_size"))
    add_row("总体", "划分", "random_state", payload.get("split_random_state"))

    gw_params = payload.get("gw_params") or {}
    for key in (
        "bandwidth",
        "bandwidth_source",
        "kernel",
        "spatial_weights",
        "distance_metric",
        "distance_unit",
        "coord_order",
        "alpha_weight_type",
        "alpha_weight_value",
        "optimize_bw",
        "bw_min",
        "bw_max",
        "bw_step",
        "gw_seed",
        "gw_n_splits",
    ):
        add_row("总体", "地理加权设定", key, gw_params.get(key))

    for key, value in (payload.get("xgb_params") or {}).items():
        add_row("总体", "XGBoost 参数", str(key), value)

    for perf_row in payload.get("performance_rows") or []:
        model_name = str(perf_row.get("Model"))
        add_row(model_name, "主性能", "R2", perf_row.get("R2"), "同一 holdout 测试集整体计算")
        add_row(model_name, "主性能", "RMSE", perf_row.get("RMSE"), "同一 holdout 测试集整体计算")
        add_row(model_name, "主性能", "MAE", perf_row.get("MAE"), "同一 holdout 测试集整体计算")

    for key, value in (payload.get("local_summary") or {}).items():
        add_row("GW-XGBoost-local", "局部诊断汇总", str(key), value, "诊断项，不作为主性能")

    bw_results_file = payload.get("bw_results_file")
    if bw_results_file:
        add_row("总体", "来源文件", "bw_results_file", bw_results_file)
    add_row("总体", "来源文件", "benchmark_dir", payload.get("benchmark_dir"))
    add_row("总体", "来源文件", "generated_at", payload.get("generated_at"))
    return rows


def _write_dataset_performance_outputs(
    *,
    dataset_key: str,
    dataset_root: Path,
    data_path: Path,
) -> dict[str, Any]:
    logging.info("开始输出论文式 holdout benchmark：dataset=%s", dataset_key)
    payload = _build_holdout_benchmark_payload(
        dataset_key=dataset_key,
        dataset_root=dataset_root,
        data_path=data_path,
    )

    matrix_path = dataset_root / MODEL_PERFORMANCE_MATRIX_FILE
    details_path = dataset_root / MODEL_PERFORMANCE_DETAILS_FILE
    diagnostics_path = dataset_root / GWXGB_LOCAL_DIAGNOSTICS_FILE

    matrix_df = pd.DataFrame(payload["performance_rows"])[
        ["Model", "R2", "RMSE", "MAE", "TrainSamples", "TestSamples", "TestSize", "RandomState"]
    ]
    details_df = pd.DataFrame(_performance_details_rows_from_payload(payload))
    diagnostics_df = pd.DataFrame(payload["local_diagnostics_rows"])

    matrix_df.to_csv(matrix_path, index=False, encoding="utf-8-sig")
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")
    diagnostics_df.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    logging.info("已保存 holdout benchmark 主性能矩阵：%s", matrix_path.resolve())
    logging.info("已保存 holdout benchmark 明细表：%s", details_path.resolve())
    logging.info("已保存 GW-XGBoost local diagnostics：%s", diagnostics_path.resolve())
    return payload


def _write_run_root_performance_outputs(
    *,
    run_root: Path,
    dataset_keys: Iterable[str],
    datasets: dict[str, tuple[str, Path]],
) -> Path | None:
    matrix_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []

    for dataset_key in dataset_keys:
        folder_name, data_path = datasets[dataset_key]
        dataset_root = _resolve_existing_dataset_root(run_root, folder_name)
        if not dataset_root.exists():
            continue

        matrix_path = dataset_root / MODEL_PERFORMANCE_MATRIX_FILE
        details_path = dataset_root / MODEL_PERFORMANCE_DETAILS_FILE
        diagnostics_path = dataset_root / GWXGB_LOCAL_DIAGNOSTICS_FILE
        if not (matrix_path.exists() and details_path.exists() and diagnostics_path.exists()):
            _write_dataset_performance_outputs(
                dataset_key=dataset_key,
                dataset_root=dataset_root,
                data_path=data_path,
            )

        matrix_df = pd.read_csv(matrix_path, encoding="utf-8-sig")
        matrix_df.insert(0, "DatasetLabel", dataset_root.name)
        matrix_df.insert(0, "DatasetKey", dataset_key)
        matrix_frames.append(matrix_df)

        detail_frames.append(pd.read_csv(details_path, encoding="utf-8-sig"))
        diagnostic_frames.append(pd.read_csv(diagnostics_path, encoding="utf-8-sig"))

    if not matrix_frames:
        return None

    matrix_output = run_root / RUN_ROOT_MODEL_PERFORMANCE_MATRIX_FILE
    details_output = run_root / RUN_ROOT_MODEL_PERFORMANCE_DETAILS_FILE
    diagnostics_output = run_root / RUN_ROOT_GWXGB_LOCAL_DIAGNOSTICS_FILE

    pd.concat(matrix_frames, ignore_index=True, sort=False).to_csv(
        matrix_output,
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(detail_frames, ignore_index=True, sort=False).to_csv(
        details_output,
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(diagnostic_frames, ignore_index=True, sort=False).to_csv(
        diagnostics_output,
        index=False,
        encoding="utf-8-sig",
    )
    logging.info("已保存批处理 holdout benchmark 主性能矩阵：%s", matrix_output.resolve())
    logging.info("已保存批处理 holdout benchmark 明细表：%s", details_output.resolve())
    logging.info("已保存批处理 GW-XGBoost local diagnostics：%s", diagnostics_output.resolve())
    return matrix_output


def _overview_rows_from_summary(summary_row: Dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "数据集键": summary_row.get("dataset_key"),
            "数据集目录": summary_row.get("dataset_label"),
            "样本数": _summary_value(summary_row, "gwxgb_n_spatial_units")
            or _summary_value(summary_row, "xgb_baseline_samples"),
            "特征数": _summary_value(summary_row, "gwxgb_n_features")
            or _summary_value(summary_row, "xgb_baseline_features"),
            "GeoXGBoost_指标口径": _gwxgb_metric_basis(summary_row),
            "GeoXGBoost_R2": _summary_value(summary_row, "gwxgb_metric_r2_oob"),
            "GeoXGBoost_MAE": _summary_value(summary_row, "gwxgb_metric_mae_oob"),
            "GeoXGBoost_RMSE": _summary_value(summary_row, "gwxgb_metric_rmse_oob"),
            "XGBbaseline_CV均值_R2": _summary_value(summary_row, "xgb_baseline_cv_mean_r2"),
            "XGBbaseline_CV标准差_R2": _summary_value(summary_row, "xgb_baseline_cv_std_r2"),
            "XGBbaseline_CV均值_MAE": _summary_value(summary_row, "xgb_baseline_cv_mean_mae"),
            "XGBbaseline_CV标准差_MAE": _summary_value(summary_row, "xgb_baseline_cv_std_mae"),
            "XGBbaseline_CV均值_RMSE": _summary_value(summary_row, "xgb_baseline_cv_mean_rmse"),
            "XGBbaseline_CV标准差_RMSE": _summary_value(summary_row, "xgb_baseline_cv_std_rmse"),
            "XGBbaseline_OOF_R2": _summary_value(summary_row, "xgb_baseline_oof_r2"),
            "XGBbaseline_OOF_MAE": _summary_value(summary_row, "xgb_baseline_oof_mae"),
            "XGBbaseline_OOF_RMSE": _summary_value(summary_row, "xgb_baseline_oof_rmse"),
            "对照_R2差值_GeoXGBoost减BaselineOOF": _metric_diff(
                _summary_value(summary_row, "gwxgb_metric_r2_oob"),
                _summary_value(summary_row, "xgb_baseline_oof_r2"),
            ),
            "对照_MAE差值_GeoXGBoost减BaselineOOF": _metric_diff(
                _summary_value(summary_row, "gwxgb_metric_mae_oob"),
                _summary_value(summary_row, "xgb_baseline_oof_mae"),
            ),
            "对照_RMSE差值_GeoXGBoost减BaselineOOF": _metric_diff(
                _summary_value(summary_row, "gwxgb_metric_rmse_oob"),
                _summary_value(summary_row, "xgb_baseline_oof_rmse"),
            ),
            "带宽": _summary_value(summary_row, "gwxgb_gw_bandwidth"),
            "核函数": _summary_value(summary_row, "gwxgb_gw_kernel"),
            "空间权重": _summary_value(summary_row, "gwxgb_gw_spatial_weights"),
            "距离口径": _summary_value(summary_row, "gwxgb_gw_distance_metric_config"),
            "距离单位": _summary_value(summary_row, "gwxgb_gw_distance_unit_config"),
            "经纬顺序": _summary_value(summary_row, "gwxgb_gw_coord_order_config"),
            "Alpha权重类型": _summary_value(summary_row, "gwxgb_gw_alpha_weight_type"),
            "Alpha权重值": _summary_value(summary_row, "gwxgb_gw_alpha_weight_value"),
            "XGB_n_estimators": _summary_value(summary_row, "gwxgb_xgb_n_estimators"),
            "XGB_learning_rate": _summary_value(summary_row, "gwxgb_xgb_learning_rate"),
            "XGB_max_depth": _summary_value(summary_row, "gwxgb_xgb_max_depth"),
            "XGB_min_child_weight": _summary_value(summary_row, "gwxgb_xgb_min_child_weight"),
            "XGB_subsample": _summary_value(summary_row, "gwxgb_xgb_subsample"),
            "XGB_colsample_bytree": _summary_value(summary_row, "gwxgb_xgb_colsample_bytree"),
            "XGB_reg_alpha": _summary_value(summary_row, "gwxgb_xgb_reg_alpha"),
            "XGB_reg_lambda": _summary_value(summary_row, "gwxgb_xgb_reg_lambda"),
            "Baseline_CV折数": _summary_value(summary_row, "xgb_baseline_cv_n_splits"),
            "Baseline_CV随机种子": _summary_value(summary_row, "xgb_baseline_cv_random_state"),
            "备注": "表内对照以 GeoXGBoost 的 OOB 指标与 XGBoost baseline 的 OOF/CV 指标为主；不再纳入分析阶段模型性能。",
        }
    ]


def _details_rows_from_summary(summary_row: Dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gwxgb_source = _combine_sources(
        summary_row.get("gwxgb_stats_file"),
        summary_row.get("gwxgb_bw_results_file"),
    )
    baseline_source = _combine_sources(summary_row.get("dataset_path"))

    _append_detail_rows(
        rows,
        summary_row=summary_row,
        stage_code="gwxgb",
        stage_name="GeoXGBoost 主模型",
        source_file=gwxgb_source,
        fields=[
            ("对照性能", "r2_oob", "GeoXGBoost OOB R2", "gwxgb_metric_r2_oob", "主对照指标，来自 LW_GXGB.xlsx"),
            ("对照性能", "mae_oob", "GeoXGBoost OOB MAE", "gwxgb_metric_mae_oob", "主对照指标，来自 LW_GXGB.xlsx"),
            ("对照性能", "rmse_oob", "GeoXGBoost OOB RMSE", "gwxgb_metric_rmse_oob", "主对照指标，来自 LW_GXGB.xlsx"),
            ("补充指标", "r2_pred", "Pred R2", "gwxgb_metric_r2_pred", "全样本预测，仅供参考"),
            ("补充指标", "mae_pred", "Pred MAE", "gwxgb_metric_mae_pred", "全样本预测，仅供参考"),
            ("补充指标", "rmse_pred", "Pred RMSE", "gwxgb_metric_rmse_pred", "全样本预测，仅供参考"),
            ("补充指标", "r2_ensemble", "Ensemble R2", "gwxgb_metric_r2_ensemble", "局部集成结果"),
            ("补充指标", "mae_ensemble", "Ensemble MAE", "gwxgb_metric_mae_ensemble", "局部集成结果"),
            ("补充指标", "rmse_ensemble", "Ensemble RMSE", "gwxgb_metric_rmse_ensemble", "局部集成结果"),
            ("地理加权参数", "bandwidth", "Bandwidth", "gwxgb_gw_bandwidth", "Adaptive 核下表示邻居数"),
            ("地理加权参数", "kernel", "Kernel", "gwxgb_gw_kernel", ""),
            ("地理加权参数", "spatial_weights", "Spatial Weights", "gwxgb_gw_spatial_weights", ""),
            ("地理加权参数", "distance_metric", "Distance Metric", "gwxgb_gw_distance_metric_config", "GW 局部采样距离口径"),
            ("地理加权参数", "distance_unit", "Distance Unit", "gwxgb_gw_distance_unit_config", ""),
            ("地理加权参数", "coord_order", "Coord Order", "gwxgb_gw_coord_order_config", "haversine 时用于识别经纬度顺序"),
            ("地理加权参数", "alpha_weight_type", "Alpha Weight type", "gwxgb_gw_alpha_weight_type", ""),
            ("地理加权参数", "alpha_weight_value", "Alpha Weight value", "gwxgb_gw_alpha_weight_value", ""),
            ("地理加权参数", "test_size", "Test Size", "gwxgb_gw_test_size", ""),
            ("地理加权参数", "seed", "Seed", "gwxgb_gw_seed", ""),
            ("带宽搜索", "candidate_rows", "带宽搜索候选数", "gwxgb_bw_search_rows", "来自 BW_results.csv"),
            ("带宽搜索", "best_cv_bw", "CV 最优带宽", "gwxgb_bw_search_best_cv_bw", "来自 BW_results.csv"),
            ("带宽搜索", "best_cv_r2", "CV 最优 R2", "gwxgb_bw_search_best_cv_r2", "来自 BW_results.csv"),
            ("带宽搜索", "best_cv_rmse", "CV 最优 RMSE", "gwxgb_bw_search_best_cv_rmse", "来自 BW_results.csv"),
            ("XGBoost 参数", "n_estimators", "n_estimators", "gwxgb_xgb_n_estimators", ""),
            ("XGBoost 参数", "learning_rate", "learning_rate", "gwxgb_xgb_learning_rate", ""),
            ("XGBoost 参数", "max_depth", "max_depth", "gwxgb_xgb_max_depth", ""),
            ("XGBoost 参数", "min_child_weight", "min_child_weight", "gwxgb_xgb_min_child_weight", ""),
            ("XGBoost 参数", "subsample", "subsample", "gwxgb_xgb_subsample", ""),
            ("XGBoost 参数", "colsample_bytree", "colsample_bytree", "gwxgb_xgb_colsample_bytree", ""),
            ("XGBoost 参数", "reg_alpha", "reg_alpha", "gwxgb_xgb_reg_alpha", ""),
            ("XGBoost 参数", "reg_lambda", "reg_lambda", "gwxgb_xgb_reg_lambda", ""),
            ("网格搜索配置", "enabled", "是否启用网格搜索", "gwxgb_grid_search_enabled", "配置项"),
            ("网格搜索配置", "scoring", "评分指标", "gwxgb_grid_search_scoring", "配置项"),
            ("网格搜索配置", "cv", "CV 折数", "gwxgb_grid_search_cv", "配置项"),
        ],
    )

    _append_detail_rows(
        rows,
        summary_row=summary_row,
        stage_code="xgb_baseline",
        stage_name="XGBoost baseline",
        source_file=baseline_source,
        fields=[
            ("交叉验证均值", "cv_mean_r2", "CV mean R2", "xgb_baseline_cv_mean_r2", "KFold 平均"),
            ("交叉验证均值", "cv_mean_mae", "CV mean MAE", "xgb_baseline_cv_mean_mae", "KFold 平均"),
            ("交叉验证均值", "cv_mean_rmse", "CV mean RMSE", "xgb_baseline_cv_mean_rmse", "KFold 平均"),
            ("交叉验证标准差", "cv_std_r2", "CV std R2", "xgb_baseline_cv_std_r2", ""),
            ("交叉验证标准差", "cv_std_mae", "CV std MAE", "xgb_baseline_cv_std_mae", ""),
            ("交叉验证标准差", "cv_std_rmse", "CV std RMSE", "xgb_baseline_cv_std_rmse", ""),
            ("OOF对照指标", "oof_r2", "OOF R2", "xgb_baseline_oof_r2", "与 GeoXGBoost OOB 指标对照"),
            ("OOF对照指标", "oof_mae", "OOF MAE", "xgb_baseline_oof_mae", "与 GeoXGBoost OOB 指标对照"),
            ("OOF对照指标", "oof_rmse", "OOF RMSE", "xgb_baseline_oof_rmse", "与 GeoXGBoost OOB 指标对照"),
            ("交叉验证配置", "cv_n_splits", "CV 折数", "xgb_baseline_cv_n_splits", ""),
            ("交叉验证配置", "cv_random_state", "CV 随机种子", "xgb_baseline_cv_random_state", ""),
            ("XGBoost 参数", "n_estimators", "n_estimators", "xgb_baseline_xgb_n_estimators", "与 GeoXGBoost 使用相同超参数"),
            ("XGBoost 参数", "learning_rate", "learning_rate", "xgb_baseline_xgb_learning_rate", "与 GeoXGBoost 使用相同超参数"),
            ("XGBoost 参数", "max_depth", "max_depth", "xgb_baseline_xgb_max_depth", "与 GeoXGBoost 使用相同超参数"),
            ("XGBoost 参数", "min_child_weight", "min_child_weight", "xgb_baseline_xgb_min_child_weight", "与 GeoXGBoost 使用相同超参数"),
            ("XGBoost 参数", "subsample", "subsample", "xgb_baseline_xgb_subsample", "与 GeoXGBoost 使用相同超参数"),
            ("XGBoost 参数", "colsample_bytree", "colsample_bytree", "xgb_baseline_xgb_colsample_bytree", "与 GeoXGBoost 使用相同超参数"),
            ("XGBoost 参数", "reg_alpha", "reg_alpha", "xgb_baseline_xgb_reg_alpha", "与 GeoXGBoost 使用相同超参数"),
            ("XGBoost 参数", "reg_lambda", "reg_lambda", "xgb_baseline_xgb_reg_lambda", "与 GeoXGBoost 使用相同超参数"),
        ],
    )
    return rows


def _write_readable_dataset_tables(
    *,
    dataset_root: Path,
    summary_row: Dict[str, Any],
) -> None:
    overview_df = pd.DataFrame(_overview_rows_from_summary(summary_row))
    details_df = pd.DataFrame(_details_rows_from_summary(summary_row))
    overview_path = dataset_root / MODEL_OVERVIEW_FILE
    details_path = dataset_root / MODEL_DETAILS_FILE
    overview_df.to_csv(overview_path, index=False, encoding="utf-8-sig")
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")
    logging.info("已保存可读版模型概览表：%s", overview_path.resolve())
    logging.info("已保存可读版模型明细表：%s", details_path.resolve())


def _write_readable_run_root_tables(
    *,
    run_root: Path,
    summary_rows: list[Dict[str, Any]],
) -> None:
    overview_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for summary_row in summary_rows:
        overview_rows.extend(_overview_rows_from_summary(summary_row))
        detail_rows.extend(_details_rows_from_summary(summary_row))

    overview_df = pd.DataFrame(overview_rows)
    details_df = pd.DataFrame(detail_rows)
    overview_path = run_root / RUN_ROOT_MODEL_OVERVIEW_FILE
    details_path = run_root / RUN_ROOT_MODEL_DETAILS_FILE
    overview_df.to_csv(overview_path, index=False, encoding="utf-8-sig")
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")
    logging.info("已保存批处理可读版模型概览表：%s", overview_path.resolve())
    logging.info("已保存批处理可读版模型明细表：%s", details_path.resolve())


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    shared_strings: list[str] = []
    for item in root.findall("m:si", _XLSX_NS):
        texts = [node.text or "" for node in item.iterfind(".//m:t", _XLSX_NS)]
        shared_strings.append("".join(texts))
    return shared_strings


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    inline = cell.find("m:is", _XLSX_NS)
    if inline is not None:
        texts = [node.text or "" for node in inline.iterfind(".//m:t", _XLSX_NS)]
        return "".join(texts)

    value_node = cell.find("m:v", _XLSX_NS)
    if value_node is None:
        return ""

    text = value_node.text or ""
    if cell.attrib.get("t") == "s":
        try:
            idx = int(text)
        except ValueError:
            return text
        if 0 <= idx < len(shared_strings):
            return shared_strings[idx]
    return text


def _xlsx_sheet_rows(xlsx_path: Path, sheet_name: str) -> list[list[str]]:
    with zipfile.ZipFile(xlsx_path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        shared_strings = _xlsx_shared_strings(archive)

        target_path: str | None = None
        sheets = workbook.find("m:sheets", _XLSX_NS)
        if sheets is None:
            return []
        rel_attr = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        for sheet in sheets:
            if sheet.attrib.get("name") != sheet_name:
                continue
            rel_id = sheet.attrib.get(rel_attr)
            if not rel_id:
                break
            raw_target = rel_map.get(rel_id, "").lstrip("/")
            target_path = (
                raw_target
                if raw_target.startswith("xl/")
                else posixpath.normpath(posixpath.join("xl", raw_target))
            )
            break

        if not target_path:
            return []

        sheet_root = ET.fromstring(archive.read(target_path))
        rows: list[list[str]] = []
        for row in sheet_root.findall(".//m:sheetData/m:row", _XLSX_NS):
            rows.append(
                [
                    _xlsx_cell_value(cell, shared_strings)
                    for cell in row.findall("m:c", _XLSX_NS)
                ]
            )
        return rows


def _parse_gwxgb_stats_sheet(stats_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not stats_path.exists():
        return {}, {}

    try:
        rows = _xlsx_sheet_rows(stats_path, "Stats")
    except (KeyError, ET.ParseError, OSError, zipfile.BadZipFile) as exc:
        logging.warning("读取 LW_GXGB.xlsx 失败，无法生成模型汇总：%s；原因：%s", stats_path, exc)
        return {}, {}

    if len(rows) < 2:
        return {}, {}

    metrics: dict[str, Any] = {}
    params: dict[str, Any] = {}

    metric_headers = [str(item).strip() for item in rows[0] if str(item).strip()]
    raw_metric_values = rows[1]
    metric_values = (
        raw_metric_values[1:]
        if len(raw_metric_values) >= len(metric_headers) + 1
        else raw_metric_values[: len(metric_headers)]
    )
    for header, value in zip(metric_headers, metric_values):
        key = GWXGB_STATS_METRIC_MAP.get(str(header).strip())
        if key:
            metrics[key] = _coerce_summary_value(value)

    for row in rows[2:]:
        if not row:
            continue
        raw_key = str(row[0]).strip()
        if not raw_key or raw_key == "Value":
            continue
        if raw_key in {"Notes:", "Description"}:
            break
        value = row[1] if len(row) > 1 else None
        key = GWXGB_STATS_PARAM_MAP.get(raw_key, f"stats_{_slug_summary_name(raw_key)}")
        params[key] = _coerce_summary_value(value)

    return metrics, params


def _load_bw_search_summary(
    bw_results_path: Path,
    *,
    selected_bw: Any,
) -> dict[str, Any]:
    if not bw_results_path.exists():
        return {}

    try:
        df = pd.read_csv(bw_results_path, encoding="utf-8-sig")
    except (OSError, pd.errors.EmptyDataError) as exc:
        logging.warning("读取 BW_results.csv 失败，无法补充带宽搜索汇总：%s；原因：%s", bw_results_path, exc)
        return {}

    if df.empty or "BW" not in df.columns:
        return {}

    summary: dict[str, Any] = {"bw_search_rows": int(df.shape[0])}
    bw_series = pd.to_numeric(df["BW"], errors="coerce")

    selected_row: pd.Series | None = None
    try:
        selected_bw_float = float(selected_bw)
    except (TypeError, ValueError):
        selected_bw_float = float("nan")
    if np.isfinite(selected_bw_float):
        matches = df.loc[bw_series.notna() & np.isclose(bw_series, selected_bw_float)]
        if not matches.empty:
            selected_row = matches.iloc[0]

    if selected_row is not None:
        summary.update(
            {
                "bw_search_selected_bw": _coerce_summary_value(selected_row.get("BW")),
                "bw_search_selected_r2": _coerce_summary_value(selected_row.get("R2")),
                "bw_search_selected_mae": _coerce_summary_value(selected_row.get("MAE")),
                "bw_search_selected_rmse": _coerce_summary_value(selected_row.get("RMSE")),
                "bw_search_selected_cv": _coerce_summary_value(selected_row.get("CV")),
            }
        )

    if "CV" in df.columns:
        cv_series = pd.to_numeric(df["CV"], errors="coerce")
        valid = df.loc[cv_series.notna()]
        if not valid.empty:
            best_row = valid.loc[cv_series.loc[valid.index].idxmin()]
            summary.update(
                {
                    "bw_search_best_cv_bw": _coerce_summary_value(best_row.get("BW")),
                    "bw_search_best_cv_r2": _coerce_summary_value(best_row.get("R2")),
                    "bw_search_best_cv_mae": _coerce_summary_value(best_row.get("MAE")),
                    "bw_search_best_cv_rmse": _coerce_summary_value(best_row.get("RMSE")),
                    "bw_search_best_cv": _coerce_summary_value(best_row.get("CV")),
                }
            )

    return summary


def _gwxgb_config_summary(config: Dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    model_cfg = config.get("model") or {}
    for key, value in (model_cfg.get("params") or {}).items():
        summary[f"xgb_{_slug_summary_name(key)}"] = _coerce_summary_value(value)
    summary["model_random_state"] = _coerce_summary_value(model_cfg.get("random_state"))

    grid_cfg = config.get("grid_search") or {}
    summary["grid_search_enabled"] = int(bool(grid_cfg.get("enabled", False)))
    summary["grid_search_scoring"] = _coerce_summary_value(grid_cfg.get("scoring"))
    summary["grid_search_cv"] = _coerce_summary_value(grid_cfg.get("cv"))
    summary["grid_search_verbose"] = _coerce_summary_value(grid_cfg.get("verbose"))

    gw_cfg = config.get("gw") or {}
    summary["gw_optimize_bw_config"] = _coerce_summary_value(gw_cfg.get("optimize_bw"))
    summary["gw_bw_min_config"] = _coerce_summary_value(gw_cfg.get("bw_min"))
    summary["gw_bw_max_config"] = _coerce_summary_value(gw_cfg.get("bw_max"))
    summary["gw_bw_step_config"] = _coerce_summary_value(gw_cfg.get("bw_step"))
    summary["gw_n_splits_config"] = _coerce_summary_value(gw_cfg.get("n_splits"))
    summary["gw_distance_metric_config"] = _coerce_summary_value(gw_distance_metric_label(config))
    summary["gw_distance_unit_config"] = _coerce_summary_value(gw_distance_unit(config))
    summary["gw_coord_order_config"] = _coerce_summary_value(gw_cfg.get("coord_order", "auto"))
    return summary


def _interaction_config_summary(config: Dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    model_cfg = config.get("model") or {}
    for key, value in (model_cfg.get("params") or {}).items():
        summary[f"xgb_{_slug_summary_name(key)}"] = _coerce_summary_value(value)

    matrix_cfg = _interaction_matrix_cfg(config)
    summary["test_size"] = _coerce_summary_value(matrix_cfg.get("test_size"))
    summary["random_state"] = _coerce_summary_value(matrix_cfg.get("random_state"))
    summary["use_grid_search"] = _coerce_summary_value(matrix_cfg.get("use_grid_search"))
    summary["scoring"] = _coerce_summary_value(matrix_cfg.get("scoring"))
    summary["cv_splits"] = _coerce_summary_value(matrix_cfg.get("cv_splits"))
    summary["grid_n_jobs"] = _coerce_summary_value(matrix_cfg.get("grid_n_jobs"))
    summary["model_n_jobs"] = _coerce_summary_value(matrix_cfg.get("model_n_jobs"))
    summary["scheme_index"] = _coerce_summary_value(matrix_cfg.get("scheme_index"))
    summary["style_index"] = _coerce_summary_value(matrix_cfg.get("style_index"))
    return summary


def _parse_interaction_log(
    log_path: Path,
    *,
    config: Dict[str, Any],
) -> dict[str, Any]:
    summary = _prefixed_summary("interaction_", _interaction_config_summary(config))
    if not log_path.exists():
        summary["interaction_training_logged"] = 0
        return summary

    summary["interaction_training_logged"] = 0
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    best_param_matches = list(_INTERACTION_BEST_PARAMS_RE.finditer(text))
    if best_param_matches:
        raw_text = best_param_matches[-1].group(1)
        try:
            parsed = ast.literal_eval(raw_text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                summary[f"interaction_xgb_{_slug_summary_name(key)}"] = _coerce_summary_value(value)

    best_score_matches = list(_INTERACTION_BEST_SCORE_RE.finditer(text))
    if best_score_matches:
        metric_name, best_score = best_score_matches[-1].groups()
        summary["interaction_grid_search_best_score_metric"] = metric_name
        summary["interaction_grid_search_best_score"] = _coerce_summary_value(best_score)

    metric_matches = list(_INTERACTION_METRICS_RE.finditer(text))
    if metric_matches:
        match = metric_matches[-1]
        summary["interaction_training_logged"] = 1
        summary["interaction_train_samples"] = int(match.group(1))
        summary["interaction_test_samples"] = int(match.group(2))
        summary["interaction_metric_test_rmse"] = float(match.group(3))
        summary["interaction_metric_test_r2"] = float(match.group(4))

    return summary


def _build_dataset_model_summary(
    *,
    dataset_key: str,
    dataset_root: Path,
    data_path: Path,
) -> dict[str, Any]:
    gwxgb_dir = dataset_root / CURATED_STAGE_DIRS["gwxgb"]
    gwxgb_stats_path = gwxgb_dir / "LW_GXGB.xlsx"
    bw_results_path = gwxgb_dir / "BW_results.csv"

    gwxgb_metrics, gwxgb_params = _parse_gwxgb_stats_sheet(gwxgb_stats_path)
    gwxgb_config = _configure_gwxgb(data_path=data_path, output_dir=gwxgb_dir)

    row: dict[str, Any] = {
        "dataset_key": dataset_key,
        "dataset_label": dataset_root.name,
        "dataset_path": str(data_path.resolve()),
        "summary_generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "gwxgb_stats_file": str(gwxgb_stats_path.resolve()) if gwxgb_stats_path.exists() else "",
        "gwxgb_bw_results_file": str(bw_results_path.resolve()) if bw_results_path.exists() else "",
        "gwxgb_stats_found": int(bool(gwxgb_metrics or gwxgb_params)),
    }
    row.update(_prefixed_summary("gwxgb_metric_", gwxgb_metrics))
    row.update(_prefixed_summary("gwxgb_", gwxgb_params))
    for key, value in _prefixed_summary("gwxgb_", _gwxgb_config_summary(gwxgb_config)).items():
        row.setdefault(key, value)

    selected_bw = row.get("gwxgb_gw_bandwidth")
    row.update(_prefixed_summary("gwxgb_", _load_bw_search_summary(bw_results_path, selected_bw=selected_bw)))
    row.update(
        _compute_xgb_baseline_summary(
            dataset_key=dataset_key,
            data_path=data_path,
            gwxgb_config=gwxgb_config,
            summary_row=row,
        )
    )
    return row


def _write_dataset_model_summary(
    *,
    dataset_key: str,
    dataset_root: Path,
    data_path: Path,
) -> Path:
    summary_row = _build_dataset_model_summary(
        dataset_key=dataset_key,
        dataset_root=dataset_root,
        data_path=data_path,
    )
    summary_path = dataset_root / MODEL_SUMMARY_FILE
    summary_df = pd.DataFrame([summary_row])
    summary_df = summary_df.reindex(columns=_summary_column_order(summary_df.columns))
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    _write_readable_dataset_tables(
        dataset_root=dataset_root,
        summary_row=summary_row,
    )
    logging.info("已保存数据集模型指标与超参数汇总表：%s", summary_path.resolve())
    return summary_path


def _write_run_root_model_summary(
    *,
    run_root: Path,
    dataset_keys: Iterable[str],
    datasets: dict[str, tuple[str, Path]],
) -> Path | None:
    frames: list[pd.DataFrame] = []
    summary_rows: list[Dict[str, Any]] = []
    for dataset_key in dataset_keys:
        folder_name, data_path = datasets[dataset_key]
        dataset_root = _resolve_existing_dataset_root(run_root, folder_name)
        if not dataset_root.exists():
            continue

        summary_path = dataset_root / MODEL_SUMMARY_FILE
        if summary_path.exists():
            loaded_df = pd.read_csv(summary_path, encoding="utf-8-sig")
            frames.append(loaded_df)
            summary_rows.extend(loaded_df.to_dict(orient="records"))
            continue

        summary_row = _build_dataset_model_summary(
            dataset_key=dataset_key,
            dataset_root=dataset_root,
            data_path=data_path,
        )
        summary_rows.append(summary_row)
        frames.append(pd.DataFrame([summary_row]))

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.reindex(columns=_summary_column_order(combined.columns))
    output_path = run_root / RUN_ROOT_MODEL_SUMMARY_FILE
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    _write_readable_run_root_tables(
        run_root=run_root,
        summary_rows=summary_rows or combined.to_dict(orient="records"),
    )
    logging.info("已保存批处理总模型指标与超参数汇总表：%s", output_path.resolve())
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按指定目录结构批量输出："
            "sample_aggregated_local shap_mean_value/shap_sum/dependence、"
            "全局全部两两交互图、交互矩阵图、"
            "local_shap_values/local_feature_importance_wide、"
            "Spearman/Pearson 相关性热力图。"
        )
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default=None,
        help=(
            "输出总目录；若不填，则自动创建 Output/YYMMDD_n。"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=list(DEFAULT_DATASETS.keys()),
        default=None,
        help="需要处理的数据集键；默认依次处理 2005/2010/2015/2020/full。",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=(
            "包含 5 个输入 CSV 的目录；若提供，则从该目录读取 "
            "`终市级指标数据_with_latlon.csv` 和四个年份文件。"
        ),
    )
    parser.add_argument(
        "--force-rebuild",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否忽略已有批处理结果并强制重建；默认优先复用已有完整输出，避免重复训练模型。",
    )
    return parser.parse_args()


def _date_prefix(today: dt.date | None = None) -> str:
    date_obj = today or dt.date.today()
    return date_obj.strftime("%y%m%d")


def _next_run_root(output_root: Path, date_prefix: str) -> Path:
    pattern = re.compile(rf"^{re.escape(date_prefix)}_(\d+)$")
    max_index = 0
    if output_root.exists():
        for child in output_root.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
    run_root = output_root / f"{date_prefix}_{max_index + 1}"
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def _dataset_keys(selected: Iterable[str] | None) -> list[str]:
    if selected:
        return [str(item) for item in selected]
    return ["2005", "2010", "2015", "2020", "full"]


def _resolve_datasets(data_root: str | None) -> dict[str, tuple[str, Path]]:
    if not data_root:
        return dict(DEFAULT_DATASETS)

    root = Path(data_root).expanduser().resolve()
    return {
        "2005": ("2005", root / "终市级指标数据_with_latlon_2005.csv"),
        "2010": ("2010", root / "终市级指标数据_with_latlon_2010.csv"),
        "2015": ("2015", root / "终市级指标数据_with_latlon_2015.csv"),
        "2020": ("2020", root / "终市级指标数据_with_latlon_2020.csv"),
        "full": ("2002_2020_full", root / "终市级指标数据_with_latlon.csv"),
    }


def _stage_output_dir(dataset_root: Path, name: str) -> Path:
    path = dataset_root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _curated_stage_output_dir(dataset_root: Path, key: str) -> Path:
    return _stage_output_dir(dataset_root, CURATED_STAGE_DIRS.get(key, key))


def _curated_run_root_output_dir(run_root: Path, key: str) -> Path:
    return _stage_output_dir(run_root, CURATED_RUN_ROOT_DIRS.get(key, key))


def _resolve_existing_dataset_root(run_root: Path, dataset_label: str) -> Path:
    exact = run_root / dataset_label
    if exact.exists():
        return exact
    if not run_root.exists():
        return exact

    prefix = f"{dataset_label}("
    candidates = [child for child in run_root.iterdir() if child.is_dir() and child.name.startswith(prefix)]
    if not candidates:
        return exact

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if len(candidates) > 1:
        logging.warning(
            "run_root=%s 下发现多个与数据集 %s 匹配的目录，采用最新修改项：%s",
            run_root,
            dataset_label,
            candidates[0].name,
        )
    return candidates[0]


def _sample_aggregated_dir(dataset_root: Path) -> Path:
    return dataset_root / CURATED_STAGE_DIRS["sample_aggregated_local"]


def _sample_mean_value_path(dataset_root: Path) -> Path:
    return _sample_aggregated_dir(dataset_root) / "sample_aggregated_local_shap_mean_value.csv"


def _sample_rows_path(dataset_root: Path) -> Path:
    return _sample_aggregated_dir(dataset_root) / "sample_aggregated_local_shap_rows.csv"


def _interaction_cache_paths(results_dir: Path) -> dict[str, Path]:
    return {
        "values": results_dir / "interaction_values.npy",
        "features": results_dir / "interaction_test_features.csv",
    }


def _setup_batch_summary_logging(run_root: Path) -> None:
    log_config = {
        "output": {
            "log_file": "batch_summary_log.txt",
            "capture_prints": 0,
        }
    }
    setup_logging(log_config, run_root)


def _rebuild_sample_mean_value_from_rows(dataset_root: Path) -> Path:
    rows_path = _sample_rows_path(dataset_root)
    if not rows_path.exists():
        raise FileNotFoundError(f"缺少 sample 聚合 SHAP 明细：{rows_path}")

    rows_df = pd.read_csv(rows_path, encoding="utf-8-sig")
    shap_cols = [col for col in rows_df.columns if str(col).startswith("shap_")]
    if not shap_cols:
        raise ValueError(f"sample 聚合 SHAP 明细中未找到 shap_ 列：{rows_path}")

    feature_names = [str(col)[len("shap_") :] for col in shap_cols]
    shap_matrix = rows_df.loc[:, shap_cols].to_numpy(dtype=float, copy=False)
    mean_value_df = build_importance_table(
        feature_names,
        np.mean(np.abs(shap_matrix), axis=0),
        prefix="sample_aggregated_local",
    )
    mean_value_path = _sample_mean_value_path(dataset_root)
    mean_value_df.to_csv(mean_value_path, index=False, encoding="utf-8-sig")
    logging.info("已根据 sample_aggregated_local_shap_rows.csv 重建 mean value 表：%s", mean_value_path.resolve())
    return mean_value_path


def _load_sample_importance_share(dataset_root: Path) -> pd.Series:
    mean_value_path = _sample_mean_value_path(dataset_root)
    if not mean_value_path.exists():
        mean_value_path = _rebuild_sample_mean_value_from_rows(dataset_root)

    mean_value_df = pd.read_csv(mean_value_path, encoding="utf-8-sig")
    share_col = "sample_aggregated_local_importance_share"
    if "feature" not in mean_value_df.columns or share_col not in mean_value_df.columns:
        raise KeyError(f"mean value 表缺少必要列：{mean_value_path}")
    return mean_value_df.set_index("feature")[share_col].astype(float)


def _collect_sankey_share_df(
    *,
    run_root: Path,
    datasets: dict[str, tuple[str, Path]],
) -> pd.DataFrame | None:
    share_map: dict[str, pd.Series] = {}
    missing_keys: list[str] = []
    for dataset_key in SANKEY_REQUIRED_DATASET_KEYS:
        folder_name, _ = datasets[dataset_key]
        dataset_root = _resolve_existing_dataset_root(run_root, folder_name)
        if not dataset_root.exists():
            missing_keys.append(dataset_key)
            continue
        try:
            share_map[SANKEY_DATASET_LABELS[dataset_key]] = _load_sample_importance_share(dataset_root)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            logging.warning("跳过跨数据集桑基图：%s 数据集缺少可用的 sample_aggregated_local mean value 输出。原因：%s", dataset_key, exc)
            return None

    if missing_keys:
        logging.info("跳过跨数据集桑基图：run_root 中尚未集齐所需数据集 %s。", missing_keys)
        return None

    share_df = pd.DataFrame(share_map).fillna(0.0)
    share_df["avg_share"] = share_df.mean(axis=1)
    share_df.sort_values("avg_share", ascending=False, inplace=True)
    share_df.drop(columns="avg_share", inplace=True)
    return share_df


def _configure_cjk_plot_style() -> None:
    cjk_font = _pick_installed_font(
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
    latin_font = _pick_installed_font(
        ["Times New Roman", "Cambria", "Georgia", "DejaVu Serif"]
    ) or "DejaVu Serif"
    family = ([cjk_font] if cjk_font else []) + [latin_font, "DejaVu Sans", "DejaVu Serif"]
    plt.rcParams["font.family"] = family
    plt.rcParams["font.sans-serif"] = ([cjk_font] if cjk_font else []) + ["DejaVu Sans"]
    plt.rcParams["font.serif"] = [latin_font, "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False


def _configure_sankey_plot_style() -> None:
    _configure_cjk_plot_style()


def _smoothstep(values: np.ndarray) -> np.ndarray:
    return values * values * (3.0 - 2.0 * values)


def _export_sankey_data_table(
    *,
    share_df: pd.DataFrame,
    output_path: Path,
    display_share_df: pd.DataFrame | None = None,
) -> None:
    export_df = share_df.copy()
    export_df.insert(
        0,
        "feature_short",
        [SANKEY_FEATURE_LABELS.get(str(feature), str(feature)) for feature in export_df.index],
    )
    export_df.insert(0, "feature", export_df.index)
    if display_share_df is not None:
        for column in display_share_df.columns:
            export_df[f"{column}_display_share"] = display_share_df.loc[export_df.index, column]
    export_df.reset_index(drop=True, inplace=True)
    export_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info("跨数据集桑基图数据表已保存：%s", output_path.resolve())


def _save_sankey_chart(
    *,
    share_df: pd.DataFrame,
    output_dir: Path,
    filename_stem: str,
    threshold: float | None = None,
    equalize_visible_height: bool = False,
) -> None:
    _configure_sankey_plot_style()

    visible_features = [
        feature
        for feature in share_df.index
        if threshold is None or bool((share_df.loc[feature] > float(threshold)).any())
    ]
    if not visible_features:
        raise ValueError("没有满足条件的特征用于绘制桑基图。")

    filtered_df = share_df.loc[visible_features].copy()
    datasets = list(filtered_df.columns)
    colors = {
        str(feature): SANKEY_COLORS[idx % len(SANKEY_COLORS)]
        for idx, feature in enumerate(filtered_df.index)
    }

    display_share_df = filtered_df.copy()
    if threshold is not None:
        display_share_df.loc[:, :] = 0.0
        for dataset in datasets:
            mask = filtered_df[dataset] > float(threshold)
            if not bool(mask.any()):
                continue
            visible_total = float(filtered_df.loc[mask, dataset].sum())
            if visible_total <= 0.0:
                continue
            if equalize_visible_height:
                gap = 0.008
                gap_total = gap * max(0, int(mask.sum()) - 1)
                available_height = max(0.0, 1.0 - gap_total)
                display_share_df.loc[mask, dataset] = (
                    filtered_df.loc[mask, dataset] / visible_total * available_height
                )
            else:
                display_share_df.loc[mask, dataset] = filtered_df.loc[mask, dataset]
    else:
        gap = 0.005
        scale = 1.0 - gap * max(0, len(filtered_df.index) - 1)
        display_share_df *= scale

    positions: dict[str, dict[str, dict[str, float]]] = {}
    for dataset in datasets:
        if threshold is None:
            ordered = filtered_df[dataset].sort_values(ascending=False)
        else:
            ordered = filtered_df.loc[filtered_df[dataset] > float(threshold), dataset].sort_values(
                ascending=False
            )
        current_top = 1.0
        pos_map: dict[str, dict[str, float]] = {}
        for rank, (feature, raw_share) in enumerate(ordered.items(), start=1):
            feature_name = str(feature)
            height = float(display_share_df.at[feature, dataset])
            top = current_top
            bottom = top - height
            pos_map[feature_name] = {
                "raw_share": float(raw_share),
                "height": height,
                "top": top,
                "bottom": bottom,
                "rank": float(rank),
            }
            current_top = bottom - gap
        positions[dataset] = pos_map

    fig_height = 8.8 if threshold is not None else 9.5
    column_width = 0.62 if threshold is not None else 0.55
    fig, ax = plt.subplots(figsize=(16, fig_height), dpi=220)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    xs = np.linspace(0.0, 8.4, len(datasets))
    t = np.linspace(0.0, 1.0, 80)
    s = _smoothstep(t)

    for left_idx in range(len(datasets) - 1):
        left_name = datasets[left_idx]
        right_name = datasets[left_idx + 1]
        x0 = xs[left_idx] + column_width / 2
        x1 = xs[left_idx + 1] - column_width / 2
        x_curve = x0 + (x1 - x0) * t
        common_features = [
            str(feature)
            for feature in filtered_df.index
            if str(feature) in positions[left_name] and str(feature) in positions[right_name]
        ]
        for feature_name in common_features:
            left = positions[left_name][feature_name]
            right = positions[right_name][feature_name]
            y_top = left["top"] + (right["top"] - left["top"]) * s
            y_bottom = left["bottom"] + (right["bottom"] - left["bottom"]) * s
            ax.add_patch(
                Polygon(
                    np.column_stack(
                        [
                            np.concatenate([x_curve, x_curve[::-1]]),
                            np.concatenate([y_top, y_bottom[::-1]]),
                        ]
                    ),
                    closed=True,
                    facecolor=colors[feature_name],
                    edgecolor="none",
                    alpha=0.38 if threshold is not None else 0.36,
                    zorder=1,
                )
            )

    for idx, dataset in enumerate(datasets):
        x_left = xs[idx] - column_width / 2
        for feature_name, meta in positions[dataset].items():
            ax.add_patch(
                Rectangle(
                    (x_left, meta["bottom"]),
                    column_width,
                    meta["height"],
                    facecolor=colors[feature_name],
                    edgecolor=(1, 1, 1, 0.9),
                    linewidth=1.0,
                    zorder=3,
                )
            )
            raw_pct = meta["raw_share"] * 100.0
            if threshold is not None or raw_pct >= 2.4:
                ax.text(
                    xs[idx],
                    (meta["top"] + meta["bottom"]) / 2,
                    f"{raw_pct:.2f}%",
                    ha="center",
                    va="center",
                    fontsize=11 if threshold is not None else (12 if raw_pct >= 14 else 10 if raw_pct >= 7 else 8),
                    color="black",
                    zorder=4,
                )

    legend_features = [str(feature) for feature in filtered_df.index]
    legend_cols = 4 if threshold is not None else 5
    legend_rows = int(math.ceil(len(legend_features) / legend_cols))
    legend_xs = np.linspace(0.10, 0.90, legend_cols)
    for row in range(legend_rows):
        y_text = 1.12 - row * (0.075 if threshold is not None else 0.072)
        y_box = 1.086 - row * (0.075 if threshold is not None else 0.072)
        start = row * legend_cols
        stop = min(len(legend_features), (row + 1) * legend_cols)
        row_features = legend_features[start:stop]
        row_xs = legend_xs[: len(row_features)]
        for x_frac, feature_name in zip(row_xs, row_features):
            ax.text(
                x_frac,
                y_text,
                SANKEY_FEATURE_LABELS.get(feature_name, feature_name),
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=11 if threshold is not None else 12,
                color="black",
            )
            ax.add_patch(
                Rectangle(
                    (x_frac - 0.03, y_box),
                    0.06,
                    0.022,
                    transform=ax.transAxes,
                    clip_on=False,
                    facecolor=colors[feature_name],
                    edgecolor=(0, 0, 0, 0.35),
                    linewidth=0.8,
                    zorder=5,
                )
            )

    for idx, dataset in enumerate(datasets):
        ax.text(
            xs[idx],
            -0.045 if threshold is not None else -0.035,
            dataset,
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_xlim(xs[0] - 0.75, xs[-1] + 0.75)
    ax.set_ylim(-0.08 if threshold is not None else -0.06, 1.16)
    ax.axis("off")

    png_path = output_dir / f"{filename_stem}.png"
    svg_path = output_dir / f"{filename_stem}.svg"
    fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logging.info("跨数据集桑基图已保存：%s", png_path.resolve())
    logging.info("跨数据集桑基图已保存：%s", svg_path.resolve())

    data_name = "shap_mean_value_sankey_gt9_data.csv" if threshold is not None else "shap_mean_value_sankey_data.csv"
    _export_sankey_data_table(
        share_df=filtered_df,
        output_path=output_dir / data_name,
        display_share_df=display_share_df if threshold is not None else None,
    )


def _generate_batch_sankey_outputs(
    *,
    run_root: Path,
    datasets: dict[str, tuple[str, Path]],
) -> None:
    _setup_batch_summary_logging(run_root)
    share_df = _collect_sankey_share_df(run_root=run_root, datasets=datasets)
    if share_df is None:
        return

    output_dir = _curated_run_root_output_dir(run_root, "sankey")
    _save_sankey_chart(
        share_df=share_df,
        output_dir=output_dir,
        filename_stem="sample_aggregated_local_shap_mean_value_sankey",
    )
    _save_sankey_chart(
        share_df=share_df,
        output_dir=output_dir,
        filename_stem="sample_aggregated_local_shap_mean_value_sankey_gt9",
        threshold=0.09,
        equalize_visible_height=True,
    )


def _feature_count_from_config(config: Dict[str, Any]) -> int:
    data_cfg = config.get("data") or {}
    features = data_cfg.get("features") or []
    return len(features)


def _pair_count(n_features: int) -> int:
    if n_features < 2:
        return 0
    return math.comb(n_features, 2)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _pick_installed_font(candidates: Iterable[str]) -> str | None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def _latest_matching_dir(base_dir: Path, pattern: str) -> Path | None:
    matches = sorted(base_dir.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def _sorted_curated_run_roots(output_root: Path) -> list[Path]:
    pattern = re.compile(r"^(\d{6})_(\d+)$")
    parsed: list[tuple[str, int, Path]] = []
    if not output_root.exists():
        return []
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if not match:
            continue
        parsed.append((match.group(1), int(match.group(2)), child))
    parsed.sort(key=lambda item: (item[0], item[1]))
    return [path for _, _, path in parsed]


def _has_stage_png(stage_dir: Path) -> bool:
    return stage_dir.exists() and any(stage_dir.glob("*.png"))


def _is_complete_curated_dataset_dir(dataset_dir: Path) -> bool:
    if not dataset_dir.exists():
        return False

    dataset_label = dataset_dir.name
    corr_dir = dataset_dir / CURATED_STAGE_DIRS["correlation_heatmap"]
    required_files = [
        dataset_dir / CURATED_STAGE_DIRS["gwxgb"] / "LW_GXGB.xlsx",
        dataset_dir / CURATED_STAGE_DIRS["gwxgb"] / "gw_shap_interactions.csv",
        dataset_dir / CURATED_STAGE_DIRS["local_shap"] / "local_shap_values.csv",
        dataset_dir / CURATED_STAGE_DIRS["local_shap"] / "local_feature_importance_wide.csv",
        dataset_dir / CURATED_STAGE_DIRS["sample_aggregated_local"] / "sample_aggregated_local_shap_rows.csv",
        dataset_dir / CURATED_STAGE_DIRS["sample_aggregated_local"] / "sample_aggregated_local_shap_mean_value.csv",
        dataset_dir / CURATED_STAGE_DIRS["sample_aggregated_local"] / "sample_aggregated_local_shap_sum.png",
        corr_dir / "correlation_heatmap_manifest.csv",
        corr_dir / f"spearman_correlation_{dataset_label}.csv",
        corr_dir / f"spearman_correlation_{dataset_label}.png",
        corr_dir / f"pearson_correlation_{dataset_label}.csv",
        corr_dir / f"pearson_correlation_{dataset_label}.png",
    ]
    if not all(path.exists() for path in required_files):
        return False

    interaction_dir = dataset_dir / CURATED_STAGE_DIRS["interaction_matrix"]
    return _has_stage_png(interaction_dir)


def _latest_curated_dataset_source(
    *,
    dataset_label: str,
    exclude_run_root: Path | None,
) -> Path | None:
    excluded_root = exclude_run_root.resolve() if exclude_run_root is not None else None
    for run_root in reversed(_sorted_curated_run_roots(OUTPUT_ROOT)):
        if excluded_root is not None and run_root.resolve() == excluded_root:
            continue
        candidate = _resolve_existing_dataset_root(run_root, dataset_label)
        if _is_complete_curated_dataset_dir(candidate):
            return candidate
    return None


def _start_reuse_logging(dataset_root: Path) -> None:
    reuse_log_dir = _curated_stage_output_dir(dataset_root, "reuse_logs")
    log_config = {
        "output": {
            "log_file": "reuse_run_log.txt",
            "capture_prints": 0,
        }
    }
    setup_logging(log_config, reuse_log_dir)


def _configure_gwxgb(
    *,
    data_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    config = load_xgb_config(GWXGB_BASE_CONFIG)
    config = copy.deepcopy(config)
    data_cfg = dict(config.get("data") or {})
    data_cfg["path"] = str(data_path.resolve())
    config["data"] = data_cfg

    output_cfg = dict(config.get("output") or {})
    output_cfg["output_dir"] = str(output_dir.resolve())
    output_cfg["timestamp_subdir"] = 0
    output_cfg["capture_prints"] = 0
    output_cfg["log_file"] = "gwxgb_run_log.txt"
    config["output"] = output_cfg

    shap_cfg = dict(config.get("shap") or {})
    shap_cfg["use_summary"] = 0
    shap_cfg["use_dependence"] = 0
    shap_cfg["compute_interactions"] = 1
    shap_cfg["interaction_base"] = ""
    shap_cfg["interaction_with"] = []
    shap_cfg["interaction_top_n_pairs"] = _pair_count(
        _feature_count_from_config(config)
    )
    config["shap"] = shap_cfg

    local_cfg = dict(config.get("local_shap") or {})
    local_cfg["enabled"] = 1
    local_cfg["save_summary_plots"] = 0
    local_cfg["save_shap_csv"] = 1
    local_cfg["save_feature_importance_table"] = 1
    config["local_shap"] = local_cfg
    return config


def _configure_matrix(
    *,
    data_path: Path,
    dataset_root: Path,
) -> Dict[str, Any]:
    config = load_xgb_config(XGB_BASE_CONFIG)
    config = copy.deepcopy(config)
    data_cfg = dict(config.get("data") or {})
    data_cfg["path"] = str(data_path.resolve())
    config["data"] = data_cfg

    output_cfg = dict(config.get("output") or {})
    output_cfg["output_dir"] = str(dataset_root.resolve())
    output_cfg["timestamp_subdir"] = 0
    output_cfg["capture_prints"] = 0
    output_cfg["log_file"] = "interaction_matrix_run_log.txt"
    config["output"] = output_cfg
    matrix_cfg = dict(config.get("interaction_matrix") or {})
    matrix_cfg["output_subdir"] = CURATED_STAGE_DIRS["interaction_matrix"]
    config["interaction_matrix"] = matrix_cfg
    return config


def _configure_correlation(
    *,
    data_path: Path,
    dataset_root: Path,
) -> Dict[str, Any]:
    config = load_xgb_config(XGB_BASE_CONFIG)
    config = copy.deepcopy(config)
    data_cfg = dict(config.get("data") or {})
    data_cfg["path"] = str(data_path.resolve())
    config["data"] = data_cfg

    output_cfg = dict(config.get("output") or {})
    output_cfg["output_dir"] = str(dataset_root.resolve())
    output_cfg["timestamp_subdir"] = 0
    output_cfg["capture_prints"] = 0
    output_cfg["log_file"] = "correlation_heatmap_run_log.txt"
    config["output"] = output_cfg
    return config


def _save_local_tables(
    *,
    dataset_root: Path,
    local_pooled_df,
    local_importance_df,
) -> None:
    local_dir = _curated_stage_output_dir(dataset_root, "local_shap")
    pooled_path = local_dir / "local_shap_values.csv"
    importance_path = local_dir / "local_feature_importance_wide.csv"
    local_pooled_df.to_csv(pooled_path, index=False, encoding="utf-8-sig")
    local_importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")
    logging.info("已保存 local_shap_values: %s", pooled_path.resolve())
    logging.info(
        "已保存 local_feature_importance_wide: %s",
        importance_path.resolve(),
    )


def _sample_artifacts_from_local_shap(
    *,
    data_path: Path,
    local_pooled_df: pd.DataFrame,
    local_importance_df: pd.DataFrame,
    output_dir: Path,
) -> CompareArtifacts:
    config = load_demo_config(COMPARE_BASE_CONFIG)
    config = copy.deepcopy(config)
    data_cfg = dict(config.get("data") or {})
    data_cfg["path"] = str(data_path.resolve())
    config["data"] = data_cfg
    output_cfg = dict(config.get("output") or {})
    output_cfg["output_dir"] = str(output_dir.resolve())
    output_cfg["timestamp_subdir"] = 0
    output_cfg["capture_prints"] = 0
    config["output"] = output_cfg
    shap_cfg = dict(config.get("shap") or {})
    shap_cfg["use_dependence"] = 1
    shap_cfg["dependence_all"] = 1
    config["shap"] = shap_cfg

    df, X, y, coords = load_gwxgb_dataset(config)
    empty_2d = np.empty((0, X.shape[1]), dtype=float)
    empty_df = pd.DataFrame()
    return CompareArtifacts(
        config=config,
        output_dir=output_dir,
        df=df,
        X=X,
        y=y,
        coords=coords,
        global_shap_values=empty_2d,
        global_interaction_values=None,
        global_importance_df=empty_df,
        local_pooled_df=local_pooled_df,
        local_center_df=empty_df,
        local_importance_df=local_importance_df,
        local_signed_df=empty_df,
        local_positive_df=empty_df,
        local_negative_df=empty_df,
    )


def _generate_sample_aggregated_outputs(
    *,
    dataset_root: Path,
    data_path: Path,
    local_pooled_df: pd.DataFrame,
    local_importance_df: pd.DataFrame,
) -> None:
    _configure_cjk_plot_style()
    sample_dir = _curated_stage_output_dir(dataset_root, "sample_aggregated_local")
    artifacts = _sample_artifacts_from_local_shap(
        data_path=data_path,
        local_pooled_df=local_pooled_df,
        local_importance_df=local_importance_df,
        output_dir=sample_dir,
    )
    sample_df = build_sample_aggregated_local_df(artifacts)
    sample_rows_path = sample_dir / "sample_aggregated_local_shap_rows.csv"
    sample_df.to_csv(sample_rows_path, index=False, encoding="utf-8-sig")
    logging.info("已保存 sample 聚合 SHAP 明细: %s", sample_rows_path.resolve())
    compare_cfg = {"plot_top_n": compare_plot_max_display({}, artifacts.X.shape[1])}
    save_sample_aggregated_local_native_plots(
        artifacts,
        sample_df,
        output_dir=sample_dir,
        compare_cfg=compare_cfg,
    )
    save_sample_aggregated_local_dependence_plots(
        artifacts,
        sample_df,
        output_dir=sample_dir,
    )


def _refresh_non_training_outputs_from_cache(
    *,
    dataset_root: Path,
    data_path: Path,
) -> None:
    local_dir = _curated_stage_output_dir(dataset_root, "local_shap")
    pooled_path = local_dir / "local_shap_values.csv"
    importance_path = local_dir / "local_feature_importance_wide.csv"
    if not pooled_path.exists() or not importance_path.exists():
        raise FileNotFoundError(
            f"缺少缓存 local_shap 表，无法刷新轻量输出：{pooled_path} / {importance_path}"
        )

    local_pooled_df = pd.read_csv(pooled_path, encoding="utf-8-sig")
    local_importance_df = pd.read_csv(importance_path, encoding="utf-8-sig")
    _generate_sample_aggregated_outputs(
        dataset_root=dataset_root,
        data_path=data_path,
        local_pooled_df=local_pooled_df,
        local_importance_df=local_importance_df,
    )
    if not _refresh_interaction_matrix_from_cache(
        dataset_root=dataset_root,
        data_path=data_path,
    ):
        logging.info("缺少 interaction matrix 缓存，补跑该阶段并写入缓存。")
        _run_interaction_matrix_dataset(dataset_root=dataset_root, data_path=data_path)
    _run_correlation_heatmap_dataset(dataset_root=dataset_root, data_path=data_path)


def _copy_existing_gwxgb_outputs(*, source_dir: Path, dataset_root: Path) -> None:
    target_dir = _curated_stage_output_dir(dataset_root, "gwxgb")
    for name in ("gw_shap_interactions.csv", "BW_results.csv", "LW_GXGB.xlsx"):
        src = source_dir / name
        if src.exists():
            _copy_file(src, target_dir / name)
    for src in source_dir.glob("gw_shap_interaction_top_*.png"):
        _copy_file(src, target_dir / src.name)

    src_local = source_dir / "local_shap"
    if not src_local.exists():
        raise FileNotFoundError(f"未找到现有 local_shap 目录：{src_local}")
    local_tables_dir = _curated_stage_output_dir(dataset_root, "local_shap")
    _copy_file(src_local / "local_shap_values.csv", local_tables_dir / "local_shap_values.csv")
    _copy_file(
        src_local / "local_feature_importance_wide.csv",
        local_tables_dir / "local_feature_importance_wide.csv",
    )


def _reuse_existing_dataset(
    *,
    dataset_key: str,
    dataset_root: Path,
    data_path: Path,
    run_root: Path,
) -> bool:
    curated_source = _latest_curated_dataset_source(
        dataset_label=dataset_root.name,
        exclude_run_root=run_root,
    )
    if curated_source is not None:
        _copy_tree(curated_source, dataset_root)
        _start_reuse_logging(dataset_root)
        logging.info("复用既有完整批处理数据集目录: %s", curated_source.resolve())
        logging.info("当前数据集将直接复用既有阶段结果，不重新训练模型。")
        logging.info("将优先基于缓存刷新 sample_aggregated_local、interaction matrix 与相关性热力图，以应用当前绘图修复。")
        _refresh_non_training_outputs_from_cache(
            dataset_root=dataset_root,
            data_path=data_path,
        )
        return True

    if dataset_key == "full":
        return False

    gwxgb_source = _latest_matching_dir(
        OUTPUT_ROOT / "output_gwxgb",
        f"gwxgb_终市级指标数据_with_latlon_{dataset_key}_*",
    )
    if gwxgb_source is None:
        return False

    _start_reuse_logging(dataset_root)
    logging.info("复用既有 gwxgb 输出目录: %s", gwxgb_source.resolve())
    logging.info("当前缺少完整批处理缓存，sample_aggregated_local 将重建，interaction matrix 将继续按当前脚本生成。")

    _copy_existing_gwxgb_outputs(source_dir=gwxgb_source, dataset_root=dataset_root)

    local_pooled_df = pd.read_csv(
        gwxgb_source / "local_shap" / "local_shap_values.csv",
        encoding="utf-8-sig",
    )
    local_importance_df = pd.read_csv(
        gwxgb_source / "local_shap" / "local_feature_importance_wide.csv",
        encoding="utf-8-sig",
    )
    _generate_sample_aggregated_outputs(
        dataset_root=dataset_root,
        data_path=data_path,
        local_pooled_df=local_pooled_df,
        local_importance_df=local_importance_df,
    )
    return True


def _run_gwxgb_dataset(*, dataset_root: Path, data_path: Path) -> None:
    gwxgb_dir = _curated_stage_output_dir(dataset_root, "gwxgb")
    config = _configure_gwxgb(data_path=data_path, output_dir=gwxgb_dir)
    _patch_geoxgboost_parallelism()
    setup_logging(config, gwxgb_dir)
    logging.info("开始 gwxgb 阶段：%s", data_path)
    df, X, y, coords = load_gwxgb_dataset(config)
    logging.info("gwxgb 输入数据已加载：rows=%s, cols=%s。", df.shape[0], df.shape[1])
    optimize_global_model(config, X, y)
    model_global = build_and_train_model(config, X, y)
    _, interaction_values_global = compute_shap_and_interactions(model_global, X, config)
    summarize_and_save_interactions(interaction_values_global, X, config, gwxgb_dir)
    plot_top_interactions(interaction_values_global, X, config, gwxgb_dir)
    plot_fixed_base_interactions(interaction_values_global, X, config, gwxgb_dir)
    bw_opt = optimize_bandwidth(config, X, y, coords, output_dir=gwxgb_dir)
    result_local = run_gxgb(config, X, y, coords, bw=bw_opt, output_dir=gwxgb_dir)
    export_local_models_shap(
        result_local,
        config=config,
        df=df,
        X=X,
        y=y,
        coords=coords,
        bw=bw_opt,
        output_dir=gwxgb_dir,
    )

    local_pooled_df = pd.read_csv(
        gwxgb_dir / "local_shap" / "local_shap_values.csv",
        encoding="utf-8-sig",
    )
    local_importance_df = pd.read_csv(
        gwxgb_dir / "local_shap" / "local_feature_importance_wide.csv",
        encoding="utf-8-sig",
    )
    _save_local_tables(
        dataset_root=dataset_root,
        local_pooled_df=local_pooled_df,
        local_importance_df=local_importance_df,
    )
    _generate_sample_aggregated_outputs(
        dataset_root=dataset_root,
        data_path=data_path,
        local_pooled_df=local_pooled_df,
        local_importance_df=local_importance_df,
    )


def _run_interaction_matrix_dataset(*, dataset_root: Path, data_path: Path) -> None:
    config = _configure_matrix(data_path=data_path, dataset_root=dataset_root)
    setup_logging(config, dataset_root)
    matrix_cfg = _interaction_matrix_cfg(config)
    results_dir = _stage_output_dir(
        dataset_root,
        str(matrix_cfg.get("output_subdir", CURATED_STAGE_DIRS["interaction_matrix"])).strip()
        or CURATED_STAGE_DIRS["interaction_matrix"],
    )

    logging.info("开始 interaction matrix 阶段：%s", data_path)
    df, X, y = load_xgb_dataset(config)
    logging.info(
        "矩阵图输入数据已加载：rows=%s, cols=%s, features=%s",
        df.shape[0],
        df.shape[1],
        list(X.columns),
    )
    if X.shape[1] < 2:
        raise ValueError("至少需要 2 个特征才能绘制交互矩阵图。")

    model, X_train, X_test, y_train, y_test = _fit_best_model(config, X, y, matrix_cfg)
    y_pred = np.asarray(model.predict(X_test), dtype=float)
    rmse, r2 = _regression_metrics(y_test.to_numpy(dtype=float), y_pred)
    logging.info(
        "interaction matrix 模型已完成训练/选参：train_samples=%s, test_samples=%s, test_RMSE=%.4f, test_R2=%.4f",
        len(X_train),
        len(X_test),
        rmse,
        r2,
    )

    shap_config = copy.deepcopy(config)
    shap_cfg = dict(shap_config.get("shap") or {})
    shap_cfg["compute_interactions"] = 1
    shap_config["shap"] = shap_cfg
    _, interaction_values = compute_shap_and_interactions(model, X_test, shap_config)
    if interaction_values is None:
        raise RuntimeError("SHAP interaction values 计算失败。")

    interaction_array = np.asarray(interaction_values, dtype=float)
    _write_interaction_matrix_cache(
        results_dir=results_dir,
        interaction_array=interaction_array,
        X_test=X_test.reset_index(drop=True),
    )
    mean_signed_interaction_matrix = interaction_array.mean(axis=0)
    mean_abs_interaction_matrix = np.abs(mean_signed_interaction_matrix)
    scheme_index, cmap_name = _selected_cmap_name(matrix_cfg)
    style_index, marker_symbol = _selected_marker(matrix_cfg)
    _remove_stale_interaction_pngs(results_dir=results_dir, matrix_cfg=matrix_cfg)
    plot_shap_interaction_matrix(
        interaction_array,
        mean_abs_interaction_matrix,
        mean_signed_interaction_matrix,
        X_test.reset_index(drop=True),
        [str(name) for name in X_test.columns],
        cmap_name=cmap_name,
        marker_symbol=marker_symbol,
        results_dir=results_dir,
        matrix_cfg=matrix_cfg,
        scheme_index=scheme_index,
        style_index=style_index,
    )


def _remove_stale_interaction_pngs(*, results_dir: Path, matrix_cfg: Dict[str, Any]) -> None:
    output_stem = str(matrix_cfg.get("output_stem", "shap_int")).strip() or "shap_int"
    for png_path in results_dir.glob(f"{output_stem}_*.png"):
        png_path.unlink(missing_ok=True)


def _write_interaction_matrix_cache(
    *,
    results_dir: Path,
    interaction_array: np.ndarray,
    X_test: pd.DataFrame,
) -> None:
    cache_paths = _interaction_cache_paths(results_dir)
    np.save(cache_paths["values"], np.asarray(interaction_array, dtype=float))
    X_test.to_csv(cache_paths["features"], index=False, encoding="utf-8-sig")
    logging.info(
        "interaction matrix 缓存已保存：values=%s, features=%s",
        cache_paths["values"].resolve(),
        cache_paths["features"].resolve(),
    )


def _refresh_interaction_matrix_from_cache(
    *,
    dataset_root: Path,
    data_path: Path,
) -> bool:
    config = _configure_matrix(data_path=data_path, dataset_root=dataset_root)
    setup_logging(config, dataset_root)
    matrix_cfg = _interaction_matrix_cfg(config)
    results_dir = _stage_output_dir(
        dataset_root,
        str(matrix_cfg.get("output_subdir", CURATED_STAGE_DIRS["interaction_matrix"])).strip()
        or CURATED_STAGE_DIRS["interaction_matrix"],
    )
    cache_paths = _interaction_cache_paths(results_dir)
    if not cache_paths["values"].exists() or not cache_paths["features"].exists():
        return False

    interaction_array = np.load(cache_paths["values"])
    X_test = pd.read_csv(cache_paths["features"], encoding="utf-8-sig")
    mean_signed_interaction_matrix = interaction_array.mean(axis=0)
    mean_abs_interaction_matrix = np.abs(mean_signed_interaction_matrix)
    scheme_index, cmap_name = _selected_cmap_name(matrix_cfg)
    style_index, marker_symbol = _selected_marker(matrix_cfg)
    _remove_stale_interaction_pngs(results_dir=results_dir, matrix_cfg=matrix_cfg)
    plot_shap_interaction_matrix(
        interaction_array,
        mean_abs_interaction_matrix,
        mean_signed_interaction_matrix,
        X_test.reset_index(drop=True),
        [str(name) for name in X_test.columns],
        cmap_name=cmap_name,
        marker_symbol=marker_symbol,
        results_dir=results_dir,
        matrix_cfg=matrix_cfg,
        scheme_index=scheme_index,
        style_index=style_index,
    )
    logging.info(
        "已基于缓存刷新 interaction matrix：values=%s, features=%s",
        cache_paths["values"].resolve(),
        cache_paths["features"].resolve(),
    )
    return True


def _run_correlation_heatmap_dataset(
    *,
    dataset_root: Path,
    data_path: Path,
) -> None:
    config = _configure_correlation(data_path=data_path, dataset_root=dataset_root)
    setup_logging(config, dataset_root)
    corr_dir = _curated_stage_output_dir(dataset_root, "correlation_heatmap")

    dataset_label = dataset_root.name
    logging.info(
        "开始相关性热力图阶段：%s；methods=%s",
        data_path,
        ", ".join(CURATED_CORRELATION_METHODS),
    )
    manifest_rows: list[dict[str, Any]] = []
    logged_skipped = False
    for method in CURATED_CORRELATION_METHODS:
        corr, skipped, n_rows = compute_feature_corr(
            config,
            data_path,
            include_target=False,
            method=method,
        )
        if skipped and not logged_skipped:
            logging.warning("相关性热力图阶段发现非数值列未参与计算：%s", skipped)
            logged_skipped = True

        corr_csv = corr_dir / f"{method}_correlation_{dataset_label}.csv"
        corr_png = corr_dir / f"{method}_correlation_{dataset_label}.png"
        corr.to_csv(corr_csv, encoding="utf-8-sig")
        plot_corr_heatmap(
            corr,
            title=f"{dataset_label} {method.capitalize()} Correlation Heatmap",
            out_path=corr_png,
            dpi=300,
            annot=True,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            colorbar_label=format_correlation_label(method),
        )
        manifest_rows.append(
            {
                "dataset": dataset_label,
                "method": method,
                "dataset_path": str(data_path.resolve()),
                "n_rows": n_rows,
                "n_variables": int(corr.shape[0]),
                "variables": " | ".join(corr.columns.tolist()),
                "skipped_non_numeric": " | ".join(str(item) for item in skipped),
                "corr_csv": str(corr_csv.resolve()),
                "heatmap_png": str(corr_png.resolve()),
            }
        )
        logging.info(
            "%s 相关性热力图已完成：rows=%s, vars=%s, csv=%s, png=%s",
            method.capitalize(),
            n_rows,
            corr.shape[0],
            corr_csv.resolve(),
            corr_png.resolve(),
        )

    manifest_path = corr_dir / "correlation_heatmap_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    logging.info(
        "相关性热力图阶段完成：dataset=%s, methods=%s, manifest=%s",
        dataset_label,
        ", ".join(CURATED_CORRELATION_METHODS),
        manifest_path.resolve(),
    )


def _run_one_dataset(
    *,
    run_root: Path,
    dataset_key: str,
    datasets: dict[str, tuple[str, Path]],
    force_rebuild: bool,
) -> None:
    folder_name, data_path = datasets[dataset_key]
    dataset_root = _stage_output_dir(run_root, folder_name)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{data_path}")

    print(f"[START] {dataset_key} -> {dataset_root}")
    if not force_rebuild and _is_complete_curated_dataset_dir(dataset_root):
        _start_reuse_logging(dataset_root)
        logging.info("当前 run_root 已存在完整数据集输出，直接跳过重建：%s", dataset_root.resolve())
        _write_dataset_model_summary(
            dataset_key=dataset_key,
            dataset_root=dataset_root,
            data_path=data_path,
        )
        _write_dataset_performance_outputs(
            dataset_key=dataset_key,
            dataset_root=dataset_root,
            data_path=data_path,
        )
        print(f"[DONE] {dataset_key} -> {dataset_root}")
        return

    reused = False
    if not force_rebuild:
        reused = _reuse_existing_dataset(
            dataset_key=dataset_key,
            dataset_root=dataset_root,
            data_path=data_path,
            run_root=run_root,
        )
        if reused and _is_complete_curated_dataset_dir(dataset_root):
            _write_dataset_model_summary(
                dataset_key=dataset_key,
                dataset_root=dataset_root,
                data_path=data_path,
            )
            _write_dataset_performance_outputs(
                dataset_key=dataset_key,
                dataset_root=dataset_root,
                data_path=data_path,
            )
            print(f"[DONE] {dataset_key} -> {dataset_root}")
            return
    if not reused:
        _run_gwxgb_dataset(dataset_root=dataset_root, data_path=data_path)
    _run_interaction_matrix_dataset(dataset_root=dataset_root, data_path=data_path)
    _run_correlation_heatmap_dataset(dataset_root=dataset_root, data_path=data_path)
    _write_dataset_model_summary(
        dataset_key=dataset_key,
        dataset_root=dataset_root,
        data_path=data_path,
    )
    _write_dataset_performance_outputs(
        dataset_key=dataset_key,
        dataset_root=dataset_root,
        data_path=data_path,
    )
    print(f"[DONE] {dataset_key} -> {dataset_root}")


def main() -> None:
    args = parse_args()
    datasets = _resolve_datasets(args.data_root)
    selected_dataset_keys = _dataset_keys(args.datasets)
    if args.run_root:
        run_root = Path(args.run_root).expanduser().resolve()
        run_root.mkdir(parents=True, exist_ok=True)
    else:
        run_root = _next_run_root(OUTPUT_ROOT, _date_prefix())

    print(f"Run root: {run_root}")
    for dataset_key in selected_dataset_keys:
        _run_one_dataset(
            run_root=run_root,
            dataset_key=dataset_key,
            datasets=datasets,
            force_rebuild=bool(args.force_rebuild),
        )
    _generate_batch_sankey_outputs(run_root=run_root, datasets=datasets)
    _write_run_root_model_summary(
        run_root=run_root,
        dataset_keys=selected_dataset_keys,
        datasets=datasets,
    )
    _write_run_root_performance_outputs(
        run_root=run_root,
        dataset_keys=selected_dataset_keys,
        datasets=datasets,
    )


if __name__ == "__main__":
    main()
