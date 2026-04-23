from __future__ import annotations

import argparse
import atexit
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Tuple
import logging
import datetime
import xgboost as xgb
from sklearn.model_selection import GridSearchCV as SklearnGridSearchCV, KFold
import numpy as np
from scipy.spatial import distance_matrix as scipy_distance_matrix

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

import geoxgboost.geoxgboost as gx

_GEOXGBOOST_ORIGINAL_DISTANCE_MATRIX = gx.distance_matrix
_EARTH_RADIUS_KM = 6371.0088

_THIS_DIR = Path(__file__).resolve().parent
_XGB_DIR = _THIS_DIR.parent / "xgb"
if _XGB_DIR.exists() and str(_XGB_DIR) not in sys.path:
    sys.path.insert(0, str(_XGB_DIR))

from xgb_shap import (
    _resolve_path,
    load_config,
    ensure_output_dir,
    ensure_run_output_dir,
    build_and_train_model,
    compute_shap_and_interactions,
    plot_shap_summary,
    plot_shap_dependence,
    plot_top_interactions,
    plot_fixed_base_interactions,
    summarize_and_save_interactions,
)

rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Arial Unicode MS",
]
rcParams["axes.unicode_minus"] = False


def _patch_geoxgboost_parallelism() -> None:
    """将 geoxgboost 内部的 GridSearchCV 强制改为单线程，避免受限环境下的多进程/线程池权限问题。"""

    def _grid_search_cv_sequential(*args: Any, **kwargs: Any) -> Any:
        kwargs["n_jobs"] = 1
        return SklearnGridSearchCV(*args, **kwargs)

    gx.GridSearchCV = _grid_search_cv_sequential  # type: ignore[attr-defined]


def _geoxgboost_path_save(output_dir: Path) -> str:
    """geoxgboost 内部通过字符串拼接 `path_save + filename` 写文件，因此需要尾随分隔符。"""
    path = str(output_dir.resolve())
    if not path.endswith(os.sep):
        path += os.sep
    return path


def _gw_distance_metric(config: Dict[str, Any]) -> str:
    gw_cfg = config.get("gw") or {}
    metric = str(gw_cfg.get("distance_metric", "euclidean")).strip().lower()
    aliases = {
        "geo": "haversine",
        "geodesic": "haversine",
        "great_circle": "haversine",
        "surface": "haversine",
        "surface_km": "haversine",
        "wgs84": "haversine",
    }
    return aliases.get(metric, metric or "euclidean")


def gw_distance_metric_label(config: Dict[str, Any]) -> str:
    metric = _gw_distance_metric(config)
    if metric == "haversine":
        return "haversine_km"
    return "euclidean_coord_units"


def gw_distance_unit(config: Dict[str, Any]) -> str:
    return "km" if _gw_distance_metric(config) == "haversine" else "coord_units"


def _coord_names_from_config(config: Dict[str, Any]) -> list[str]:
    data_cfg = config.get("data") or {}
    return [str(name) for name in (data_cfg.get("coords") or [])]


def _resolve_lat_lon_indices(
    config: Dict[str, Any], coords: Any | None = None
) -> tuple[int, int]:
    gw_cfg = config.get("gw") or {}
    coord_order = str(gw_cfg.get("coord_order", "auto")).strip().lower()
    coord_order = coord_order.replace("-", "_").replace(" ", "_")
    if coord_order in {"lat_lon", "latitude_longitude"}:
        return 0, 1
    if coord_order in {"lon_lat", "longitude_latitude", "lng_lat"}:
        return 1, 0
    if coord_order not in {"", "auto"}:
        raise ValueError(
            "gw.coord_order 仅支持 auto / lat_lon / lon_lat，"
            f"当前为 {coord_order!r}。"
        )

    if isinstance(coords, pd.DataFrame):
        names = [str(name) for name in coords.columns]
    else:
        names = _coord_names_from_config(config)

    lat_idx: int | None = None
    lon_idx: int | None = None
    for idx, name in enumerate(names):
        normalized = name.strip().lower()
        if lat_idx is None and ("纬" in normalized or "lat" in normalized):
            lat_idx = idx
        if lon_idx is None and (
            "经" in normalized or "lon" in normalized or "lng" in normalized
        ):
            lon_idx = idx

    if (lat_idx is None or lon_idx is None) and coords is not None:
        arr = _as_coord_array(coords)
        if arr.shape[1] >= 2:
            col0 = arr[:, 0]
            col1 = arr[:, 1]
            col0_lat_like = bool(np.nanmin(col0) >= -90.0 and np.nanmax(col0) <= 90.0)
            col1_lat_like = bool(np.nanmin(col1) >= -90.0 and np.nanmax(col1) <= 90.0)
            col0_lon_like = bool(np.nanmin(col0) >= -180.0 and np.nanmax(col0) <= 180.0)
            col1_lon_like = bool(np.nanmin(col1) >= -180.0 and np.nanmax(col1) <= 180.0)
            if col0_lat_like and col1_lon_like and not col1_lat_like:
                return 0, 1
            if col1_lat_like and col0_lon_like and not col0_lat_like:
                return 1, 0

    if lat_idx is None or lon_idx is None:
        raise ValueError(
            "gw.distance_metric=haversine 需要识别经纬度列。"
            "请在 gw.coord_order 中显式设置 lat_lon 或 lon_lat。"
        )
    if lat_idx == lon_idx:
        raise ValueError("经纬度列识别结果重复，请检查 data.coords 或 gw.coord_order。")
    return lat_idx, lon_idx


def _as_coord_array(coords: Any) -> np.ndarray:
    if isinstance(coords, pd.DataFrame):
        arr = coords.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(coords, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("坐标数组必须是二维结构，且至少包含两列。")
    return arr


def _haversine_distance_matrix(
    left_coords: Any,
    right_coords: Any,
    *,
    lat_idx: int,
    lon_idx: int,
    radius_km: float = _EARTH_RADIUS_KM,
) -> np.ndarray:
    left = _as_coord_array(left_coords)
    right = _as_coord_array(right_coords)
    left_lat = np.radians(left[:, lat_idx])[:, None]
    left_lon = np.radians(left[:, lon_idx])[:, None]
    right_lat = np.radians(right[:, lat_idx])[None, :]
    right_lon = np.radians(right[:, lon_idx])[None, :]

    d_lat = right_lat - left_lat
    d_lon = right_lon - left_lon
    a = (
        np.sin(d_lat / 2.0) ** 2
        + np.cos(left_lat) * np.cos(right_lat) * np.sin(d_lon / 2.0) ** 2
    )
    angular = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return (float(radius_km) * angular).astype(float, copy=False)


def compute_gw_distance_matrix(
    config: Dict[str, Any], left_coords: Any, right_coords: Any
) -> np.ndarray:
    """按 gw.distance_metric 计算 GW 局部采样距离矩阵。"""
    metric = _gw_distance_metric(config)
    if metric == "haversine":
        lat_idx, lon_idx = _resolve_lat_lon_indices(config, left_coords)
        return _haversine_distance_matrix(
            left_coords,
            right_coords,
            lat_idx=lat_idx,
            lon_idx=lon_idx,
        )
    if metric == "euclidean":
        return np.asarray(scipy_distance_matrix(left_coords, right_coords), dtype=float)
    raise ValueError(
        "gw.distance_metric 仅支持 euclidean / haversine，"
        f"当前为 {metric!r}。"
    )


def _patch_geoxgboost_distance_metric(config: Dict[str, Any], coords: Any) -> None:
    """让 geoxgboost 内部的距离矩阵与本项目 GW 距离口径一致。"""
    metric = _gw_distance_metric(config)
    if metric == "haversine":
        lat_idx, lon_idx = _resolve_lat_lon_indices(config, coords)

        def _distance_matrix_haversine(
            left_coords: Any, right_coords: Any, *args: Any, **kwargs: Any
        ) -> np.ndarray:
            return _haversine_distance_matrix(
                left_coords,
                right_coords,
                lat_idx=lat_idx,
                lon_idx=lon_idx,
            )

        gx.distance_matrix = _distance_matrix_haversine  # type: ignore[attr-defined]
        logging.info(
            "GW 距离口径: haversine 地表距离（单位 km, coord_order=%s）。",
            (config.get("gw") or {}).get("coord_order", "auto"),
        )
        return

    if metric == "euclidean":
        gx.distance_matrix = _GEOXGBOOST_ORIGINAL_DISTANCE_MATRIX  # type: ignore[attr-defined]
        logging.info("GW 距离口径: 经纬度/坐标欧氏距离（原始坐标单位）。")
        return

    raise ValueError(
        "gw.distance_metric 仅支持 euclidean / haversine，"
        f"当前为 {metric!r}。"
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回命名空间。"""
    parser = argparse.ArgumentParser(
        description=(
            "基于 geoxgboost 的地理加权 XGBoost（GXGB），"
            "包含全局 XGBoost 基线训练与带宽优化，"
            "并仅输出全局模型的 SHAP 图。"
        )
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default=None,
        help=(
            "YAML 配置文件路径；"
            "若不指定则默认使用脚本同目录下的 gwxgb_config.yaml"
        ),
    )
    return parser.parse_args()


def load_dataset(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """根据配置加载数据集和坐标列。"""
    data_cfg = config["data"]
    base_dir = Path(str(config.get("_config_dir") or Path.cwd()))
    data_path = _resolve_path(data_cfg["path"], base_dir)
    data_cfg["path"] = str(data_path)
    df = pd.read_csv(
        data_path,
        sep=data_cfg.get("sep", ","),
        encoding=data_cfg.get("encoding", "utf-8"),
    )
    target = data_cfg["target"]
    features = data_cfg.get("features") or [c for c in df.columns if c != target]
    coord_cols: list[str] = data_cfg.get("coords") or []
    if len(coord_cols) != 2:
        raise ValueError("`data.coords` 必须是长度为 2 的列表，例如 ['lon', 'lat']。")

    X = df[features]
    y = df[target]
    coords = df[coord_cols]
    return df, X, y, coords


def _resolve_city_col(config: Dict[str, Any], df: pd.DataFrame) -> str | None:
    data_cfg = config.get("data") or {}
    cfg_col = str(data_cfg.get("city_col", "")).strip()
    if cfg_col:
        if cfg_col in df.columns:
            return cfg_col
        logging.warning(f"data.city_col={cfg_col} 不在数据列中，将忽略。")

    candidates = [
        "城市",
        "city",
        "City",
        "CITY",
        "name",
        "Name",
        "NAME",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_city_name(
    df: pd.DataFrame,
    *,
    city_col: str,
    center_index: Any,
    center_pos: int,
) -> str | None:
    if city_col not in df.columns:
        return None

    value: Any
    try:
        value = df.loc[center_index, city_col]
    except Exception:
        try:
            value = df.iloc[int(center_pos)][city_col]
        except Exception:
            return None

    # 若索引不唯一，loc 可能返回 Series；取第一项即可
    if isinstance(value, pd.Series):
        if len(value) == 0:
            return None
        value = value.iloc[0]

    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(value)


def _spatial_weights_for_local_data(
    distances: np.ndarray,
    *,
    bw: Any,
    kernel: str,
) -> np.ndarray:
    """计算 bi-square 空间权重 w=(1-(d^2/h^2))^2。

    - Adaptive kernel：h = max(distance)（基于局部邻域最远点）
    - Fixed kernel：h = bw（距离阈值）
    """
    d = np.asarray(distances, dtype=float)
    if d.ndim != 1:
        d = d.reshape(-1)
    kernel_lower = str(kernel).strip().lower()
    if kernel_lower == "adaptive":
        h = float(np.max(d)) if d.size > 0 else 0.0
    else:
        h = float(bw)
    if h <= 0.0:
        return np.ones_like(d, dtype=float)
    w = (1.0 - (d**2) / (h**2)) ** 2
    w = np.clip(w, 0.0, 1.0)
    return w.astype(float, copy=False)


def optimize_bandwidth(
    config: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    coords: pd.DataFrame,
    output_dir: Path,
) -> Any:
    """根据配置（或默认范围）优化带宽 bw，并返回优化后的 bw。"""
    _patch_geoxgboost_distance_metric(config, coords)
    gw_cfg = config.get("gw") or {}
    bw0 = gw_cfg["bw"]
    if int(gw_cfg.get("optimize_bw", 1)) == 0:
        logging.info(f"gw.optimize_bw 为 0，将直接使用配置的 bw: {bw0}")
        return bw0

    model_cfg = config.get("model") or {}
    params: Dict[str, Any] = model_cfg.get("params") or {}

    kernel = str(gw_cfg.get("kernel", "Adaptive"))
    spatial_weights = bool(gw_cfg.get("spatial_weights", False))

    n_samples = int(len(X))
    bw_min_cfg = gw_cfg.get("bw_min")
    bw_max_cfg = gw_cfg.get("bw_max")
    bw_step_cfg = gw_cfg.get("bw_step")

    kernel_lower = kernel.lower()
    if kernel_lower == "adaptive":
        bw0_int = int(bw0)
        span = max(5, int(round(bw0_int * 0.3)))
        bw_min = int(bw_min_cfg) if bw_min_cfg is not None else max(2, bw0_int - span)
        bw_max = int(bw_max_cfg) if bw_max_cfg is not None else min(n_samples - 1, bw0_int + span)
        step = int(bw_step_cfg) if bw_step_cfg is not None else 1

        if bw_max <= bw_min:
            logging.warning(
                f"bw 优化范围无效（bw_min={bw_min}, bw_max={bw_max}），将使用 bw={bw0_int}"
            )
            return bw0_int
    else:
        bw0_float = float(bw0)
        bw_min = float(bw_min_cfg) if bw_min_cfg is not None else bw0_float * 0.7
        bw_max = float(bw_max_cfg) if bw_max_cfg is not None else bw0_float * 1.3
        step = float(bw_step_cfg) if bw_step_cfg is not None else (bw_max - bw_min) / 10.0

        if bw_max <= bw_min or step <= 0:
            logging.warning(
                f"bw 优化范围无效（bw_min={bw_min}, bw_max={bw_max}, step={step}），将使用 bw={bw0_float}"
            )
            return bw0_float

    logging.info(
        "开始优化带宽 bw..."
        f" (Kernel={kernel}, spatial_weights={spatial_weights}, "
        f"bw_min={bw_min}, bw_max={bw_max}, step={step})"
    )
    # geoxgboost 内部的 GridSearchCV 会被强制为 n_jobs=1（见 `_patch_geoxgboost_parallelism`），
    # 以避免在受限环境下触发多进程/线程池的权限错误。
    bw_opt = gx.optimize_bw(
        X=X,
        y=y,
        Coords=coords,
        params=params,
        bw_min=bw_min,
        bw_max=bw_max,
        step=step,
        Kernel=kernel,
        spatial_weights=spatial_weights,
        path_save=_geoxgboost_path_save(output_dir),
    )
    logging.info(f"带宽优化完成，bw = {bw_opt}")
    return bw_opt


def optimize_global_model(
    config: Dict[str, Any], X: pd.DataFrame, y: pd.Series
) -> None:
    """使用 GridSearchCV 优化全局 XGBoost 模型参数，并将最优参数通过 config 返回。"""
    grid_cfg = config.get("grid_search") or {}
    if not bool(grid_cfg.get("enabled", False)):
        return

    logging.info("开始全局 XGBoost 模型超参数网格搜索...")
    param_grid = grid_cfg.get("param_grid") or {}
    if not param_grid:
        logging.warning("grid_search.enabled 为 True 但 param_grid 为空，跳过搜索。")
        return

    scoring = grid_cfg.get("scoring", "neg_root_mean_squared_error")
    n_splits = int(grid_cfg.get("cv", 5))
    verbose = int(grid_cfg.get("verbose", 1))

    # 基础模型
    model_cfg = config.get("model") or {}
    base_params = dict(model_cfg.get("params") or {})
    random_state = int(model_cfg.get("random_state", 42))

    # 在部分受限环境（Windows 受控目录/权限）中，sklearn/joblib 的多进程并行会触发
    # PermissionError: [WinError 5] 拒绝访问，因此这里固定使用单进程网格搜索。
    gs_n_jobs = 1

    # 构造 XGBRegressor。为提升可复现性和避免小数据集下多线程开销，默认使用 n_jobs=1。
    estimator_params = dict(base_params)
    estimator_params.setdefault("random_state", random_state)
    estimator_params.setdefault("n_jobs", 1)
    estimator = xgb.XGBRegressor(**estimator_params)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    grid_search = SklearnGridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=gs_n_jobs,
    )

    grid_search.fit(X, y)

    logging.info(f"网格搜索完成。最佳得分 ({scoring}): {grid_search.best_score_:.4f}")
    
    logging.info("最佳参数组合:")
    for k, v in grid_search.best_params_.items():
        logging.info(f"  {k}: {v}")

    # 将最佳参数更新回 config，以便后续构建最终模型使用
    # 注意：这里直接修改了字典引用，config 在 main 中是共享的
    if "params" not in config["model"]:
        config["model"]["params"] = {}
    
    # 更新参数
    config["model"]["params"].update(grid_search.best_params_)
    logging.info("已将最佳参数更新至当前配置。")


def run_gxgb(
    config: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    coords: pd.DataFrame,
    bw: Any,
    output_dir: Path,
) -> Dict[str, Any]:
    """调用 geoxgboost.gxgb 训练地理加权 XGBoost，并返回结果字典。"""
    _patch_geoxgboost_distance_metric(config, coords)
    model_cfg = config.get("model") or {}
    params: Dict[str, Any] = model_cfg.get("params") or {}

    gw_cfg = config.get("gw") or {}
    kernel = gw_cfg.get("kernel", "Adaptive")
    spatial_weights = bool(gw_cfg.get("spatial_weights", False))
    feat_importance = gw_cfg.get("feat_importance", "gain")
    alpha_wt_type = gw_cfg.get("alpha_wt_type", "varying")
    alpha_wt = float(gw_cfg.get("alpha_wt", 1.0))
    test_size = float(gw_cfg.get("test_size", 0.3))
    seed = int(gw_cfg.get("seed", 7))
    n_splits = int(gw_cfg.get("n_splits", 5))

    # geoxgboost 会写出 `BW_results.csv` / `LW_GXGB.xlsx`，此处统一写入 output_dir
    path_save: str = _geoxgboost_path_save(output_dir)

    logging.info("开始训练 Geographical XGBoost (geoxgboost.gxgb)...")
    result: Dict[str, Any] = gx.gxgb(
        X=X,
        y=y,
        Coords=coords,
        params=params,
        bw=bw,
        Kernel=kernel,
        spatial_weights=spatial_weights,
        feat_importance=feat_importance,
        alpha_wt_type=alpha_wt_type,
        alpha_wt=alpha_wt,
        test_size=test_size,
        seed=seed,
        n_splits=n_splits,
        path_save=path_save,
    )
    logging.info("Geographical XGBoost 训练完成。")
    return result


def _local_shap_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    local_cfg = config.get("local_shap") or {}
    return local_cfg if isinstance(local_cfg, dict) else {}


def _local_shap_enabled(config: Dict[str, Any]) -> bool:
    cfg = _local_shap_cfg(config)
    return int(cfg.get("enabled", 0)) == 1


def _iter_local_models(result: Dict[str, Any]) -> Iterable[tuple[int, Any]]:
    models = result.get("bestLocalModel")
    if models is None:
        return iter(())
    if not isinstance(models, (list, tuple)):
        return iter(())
    return ((i, m) for i, m in enumerate(models) if m is not None)


def _select_local_positions(
    dist_col: np.ndarray,
    *,
    bw: Any,
    kernel: str,
) -> np.ndarray:
    order = np.argsort(dist_col, kind="mergesort")
    if str(kernel).strip().lower() == "adaptive":
        k = max(2, int(bw) + 1)  # 至少包含自己+1个邻居
        return order[:k]

    # Fixed kernel：按距离阈值筛选；如果过少则回退到最近的 2 个点
    threshold = float(bw)
    pos = np.flatnonzero(dist_col <= threshold)
    if len(pos) < 2:
        return order[:2]
    return pos[np.argsort(dist_col[pos], kind="mergesort")]


def export_local_models_shap(
    result: Dict[str, Any],
    *,
    config: Dict[str, Any],
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    coords: pd.DataFrame,
    bw: Any,
    output_dir: Path,
) -> None:
    """为每个局部模型输出 SHAP summary 图，并汇总导出所有局部 SHAP 值到一个 CSV。

    说明：该功能默认关闭（local_shap.enabled=0），因为会生成大量图片并耗时较久。
    """
    if not _local_shap_enabled(config):
        return

    local_cfg = _local_shap_cfg(config)
    subdir = str(local_cfg.get("output_subdir", "local_shap")).strip() or "local_shap"
    local_root = (output_dir / subdir).resolve()
    plots_dir = local_root / str(local_cfg.get("plots_dir", "summary_plots")).strip()
    csv_file = str(local_cfg.get("csv_file", "local_shap_values.csv")).strip() or "local_shap_values.csv"
    csv_path = local_root / csv_file
    importance_file = (
        str(local_cfg.get("feature_importance_file", "local_feature_importance_wide.csv")).strip()
        or "local_feature_importance_wide.csv"
    )
    importance_path = local_root / importance_file

    save_plots = int(local_cfg.get("save_summary_plots", 1)) == 1
    save_csv = int(local_cfg.get("save_shap_csv", 1)) == 1
    save_importance = int(local_cfg.get("save_feature_importance_table", 1)) == 1
    max_models = int(local_cfg.get("max_models", 0))  # 0 表示全部
    log_every = int(local_cfg.get("log_every", 10))
    model_variant = str(local_cfg.get("model_variant", "oob")).strip().lower() or "oob"
    if model_variant not in {"oob", "in_sample"}:
        logging.warning(
            f"local_shap.model_variant={model_variant} 无效，将回退为 oob。支持: oob / in_sample"
        )
        model_variant = "oob"

    local_root.mkdir(parents=True, exist_ok=True)
    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "local_shap: enabled=1, save_summary_plots=%s, save_shap_csv=%s, save_feature_importance_table=%s, max_models=%s",
        int(save_plots),
        int(save_csv),
        int(save_importance),
        max_models,
    )
    logging.info(f"local_shap: plots_dir={plots_dir.resolve()}")
    logging.info(f"local_shap: shap_csv={csv_path.resolve()}")
    logging.info(f"local_shap: feature_importance={importance_path.resolve()}")

    n = len(X)
    if n == 0:
        logging.info("local_shap: X 为空，跳过。")
        return

    feature_names = [str(c) for c in X.columns]
    importance_rows: list[Dict[str, Any]] = []

    models = result.get("bestLocalModel")
    if not isinstance(models, (list, tuple)) or len(models) == 0:
        keys = sorted([str(k) for k in result.keys()])
        logging.warning(
            "local_shap: result 中未找到 `bestLocalModel`（或为空），无法计算局部 SHAP。"
            f"可用 keys: {keys}"
        )
        return

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
            "local_shap: 将使用包含中心点的局部模型进行 SHAP（in-sample）。"
            "该解释更偏“拟合描述”，不等价于 oob 泛化解释。"
        )
        logging.info(
            "local_shap: in_sample_use_spatial_weights=%s（中心点距离=0，权重为 1）",
            int(in_sample_use_weights),
        )
    else:
        logging.info("local_shap: model_variant=oob（使用 geoxgboost 输出的 bestLocalModel）")

    # 为局部 SHAP 强制使用 xgboost 原生 SHAP（更稳健，也可避免 shap.TreeExplainer 兼容性问题）
    # 且不计算交互作用（局部模型数量多，交互作用计算非常慢）
    shap_cfg = dict(config.get("shap") or {})
    shap_cfg["engine"] = "xgboost"
    shap_cfg["compute_interactions"] = 0
    local_config = dict(config)
    local_config["shap"] = shap_cfg

    city_col = _resolve_city_col(config, df)
    if city_col is None:
        logging.info(
            "local_shap: 未配置/未检测到城市列（data.city_col），将不在局部模型表格中附加城市名。"
        )
    else:
        logging.info(f"local_shap: 城市列 = {city_col}（将写入 center_city/center_label）")

    first_write = True
    total_exported_rows = 0

    for pos, model in _iter_local_models(result):
        if max_models > 0 and pos >= max_models:
            break
        if pos >= n:
            continue

        try:
            dist_col = np.asarray(dist_mat[:, pos], dtype=float)
            local_positions = _select_local_positions(
                dist_col, bw=bw, kernel=kernel
            )
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
                    w = _spatial_weights_for_local_data(
                        dist_col[local_positions], bw=bw, kernel=kernel
                    )
                    model_for_shap.fit(local_X, local_y_full, sample_weight=w)
                else:
                    model_for_shap.fit(local_X, local_y_full)

            shap_values, _ = compute_shap_and_interactions(
                model_for_shap, local_X, local_config
            )

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

            if save_importance:
                mean_abs = np.mean(np.abs(np.asarray(shap_values, dtype=float)), axis=0)
                rec = {
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
                for i, fname in enumerate(feature_names):
                    rec[fname] = float(mean_abs[i])
                importance_rows.append(rec)

            if save_plots:
                plot_path = plots_dir / f"shap_local_{pos + 1:04d}.png"
                plt.figure()
                import shap  # 延迟导入以减少脚本启动时开销

                shap.summary_plot(shap_values, local_X, show=False)
                plt.tight_layout()
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

            if save_csv:
                df_out = pd.DataFrame(
                    shap_values,
                    columns=[f"shap_{c}" for c in local_X.columns],
                )
                df_out.insert(0, "model_variant", model_variant)
                df_out.insert(0, "center_in_train", int(center_in_train))
                df_out.insert(0, "train_use_spatial_weights", int(train_use_spatial_weights))
                df_out.insert(0, "distance", dist_col[local_positions])
                df_out.insert(0, "distance_unit", gw_distance_unit(config))
                df_out.insert(0, "distance_metric", gw_distance_metric_label(config))
                df_out.insert(0, "sample_pos", local_positions)
                df_out.insert(0, "sample_index", local_X.index.to_numpy())
                df_out.insert(0, "y", y.iloc[local_positions].to_numpy())
                df_out.insert(0, "center_pos", pos)
                df_out.insert(0, "center_index", center_index)
                df_out.insert(0, "center_city", center_city)
                df_out.insert(0, "center_label", center_label)
                df_out.insert(0, "kernel", kernel)
                df_out.insert(0, "bw", bw)

                mode = "w" if first_write else "a"
                df_out.to_csv(
                    csv_path,
                    mode=mode,
                    index=False,
                    header=first_write,
                    encoding="utf-8-sig",
                )
                first_write = False
                total_exported_rows += int(df_out.shape[0])

        except Exception as exc:
            logging.warning(f"local_shap: 位置 pos={pos} 计算失败，已跳过。原因: {exc}")
            continue

        if log_every > 0 and (pos + 1) % log_every == 0:
            logging.info(
                f"local_shap: 已完成 {pos + 1}/{n} 个局部模型，累计导出 {total_exported_rows} 行。"
            )

    if save_plots:
        logging.info(f"local_shap: summary 图输出目录: {plots_dir.resolve()}")
    if save_csv:
        logging.info(f"local_shap: SHAP 值 CSV: {csv_path.resolve()}（共 {total_exported_rows} 行）")
    if save_importance and importance_rows:
        meta_cols = [
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
        df_imp = pd.DataFrame(importance_rows)
        ordered_cols = [c for c in meta_cols if c in df_imp.columns] + [
            c for c in feature_names if c in df_imp.columns
        ]
        df_imp = df_imp.reindex(columns=ordered_cols)
        df_imp.to_csv(importance_path, index=False, encoding="utf-8-sig")
        logging.info(f"local_shap: 特征重要性汇总表: {importance_path.resolve()}")



class _TeeStream:
    def __init__(self, primary: Any, secondary: Any) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        self._primary.write(data)
        try:
            self._secondary.write(data)
        except Exception:
            pass
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        try:
            self._secondary.flush()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)


def setup_logging(config: Dict[str, Any]) -> None:
    """配置日志记录，将日志同时输出到控制台和文件。"""
    output_path = ensure_output_dir(config)
    output_cfg = config.get("output") or {}
    
    log_file = output_cfg.get("log_file", "run_log.txt")
    log_path = output_path / log_file

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.info("=" * 50)
    logging.info(f"本次运行开始: {datetime.datetime.now()}")
    logging.info(f"命令行: {' '.join(sys.argv)}")
    logging.info(f"输出目录: {output_path.resolve()}")
    config_path = config.get("_config_path")
    if config_path:
        logging.info(f"配置文件: {config_path}")
    logging.info(f"配置文件内容: {config}")

    capture_prints = int(output_cfg.get("capture_prints", 1)) == 1
    if capture_prints:
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        log_stream = log_path.open("a", encoding="utf-8")
        atexit.register(log_stream.close)
        sys.stdout = _TeeStream(orig_stdout, log_stream)
        sys.stderr = _TeeStream(orig_stderr, log_stream)
        logging.info("已启用 stdout/stderr 捕获到日志文件。")


def main() -> None:
    """脚本主入口：GeoXGBoost + SHAP 分析。"""
    run_start = datetime.datetime.now()
    args = parse_args()
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).with_name("gwxgb_config.yaml")

    config = load_config(config_path)

    _patch_geoxgboost_parallelism()

    output_dir = ensure_run_output_dir(config, prefix="gwxgb_")
    
    # 初始化日志
    setup_logging(config)
    logging.info(f"输出目录: {output_dir.resolve()}")
    df, X, y, coords = load_dataset(config)
    
    logging.info(
        f"已加载数据 `{config['data']['path']}`，"
        f"共 {df.shape[0]} 行，{df.shape[1]} 列（含坐标列）。"
    )

    # 在训练最终模型前，尝试进行网格搜索优化参数
    optimize_global_model(config, X, y)

    logging.info("开始训练全局 XGBoost 基线模型...")
    model_global = build_and_train_model(config, X, y)
    logging.info("全局 XGBoost 基线模型训练完成。")
    
    shap_values_global, interaction_values_global = compute_shap_and_interactions(
        model_global, X, config
    )
    logging.info("全局 XGBoost 基线模型训练完成，并已计算 SHAP 值。")

    plot_shap_summary(shap_values_global, X, config, output_dir)
    plot_shap_dependence(shap_values_global, X, config, output_dir)
    summarize_and_save_interactions(interaction_values_global, X, config, output_dir)
    plot_top_interactions(interaction_values_global, X, config, output_dir)
    plot_fixed_base_interactions(interaction_values_global, X, config, output_dir)

    bw_opt = optimize_bandwidth(config, X, y, coords, output_dir=output_dir)
    logging.info(f"最优带宽 bw = {bw_opt}")
    
    result_local = run_gxgb(config, X, y, coords, bw=bw_opt, output_dir=output_dir)
    export_local_models_shap(
        result_local,
        config=config,
        df=df,
        X=X,
        y=y,
        coords=coords,
        bw=bw_opt,
        output_dir=output_dir,
    )

    logging.info(f"所有全局 XGBoost SHAP 输出文件已保存到目录: {output_dir.resolve()}")
    run_end = datetime.datetime.now()
    logging.info(
        f"本次运行结束: {run_end}（耗时 {(run_end - run_start).total_seconds():.2f} 秒）"
    )


if __name__ == "__main__":
    main()
