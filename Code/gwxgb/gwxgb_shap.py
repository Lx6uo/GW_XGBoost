from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Dict, Tuple
import logging
import datetime
import xgboost as xgb
from sklearn.model_selection import GridSearchCV as SklearnGridSearchCV, KFold
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

import geoxgboost.geoxgboost as gx

_THIS_DIR = Path(__file__).resolve().parent
_XGB_DIR = _THIS_DIR.parent / "xgb"
if _XGB_DIR.exists() and str(_XGB_DIR) not in sys.path:
    sys.path.insert(0, str(_XGB_DIR))

from xgb_shap import (
    load_config,
    ensure_output_dir,
    build_and_train_model,
    compute_shap_and_interactions,
    plot_shap_summary,
    plot_shap_dependence,
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
    data_path = Path(str(data_cfg["path"])).expanduser()
    if not data_path.is_absolute():
        data_path = (base_dir / data_path).resolve()
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


def optimize_bandwidth(
    config: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    coords: pd.DataFrame,
    output_dir: Path,
) -> Any:
    """根据配置（或默认范围）优化带宽 bw，并返回优化后的 bw。"""
    gw_cfg = config.get("gw") or {}
    bw0 = gw_cfg["bw"]
    if int(gw_cfg.get("optimize_bw", 1)) == 0:
        print(f"gw.optimize_bw 为 0，将直接使用配置的 bw: {bw0}")
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
            print(f"bw 优化范围无效（bw_min={bw_min}, bw_max={bw_max}），将使用 bw={bw0_int}")
            return bw0_int
    else:
        bw0_float = float(bw0)
        bw_min = float(bw_min_cfg) if bw_min_cfg is not None else bw0_float * 0.7
        bw_max = float(bw_max_cfg) if bw_max_cfg is not None else bw0_float * 1.3
        step = float(bw_step_cfg) if bw_step_cfg is not None else (bw_max - bw_min) / 10.0

        if bw_max <= bw_min or step <= 0:
            print(f"bw 优化范围无效（bw_min={bw_min}, bw_max={bw_max}, step={step}），将使用 bw={bw0_float}")
            return bw0_float

    print(
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
    print(f"带宽优化完成，bw = {bw_opt}")
    return bw_opt


def optimize_global_model(
    config: Dict[str, Any], X: pd.DataFrame, y: pd.Series
) -> None:
    """使用 GridSearchCV 优化全局 XGBoost 模型参数，并将最优参数通过 config 返回。"""
    grid_cfg = config.get("grid_search") or {}
    if not bool(grid_cfg.get("enabled", False)):
        return

    print("开始全局 XGBoost 模型超参数网格搜索...")
    param_grid = grid_cfg.get("param_grid") or {}
    if not param_grid:
        print("警告: grid_search.enabled 为 True 但 param_grid 为空，跳过搜索。")
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

    print(f"网格搜索完成。最佳得分 ({scoring}): {grid_search.best_score_:.4f}")
    logging.info(f"网格搜索完成。最佳得分 ({scoring}): {grid_search.best_score_:.4f}")
    
    print("最佳参数组合:")
    logging.info("最佳参数组合:")
    for k, v in grid_search.best_params_.items():
        print(f"  {k}: {v}")
        logging.info(f"  {k}: {v}")

    # 将最佳参数更新回 config，以便后续构建最终模型使用
    # 注意：这里直接修改了字典引用，config 在 main 中是共享的
    if "params" not in config["model"]:
        config["model"]["params"] = {}
    
    # 更新参数
    config["model"]["params"].update(grid_search.best_params_)
    print("已将最佳参数更新至当前配置。")


def run_gxgb(
    config: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    coords: pd.DataFrame,
    bw: Any,
    output_dir: Path,
) -> Dict[str, Any]:
    """调用 geoxgboost.gxgb 训练地理加权 XGBoost，并返回结果字典。"""
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

    print("开始训练 Geographical XGBoost (geoxgboost.gxgb)...")
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
    print("Geographical XGBoost 训练完成。")
    return result



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
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("=" * 50)
    logging.info(f"本次运行开始: {datetime.datetime.now()}")
    logging.info(f"配置文件内容: {config}")


def main() -> None:
    """脚本主入口：GeoXGBoost + SHAP 分析。"""
    args = parse_args()
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).with_name("gwxgb_config.yaml")
        print(f"未指定配置文件路径，将使用默认配置文件: {config_path}")

    config = load_config(config_path)

    _patch_geoxgboost_parallelism()
    
    # 初始化日志
    setup_logging(config)

    print(f"使用配置文件: {config_path}")
    df, X, y, coords = load_dataset(config)
    
    logging.info(
        f"已加载数据 `{config['data']['path']}`，"
        f"共 {df.shape[0]} 行，{df.shape[1]} 列（含坐标列）。"
    )
    print(
        f"已加载数据 `{config['data']['path']}`，"
        f"共 {df.shape[0]} 行，{df.shape[1]} 列（含坐标列）。"
    )

    output_dir = ensure_output_dir(config)

    # 在训练最终模型前，尝试进行网格搜索优化参数
    optimize_global_model(config, X, y)

    logging.info("开始训练全局 XGBoost 基线模型...")
    print("开始训练全局 XGBoost 基线模型...")
    model_global = build_and_train_model(config, X, y)
    logging.info("全局 XGBoost 基线模型训练完成。")
    
    shap_values_global, interaction_values_global = compute_shap_and_interactions(
        model_global, X, config
    )
    print("全局 XGBoost 基线模型训练完成，并已计算 SHAP 值。")

    plot_shap_summary(shap_values_global, X, config, output_dir)
    plot_shap_dependence(shap_values_global, X, config, output_dir)
    plot_fixed_base_interactions(interaction_values_global, X, config, output_dir)
    summarize_and_save_interactions(interaction_values_global, X, config, output_dir)

    bw_opt = optimize_bandwidth(config, X, y, coords, output_dir=output_dir)
    logging.info(f"最优带宽 bw = {bw_opt}")
    
    _ = run_gxgb(config, X, y, coords, bw=bw_opt, output_dir=output_dir)

    logging.info(f"所有全局 XGBoost SHAP 输出文件已保存到目录: {output_dir.resolve()}")
    print(f"所有全局 XGBoost SHAP 输出文件已保存到目录: {output_dir.resolve()}")
    logging.info("本次运行结束。")


if __name__ == "__main__":
    main()
