from __future__ import annotations

import argparse
import copy
import datetime as dt
import logging
import math
import re
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

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
    build_sample_aggregated_local_df,
    compare_plot_max_display,
    load_demo_config,
    save_sample_aggregated_local_dependence_plots,
    save_sample_aggregated_local_native_plots,
)
from gwxgb_shap import (
    _patch_geoxgboost_parallelism,
    export_local_models_shap,
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
    "local_shap": "local_shap_tables",
    "sample_aggregated_local": "sample_aggregated_local_shap",
    "reuse_logs": "reused_source_logs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按指定目录结构批量输出："
            "sample_aggregated_local shap_mean_value/shap_sum/dependence、"
            "全局全部两两交互图、交互矩阵图、"
            "local_shap_values/local_feature_importance_wide。"
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


def _latest_matching_dir(base_dir: Path, pattern: str) -> Path | None:
    matches = sorted(base_dir.glob(pattern))
    if not matches:
        return None
    return matches[-1]


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


def _reuse_existing_dataset(*, dataset_key: str, dataset_root: Path, data_path: Path) -> bool:
    if dataset_key == "full":
        return False

    gwxgb_source = _latest_matching_dir(
        OUTPUT_ROOT / "output_gwxgb",
        f"gwxgb_终市级指标数据_with_latlon_{dataset_key}_*",
    )
    if gwxgb_source is None:
        return False

    reuse_log_dir = _curated_stage_output_dir(dataset_root, "reuse_logs")
    log_config = {
        "output": {
            "log_file": "reuse_run_log.txt",
            "capture_prints": 0,
        }
    }
    setup_logging(log_config, reuse_log_dir)
    logging.info("复用既有 gwxgb 输出目录: %s", gwxgb_source.resolve())
    logging.info("交互矩阵图将按当前脚本重新生成，不复用旧矩阵图文件。")

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
    mean_signed_interaction_matrix = interaction_array.mean(axis=0)
    mean_abs_interaction_matrix = np.abs(mean_signed_interaction_matrix)
    scheme_index, cmap_name = _selected_cmap_name(matrix_cfg)
    style_index, marker_symbol = _selected_marker(matrix_cfg)
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


def _run_one_dataset(
    *,
    run_root: Path,
    dataset_key: str,
    datasets: dict[str, tuple[str, Path]],
) -> None:
    folder_name, data_path = datasets[dataset_key]
    dataset_root = _stage_output_dir(run_root, folder_name)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{data_path}")

    print(f"[START] {dataset_key} -> {dataset_root}")
    reused = _reuse_existing_dataset(
        dataset_key=dataset_key,
        dataset_root=dataset_root,
        data_path=data_path,
    )
    if not reused:
        _run_gwxgb_dataset(dataset_root=dataset_root, data_path=data_path)
    _run_interaction_matrix_dataset(dataset_root=dataset_root, data_path=data_path)
    print(f"[DONE] {dataset_key} -> {dataset_root}")


def main() -> None:
    args = parse_args()
    datasets = _resolve_datasets(args.data_root)
    if args.run_root:
        run_root = Path(args.run_root).expanduser().resolve()
        run_root.mkdir(parents=True, exist_ok=True)
    else:
        run_root = _next_run_root(OUTPUT_ROOT, _date_prefix())

    print(f"Run root: {run_root}")
    for dataset_key in _dataset_keys(args.datasets):
        _run_one_dataset(
            run_root=run_root,
            dataset_key=dataset_key,
            datasets=datasets,
        )


if __name__ == "__main__":
    main()
