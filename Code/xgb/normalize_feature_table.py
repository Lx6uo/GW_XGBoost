from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from xgb_shap import ensure_run_output_dir, load_config, load_dataset, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "对当前数据表中的自变量做按列 min-max 归一化；"
            "默认只处理 data.features 中的列，并保留整张原表输出。"
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


def _resolve_normalize_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = config.get("preprocess_normalize")
    if isinstance(cfg, dict):
        return cfg
    cfg = {}
    config["preprocess_normalize"] = cfg
    return cfg


def _feature_selection(
    X: pd.DataFrame,
    normalize_cfg: Dict[str, Any],
) -> tuple[List[str], List[str]]:
    include_cfg = normalize_cfg.get("include_features") or []
    if include_cfg:
        include_features = [str(value).strip() for value in include_cfg if str(value).strip()]
    else:
        include_features = [str(column) for column in X.columns]

    exclude_cfg = normalize_cfg.get("exclude_features") or []
    exclude_features = [str(value).strip() for value in exclude_cfg if str(value).strip()]
    return include_features, exclude_features


def _normalize_selected_features(
    df: pd.DataFrame,
    *,
    include_features: Sequence[str],
    exclude_features: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_df = df.copy()
    exclude_set = set(exclude_features)
    manifest_rows: List[Dict[str, Any]] = []

    for feature in include_features:
        manifest: Dict[str, Any] = {
            "feature": str(feature),
            "method": "minmax",
            "status": "",
            "reason": "",
            "min_value": np.nan,
            "max_value": np.nan,
            "range_value": np.nan,
            "n_missing": 0,
        }

        if feature in exclude_set:
            manifest["status"] = "excluded"
            manifest["reason"] = "configured_exclude_feature"
            manifest_rows.append(manifest)
            continue

        if feature not in output_df.columns:
            manifest["status"] = "missing"
            manifest["reason"] = "feature_not_found_in_table"
            manifest_rows.append(manifest)
            continue

        numeric_series = pd.to_numeric(output_df[feature], errors="coerce")
        n_missing = int(numeric_series.isna().sum())
        finite_values = numeric_series[np.isfinite(numeric_series.to_numpy(dtype=float))]
        manifest["n_missing"] = n_missing

        if len(finite_values) == 0:
            manifest["status"] = "skipped"
            manifest["reason"] = "no_numeric_values"
            manifest_rows.append(manifest)
            continue

        min_value = float(finite_values.min())
        max_value = float(finite_values.max())
        range_value = max_value - min_value
        manifest["min_value"] = min_value
        manifest["max_value"] = max_value
        manifest["range_value"] = range_value

        if range_value <= 0.0:
            output_df[feature] = numeric_series - min_value
            manifest["status"] = "constant_zero"
            manifest["reason"] = "feature_has_constant_value"
            manifest_rows.append(manifest)
            continue

        output_df[feature] = (numeric_series - min_value) / range_value
        manifest["status"] = "normalized"
        manifest["reason"] = "applied_minmax"
        manifest_rows.append(manifest)

    manifest_df = pd.DataFrame(manifest_rows)
    return output_df, manifest_df


def main() -> None:
    run_start = datetime.datetime.now()
    args = parse_args()
    config_path = (
        Path(args.config) if args.config else Path(__file__).with_name("config.yaml")
    )
    config = load_config(config_path)

    output_dir = ensure_run_output_dir(config, prefix="xgb_preprocess_")
    setup_logging(config, output_dir)

    normalize_cfg = _resolve_normalize_cfg(config)
    results_subdir = (
        str(normalize_cfg.get("output_subdir", "preprocess_normalize")).strip()
        or "preprocess_normalize"
    )
    results_dir = output_dir / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = str(normalize_cfg.get("output_file", "normalized_feature_table.csv")).strip()
    if not output_file:
        output_file = "normalized_feature_table.csv"
    manifest_file = str(normalize_cfg.get("manifest_file", "normalized_feature_manifest.csv")).strip()
    if not manifest_file:
        manifest_file = "normalized_feature_manifest.csv"

    logging.info(f"使用配置文件: {config_path}")
    logging.info(f"归一化预处理输出目录: {results_dir.resolve()}")

    df, X, y = load_dataset(config)
    logging.info(
        "已加载数据 `%s`，共 %s 行，%s 列；目标列 `%s` 不参与归一化。",
        config["data"]["path"],
        df.shape[0],
        df.shape[1],
        config["data"]["target"],
    )

    include_features, exclude_features = _feature_selection(X, normalize_cfg)
    logging.info("候选归一化特征: %s", include_features)
    logging.info("排除归一化特征: %s", exclude_features)

    normalized_df, manifest_df = _normalize_selected_features(
        df,
        include_features=include_features,
        exclude_features=exclude_features,
    )

    output_path = results_dir / output_file
    manifest_path = results_dir / manifest_file
    normalized_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    normalized_count = int((manifest_df["status"] == "normalized").sum()) if not manifest_df.empty else 0
    excluded_count = int((manifest_df["status"] == "excluded").sum()) if not manifest_df.empty else 0
    constant_count = int((manifest_df["status"] == "constant_zero").sum()) if not manifest_df.empty else 0

    logging.info("已保存归一化后的表格: %s", output_path.resolve())
    logging.info("已保存归一化清单: %s", manifest_path.resolve())
    logging.info(
        "归一化完成：normalized=%s, excluded=%s, constant_zero=%s, total_features=%s",
        normalized_count,
        excluded_count,
        constant_count,
        len(include_features),
    )

    run_end = datetime.datetime.now()
    logging.info(
        "数据预处理完成: %s（耗时 %.2f 秒）",
        run_end,
        (run_end - run_start).total_seconds(),
    )


if __name__ == "__main__":
    main()
