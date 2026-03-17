from __future__ import annotations

import argparse
import copy
import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from xgb_shap import (
    compute_shap_and_interactions,
    ensure_run_output_dir,
    load_config,
    load_dataset,
    setup_logging,
)


rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Arial Unicode MS",
]
rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class SpecDefinition:
    name: str
    encode_categorical: bool = False
    log_skewed_x: bool = False
    zscore_x: bool = False
    minmax_x: bool = False
    log_y: bool = False
    diagnostic_only: bool = False


@dataclass
class FoldPreparedData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val_raw: pd.Series
    feature_group_map: Dict[str, str]
    manifest_rows: List[Dict[str, Any]]
    inverse_target_fn: Callable[[np.ndarray], np.ndarray]
    target_scale: str
    uses_log_y: bool


DEFAULT_SPECS: Dict[str, SpecDefinition] = {
    "raw_x__raw_y": SpecDefinition(name="raw_x__raw_y"),
    "encoded_cat__raw_y": SpecDefinition(
        name="encoded_cat__raw_y", encode_categorical=True
    ),
    "encoded_cat__log_skewed_x__raw_y": SpecDefinition(
        name="encoded_cat__log_skewed_x__raw_y",
        encode_categorical=True,
        log_skewed_x=True,
    ),
    "encoded_cat__log_y": SpecDefinition(
        name="encoded_cat__log_y",
        encode_categorical=True,
        log_y=True,
    ),
    "encoded_cat__log_skewed_x__log_y": SpecDefinition(
        name="encoded_cat__log_skewed_x__log_y",
        encode_categorical=True,
        log_skewed_x=True,
        log_y=True,
    ),
    "encoded_cat__zscore_x__raw_y": SpecDefinition(
        name="encoded_cat__zscore_x__raw_y",
        encode_categorical=True,
        zscore_x=True,
        diagnostic_only=True,
    ),
    "encoded_cat__minmax_x__raw_y": SpecDefinition(
        name="encoded_cat__minmax_x__raw_y",
        encode_categorical=True,
        minmax_x=True,
        diagnostic_only=True,
    ),
}


SPEC_DISPLAY_NAMES: Dict[str, str] = {
    "raw_x__raw_y": "原始X_原始Y_不编码类别",
    "encoded_cat__raw_y": "类别编码_原始X_原始Y",
    "encoded_cat__log_skewed_x__raw_y": "类别编码_偏态X_log1p_原始Y",
    "encoded_cat__log_y": "类别编码_原始X_Y_log1p",
    "encoded_cat__log_skewed_x__log_y": "类别编码_偏态X_log1p_Y_log1p",
    "encoded_cat__zscore_x__raw_y": "类别编码_X_zscore_原始Y",
    "encoded_cat__minmax_x__raw_y": "类别编码_X_minmax_原始Y",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "归因优先的 XGBoost + SHAP 稳健性分析："
            "比较不同预处理口径下的验证集 SHAP 排名稳定性，并输出正式推荐口径。"
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


def _resolve_attribution_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    attr_cfg = config.get("attribution")
    if isinstance(attr_cfg, dict):
        return attr_cfg
    attr_cfg = {}
    config["attribution"] = attr_cfg
    return attr_cfg


def _candidate_specs(attr_cfg: Dict[str, Any]) -> List[SpecDefinition]:
    raw = attr_cfg.get("candidate_specs")
    if not isinstance(raw, list) or not raw:
        names = list(DEFAULT_SPECS.keys())
    else:
        names = [str(x).strip() for x in raw if str(x).strip()]

    specs: List[SpecDefinition] = []
    for name in names:
        spec = DEFAULT_SPECS.get(name)
        if spec is None:
            raise ValueError(f"未知 attribution.candidate_specs: {name}")
        specs.append(spec)
    if not specs:
        raise ValueError("attribution.candidate_specs 为空，至少需要一个方案。")
    return specs


def _model_params(config: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = config.get("model") or {}
    params: Dict[str, Any] = dict(model_cfg.get("params") or {})
    if "random_state" not in params and model_cfg.get("random_state") is not None:
        params["random_state"] = int(model_cfg["random_state"])
    if "n_jobs" not in params and model_cfg.get("n_jobs") is not None:
        params["n_jobs"] = int(model_cfg["n_jobs"])
    return params


def _resolve_feature_roles(
    X: pd.DataFrame,
    attr_cfg: Dict[str, Any],
) -> tuple[List[str], List[str]]:
    columns = [str(c) for c in X.columns]
    categorical_cfg = attr_cfg.get("categorical_features") or []
    categorical_features = [str(c) for c in categorical_cfg if str(c) in columns]

    continuous_cfg = attr_cfg.get("continuous_features") or []
    if continuous_cfg:
        continuous_features = [str(c) for c in continuous_cfg if str(c) in columns]
    else:
        categorical_set = set(categorical_features)
        continuous_features = [c for c in columns if c not in categorical_set]

    return categorical_features, continuous_features


def _augment_feature_frame_from_attribution(
    df: pd.DataFrame,
    X: pd.DataFrame,
    config: Dict[str, Any],
    attr_cfg: Dict[str, Any],
) -> tuple[pd.DataFrame, List[str], List[str]]:
    data_cfg = config.get("data") or {}
    target = str(data_cfg.get("target", "")).strip()
    requested_names: List[str] = []
    for key in ("categorical_features", "continuous_features"):
        raw = attr_cfg.get(key) or []
        for value in raw:
            name = str(value).strip()
            if not name or name == target or name in requested_names:
                continue
            requested_names.append(name)

    if not requested_names:
        return X.copy(), [], []

    X_out = X.copy()
    added_columns: List[str] = []
    missing_columns: List[str] = []

    for name in requested_names:
        if name in X_out.columns:
            continue
        if name not in df.columns:
            missing_columns.append(name)
            continue
        X_out[name] = df[name].reset_index(drop=True)
        added_columns.append(name)

    ordered_columns = list(X.columns) + [name for name in added_columns if name not in X.columns]
    X_out = X_out[ordered_columns]
    return X_out, added_columns, missing_columns


def _resolve_repeat_seeds(attr_cfg: Dict[str, Any], *, fallback_seed: int) -> List[int]:
    raw = attr_cfg.get("cv_seeds")
    if isinstance(raw, list) and raw:
        seeds = [int(x) for x in raw]
    else:
        repeats = int(attr_cfg.get("cv_repeats", 5))
        seeds = [fallback_seed + i for i in range(max(1, repeats))]

    repeats = int(attr_cfg.get("cv_repeats", len(seeds)))
    if repeats <= 0:
        repeats = len(seeds)

    if len(seeds) < repeats:
        last_seed = seeds[-1] if seeds else fallback_seed
        seeds.extend(last_seed + i + 1 for i in range(repeats - len(seeds)))

    return seeds[:repeats]


def _determine_log_skewed_features(
    X: pd.DataFrame,
    continuous_features: Sequence[str],
    *,
    ratio_threshold: float,
) -> List[str]:
    selected: List[str] = []
    for feature in continuous_features:
        series = pd.to_numeric(X[feature], errors="coerce")
        valid = series[np.isfinite(series.to_numpy(dtype=float))]
        if len(valid) == 0:
            continue
        min_value = float(valid.min())
        p50 = float(valid.quantile(0.50))
        p95 = float(valid.quantile(0.95))
        if min_value <= 0.0 or p50 <= 0.0:
            continue
        if p95 / p50 >= ratio_threshold:
            selected.append(feature)
    return selected


def _kfold_indices(n_samples: int, n_splits: int, seed: int) -> List[np.ndarray]:
    if n_splits < 2 or n_splits > n_samples:
        raise ValueError(f"无效的 n_splits={n_splits}，样本数为 {n_samples}。")
    indices = np.random.RandomState(seed).permutation(n_samples)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    folds: List[np.ndarray] = []
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop = current + int(fold_size)
        folds.append(indices[start:stop])
        current = stop
    return folds


def _safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out.isna().any().any():
        missing_cols = [str(c) for c in out.columns[out.isna().any(axis=0)]]
        raise ValueError(f"变换后存在无法转为数值的列: {missing_cols}")
    return out


def _encode_categorical_columns(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    categorical_features: Sequence[str],
    *,
    spec_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, str], List[Dict[str, Any]]]:
    categorical_set = set(categorical_features)
    feature_group_map: Dict[str, str] = {}
    manifest_rows: List[Dict[str, Any]] = []
    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []

    passthrough_columns = [c for c in X_train.columns if c not in categorical_set]
    if passthrough_columns:
        passthrough_train = X_train[passthrough_columns].copy()
        passthrough_val = X_val[passthrough_columns].copy()
        train_parts.append(passthrough_train)
        val_parts.append(passthrough_val)
        for column in passthrough_columns:
            feature_group_map[str(column)] = str(column)
            manifest_rows.append(
                {
                    "spec_name": spec_name,
                    "original_feature": str(column),
                    "transformed_feature": str(column),
                    "role": "continuous",
                    "transforms": "passthrough",
                }
            )

    for feature in categorical_features:
        train_series = X_train[feature].astype("string").fillna("__nan__")
        val_series = X_val[feature].astype("string").fillna("__nan__")
        prefix = str(feature)
        train_dummies = pd.get_dummies(train_series, prefix=prefix, prefix_sep="__")
        val_dummies = pd.get_dummies(val_series, prefix=prefix, prefix_sep="__")
        val_dummies = val_dummies.reindex(columns=train_dummies.columns, fill_value=0)
        train_dummies = train_dummies.astype(float)
        val_dummies = val_dummies.astype(float)
        train_parts.append(train_dummies)
        val_parts.append(val_dummies)

        for column in train_dummies.columns:
            feature_group_map[str(column)] = prefix
            manifest_rows.append(
                {
                    "spec_name": spec_name,
                    "original_feature": prefix,
                    "transformed_feature": str(column),
                    "role": "categorical_encoded",
                    "transforms": "one_hot",
                }
            )

    X_train_out = pd.concat(train_parts, axis=1) if train_parts else pd.DataFrame(index=X_train.index)
    X_val_out = pd.concat(val_parts, axis=1) if val_parts else pd.DataFrame(index=X_val.index)
    return X_train_out, X_val_out, feature_group_map, manifest_rows


def _prepare_fold_data(
    *,
    X_train_raw: pd.DataFrame,
    X_val_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    y_val_raw: pd.Series,
    spec: SpecDefinition,
    categorical_features: Sequence[str],
    continuous_features: Sequence[str],
    log_skewed_features: Sequence[str],
) -> FoldPreparedData:
    X_train = X_train_raw.copy()
    X_val = X_val_raw.copy()

    feature_group_map: Dict[str, str] = {}
    manifest_rows: List[Dict[str, Any]] = []
    transforms_by_feature: Dict[str, List[str]] = {
        str(feature): [] for feature in X_train.columns
    }

    for feature in continuous_features:
        if feature not in X_train.columns:
            continue
        train_series = pd.to_numeric(X_train[feature], errors="coerce")
        val_series = pd.to_numeric(X_val[feature], errors="coerce")

        if spec.log_skewed_x and feature in log_skewed_features:
            train_series = np.log1p(train_series)
            val_series = np.log1p(val_series)
            transforms_by_feature[str(feature)].append("log1p")

        if spec.zscore_x:
            mean_value = float(train_series.mean())
            std_value = float(train_series.std(ddof=0))
            if std_value > 0.0:
                train_series = (train_series - mean_value) / std_value
                val_series = (val_series - mean_value) / std_value
                transforms_by_feature[str(feature)].append("zscore")
            else:
                train_series = train_series - mean_value
                val_series = val_series - mean_value
                transforms_by_feature[str(feature)].append("center_only")

        if spec.minmax_x:
            min_value = float(train_series.min())
            max_value = float(train_series.max())
            range_value = max_value - min_value
            if range_value > 0.0:
                train_series = (train_series - min_value) / range_value
                val_series = (val_series - min_value) / range_value
                transforms_by_feature[str(feature)].append("minmax")
            else:
                train_series = train_series - min_value
                val_series = val_series - min_value
                transforms_by_feature[str(feature)].append("constant_zero")

        X_train[feature] = train_series
        X_val[feature] = val_series

    if spec.encode_categorical:
        X_train_prepared, X_val_prepared, feature_group_map, manifest_rows = _encode_categorical_columns(
            X_train,
            X_val,
            categorical_features,
            spec_name=spec.name,
        )
        for feature in continuous_features:
            if feature not in X_train_prepared.columns:
                continue
            transform_text = "+".join(transforms_by_feature.get(feature) or ["passthrough"])
            manifest_rows = [
                row if row["transformed_feature"] != feature else {**row, "transforms": transform_text}
                for row in manifest_rows
            ]
    else:
        X_train_prepared = _safe_numeric_frame(X_train)
        X_val_prepared = _safe_numeric_frame(X_val)
        categorical_set = set(categorical_features)
        for feature in X_train_prepared.columns:
            feature_group_map[str(feature)] = str(feature)
            role = "categorical_raw" if str(feature) in categorical_set else "continuous"
            transform_text = "+".join(transforms_by_feature.get(str(feature)) or ["passthrough"])
            manifest_rows.append(
                {
                    "spec_name": spec.name,
                    "original_feature": str(feature),
                    "transformed_feature": str(feature),
                    "role": role,
                    "transforms": transform_text,
                }
            )

    if spec.log_y:
        if float(pd.to_numeric(y_train_raw, errors="coerce").min()) <= -1.0:
            raise ValueError("log_y 需要 y > -1。")
        y_train = np.log1p(pd.to_numeric(y_train_raw, errors="raise"))
        inverse_target_fn = np.expm1
        target_scale = "log1p_y"
    else:
        y_train = pd.to_numeric(y_train_raw, errors="raise")
        inverse_target_fn = lambda arr: np.asarray(arr, dtype=float)
        target_scale = "raw_y"

    return FoldPreparedData(
        X_train=X_train_prepared,
        X_val=X_val_prepared,
        y_train=pd.Series(y_train, index=y_train_raw.index),
        y_val_raw=pd.to_numeric(y_val_raw, errors="raise"),
        feature_group_map=feature_group_map,
        manifest_rows=manifest_rows,
        inverse_target_fn=inverse_target_fn,
        target_scale=target_scale,
        uses_log_y=spec.log_y,
    )


def _regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((true_arr - pred_arr) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(true_arr - pred_arr)))
    ss_res = float(np.sum((true_arr - pred_arr) ** 2))
    ss_tot = float(np.sum((true_arr - np.mean(true_arr)) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _rank_desc(values: pd.Series) -> pd.Series:
    return values.rank(method="average", ascending=False)


def _pairwise_spearman_mean(df_spec_feature: pd.DataFrame) -> float:
    pivot = df_spec_feature.pivot_table(
        index="run_id",
        columns="feature",
        values="importance",
        aggfunc="mean",
    )
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return float("nan")
    corr = pivot.T.corr(method="spearman")
    if corr.shape[0] < 2:
        return float("nan")
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    values = corr.to_numpy(dtype=float)[mask]
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _top_k_stability(df_spec_feature_summary: pd.DataFrame, top_k: int) -> float:
    top_df = df_spec_feature_summary.sort_values(
        ["median_rank", "median_importance"], ascending=[True, False]
    ).head(top_k)
    if len(top_df) == 0:
        return float("nan")
    return float(top_df["top_k_frequency"].mean())


def _shap_config_for_validation(config: Dict[str, Any]) -> Dict[str, Any]:
    config_copy = copy.deepcopy(config)
    shap_cfg = dict(config_copy.get("shap") or {})
    shap_cfg["compute_interactions"] = 0
    config_copy["shap"] = shap_cfg
    return config_copy


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info(f"已保存: {path.resolve()}")


def _plot_cfg(attr_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "save_plots": int(attr_cfg.get("save_plots", 1)) == 1,
        "plots_dir": str(attr_cfg.get("plots_dir", "plots")).strip() or "plots",
        "plot_top_n": max(1, int(attr_cfg.get("plot_top_n", 10))),
        "comparison_max_features": max(
            1, int(attr_cfg.get("comparison_max_features", 15))
        ),
        "summary_max_display": max(1, int(attr_cfg.get("summary_max_display", 15))),
        "dpi": max(72, int(attr_cfg.get("plot_dpi", 300))),
    }


def _slugify(text: str) -> str:
    parts: List[str] = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            parts.append(ch)
        else:
            parts.append("_")
    return "".join(parts).strip("_") or "plot"


def _spec_display_name(spec_name: str) -> str:
    return SPEC_DISPLAY_NAMES.get(spec_name, spec_name)


def _spec_filename_token(spec_name: str) -> str:
    return _slugify(_spec_display_name(spec_name))


def _save_plot(path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logging.info(f"已保存图像: {path.resolve()}")


def _ordered_feature_subset(
    feature_summary_df: pd.DataFrame,
    *,
    max_features: int,
) -> List[str]:
    if feature_summary_df.empty:
        return []
    order_df = (
        feature_summary_df.groupby("feature", as_index=False)
        .agg(
            avg_share=("median_importance_share", "mean"),
            avg_rank=("median_rank", "mean"),
        )
        .sort_values(["avg_share", "avg_rank", "feature"], ascending=[False, True, True])
    )
    return [str(x) for x in order_df["feature"].head(max_features).tolist()]


def _plot_spec_top_features(
    feature_summary_df: pd.DataFrame,
    plots_dir: Path,
    *,
    top_n: int,
    top_k: int,
    dpi: int,
) -> None:
    for spec_name, group in feature_summary_df.groupby("spec_name"):
        display_name = _spec_display_name(spec_name)
        plot_df = (
            group.sort_values(
                ["median_rank", "median_importance_share", "feature"],
                ascending=[True, False, True],
            )
            .head(top_n)
            .sort_values("median_importance_share", ascending=True)
        )
        if plot_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, max(4.5, 0.6 * len(plot_df) + 1.5)))
        values = plot_df["median_importance_share"].to_numpy(dtype=float)
        ax.barh(plot_df["feature"], values, color="#4C78A8")
        ax.set_title(f"{display_name}：Top {len(plot_df)} 相对 SHAP 重要性")
        ax.set_xlabel("Median SHAP Importance Share")
        ax.set_ylabel("Feature")

        max_value = float(values.max()) if len(values) else 0.0
        offset = max(max_value * 0.02, 0.002)
        for idx, row in enumerate(plot_df.itertuples(index=False)):
            ax.text(
                float(row.median_importance_share) + offset,
                idx,
                f"Top{top_k}={float(row.top_k_frequency):.2f}, IQR={float(row.rank_iqr):.2f}",
                va="center",
                fontsize=8,
            )

        path = plots_dir / f"top_features_{_spec_filename_token(spec_name)}.png"
        _save_plot(path, dpi=dpi)


def _plot_spec_comparison_heatmaps(
    feature_summary_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    plots_dir: Path,
    *,
    max_features: int,
    dpi: int,
) -> None:
    selected_features = _ordered_feature_subset(
        feature_summary_df,
        max_features=max_features,
    )
    if not selected_features:
        return

    spec_order = summary_df.sort_values(
        ["recommended", "eligible_for_official", "pairwise_spearman_mean", "spec_name"],
        ascending=[False, False, False, True],
    )["spec_name"].tolist()
    spec_label_map = {spec_name: _spec_display_name(spec_name) for spec_name in spec_order}

    share_pivot = (
        feature_summary_df.pivot_table(
            index="feature",
            columns="spec_name",
            values="median_importance_share",
            aggfunc="mean",
        )
        .reindex(index=selected_features, columns=spec_order)
        .fillna(0.0)
    )
    rank_pivot = (
        feature_summary_df.pivot_table(
            index="feature",
            columns="spec_name",
            values="median_rank",
            aggfunc="mean",
        )
        .reindex(index=selected_features, columns=spec_order)
    )
    share_pivot.rename(columns=spec_label_map, inplace=True)
    rank_pivot.rename(columns=spec_label_map, inplace=True)

    for pivot_df, title, filename, cmap in [
        (
            share_pivot,
            "各方案特征相对 SHAP 重要性对比",
            "spec_comparison_importance_share_heatmap.png",
            "viridis",
        ),
        (
            rank_pivot,
            "各方案特征中位排名对比（越小越重要）",
            "spec_comparison_rank_heatmap.png",
            "viridis_r",
        ),
    ]:
        values = pivot_df.to_numpy(dtype=float)
        fig_width = max(8.0, 1.5 * len(pivot_df.columns) + 2.0)
        fig_height = max(5.0, 0.55 * len(pivot_df.index) + 1.8)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(values, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(range(len(pivot_df.columns)))
        ax.set_xticklabels(pivot_df.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(pivot_df.index)))
        ax.set_yticklabels(pivot_df.index)

        finite_values = values[np.isfinite(values)]
        color_threshold = (
            float(finite_values.min() + (finite_values.max() - finite_values.min()) * 0.55)
            if finite_values.size > 0
            else 0.0
        )
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                if not np.isfinite(value):
                    label = "NA"
                    text_color = "black"
                else:
                    label = f"{value:.3f}" if "share" in filename else f"{value:.2f}"
                    text_color = "white" if value >= color_threshold else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(
            "Median Importance Share" if "share" in filename else "Median Rank"
        )
        path = plots_dir / filename
        _save_plot(path, dpi=dpi)


def _plot_spec_shap_summaries(
    spec_feature_frames: Dict[str, List[pd.DataFrame]],
    spec_shap_frames: Dict[str, List[pd.DataFrame]],
    plots_dir: Path,
    *,
    max_display: int,
    dpi: int,
) -> None:
    for spec_name, feature_blocks in spec_feature_frames.items():
        shap_blocks = spec_shap_frames.get(spec_name) or []
        if not feature_blocks or not shap_blocks:
            continue

        ordered_columns = list(
            dict.fromkeys(
                str(column)
                for frame in feature_blocks
                for column in frame.columns
            )
        )
        if not ordered_columns:
            continue

        X_plot = pd.concat(
            [frame.reindex(columns=ordered_columns, fill_value=0.0) for frame in feature_blocks],
            axis=0,
            ignore_index=True,
        )
        shap_plot = pd.concat(
            [frame.reindex(columns=ordered_columns, fill_value=0.0) for frame in shap_blocks],
            axis=0,
            ignore_index=True,
        )
        if X_plot.empty or shap_plot.empty:
            continue

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_plot.to_numpy(dtype=float),
            X_plot,
            show=False,
            max_display=max_display,
        )
        plt.title(f"{_spec_display_name(spec_name)}：SHAP Summary（验证 fold 汇总）")
        path = plots_dir / f"shap_summary_{_spec_filename_token(spec_name)}.png"
        _save_plot(path, dpi=dpi)


def _plot_official_summary(
    official_df: pd.DataFrame,
    plots_dir: Path,
    *,
    top_n: int,
    top_k: int,
    dpi: int,
) -> None:
    spec_name = str(official_df["spec_name"].iloc[0]) if "spec_name" in official_df.columns and not official_df.empty else ""
    display_name = _spec_display_name(spec_name) if spec_name else "正式推荐口径"
    plot_df = (
        official_df.sort_values(
            ["median_rank", "median_importance_share", "feature"],
            ascending=[True, False, True],
        )
        .head(top_n)
        .sort_values("median_importance_share", ascending=True)
    )
    if plot_df.empty:
        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, max(4.8, 0.6 * len(plot_df) + 1.5)),
        sharey=True,
    )
    axes[0].barh(plot_df["feature"], plot_df["median_importance_share"], color="#4C78A8")
    axes[0].set_title(f"{display_name}：相对重要性")
    axes[0].set_xlabel("Median SHAP Importance Share")

    axes[1].barh(plot_df["feature"], plot_df["top_k_frequency"], color="#F58518")
    axes[1].set_title(f"{display_name}：稳定性")
    axes[1].set_xlabel(f"Top-{top_k} Frequency")
    axes[1].set_xlim(0.0, 1.0)

    for idx, row in enumerate(plot_df.itertuples(index=False)):
        axes[1].text(
            min(float(row.top_k_frequency) + 0.02, 1.02),
            idx,
            f"IQR={float(row.rank_iqr):.2f}",
            va="center",
            fontsize=8,
        )

    path = plots_dir / "official_relative_importance_and_stability.png"
    _save_plot(path, dpi=dpi)

def _recommend_spec(
    summary_df: pd.DataFrame,
    *,
    baseline_name: str,
    r2_drop_tol: float,
    rmse_rise_tol: float,
    spearman_tie_tol: float,
) -> tuple[Optional[str], pd.DataFrame]:
    out = summary_df.copy()
    out["eligible_for_official"] = False
    out["recommended"] = False

    if baseline_name not in set(out["spec_name"]):
        logging.warning(f"未找到基线方案 `{baseline_name}`，将无法按性能约束筛选正式口径。")
        return None, out

    baseline_row = out.loc[out["spec_name"] == baseline_name].iloc[0]
    baseline_r2 = float(baseline_row["mean_r2"])
    baseline_rmse = float(baseline_row["mean_rmse"])

    out["eligible_for_official"] = (
        (out["mean_r2"] >= baseline_r2 - r2_drop_tol)
        & (out["mean_rmse"] <= baseline_rmse * (1.0 + rmse_rise_tol))
    )

    eligible = out.loc[out["eligible_for_official"]].copy()
    if eligible.empty:
        logging.warning("所有方案都未满足正式口径的性能约束；本次不会给出正式推荐方案。")
        return None, out

    best_spearman = float(eligible["pairwise_spearman_mean"].max())
    near_best = eligible.loc[
        eligible["pairwise_spearman_mean"] >= best_spearman - spearman_tie_tol
    ].copy()
    best_top_k_stability = float(near_best["top_k_stability"].max())
    near_best = near_best.loc[
        near_best["top_k_stability"] >= best_top_k_stability - 1e-12
    ].copy()
    near_best.sort_values(
        ["uses_log_y", "mean_r2", "mean_rmse", "spec_name"],
        ascending=[True, False, True, True],
        inplace=True,
    )

    recommended_name = str(near_best.iloc[0]["spec_name"])
    out.loc[out["spec_name"] == recommended_name, "recommended"] = True
    return recommended_name, out


def main() -> None:
    run_start = datetime.datetime.now()
    args = parse_args()
    config_path = (
        Path(args.config) if args.config else Path(__file__).with_name("config.yaml")
    )
    config = load_config(config_path)

    output_dir = ensure_run_output_dir(config, prefix="xgb_attr_")
    setup_logging(config, output_dir)

    attr_cfg = _resolve_attribution_cfg(config)
    plot_cfg = _plot_cfg(attr_cfg)
    results_subdir = (
        str(attr_cfg.get("output_subdir", "attribution_robustness")).strip()
        or "attribution_robustness"
    )
    results_dir = output_dir / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / plot_cfg["plots_dir"]
    if plot_cfg["save_plots"]:
        plots_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"使用配置文件: {config_path}")
    logging.info(f"归因稳健性输出目录: {results_dir.resolve()}")
    logging.info(
        "归因稳健性图输出: enabled=%s, plots_dir=%s, plot_top_n=%s, comparison_max_features=%s, summary_max_display=%s",
        int(plot_cfg["save_plots"]),
        plots_dir.resolve(),
        plot_cfg["plot_top_n"],
        plot_cfg["comparison_max_features"],
        plot_cfg["summary_max_display"],
    )

    df, X, y = load_dataset(config)
    X, added_attr_features, missing_attr_features = _augment_feature_frame_from_attribution(
        df,
        X,
        config,
        attr_cfg,
    )
    logging.info(
        f"已加载数据 `{config['data']['path']}`，共 {df.shape[0]} 行，{df.shape[1]} 列。"
    )
    if added_attr_features:
        logging.info(
            "已根据 attribution 配置从原始数据补入特征: %s",
            added_attr_features,
        )
    if missing_attr_features:
        logging.warning(
            "attribution 配置中的以下特征未在原始数据中找到，已跳过: %s",
            missing_attr_features,
        )
    if len(X) == 0:
        raise ValueError("数据为空，无法执行归因稳健性分析。")

    specs = _candidate_specs(attr_cfg)
    model_params = _model_params(config)
    random_state = int(model_params.get("random_state", 42))
    categorical_features, continuous_features = _resolve_feature_roles(X, attr_cfg)
    ratio_threshold = float(attr_cfg.get("log_skewed_ratio_threshold", 8.0))
    log_skewed_features = _determine_log_skewed_features(
        X,
        continuous_features,
        ratio_threshold=ratio_threshold,
    )
    cv_splits = int(attr_cfg.get("cv_splits", 5))
    seeds = _resolve_repeat_seeds(attr_cfg, fallback_seed=random_state)
    top_k = int(attr_cfg.get("top_k", 5))
    baseline_name = str(attr_cfg.get("baseline_spec", "encoded_cat__raw_y"))
    r2_drop_tol = float(attr_cfg.get("performance_r2_drop_tolerance", 0.05))
    rmse_rise_tol = float(attr_cfg.get("performance_rmse_rise_tolerance", 0.10))
    spearman_tie_tol = float(attr_cfg.get("spearman_tie_tolerance", 0.02))

    logging.info(f"候选方案: {[spec.name for spec in specs]}")
    logging.info(f"类别特征: {categorical_features}")
    logging.info(f"连续特征: {continuous_features}")
    logging.info(
        f"log_skewed_x 判定阈值: p95/p50 >= {ratio_threshold}; 命中特征: {log_skewed_features}"
    )
    logging.info(
        f"CV 设置: repeats={len(seeds)}, splits={cv_splits}, seeds={seeds}, top_k={top_k}"
    )

    shap_config = _shap_config_for_validation(config)
    fold_rows: List[Dict[str, Any]] = []
    importance_rows: List[Dict[str, Any]] = []
    manifest_rows_all: List[Dict[str, Any]] = []
    spec_feature_frames: Dict[str, List[pd.DataFrame]] = {}
    spec_shap_frames: Dict[str, List[pd.DataFrame]] = {}

    n_samples = len(X)
    if cv_splits < 2 or cv_splits > n_samples:
        raise ValueError(
            f"无效的 attribution.cv_splits={cv_splits}，样本数为 {n_samples}。"
        )

    for spec in specs:
        logging.info(
            "开始方案 `%s`: encode_categorical=%s, log_skewed_x=%s, zscore_x=%s, minmax_x=%s, log_y=%s, diagnostic_only=%s",
            spec.name,
            int(spec.encode_categorical),
            int(spec.log_skewed_x),
            int(spec.zscore_x),
            int(spec.minmax_x),
            int(spec.log_y),
            int(spec.diagnostic_only),
        )
        for repeat_idx, seed in enumerate(seeds, start=1):
            folds = _kfold_indices(n_samples, cv_splits, seed)
            for fold_idx, val_idx in enumerate(folds, start=1):
                train_idx = np.concatenate(
                    [folds[j] for j in range(len(folds)) if j != fold_idx - 1]
                )
                X_train_raw = X.iloc[train_idx].reset_index(drop=True)
                X_val_raw = X.iloc[val_idx].reset_index(drop=True)
                y_train_raw = y.iloc[train_idx].reset_index(drop=True)
                y_val_raw = y.iloc[val_idx].reset_index(drop=True)

                prepared = _prepare_fold_data(
                    X_train_raw=X_train_raw,
                    X_val_raw=X_val_raw,
                    y_train_raw=y_train_raw,
                    y_val_raw=y_val_raw,
                    spec=spec,
                    categorical_features=categorical_features,
                    continuous_features=continuous_features,
                    log_skewed_features=log_skewed_features,
                )

                estimator = xgb.XGBRegressor(**model_params)
                estimator.fit(prepared.X_train, prepared.y_train)

                pred_val_model_scale = estimator.predict(prepared.X_val)
                pred_val_raw = prepared.inverse_target_fn(pred_val_model_scale)
                metrics = _regression_metrics(prepared.y_val_raw, pred_val_raw)

                shap_values, _ = compute_shap_and_interactions(
                    estimator, prepared.X_val, shap_config
                )
                shap_array = np.asarray(shap_values, dtype=float)
                mean_abs_shap = np.mean(np.abs(shap_array), axis=0)
                if plot_cfg["save_plots"]:
                    spec_feature_frames.setdefault(spec.name, []).append(
                        prepared.X_val.reset_index(drop=True).copy()
                    )
                    spec_shap_frames.setdefault(spec.name, []).append(
                        pd.DataFrame(
                            shap_array,
                            columns=prepared.X_val.columns,
                        )
                    )

                grouped_importance: Dict[str, float] = {}
                for column, importance in zip(prepared.X_val.columns, mean_abs_shap):
                    original_feature = prepared.feature_group_map.get(
                        str(column), str(column)
                    )
                    grouped_importance[original_feature] = (
                        grouped_importance.get(original_feature, 0.0)
                        + float(importance)
                    )

                run_id = f"{spec.name}__r{repeat_idx:02d}__f{fold_idx:02d}"
                importance_series = pd.Series(grouped_importance, dtype=float).sort_values(
                    ascending=False,
                    kind="mergesort",
                )
                rank_series = _rank_desc(importance_series)
                total_importance = float(importance_series.sum())
                if total_importance > 0.0:
                    share_series = importance_series / total_importance
                else:
                    share_series = importance_series * 0.0

                fold_rows.append(
                    {
                        "spec_name": spec.name,
                        "run_id": run_id,
                        "repeat_idx": repeat_idx,
                        "seed": seed,
                        "fold_idx": fold_idx,
                        "n_train": len(prepared.X_train),
                        "n_val": len(prepared.X_val),
                        "mean_r2": float(metrics["r2"]),
                        "mean_rmse": float(metrics["rmse"]),
                        "mean_mae": float(metrics["mae"]),
                        "target_scale": prepared.target_scale,
                        "uses_log_y": int(prepared.uses_log_y),
                        "encode_categorical": int(spec.encode_categorical),
                        "log_skewed_x": int(spec.log_skewed_x),
                        "zscore_x": int(spec.zscore_x),
                        "minmax_x": int(spec.minmax_x),
                        "diagnostic_only": int(spec.diagnostic_only),
                    }
                )

                for feature, importance in importance_series.items():
                    importance_rows.append(
                        {
                            "spec_name": spec.name,
                            "run_id": run_id,
                            "repeat_idx": repeat_idx,
                            "seed": seed,
                            "fold_idx": fold_idx,
                            "feature": str(feature),
                            "importance": float(importance),
                            "importance_share": float(share_series.loc[feature]),
                            "rank": float(rank_series.loc[feature]),
                            "target_scale": prepared.target_scale,
                        }
                    )

                for row in prepared.manifest_rows:
                    manifest_rows_all.append(
                        {
                            **row,
                            "run_id": run_id,
                            "repeat_idx": repeat_idx,
                            "fold_idx": fold_idx,
                        }
                    )

                logging.info(
                    "方案 `%s` repeat=%s/%s fold=%s/%s - R2=%.4f RMSE=%.4f MAE=%.4f",
                    spec.name,
                    repeat_idx,
                    len(seeds),
                    fold_idx,
                    cv_splits,
                    float(metrics["r2"]),
                    float(metrics["rmse"]),
                    float(metrics["mae"]),
                )

    fold_df = pd.DataFrame(fold_rows)
    importance_df = pd.DataFrame(importance_rows)
    manifest_df = pd.DataFrame(manifest_rows_all)
    if fold_df.empty or importance_df.empty:
        raise RuntimeError("未生成任何归因稳健性结果。")

    feature_summary_df = (
        importance_df.groupby(["spec_name", "feature"], as_index=False)
        .agg(
            median_importance=("importance", "median"),
            mean_importance=("importance", "mean"),
            median_importance_share=("importance_share", "median"),
            mean_importance_share=("importance_share", "mean"),
            median_rank=("rank", "median"),
            mean_rank=("rank", "mean"),
            rank_q25=("rank", lambda s: float(pd.Series(s).quantile(0.25))),
            rank_q75=("rank", lambda s: float(pd.Series(s).quantile(0.75))),
            top_k_frequency=("rank", lambda s: float(np.mean(pd.Series(s) <= top_k))),
        )
    )
    feature_summary_df["rank_iqr"] = (
        feature_summary_df["rank_q75"] - feature_summary_df["rank_q25"]
    )
    feature_summary_df["stable_top_k"] = (
        feature_summary_df["top_k_frequency"] >= 0.8
    ).astype(int)

    spec_summary_rows: List[Dict[str, Any]] = []
    for spec_name, fold_group in fold_df.groupby("spec_name"):
        feature_group = feature_summary_df.loc[
            feature_summary_df["spec_name"] == spec_name
        ].copy()
        spec_summary_rows.append(
            {
                "spec_name": spec_name,
                "n_runs": int(len(fold_group)),
                "mean_r2": float(fold_group["mean_r2"].mean()),
                "std_r2": float(fold_group["mean_r2"].std(ddof=0)),
                "mean_rmse": float(fold_group["mean_rmse"].mean()),
                "std_rmse": float(fold_group["mean_rmse"].std(ddof=0)),
                "mean_mae": float(fold_group["mean_mae"].mean()),
                "std_mae": float(fold_group["mean_mae"].std(ddof=0)),
                "pairwise_spearman_mean": _pairwise_spearman_mean(
                    importance_df.loc[importance_df["spec_name"] == spec_name]
                ),
                "top_k_stability": _top_k_stability(feature_group, top_k),
                "uses_log_y": int(bool(fold_group["uses_log_y"].iloc[0])),
                "encode_categorical": int(
                    bool(fold_group["encode_categorical"].iloc[0])
                ),
                "log_skewed_x": int(bool(fold_group["log_skewed_x"].iloc[0])),
                "zscore_x": int(bool(fold_group["zscore_x"].iloc[0])),
                "minmax_x": int(bool(fold_group["minmax_x"].iloc[0])),
                "diagnostic_only": int(bool(fold_group["diagnostic_only"].iloc[0])),
            }
        )

    summary_df = pd.DataFrame(spec_summary_rows)
    recommended_name, summary_df = _recommend_spec(
        summary_df,
        baseline_name=baseline_name,
        r2_drop_tol=r2_drop_tol,
        rmse_rise_tol=rmse_rise_tol,
        spearman_tie_tol=spearman_tie_tol,
    )

    fold_csv = results_dir / "attribution_fold_metrics.csv"
    importance_csv = results_dir / "attribution_fold_shap_importance.csv"
    feature_summary_csv = results_dir / "feature_stability_by_spec.csv"
    summary_csv = results_dir / "robustness_summary.csv"
    manifest_csv = results_dir / "transform_manifest.csv"
    official_csv = results_dir / "official_relative_importance.csv"
    recommendation_txt = results_dir / "recommendation.txt"

    _write_csv(
        fold_csv,
        fold_df.sort_values(["spec_name", "repeat_idx", "fold_idx"]),
    )
    _write_csv(
        importance_csv,
        importance_df.sort_values(
            ["spec_name", "repeat_idx", "fold_idx", "rank", "feature"]
        ),
    )
    _write_csv(
        feature_summary_csv,
        feature_summary_df.sort_values(["spec_name", "median_rank", "feature"]),
    )
    _write_csv(
        summary_csv,
        summary_df.sort_values(
            ["recommended", "eligible_for_official", "pairwise_spearman_mean"],
            ascending=[False, False, False],
        ),
    )
    _write_csv(
        manifest_csv,
        manifest_df.sort_values(
            ["spec_name", "repeat_idx", "fold_idx", "original_feature", "transformed_feature"]
        ),
    )

    if plot_cfg["save_plots"]:
        _plot_spec_shap_summaries(
            spec_feature_frames,
            spec_shap_frames,
            plots_dir,
            max_display=plot_cfg["summary_max_display"],
            dpi=plot_cfg["dpi"],
        )
        _plot_spec_top_features(
            feature_summary_df,
            plots_dir,
            top_n=plot_cfg["plot_top_n"],
            top_k=top_k,
            dpi=plot_cfg["dpi"],
        )
        _plot_spec_comparison_heatmaps(
            feature_summary_df,
            summary_df,
            plots_dir,
            max_features=plot_cfg["comparison_max_features"],
            dpi=plot_cfg["dpi"],
        )

    if recommended_name is None:
        empty_official = pd.DataFrame(
            columns=[
                "spec_name",
                "feature",
                "median_importance",
                "mean_importance",
                "median_importance_share",
                "median_rank",
                "rank_iqr",
                "top_k_frequency",
                "stable_top_k",
            ]
        )
        _write_csv(official_csv, empty_official)
        recommendation_txt.write_text(
            "本次没有满足性能约束且稳定性更优的正式推荐口径，请查看 robustness_summary.csv。\n",
            encoding="utf-8",
        )
        logging.warning("未生成正式推荐方案，请查看 robustness_summary.csv。")
    else:
        official_df = feature_summary_df.loc[
            feature_summary_df["spec_name"] == recommended_name
        ].copy()
        official_df.sort_values(
            ["median_rank", "median_importance", "feature"],
            ascending=[True, False, True],
            inplace=True,
        )
        _write_csv(official_csv, official_df)
        if plot_cfg["save_plots"]:
            _plot_official_summary(
                official_df,
                plots_dir,
                top_n=plot_cfg["plot_top_n"],
                top_k=top_k,
                dpi=plot_cfg["dpi"],
            )

        recommended_row = summary_df.loc[
            summary_df["spec_name"] == recommended_name
        ].iloc[0]
        recommendation_lines = [
            f"recommended_spec: {recommended_name}",
            f"pairwise_spearman_mean: {float(recommended_row['pairwise_spearman_mean']):.6f}",
            f"top_k_stability: {float(recommended_row['top_k_stability']):.6f}",
            f"mean_r2: {float(recommended_row['mean_r2']):.6f}",
            f"mean_rmse: {float(recommended_row['mean_rmse']):.6f}",
            f"uses_log_y: {int(recommended_row['uses_log_y'])}",
            "",
            "说明：official_relative_importance.csv 中的 SHAP 重要性只来自验证 fold 汇总，不来自训练 fold。",
        ]
        recommendation_txt.write_text(
            "\n".join(recommendation_lines) + "\n",
            encoding="utf-8",
        )
        logging.info(
            "正式推荐方案: %s（mean_spearman=%.4f, top_k_stability=%.4f, mean_R2=%.4f, mean_RMSE=%.4f）",
            recommended_name,
            float(recommended_row["pairwise_spearman_mean"]),
            float(recommended_row["top_k_stability"]),
            float(recommended_row["mean_r2"]),
            float(recommended_row["mean_rmse"]),
        )

    run_end = datetime.datetime.now()
    logging.info(
        f"归因稳健性分析完成: {run_end}（耗时 {(run_end - run_start).total_seconds():.2f} 秒）"
    )


if __name__ == "__main__":
    main()
