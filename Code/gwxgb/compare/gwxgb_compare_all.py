from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from gwxgb_compare_demo_common import (
    build_arg_parser,
    build_importance_table,
    build_local_importance_outputs,
    compare_metric_dict,
    compare_signed_metric_dict,
    prepare_run,
    run_compare_pipeline,
    save_center_local_native_plots,
    save_global_native_plots,
    save_local_importance_native_plots,
    save_metrics_report,
    save_pooled_local_native_plots,
)


def _enabled(compare_cfg: Dict[str, Any], key: str, default: int = 1) -> bool:
    return int(compare_cfg.get(key, default)) == 1


def _ensure_section_dir(output_dir: Path, name: str) -> Path:
    section_dir = output_dir / name
    section_dir.mkdir(parents=True, exist_ok=True)
    return section_dir


def _run_center_local(
    artifacts: Any,
    *,
    output_dir: Path,
    compare_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    if artifacts.local_center_df.empty:
        raise ValueError("未采集到中心样本局部 SHAP 行，无法执行中心样本口径对比。")

    section_dir = _ensure_section_dir(output_dir, "center_local")
    feature_names = list(artifacts.X.columns)
    center_mean_abs = np.array(
        [
            float(
                np.mean(
                    np.abs(
                        artifacts.local_center_df[f"shap_{feature}"].to_numpy(
                            dtype=float
                        )
                    )
                )
            )
            for feature in feature_names
        ],
        dtype=float,
    )
    center_importance_df = build_importance_table(
        feature_names,
        center_mean_abs,
        prefix="center_local",
    )

    compare_df = artifacts.global_importance_df.merge(
        center_importance_df,
        on="feature",
        how="outer",
    )
    compare_df["share_diff_center_minus_global"] = (
        compare_df["center_local_importance_share"]
        - compare_df["global_importance_share"]
    )
    compare_df.sort_values("global_importance_share", ascending=False, inplace=True)
    compare_df.reset_index(drop=True, inplace=True)

    compare_path = section_dir / "global_vs_center_local_shap.csv"
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")
    logging.info(f"中心局部 SHAP 对比表已保存: {compare_path}")

    center_rows_path = section_dir / "center_local_shap_rows.csv"
    artifacts.local_center_df.to_csv(center_rows_path, index=False, encoding="utf-8-sig")
    logging.info(f"中心样本局部 SHAP 明细已保存: {center_rows_path}")

    save_center_local_native_plots(
        artifacts,
        output_dir=section_dir,
        compare_cfg=compare_cfg,
    )

    metrics = compare_metric_dict(
        compare_df,
        left_share_col="global_importance_share",
        right_share_col="center_local_importance_share",
        top_k=int(compare_cfg.get("top_k", 5)),
        left_name="global",
        right_name="center_local",
    )
    metrics["n_center_rows"] = int(len(artifacts.local_center_df))
    report_path = section_dir / "center_local_comparison_metrics.txt"
    save_metrics_report(
        report_path,
        title="Center-local SHAP vs Global SHAP",
        metrics=metrics,
        extra_lines=[
            "说明：每个中心位置只保留一条“中心样本在该中心局部模型下”的 SHAP 向量。",
            "该口径最接近‘每个地理位置一个局部解释’，适合与全局模型做一对一聚合比较。",
        ],
    )
    metrics["method"] = "center_local"
    return metrics


def _run_local_importance(
    artifacts: Any,
    *,
    output_dir: Path,
    compare_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    local_imp_df = artifacts.local_importance_df.copy()
    if local_imp_df.empty:
        raise ValueError("未采集到局部模型重要性表，无法执行局部重要性口径对比。")

    section_dir = _ensure_section_dir(output_dir, "local_importance")
    outputs = build_local_importance_outputs(artifacts)
    long_df = outputs["local_importance_long_df"]
    compare_df = outputs["local_importance_compare_df"]
    signed_compare_df = outputs["global_vs_local_signed_df"]
    pos_neg_summary_df = outputs["local_pos_neg_summary_df"]

    compare_path = section_dir / "global_vs_local_importance_summary.csv"
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")
    logging.info(f"局部模型重要性对比表已保存: {compare_path}")

    raw_path = section_dir / "local_model_importance_wide.csv"
    local_imp_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    logging.info(f"局部模型重要性宽表已保存: {raw_path}")

    long_path = section_dir / "local_model_importance_long.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    logging.info(f"局部模型重要性长表已保存: {long_path}")

    signed_raw_path = section_dir / "local_model_signed_shap_wide.csv"
    artifacts.local_signed_df.to_csv(signed_raw_path, index=False, encoding="utf-8-sig")
    logging.info(f"局部模型 signed SHAP 宽表已保存: {signed_raw_path}")

    signed_compare_path = section_dir / "global_vs_local_signed_shap_summary.csv"
    signed_compare_df.to_csv(signed_compare_path, index=False, encoding="utf-8-sig")
    logging.info(f"全局 vs 局部 signed SHAP 对比表已保存: {signed_compare_path}")

    pos_neg_path = section_dir / "local_model_positive_negative_summary.csv"
    pos_neg_summary_df.to_csv(pos_neg_path, index=False, encoding="utf-8-sig")
    logging.info(f"局部模型正负 SHAP 汇总表已保存: {pos_neg_path}")

    save_local_importance_native_plots(
        artifacts,
        output_dir=section_dir,
        compare_cfg=compare_cfg,
    )

    metrics = compare_metric_dict(
        compare_df,
        left_share_col="global_importance_share",
        right_share_col="local_model_median_share",
        top_k=int(compare_cfg.get("top_k", 5)),
        left_name="global",
        right_name="local_model_median",
    )
    metrics.update(
        compare_signed_metric_dict(
            signed_compare_df,
            left_value_col="global_mean_shap",
            right_value_col="local_model_mean_signed_shap",
            left_name="global_mean_shap",
            right_name="local_model_mean_signed_shap",
        )
    )
    metrics["n_local_models"] = int(local_imp_df["center_label"].nunique())
    report_path = section_dir / "local_importance_comparison_metrics.txt"
    save_metrics_report(
        report_path,
        title="Local-model importance summary vs Global SHAP",
        metrics=metrics,
        extra_lines=[
            "说明：该口径同时保留强度与方向两套结果。",
            "强度部分使用每个局部模型的 mean(|SHAP|)；方向部分使用每个局部模型的 mean(SHAP)。",
            "同时输出 local_model_positive_negative_summary.csv，用于查看正向贡献与负向贡献的平均强度。",
        ],
    )
    metrics["method"] = "local_importance"
    return metrics


def _run_pooled_local(
    artifacts: Any,
    *,
    output_dir: Path,
    compare_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    pooled_df = artifacts.local_pooled_df.copy()
    if pooled_df.empty:
        raise ValueError("未采集到 pooled 局部 SHAP 行，无法执行拼池口径对比。")

    section_dir = _ensure_section_dir(output_dir, "pooled_local")
    feature_names = list(artifacts.X.columns)
    pooled_mean_abs = np.array(
        [
            float(np.mean(np.abs(pooled_df[f"shap_{feature}"].to_numpy(dtype=float))))
            for feature in feature_names
        ],
        dtype=float,
    )
    pooled_importance_df = build_importance_table(
        feature_names,
        pooled_mean_abs,
        prefix="pooled_local",
    )

    compare_df = artifacts.global_importance_df.merge(
        pooled_importance_df,
        on="feature",
        how="outer",
    )
    compare_df["share_diff_pooled_minus_global"] = (
        compare_df["pooled_local_importance_share"]
        - compare_df["global_importance_share"]
    )
    compare_df.sort_values("global_importance_share", ascending=False, inplace=True)
    compare_df.reset_index(drop=True, inplace=True)

    compare_path = section_dir / "global_vs_pooled_local_shap.csv"
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")
    logging.info(f"pooled 局部 SHAP 对比表已保存: {compare_path}")

    pooled_path = section_dir / "pooled_local_shap_rows.csv"
    pooled_df.to_csv(pooled_path, index=False, encoding="utf-8-sig")
    logging.info(f"pooled 局部 SHAP 明细已保存: {pooled_path}")

    reuse_df = (
        pooled_df.groupby("sample_index", dropna=False)
        .size()
        .rename("reuse_count")
        .reset_index()
        .sort_values("reuse_count", ascending=False)
    )
    reuse_path = section_dir / "pooled_sample_reuse_counts.csv"
    reuse_df.to_csv(reuse_path, index=False, encoding="utf-8-sig")
    logging.info(f"样本复用统计表已保存: {reuse_path}")

    save_pooled_local_native_plots(
        artifacts,
        output_dir=section_dir,
        compare_cfg=compare_cfg,
    )

    metrics = compare_metric_dict(
        compare_df,
        left_share_col="global_importance_share",
        right_share_col="pooled_local_importance_share",
        top_k=int(compare_cfg.get("top_k", 5)),
        left_name="global",
        right_name="pooled_local",
    )
    unique_samples = int(reuse_df["sample_index"].nunique())
    total_rows = int(len(pooled_df))
    metrics["n_pooled_rows"] = total_rows
    metrics["n_unique_samples"] = unique_samples
    metrics["mean_reuse_count"] = (
        float(reuse_df["reuse_count"].mean()) if unique_samples > 0 else float("nan")
    )
    metrics["max_reuse_count"] = (
        int(reuse_df["reuse_count"].max()) if unique_samples > 0 else 0
    )
    metrics["duplication_ratio_rows_over_unique_samples"] = (
        float(total_rows) / float(unique_samples)
        if unique_samples > 0
        else float("nan")
    )
    report_path = section_dir / "pooled_local_comparison_metrics.txt"
    save_metrics_report(
        report_path,
        title="Pooled local SHAP vs Global SHAP",
        metrics=metrics,
        extra_lines=[
            "说明：该口径直接把所有局部模型邻域样本的 SHAP 行拼接后再聚合。",
            "它可用于演示你的想法，但会让重复出现在多个邻域中的样本被反复计数，通常应配合 reuse_count 一起解释。",
        ],
    )
    metrics["method"] = "pooled_local"
    return metrics


def _save_overview(output_dir: Path, metrics_rows: List[Dict[str, Any]]) -> None:
    if not metrics_rows:
        logging.warning("未生成任何 comparison metrics，跳过 overview 输出。")
        return

    overview_df = pd.DataFrame(metrics_rows)
    cols = ["method"] + [col for col in overview_df.columns if col != "method"]
    overview_df = overview_df.reindex(columns=cols)
    overview_csv = output_dir / "comparison_metrics_overview.csv"
    overview_df.to_csv(overview_csv, index=False, encoding="utf-8-sig")
    logging.info(f"总览指标表已保存: {overview_csv}")

    lines = ["Global vs Local SHAP Comparison Overview", ""]
    for row in metrics_rows:
        method = str(row.get("method", "unknown"))
        lines.append(f"[{method}]")
        for key, value in row.items():
            if key == "method":
                continue
            lines.append(f"{key}: {value}")
        lines.append("")
    overview_txt = output_dir / "comparison_metrics_overview.txt"
    overview_txt.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    logging.info(f"总览指标文本已保存: {overview_txt}")


def main() -> None:
    parser = build_arg_parser(
        "一次运行同时完成 center_local、local_importance、pooled_local 三种 gwxgb SHAP 对比。"
    )
    args = parser.parse_args()
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).with_name("gwxgb_compare_all_config.yaml")

    config, output_dir = prepare_run(
        config_path,
        run_prefix="gwxgb_compare_all_",
    )
    artifacts = run_compare_pipeline(config, output_dir)
    compare_cfg = config.get("compare") or {}
    save_global_native_plots(
        artifacts,
        output_dir=output_dir,
        compare_cfg=compare_cfg,
    )
    metrics_rows: List[Dict[str, Any]] = []

    if _enabled(compare_cfg, "run_center_local", 1):
        metrics_rows.append(
            _run_center_local(
                artifacts,
                output_dir=output_dir,
                compare_cfg=compare_cfg,
            )
        )

    if _enabled(compare_cfg, "run_local_importance", 1):
        metrics_rows.append(
            _run_local_importance(
                artifacts,
                output_dir=output_dir,
                compare_cfg=compare_cfg,
            )
        )

    if _enabled(compare_cfg, "run_pooled_local", 1):
        metrics_rows.append(
            _run_pooled_local(
                artifacts,
                output_dir=output_dir,
                compare_cfg=compare_cfg,
            )
        )

    _save_overview(output_dir, metrics_rows)


if __name__ == "__main__":
    main()
