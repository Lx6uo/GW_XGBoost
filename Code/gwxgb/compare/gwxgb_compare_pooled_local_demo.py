from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gwxgb_compare_demo_common import (
    build_arg_parser,
    build_importance_table,
    compare_metric_dict,
    prepare_run,
    run_compare_pipeline,
    save_global_native_plots,
    save_metrics_report,
    save_pooled_local_native_plots,
)


def main() -> None:
    parser = build_arg_parser(
        "Demo: 将所有局部模型邻域样本的 SHAP 行直接拼接后，与全局模型 SHAP 做对比。"
    )
    args = parser.parse_args()
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).with_name("gwxgb_compare_pooled_local_config.yaml")

    config, output_dir = prepare_run(
        config_path,
        run_prefix="gwxgb_pooled_local_demo_",
    )
    artifacts = run_compare_pipeline(config, output_dir)
    compare_cfg = config.get("compare") or {}
    save_global_native_plots(
        artifacts,
        output_dir=output_dir,
        compare_cfg=compare_cfg,
    )

    pooled_df = artifacts.local_pooled_df.copy()
    if pooled_df.empty:
        raise ValueError("未采集到 pooled 局部 SHAP 行，无法执行 pooled demo。")

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
        compare_df["pooled_local_importance_share"] - compare_df["global_importance_share"]
    )
    compare_df.sort_values("global_importance_share", ascending=False, inplace=True)
    compare_df.reset_index(drop=True, inplace=True)

    compare_path = output_dir / "global_vs_pooled_local_shap.csv"
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")
    logging.info(f"pooled 局部 SHAP 对比表已保存: {compare_path}")

    pooled_path = output_dir / "pooled_local_shap_rows.csv"
    pooled_df.to_csv(pooled_path, index=False, encoding="utf-8-sig")
    logging.info(f"pooled 局部 SHAP 明细已保存: {pooled_path}")

    reuse_df = (
        pooled_df.groupby("sample_index", dropna=False)
        .size()
        .rename("reuse_count")
        .reset_index()
        .sort_values("reuse_count", ascending=False)
    )
    reuse_path = output_dir / "pooled_sample_reuse_counts.csv"
    reuse_df.to_csv(reuse_path, index=False, encoding="utf-8-sig")
    logging.info(f"样本复用统计表已保存: {reuse_path}")

    save_pooled_local_native_plots(
        artifacts,
        output_dir=output_dir,
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
        float(total_rows) / float(unique_samples) if unique_samples > 0 else float("nan")
    )
    report_path = output_dir / "pooled_local_comparison_metrics.txt"
    save_metrics_report(
        report_path,
        title="Pooled local SHAP vs Global SHAP",
        metrics=metrics,
        extra_lines=[
            "说明：该口径直接把所有局部模型邻域样本的 SHAP 行拼接后再聚合。",
            "它可用于演示你的想法，但会让重复出现在多个邻域中的样本被反复计数，通常应配合 reuse_count 一起解释。",
        ],
    )


if __name__ == "__main__":
    main()
