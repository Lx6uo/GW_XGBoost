from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gwxgb_compare_demo_common import (
    build_arg_parser,
    build_local_importance_outputs,
    compare_metric_dict,
    compare_signed_metric_dict,
    prepare_run,
    run_compare_pipeline,
    save_global_native_plots,
    save_local_importance_native_plots,
    save_metrics_report,
)


def main() -> None:
    parser = build_arg_parser(
        "Demo: 使用每个局部模型的 mean(|SHAP|) 重要性向量，与全局模型 SHAP 做对比。"
    )
    args = parser.parse_args()
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).with_name("gwxgb_compare_local_importance_config.yaml")

    config, output_dir = prepare_run(
        config_path,
        run_prefix="gwxgb_local_importance_demo_",
    )
    artifacts = run_compare_pipeline(config, output_dir)
    compare_cfg = config.get("compare") or {}
    save_global_native_plots(
        artifacts,
        output_dir=output_dir,
        compare_cfg=compare_cfg,
    )

    local_imp_df = artifacts.local_importance_df.copy()
    if local_imp_df.empty:
        raise ValueError("未采集到局部模型重要性表，无法执行 local importance demo。")

    outputs = build_local_importance_outputs(artifacts)
    long_df = outputs["local_importance_long_df"]
    compare_df = outputs["local_importance_compare_df"]
    signed_compare_df = outputs["global_vs_local_signed_df"]
    pos_neg_summary_df = outputs["local_pos_neg_summary_df"]

    compare_path = output_dir / "global_vs_local_importance_summary.csv"
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")
    logging.info(f"局部模型重要性对比表已保存: {compare_path}")

    raw_path = output_dir / "local_model_importance_wide.csv"
    local_imp_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    logging.info(f"局部模型重要性宽表已保存: {raw_path}")

    long_path = output_dir / "local_model_importance_long.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    logging.info(f"局部模型重要性长表已保存: {long_path}")

    signed_raw_path = output_dir / "local_model_signed_shap_wide.csv"
    artifacts.local_signed_df.to_csv(signed_raw_path, index=False, encoding="utf-8-sig")
    logging.info(f"局部模型 signed SHAP 宽表已保存: {signed_raw_path}")

    signed_compare_path = output_dir / "global_vs_local_signed_shap_summary.csv"
    signed_compare_df.to_csv(signed_compare_path, index=False, encoding="utf-8-sig")
    logging.info(f"全局 vs 局部 signed SHAP 对比表已保存: {signed_compare_path}")

    pos_neg_path = output_dir / "local_model_positive_negative_summary.csv"
    pos_neg_summary_df.to_csv(pos_neg_path, index=False, encoding="utf-8-sig")
    logging.info(f"局部模型正负 SHAP 汇总表已保存: {pos_neg_path}")

    save_local_importance_native_plots(
        artifacts,
        output_dir=output_dir,
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
    report_path = output_dir / "local_importance_comparison_metrics.txt"
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


if __name__ == "__main__":
    main()
