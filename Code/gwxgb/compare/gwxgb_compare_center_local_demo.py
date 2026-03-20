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
    save_center_local_native_plots,
    save_global_native_plots,
    save_metrics_report,
)


def main() -> None:
    parser = build_arg_parser(
        "Demo: 使用每个中心样本在各自局部模型下的 SHAP，与全局模型 SHAP 做对比。"
    )
    args = parser.parse_args()
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).with_name("gwxgb_compare_center_local_config.yaml")

    config, output_dir = prepare_run(
        config_path,
        run_prefix="gwxgb_center_local_demo_",
    )
    artifacts = run_compare_pipeline(config, output_dir)
    compare_cfg = config.get("compare") or {}
    save_global_native_plots(
        artifacts,
        output_dir=output_dir,
        compare_cfg=compare_cfg,
    )

    if artifacts.local_center_df.empty:
        raise ValueError("未采集到中心样本局部 SHAP 行，无法执行中心对比 demo。")

    feature_names = list(artifacts.X.columns)
    center_mean_abs = np.array(
        [
            float(np.mean(np.abs(artifacts.local_center_df[f"shap_{feature}"].to_numpy(dtype=float))))
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

    compare_path = output_dir / "global_vs_center_local_shap.csv"
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")
    logging.info(f"中心局部 SHAP 对比表已保存: {compare_path}")

    center_rows_path = output_dir / "center_local_shap_rows.csv"
    artifacts.local_center_df.to_csv(center_rows_path, index=False, encoding="utf-8-sig")
    logging.info(f"中心样本局部 SHAP 明细已保存: {center_rows_path}")

    save_center_local_native_plots(
        artifacts,
        output_dir=output_dir,
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
    report_path = output_dir / "center_local_comparison_metrics.txt"
    save_metrics_report(
        report_path,
        title="Center-local SHAP vs Global SHAP",
        metrics=metrics,
        extra_lines=[
            "说明：每个中心位置只保留一条“中心样本在该中心局部模型下”的 SHAP 向量。",
            "该口径最接近‘每个地理位置一个局部解释’，适合与全局模型做一对一聚合比较。",
        ],
    )


if __name__ == "__main__":
    main()
