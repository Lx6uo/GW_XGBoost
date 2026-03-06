from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd

from xgb_shap import ensure_run_output_dir, load_config, load_dataset, setup_logging


rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Arial Unicode MS",
]
rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对数据集特征做相关性矩阵分析，并输出矩阵 CSV + 热力图 PNG。"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default=None,
        help="YAML 配置文件路径；若不指定则默认使用脚本同目录下的 config.yaml",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["pearson", "spearman", "kendall"],
        default="pearson",
        help="相关性计算方法（默认 pearson）",
    )
    parser.add_argument(
        "--include-target",
        action="store_true",
        help="将 target 列也纳入相关性矩阵（默认不包含）。",
    )
    parser.add_argument(
        "--abs",
        action="store_true",
        help="输出绝对相关系数（0~1）。默认输出带正负号的相关系数（-1~1）。",
    )
    parser.add_argument(
        "--annot",
        action="store_true",
        help="在热力图上标注数值（特征较多时会很慢且难读）。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="输出图片的 dpi（默认 300）。",
    )
    return parser.parse_args()


def _select_numeric_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    numeric_df = df.select_dtypes(include=[np.number])
    skipped = [c for c in df.columns if c not in numeric_df.columns]
    return numeric_df, skipped


def plot_corr_heatmap(
    corr: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    dpi: int,
    annot: bool,
    vmin: float,
    vmax: float,
) -> None:
    n = int(corr.shape[0])
    if n == 0:
        raise ValueError("corr 为空，无法绘图。")

    # 自适应图尺寸：特征越多越大
    size = max(6.0, 0.35 * n)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_title(title)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns.tolist(), rotation=90)
    ax.set_yticklabels(corr.index.tolist())

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("correlation")

    if annot:
        # 注：n 较大时会很慢
        values = corr.to_numpy()
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    run_start = datetime.datetime.now()
    args = parse_args()

    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).with_name("config.yaml")

    config: Dict[str, Any] = load_config(config_path)
    output_dir = ensure_run_output_dir(config, prefix="corr_")
    setup_logging(config, output_dir)
    logging.info(f"使用配置文件: {config_path}")

    df, X, y = load_dataset(config)
    logging.info(
        f"已加载数据 `{config['data']['path']}`，共 {df.shape[0]} 行，{df.shape[1]} 列。"
    )
    logging.info(f"特征数: {X.shape[1]}（按 data.features 配置 / 自动推断）")

    data_for_corr: pd.DataFrame
    if args.include_target:
        y_name = str(config["data"]["target"])
        data_for_corr = pd.concat([X, y.rename(y_name)], axis=1)
        logging.info("已将 target 列纳入相关性矩阵。")
    else:
        data_for_corr = X

    numeric_df, skipped = _select_numeric_columns(data_for_corr)
    if skipped:
        logging.warning(
            f"以下列为非数值类型，将跳过相关性计算（共 {len(skipped)} 列）: {skipped}"
        )

    if numeric_df.shape[1] < 2:
        raise ValueError(
            f"可用于相关性计算的数值列不足 2 列（当前 {numeric_df.shape[1]} 列）。"
        )

    method = str(args.method).lower().strip()
    corr = numeric_df.corr(method=method)
    if args.abs:
        corr = corr.abs()

    corr_out_dir = output_dir / "correlation"
    corr_csv = corr_out_dir / f"feature_correlation_{method}.csv"
    corr_png = corr_out_dir / f"feature_correlation_{method}.png"

    corr_out_dir.mkdir(parents=True, exist_ok=True)
    corr.to_csv(corr_csv, encoding="utf-8-sig")

    title_parts = ["Feature correlation", f"method={method}"]
    if args.abs:
        title_parts.append("abs=1")
    title = " | ".join(title_parts)

    vmin, vmax = (0.0, 1.0) if args.abs else (-1.0, 1.0)
    plot_corr_heatmap(
        corr,
        title=title,
        out_path=corr_png,
        dpi=int(args.dpi),
        annot=bool(args.annot),
        vmin=vmin,
        vmax=vmax,
    )

    logging.info(f"相关性矩阵 CSV: {corr_csv.resolve()}")
    logging.info(f"相关性矩阵热力图: {corr_png.resolve()}")

    run_end = datetime.datetime.now()
    logging.info(
        f"本次运行结束: {run_end}（耗时 {(run_end - run_start).total_seconds():.2f} 秒）"
    )


if __name__ == "__main__":
    main()

