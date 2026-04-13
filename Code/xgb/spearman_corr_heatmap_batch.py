from __future__ import annotations

import argparse
import copy
import datetime
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xgb_shap import load_config, load_dataset, setup_logging


DEFAULT_YEARS = ["2005", "2010", "2015", "2020"]

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False


def _pick_installed_font(candidates: List[str]) -> str | None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


LATIN_FONT_NAME = _pick_installed_font(
    ["Times New Roman", "Cambria", "Georgia", "DejaVu Serif"]
) or "DejaVu Serif"
CJK_FONT_NAME = _pick_installed_font(
    [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "NSimSun",
        "KaiTi",
        "FangSong",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
)

plt.rcParams["font.family"] = LATIN_FONT_NAME
plt.rcParams["font.serif"] = [LATIN_FONT_NAME, "DejaVu Serif"]
if CJK_FONT_NAME:
    plt.rcParams["font.sans-serif"] = [CJK_FONT_NAME, "DejaVu Sans"]


def _contains_cjk(text: Any) -> bool:
    for ch in str(text):
        code = ord(ch)
        if (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
            or 0xFF00 <= code <= 0xFFEF
        ):
            return True
    return False


def _font_properties(
    text: Any,
    *,
    size: float | int | None = None,
    bold: bool = False,
) -> FontProperties:
    family = CJK_FONT_NAME if (_contains_cjk(text) and CJK_FONT_NAME) else LATIN_FONT_NAME
    return FontProperties(
        family=family,
        size=size,
        weight="bold" if bold else "normal",
    )


def _normalize_path_value(path_value: Any) -> str:
    text = str(path_value).strip()
    if (
        len(text) >= 3
        and text[0].lower() == "r"
        and text[1] in {'"', "'"}
        and text[-1] == text[1]
    ):
        return text[2:-1]
    if len(text) >= 2 and text[0] in {'"', "'"} and text[-1] == text[0]:
        return text[1:-1]
    return text


def _config_base_dir(config: Dict[str, Any]) -> Path:
    cfg_dir = config.get("_config_dir")
    if isinstance(cfg_dir, str) and cfg_dir:
        return Path(cfg_dir)
    return Path.cwd()


def _resolve_path(path_value: Any, base_dir: Path) -> Path:
    path = Path(_normalize_path_value(path_value)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "批量读取 2005/2010/2015/2020 年数据，输出 Spearman 相关性矩阵 "
            "CSV 与热力图。"
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
    parser.add_argument(
        "--years",
        nargs="+",
        default=list(DEFAULT_YEARS),
        help="要批量处理的年份列表，默认：2005 2010 2015 2020",
    )
    parser.add_argument(
        "--include-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否把 target 列也纳入 Spearman 相关性矩阵，默认不包含。",
    )
    parser.add_argument(
        "--annot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在热力图格子里写数值，默认开启。",
    )
    parser.add_argument(
        "--abs",
        dest="absolute_values",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否输出绝对相关系数（0~1），默认保留符号（-1~1）。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="输出 PNG 的 dpi，默认 300。",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="coolwarm",
        help="热力图配色，默认 coolwarm。",
    )
    return parser.parse_args()


def _year_token(year: str) -> str:
    year_text = str(year).strip()
    if not re.fullmatch(r"\d{4}", year_text):
        raise ValueError(f"年份 `{year}` 不是 4 位数字。")
    return year_text


def discover_dataset_paths(
    config: Dict[str, Any],
    years: Sequence[str],
) -> List[tuple[str, Path]]:
    data_cfg = config.get("data") or {}
    if "path" not in data_cfg:
        raise KeyError("配置缺少 `data.path`。")

    base_path = _resolve_path(data_cfg["path"], _config_base_dir(config))
    match = re.match(r"^(.*?)(\d{4})$", base_path.stem)
    if not match:
        raise ValueError(
            "无法从 `data.path` 推断年份后缀；期望文件名形如 `xxx_2005.csv`。"
        )
    stem_prefix = match.group(1)
    suffix = base_path.suffix

    discovered: List[tuple[str, Path]] = []
    missing: List[Path] = []
    for raw_year in years:
        year = _year_token(raw_year)
        candidate = base_path.with_name(f"{stem_prefix}{year}{suffix}")
        if candidate.exists():
            discovered.append((year, candidate))
        else:
            missing.append(candidate)

    if missing:
        missing_lines = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"以下年份数据未找到：\n{missing_lines}")

    return discovered


def _build_batch_output_dir(config: Dict[str, Any]) -> Path:
    output_cfg = config.get("output") or {}
    base_output_dir = _resolve_path(
        output_cfg.get("output_dir", "../../Output/output_xgb"),
        _config_base_dir(config),
    )
    timestamp_format = str(output_cfg.get("timestamp_format", "%Y%m%d_%H%M%S"))
    timestamp = datetime.datetime.now().strftime(timestamp_format)
    batch_dir = base_output_dir / f"spearman_batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    return batch_dir


def _select_numeric_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    numeric_df = df.select_dtypes(include=[np.number])
    skipped = [col for col in df.columns if col not in numeric_df.columns]
    return numeric_df, skipped


def _apply_tick_fonts(ax: plt.Axes, tick_fontsize: float) -> None:
    for label in ax.get_xticklabels():
        label.set_fontproperties(
            _font_properties(label.get_text(), size=tick_fontsize)
        )
    for label in ax.get_yticklabels():
        label.set_fontproperties(
            _font_properties(label.get_text(), size=tick_fontsize)
        )


def plot_corr_heatmap(
    corr: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    dpi: int,
    annot: bool,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    n = int(corr.shape[0])
    if n == 0:
        raise ValueError("相关性矩阵为空，无法绘图。")

    figure_size = max(7.5, n * 1.0)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    values = corr.to_numpy(dtype=float)

    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns.tolist(), rotation=45, ha="right")
    ax.set_yticklabels(corr.index.tolist())
    _apply_tick_fonts(ax, tick_fontsize=10)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    title_obj = ax.set_title(title, fontsize=15, pad=12)
    title_obj.set_fontproperties(_font_properties(title, size=15, bold=True))

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar_label = "Absolute Spearman correlation" if vmin >= 0 else "Spearman correlation"
    colorbar.set_label(colorbar_label, fontproperties=_font_properties(colorbar_label, size=11))
    for label in colorbar.ax.get_yticklabels():
        label.set_fontproperties(_font_properties(label.get_text(), size=9))

    if annot:
        for row_idx in range(n):
            for col_idx in range(n):
                value = values[row_idx, col_idx]
                text_color = "white" if abs(value) >= 0.55 else "#222222"
                cell_text = ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )
                cell_text.set_fontproperties(_font_properties(cell_text.get_text(), size=8))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def compute_spearman_corr(
    config: Dict[str, Any],
    dataset_path: Path,
    *,
    include_target: bool,
) -> tuple[pd.DataFrame, list[str], int]:
    dataset_config = copy.deepcopy(config)
    dataset_config["data"]["path"] = str(dataset_path)

    df, X, y = load_dataset(dataset_config)
    if include_target:
        y_name = str(dataset_config["data"]["target"])
        source_df = pd.concat([X, y.rename(y_name)], axis=1)
    else:
        source_df = X

    numeric_df, skipped = _select_numeric_columns(source_df)
    if numeric_df.shape[1] < 2:
        raise ValueError(
            f"`{dataset_path.name}` 可用于相关性计算的数值列不足 2 列。"
        )

    corr = numeric_df.corr(method="spearman")
    return corr, skipped, int(df.shape[0])


def _manifest_record(
    *,
    year: str,
    dataset_path: Path,
    rows: int,
    corr: pd.DataFrame,
    skipped: Iterable[str],
    csv_path: Path,
    png_path: Path,
) -> Dict[str, Any]:
    return {
        "year": year,
        "dataset_path": str(dataset_path),
        "n_rows": rows,
        "n_variables": int(corr.shape[0]),
        "variables": " | ".join(corr.columns.tolist()),
        "skipped_non_numeric": " | ".join(str(item) for item in skipped),
        "corr_csv": str(csv_path),
        "heatmap_png": str(png_path),
    }


def main() -> None:
    args = parse_args()
    config_path = (
        Path(args.config) if args.config else Path(__file__).with_name("config.yaml")
    )
    config = load_config(config_path)

    batch_output_dir = _build_batch_output_dir(config)
    log_config = copy.deepcopy(config)
    log_config.setdefault("output", {})
    log_config["output"]["output_dir"] = str(batch_output_dir)
    setup_logging(log_config, batch_output_dir)

    logging.info(f"使用配置文件: {config_path.resolve()}")
    logging.info("Spearman 批量相关性分析开始。")

    dataset_specs = discover_dataset_paths(config, args.years)
    logging.info(
        "待处理年份: %s",
        ", ".join(year for year, _ in dataset_specs),
    )

    manifest_rows: List[Dict[str, Any]] = []
    absolute_values = bool(args.absolute_values)

    for year, dataset_path in dataset_specs:
        logging.info("-" * 60)
        logging.info("开始处理 %s: %s", year, dataset_path)

        corr, skipped, n_rows = compute_spearman_corr(
            config,
            dataset_path,
            include_target=bool(args.include_target),
        )
        if absolute_values:
            corr = corr.abs()

        year_output_dir = batch_output_dir / year
        year_output_dir.mkdir(parents=True, exist_ok=True)
        corr_csv = year_output_dir / f"spearman_correlation_{year}.csv"
        corr_png = year_output_dir / f"spearman_correlation_{year}.png"
        corr.to_csv(corr_csv, encoding="utf-8-sig")

        title_parts = [f"{year} Spearman Correlation Heatmap"]
        if args.include_target:
            title_parts.append("with target")
        if absolute_values:
            title_parts.append("abs")
        title = " | ".join(title_parts)

        plot_corr_heatmap(
            corr,
            title=title,
            out_path=corr_png,
            dpi=int(args.dpi),
            annot=bool(args.annot),
            cmap=str(args.cmap),
            vmin=0.0 if absolute_values else -1.0,
            vmax=1.0,
        )

        if skipped:
            logging.warning(
                "%s 存在非数值列未参与相关性计算: %s",
                year,
                skipped,
            )

        logging.info("%s 相关矩阵 CSV: %s", year, corr_csv.resolve())
        logging.info("%s 热力图 PNG: %s", year, corr_png.resolve())

        manifest_rows.append(
            _manifest_record(
                year=year,
                dataset_path=dataset_path,
                rows=n_rows,
                corr=corr,
                skipped=skipped,
                csv_path=corr_csv.resolve(),
                png_path=corr_png.resolve(),
            )
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = batch_output_dir / "spearman_heatmap_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    logging.info("批量结果清单: %s", manifest_path.resolve())
    logging.info("Spearman 批量相关性分析完成。")


if __name__ == "__main__":
    main()
