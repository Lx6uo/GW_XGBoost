# ISML: XGBoost / GeoXGBoost / SHAP Workflow for Spatial Modeling

> A script-first, config-driven repository for global modeling, geographically weighted local modeling, SHAP interpretation, attribution robustness analysis, and global-vs-local SHAP comparison.

本仓库主要面向“城市级 / 区域级空间数据建模与解释”场景，当前实现围绕以下工作流展开：

- 全局 `XGBoost` 回归建模
- `GeoXGBoost` 空间局部建模与带宽控制
- `SHAP` 全局解释、依赖图、交互分析
- 归因优先的 SHAP 稳健性分析
- 全局 SHAP 与局部 SHAP 的多口径对比分析

如果你希望快速复现一个“从数据、训练、解释到结果输出”的脚本化工作流，而不是从 Notebook 临时拼装，这个仓库就是为此组织的。

## Contents

- [Project Snapshot](#project-snapshot)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Workflows](#core-workflows)
- [Global vs Local SHAP Comparison](#global-vs-local-shap-comparison)
- [Configuration Guide](#configuration-guide)
- [Output Convention](#output-convention)
- [Documentation Map](#documentation-map)
- [Practical Notes](#practical-notes)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Third-Party Notices](#third-party-notices)
- [License](#license)

## Project Snapshot

### What this repository does

- Uses YAML configs as the primary interface; scripts are the main entrypoints.
- Keeps `Code/`, `Data/`, and `Output/` physically separated to reduce path chaos.
- Supports both standard XGBoost and geographically weighted local modeling through `geoxgboost`.
- Produces timestamped, non-overwriting output directories for reproducibility.
- Treats SHAP as a first-class output, not an afterthought.
- Includes a dedicated compare suite for testing whether local explanations meaningfully differ from global explanations.

### What this repository is not

- Not a pip-style reusable library with stable import APIs.
- Not a notebook-centric demo collection.
- Not a fully generic AutoML framework.

它更接近“研究型 / 项目型脚本仓库”，强调：

1. 明确的输入输出约定  
2. 可复现的配置驱动运行  
3. 方便追踪的日志与结果归档  

## Repository Layout

```text
ISML/
├─ AGENTS.md                 # 仓库级 agent 入口与任务路由
├─ Code/                     # 核心代码与配置；同时也是 uv 项目根目录
│  ├─ xgb/                   # 全局 XGBoost 工作流
│  ├─ gwxgb/                 # GeoXGBoost 主流程
│  │  ├─ compare/            # 全局 SHAP vs 局部 SHAP 对比套件
│  │  ├─ gwxgb_config.yaml
│  │  └─ gwxgb_shap.py
│  ├─ PROJECT_IO.md          # 脚本与配置 I/O 详细说明
│  ├─ README.md              # Code 目录级说明
│  └─ pyproject.toml         # uv / Python 依赖声明
├─ Data/                     # 输入数据
├─ docs/                     # agent 上下文、机器可读索引、最近状态
├─ Output/                   # 运行输出
├─ Ref/                      # 参考文献与资料
├─ test/                     # Demo 与对照样例
├─ CONTRIBUTING.md           # 协作约定
├─ requirements.txt          # 从 uv 锁文件导出的 pip 依赖
└─ README.md                 # 当前文件
```

### Main code modules

| Module | Purpose | Typical entrypoint |
| --- | --- | --- |
| `Code/xgb/` | 全局 XGBoost、调参、预处理、SHAP、稳健性分析 | `xgb_shap.py` |
| `Code/gwxgb/` | GeoXGBoost 建模、带宽设置、全局 SHAP | `gwxgb_shap.py` |
| `Code/gwxgb/compare/` | 全局 SHAP vs 局部 SHAP 对比分析 | `gwxgb_compare_all.py` |
| `test/` | geoxgboost demo 与参考样例 | `GXGB_call_demo.py` |

## Installation

### Requirements

- Python `>= 3.12`
- 推荐使用 `uv`
- 也可以使用 `pip + requirements.txt`

核心依赖包括：

- `xgboost`
- `shap`
- `geoxgboost`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `scipy`

### Recommended: uv

`Code/` 是当前仓库的 `uv` 项目根目录。

```powershell
cd .\Code
uv sync
```

运行脚本时推荐：

```powershell
uv run python .\xgb\xgb_shap.py -c .\xgb\config.yaml
```

### Alternative: pip

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r .\requirements.txt
python .\Code\xgb\xgb_shap.py -c .\Code\xgb\config.yaml
```

说明：

- `requirements.txt` 由 `uv` 锁文件导出，适合无法使用 `uv` 的环境。
- README 中示例命令以 `PowerShell` 写法为主；若你在 Bash / zsh 中运行，请把 `.\` 改成 `./`。

## Quick Start

### 1. Decide your workflow

最常用的三个入口是：

```powershell
cd .\Code

uv run python .\xgb\xgb_shap.py -c .\xgb\config.yaml
uv run python .\xgb\spearman_corr_heatmap_batch.py -c .\xgb\config.yaml
uv run python .\xgb\xgb_shap_interaction_matrix.py -c .\xgb\config.yaml
uv run python .\gwxgb\gwxgb_shap.py -c .\gwxgb\gwxgb_config.yaml
uv run python .\gwxgb\compare\gwxgb_compare_all.py -c .\gwxgb\compare\gwxgb_compare_all_config.yaml
```

### 2. Update the config before running on your own data

至少检查这些字段：

- `data.path`
- `data.target`
- `data.features`
- `data.coords`（仅 `gwxgb` 需要）
- `output.output_dir`

### 3. Run the script

首次使用建议从全局 XGBoost + SHAP 开始：

```powershell
cd .\Code
uv run python .\xgb\xgb_shap.py -c .\xgb\config.yaml
```

### 4. Inspect the outputs

默认输出是“按时间戳创建的独立目录”，例如：

```text
Output/output_xgb/xgb_<data_name>_<timestamp>/
Output/output_gwxgb/gwxgb_<data_name>_<timestamp>/
Output/output_gwxgb_compare/gwxgb_compare_all_<data_name>_<timestamp>/
```

每次运行通常会生成：

- 图像文件（PNG）
- 表格文件（CSV / XLSX）
- 日志文件（`run_log.txt`）

## Core Workflows

### 1. Global XGBoost + SHAP

Entry:

- [`Code/xgb/xgb_shap.py`](Code/xgb/xgb_shap.py)
- Config: [`Code/xgb/config.yaml`](Code/xgb/config.yaml)

适合：

- 训练一个全局基线模型
- 查看整体特征重要性
- 生成 SHAP summary / dependence / interaction 输出
- 复刻矩阵式 SHAP interaction 图时，可配合 `xgb_shap_interaction_matrix.py`

主要产出：

- `shap_summary.png`
- `mean_abs_shap.png`
- `shap_dependence_<feature>.png`
- `shap_interactions.csv`

相关脚本：

- [`Code/xgb/xgb_shap_interaction_matrix.py`](Code/xgb/xgb_shap_interaction_matrix.py)
  - 用 `interaction_matrix.*` 配置单独生成“上三角热块 + 下三角蜂群”的 SHAP interaction matrix 图
  - 当前默认 `scheme_index: 2` 使用更有热力感、区分度也更稳的 `plasma`；需要更强视觉冲击时仍可切回 `turbo`
  - 上三角热块固定按 `|SHAP interaction|` 的绝对值做橙色顺序渐变，并在右侧单独显示同高度的 `|SHAP interaction|` 色标
  - 下三角蜂群与对应 colorbar 继续按原始特征值着色，但默认改为热力渐变，并放在左侧
  - `colormap_trim_low/high` 控制上三角热块端点裁剪；`scatter_colormap_trim_low/high` 控制下三角热力散点端点裁剪

### 2. Hyperparameter tuning

Entrypoints:

- [`Code/xgb/xgb_nestedcv_tune.py`](Code/xgb/xgb_nestedcv_tune.py)
- [`Code/xgb/xgb_gridcv_tune.py`](Code/xgb/xgb_gridcv_tune.py)

区别：

- `nestedcv_tune` 更适合做更稳健的评估型调参
- `gridcv_tune` 更适合快速筛选候选参数组合

辅助脚本：

- [`Code/xgb/xgb_early_stopping.py`](Code/xgb/xgb_early_stopping.py)  
  用于在 CV / GridSearch 过程中安全地启用 early stopping

### 3. Data preprocessing and diagnostics

Entrypoints:

- [`Code/xgb/normalize_feature_table.py`](Code/xgb/normalize_feature_table.py)
- [`Code/xgb/feature_corr_heatmap.py`](Code/xgb/feature_corr_heatmap.py)
- [`Code/xgb/spearman_corr_heatmap_batch.py`](Code/xgb/spearman_corr_heatmap_batch.py)
- [`Code/xgb/verify_fix.py`](Code/xgb/verify_fix.py)

适合：

- 对绝对量特征做归一化
- 画特征相关性热力图
- 一次性批量输出 2005 / 2010 / 2015 / 2020 的 Spearman 相关性热力图
- 检查 CSV 中带 `%` 或字符串数值的自动转换是否符合预期

### 4. Attribution-oriented SHAP robustness analysis

Entry:

- [`Code/xgb/xgb_shap_robustness.py`](Code/xgb/xgb_shap_robustness.py)

适合：

- 比较不同预处理方案下 SHAP 排名是否稳定
- 单独评估类别变量编码、偏态特征 log 变换、`log_y` 等处理对归因结果的影响
- 从“解释稳定性”而不仅仅是“预测精度”角度选择最终方案

这个脚本是仓库中最偏研究分析的模块之一，也是最值得保留为公开仓库亮点的部分。

### 5. GeoXGBoost + global SHAP

Entry:

- [`Code/gwxgb/gwxgb_shap.py`](Code/gwxgb/gwxgb_shap.py)
- Config: [`Code/gwxgb/gwxgb_config.yaml`](Code/gwxgb/gwxgb_config.yaml)

适合：

- 使用 `GeoXGBoost` 建立空间局部模型
- 控制或优化带宽
- 默认使用 `euclidean` 原始坐标欧氏距离作为 GW 局部采样和空间权重距离口径；如需地表距离可改为 `haversine`
- 输出全局基线模型的 SHAP 解释
- 可选输出局部模型 SHAP

主要产出：

- `BW_results.csv`
- `LW_GXGB.xlsx`
- `gw_shap_summary.png`
- `gw_shap_dependence_<feature>.png`
- `gw_shap_interactions.csv`
- `gw_shap_interaction_top_<rank>_<feature1>_x_<feature2>.png`
- `gw_shap_interaction_<base>_x_<other>.png`
- 可选：`local_shap/` 大量局部 SHAP 输出

### 6. Global vs local SHAP comparison suite

Entry:

- Unified runner: [`Code/gwxgb/compare/gwxgb_compare_all.py`](Code/gwxgb/compare/gwxgb_compare_all.py)
- Config: [`Code/gwxgb/compare/gwxgb_compare_all_config.yaml`](Code/gwxgb/compare/gwxgb_compare_all_config.yaml)

作用：

- 用统一流程同时生成 4 种“全局 SHAP vs 局部 SHAP”比较口径
- 输出总览指标、分口径 CSV 和 SHAP 原生图形

如果你只打算保留一个 compare 入口，建议保留这一套统一入口。

### 7. Curated batch export runner

Entry:

- [`Code/gwxgb/run_curated_output_batch.py`](Code/gwxgb/run_curated_output_batch.py)

适合：

- 一次批量整理 `2005` / `2010` / `2015` / `2020` / `full` 5 套数据输出
- 将 `gwxgb` 结果、全局交互图、interaction matrix、Spearman/Pearson 相关性热力图、`local_shap` 表和 `sample_aggregated_local` 结果按固定目录结构归档
- 为每个数据集额外汇总原始宽表 + 可读版概览/明细表，聚焦 `GeoXGBoost` 主模型与同参 `XGBoost baseline` 的性能对照
- 当单次 `run_root` 下集齐 5 套数据时，额外输出跨数据集 `sample_aggregated_local_shap_mean_value` 桑基图（全量版 + `>9%` 版）
- 默认优先复用最新的完整批处理数据集目录；若没有完整缓存，则对 4 个年份数据回退到复用已有 `Output/output_gwxgb/` 结果
- 命中完整批处理缓存时，批处理现在也会优先用缓存的 interaction array 重画最新的 `shap_interaction_matrix/`；若旧目录还没有该缓存，则会补跑一次 interaction matrix 阶段并写回缓存
- 如需强制重建，可追加 `--force-rebuild`

主要产出：

- `Output/YYMMDD_n/<dataset>/gwxgb_results_and_global_interactions/`
- `Output/YYMMDD_n/<dataset>/shap_interaction_matrix/`
  - 目录内会额外缓存 interaction 重画所需的 `interaction_values.npy` 与 `interaction_test_features.csv`
- `Output/YYMMDD_n/<dataset>/correlation_heatmap/`
- `Output/YYMMDD_n/<dataset>/local_shap_tables/`
- `Output/YYMMDD_n/<dataset>/sample_aggregated_local_shap/`
- `Output/YYMMDD_n/<dataset>/model_metrics_and_hyperparams.csv`
- `Output/YYMMDD_n/<dataset>/model_summary_overview.csv`
- `Output/YYMMDD_n/<dataset>/model_summary_details.csv`
- `Output/YYMMDD_n/<dataset>/model_performance_matrix.csv`
- `Output/YYMMDD_n/<dataset>/model_performance_details.csv`
- `Output/YYMMDD_n/<dataset>/gwxgb_local_diagnostics.csv`
- `Output/YYMMDD_n/<dataset>/holdout_model_benchmark/`
- `Output/YYMMDD_n/<dataset>/reused_source_logs/`（仅当复用已有输出时）
- `Output/YYMMDD_n/batch_model_metrics_and_hyperparams.csv`
- `Output/YYMMDD_n/batch_model_summary_overview.csv`
- `Output/YYMMDD_n/batch_model_summary_details.csv`
- `Output/YYMMDD_n/batch_model_performance_matrix.csv`
- `Output/YYMMDD_n/batch_model_performance_details.csv`
- `Output/YYMMDD_n/batch_gwxgb_local_diagnostics.csv`
- `Output/YYMMDD_n/shap_mean_value_sankey/`（仅当 `run_root` 内同时具备 `2005/2010/2015/2020/2002_2020_full` 5 套 `sample_aggregated_local` 结果时）

### 8. Holdout BW sweep runner

Entry:

- [`Code/gwxgb/run_holdout_bw_sweep.py`](Code/gwxgb/run_holdout_bw_sweep.py)

适合：

- 在统一 `test_size=0.2`、`random_state=42` 的 holdout 划分下，只对 `GW-XGBoost-local` 的 `bw` 做区间扫描
- 保持 `XGBoost` 本身超参数不变，仅观察 `bw` 变化对 `R2 / RMSE / MAE` 的影响
- GW 局部采样距离默认使用 `gw.distance_metric: "euclidean"`，即沿用原始经纬度坐标欧氏距离；如需地表大圆距离可改为 `haversine`
- 将每套数据的 `GW-XGBoost-local` 扫描结果与同一测试集上的 `XGBoost-global` baseline 同表对照
- 为每套数据自动输出三指标曲线图，直观看最优带宽落点与和 baseline 的差距
- 可选用 `--jobs` 启用并行扫描，并在终端显示 `BW tasks` / `Datasets` 两级进度条；受限 Windows 环境下会从多进程自动回退到线程并行；`--no-progress` 可关闭进度条
- 年份截面若将 `--year-bw-max` 设为 `0` 或超过可用训练样本数，脚本会自动按当前 holdout 训练样本数截断上限

主要产出：

- `Output/YYMMDD_n/<dataset>/bw_sweep_results.csv`
- `Output/YYMMDD_n/<dataset>/bw_sweep_best_summary.csv`
- `Output/YYMMDD_n/<dataset>/bw_sweep_local_diagnostics.csv`
- `Output/YYMMDD_n/<dataset>/bw_sweep_metrics.png`
- `Output/YYMMDD_n/batch_bw_sweep_results.csv`
- `Output/YYMMDD_n/batch_bw_sweep_best_summary.csv`
- `Output/YYMMDD_n/batch_bw_sweep_overview.png`
  - 五套数据的总览集合图；每个子图都包含 `XGBoost-global` baseline 横线及其数值标注，图下方附统一评估口径、方法与固定超参数说明

### 9. CV BW sweep runner

Entry:

- [`Code/gwxgb/run_cv_bw_sweep.py`](Code/gwxgb/run_cv_bw_sweep.py)
- [`Code/gwxgb/add_cv_ols_baseline.py`](Code/gwxgb/add_cv_ols_baseline.py)

适合：

- 在统一 `KFold(n_splits=5, shuffle=True, random_state=42)` 口径下扫描 `GW-XGBoost-local` 的 `bw`
- 年份截面默认扫描 `190~280`、步长 `5`；`full` 默认扫描 `3300~3900`、步长 `30`
- 主表和主图使用 5 折验证集指标均值，同时在结果表保留合并 OOF 预测后的整体 `R2 / RMSE / MAE`
- 将 `GW-XGBoost-local` 的 CV mean 与同折 `XGBoost-global` CV mean baseline 同图、同表对照
- 可用 `add_cv_ols_baseline.py --run-root <已有CV输出目录>` 单独计算 OLS baseline，并把 OLS 指标追加到既有 CV 扫描表和总览图，不重跑 GW-XGBoost

主要产出：

- `Output/YYMMDD_n/<dataset>/cv_bw_sweep_results.csv`
- `Output/YYMMDD_n/<dataset>/cv_bw_sweep_best_summary.csv`
- `Output/YYMMDD_n/<dataset>/cv_bw_sweep_local_diagnostics.csv`
- `Output/YYMMDD_n/<dataset>/cv_bw_sweep_fold_details.csv`
- `Output/YYMMDD_n/<dataset>/cv_bw_sweep_metrics.png`
- `Output/YYMMDD_n/batch_cv_bw_sweep_results.csv`
- `Output/YYMMDD_n/batch_cv_bw_sweep_best_summary.csv`
- `Output/YYMMDD_n/batch_cv_bw_sweep_fold_details.csv`
- `Output/YYMMDD_n/batch_cv_bw_sweep_overview.png`
- OLS 增量输出：`cv_ols_baseline.csv`、`cv_ols_fold_details.csv`、`batch_cv_ols_baseline.csv`、`batch_cv_ols_fold_details.csv`

## Global vs Local SHAP Comparison

compare 套件目前有 4 种口径，它们回答的问题不同。

| Mode | Core idea | Best use | Main caveat | Color in summary |
| --- | --- | --- | --- | --- |
| `center_local` | 每个中心位置只保留一条“中心样本在自己的局部模型下”的 SHAP | 作为全局 vs 局部解释的主对比口径 | 只看中心点，不展开邻域内部样本 | 中心样本的原始特征值 |
| `local_importance` | 每个局部模型先汇总为一条向量，再比较强度与方向 | 看空间异质性是否被全局模型平均掉 | 已从样本级解释上升到模型级汇总，方向需单独解读 | 各局部模型中心点的原始特征值 |
| `pooled_local` | 把所有局部邻域样本 SHAP 全部拼池后整体观察 | 演示“所有局部解释合并后”的整体形态 | 会重复计数同一样本 | 每个 pooled 样本的原始特征值 |
| `sample_aggregated_local` | 先保留 pooled 局部 SHAP，再按原始样本聚合成“一样本一条平均 SHAP 向量” | 在保留多次邻域解释信息的同时，避免重复计数放大 | 会把同一样本在不同局部模型下的差异平均掉 | 每个样本自己的原始特征值 |

### Interpreting `local_importance`

`local_importance` 不再只输出强度，还显式保留了方向信息：

- 强度：`mean(|SHAP|)`
- 方向：`mean(SHAP)`
- 正向/负向拆分：
  - `local_model_positive_negative_summary.csv`
  - `local_model_signed_shap_wide.csv`
  - `global_vs_local_signed_shap_summary.csv`

这使它更适合回答：

- 哪些特征在不同空间位置始终重要？
- 哪些特征虽然重要，但方向不稳定？
- 全局模型给出的平均方向，是否和局部模型平均方向一致？

## Configuration Guide

### Config files

| Workflow | Config |
| --- | --- |
| Global XGBoost | [`Code/xgb/config.yaml`](Code/xgb/config.yaml) |
| GeoXGBoost | [`Code/gwxgb/gwxgb_config.yaml`](Code/gwxgb/gwxgb_config.yaml) |
| Compare suite | [`Code/gwxgb/compare/`](Code/gwxgb/compare/) 下各 `gwxgb_compare_*.yaml` |

### Important path behavior

本仓库有一个重要约定：

> YAML 中的相对路径按“YAML 文件所在目录”解析，而不是按当前工作目录解析。

这意味着：

- 你可以从仓库根目录运行脚本
- 也可以从 `Code/` 下运行脚本
- 只要配置文件路径正确，`data.path` 与 `output.output_dir` 都会稳定解析

### Common config keys

| Key | Meaning |
| --- | --- |
| `data.path` | 输入数据 CSV 路径 |
| `data.target` | 目标列名 |
| `data.features` | 特征列列表 |
| `data.coords` | 坐标列，仅 `gwxgb` 需要 |
| `model.params` | `XGBRegressor` 参数 |
| `cv.*` | 全局模型交叉验证设置 |
| `shap.*` | SHAP summary / dependence / interaction 设置 |
| `gw.*` | GeoXGBoost 带宽、核函数、空间权重设置 |
| `local_shap.*` | GeoXGBoost 局部 SHAP 导出控制 |
| `output.*` | 输出目录、日志、文件名前缀等 |

### Inheritance in compare configs

`compare` 目录下的配置默认通过：

```yaml
base_config: "../gwxgb_config.yaml"
```

继承主 `gwxgb` 配置，因此：

- 数据路径
- 特征列
- 目标列
- 坐标列
- 带宽与局部 SHAP 口径

都会与主 `gwxgb` 配置保持同步。

## Output Convention

### Output directories are timestamped by default

仓库默认采用“不覆盖历史运行”的输出策略：

```text
Output/
├─ output_xgb/
│  └─ xgb_<data_name>_<timestamp>/
├─ output_gwxgb/
│  └─ gwxgb_<data_name>_<timestamp>/
└─ output_gwxgb_compare/
   └─ gwxgb_compare_all_<data_name>_<timestamp>/
```

批量整理脚本 `Code/gwxgb/run_curated_output_batch.py` 另外会创建：

```text
Output/
└─ YYMMDD_n/
   ├─ 2005/
   ├─ 2010/
   ├─ 2015/
   ├─ 2020/
   ├─ 2002_2020_full/
   └─ shap_mean_value_sankey/
```

每个运行目录通常包含：

- `run_log.txt`
- PNG 图像
- CSV / XLSX 结果表
- 批处理目录下还会额外包含原始宽表 `model_metrics_and_hyperparams.csv` / `batch_model_metrics_and_hyperparams.csv`
- 以及更适合人工阅读的 `model_summary_overview.csv` / `model_summary_details.csv` / `batch_model_summary_overview.csv` / `batch_model_summary_details.csv`
  - 这组表只保留 `GeoXGBoost` 主模型与“同数据、同超参数”的 `XGBoost baseline` 对照，突出 baseline 的交叉验证均值/标准差与 OOF 指标，不再混入分析阶段性能
- 还会并存输出论文式 holdout benchmark：`model_performance_matrix.csv` / `model_performance_details.csv` / `batch_model_performance_matrix.csv` / `batch_model_performance_details.csv`
  - 这组表使用统一的 `test_size=0.2`、`random_state=42` 划分，对 `XGBoost-global` 与 `GW-XGBoost-local` 在同一测试集上统一计算整体 `R2 / RMSE / MAE`
  - `GW-XGBoost-local` 默认按 `euclidean` 原始坐标欧氏距离选择局部邻域并计算空间权重
- `run_holdout_bw_sweep.py` 会在新的 `Output/YYMMDD_n/` 下输出 `bw_sweep_results.csv` / `bw_sweep_best_summary.csv` / `bw_sweep_local_diagnostics.csv` / `bw_sweep_metrics.png`
  - 年份截面默认扫描 `140~220`、步长 `10`
  - `full` 默认扫描 `2500~4000`、步长 `150`
  - 每张曲线图都同时画出 `R2`、`RMSE`、`MAE` 的 `bw` 变化曲线，并叠加 `XGBoost-global` 的虚线 baseline
- `run_cv_bw_sweep.py` 会在新的 `Output/YYMMDD_n/` 下输出 5 折 CV 口径的 `cv_bw_sweep_results.csv` / `cv_bw_sweep_best_summary.csv` / `cv_bw_sweep_local_diagnostics.csv` / `cv_bw_sweep_fold_details.csv` / `cv_bw_sweep_metrics.png`
  - 年份截面默认扫描 `190~280`、步长 `5`
  - 主性能列为 5 折验证集指标均值；表内同时保留合并 OOF 预测后的整体指标
  - run root 会额外输出 `batch_cv_bw_sweep_overview.png` 总览图
- `add_cv_ols_baseline.py` 可对已有 CV BW 输出单独追加 OLS baseline，并更新 `cv_bw_sweep_results.csv`、`cv_bw_sweep_best_summary.csv` 和总览图
- 某些工作流下的子目录（如 `local_shap/`、`center_local/`、`local_importance/`）

### Logging

大部分主脚本默认开启：

```yaml
output:
  capture_prints: 1
```

这会把 `stdout` / `stderr` 一并写入日志，方便追踪 geoxgboost 或 shap 在控制台打印的运行信息。

例外：

- compare 统一入口配置默认使用 `output.capture_prints: 0`
- `run_curated_output_batch.py` 内部也会按阶段显式关闭捕获并写独立日志文件

## Documentation Map

如果你想进一步读细节，而不是只看首页 README，请按下面顺序进入：

1. [`AGENTS.md`](AGENTS.md)
   适合新会话快速建立任务路由，避免重新扫完整仓库。

2. [`docs/AGENT_CONTEXT.md`](docs/AGENT_CONTEXT.md)
   适合快速看当前仓库状态、约束和阅读优先级。

3. [`docs/repo_index.yaml`](docs/repo_index.yaml)
   适合 agent 读取机器可读的入口、输出和约束索引。

4. [`docs/LAST_STATE.md`](docs/LAST_STATE.md)
   适合作为跨会话最近状态和最近一次文档同步的检查点。

5. [`Code/README.md`](Code/README.md)
   适合从“脚本入口视角”快速浏览代码层工作流。

6. [`Code/PROJECT_IO.md`](Code/PROJECT_IO.md)
   适合查每个脚本的输入、配置、输出文件结构。

7. [`Code/gwxgb/compare/README.md`](Code/gwxgb/compare/README.md)
   适合单独查看 compare 套件的组织方式。

8. [`CONTRIBUTING.md`](CONTRIBUTING.md)
   适合协作者了解目录约定、路径约定、输出约定。

## Practical Notes

### 1. The default configs are dataset-specific

当前仓库默认配置是围绕“地级市 / 年份建筑碳排放”类数据组织的，列名和默认路径也带有明显的项目语义。  
如果你将其复用于其他数据，请优先修改：

- `data.path`
- `data.target`
- `data.features`
- `data.coords`

### 2. Chinese column names are supported

当前脚本与配置完全支持中文列名、中文文件名。  
这对中文数据项目友好，但也意味着：

- 如果你在英文数据集上复用，请显式更新 YAML
- 某些图形在缺少 CJK 字体的系统上，可能需要额外配置字体

### 3. GeoXGBoost runs can be slow

当你启用以下内容时，运行时间会显著增加：

- 带宽优化
- `grid_search.enabled`
- `local_shap.enabled`
- compare 套件中的全量局部 SHAP 采集

建议调试时先用较小数据或限制：

- `compare.row_limit`
- `local_shap.max_models`
- `shap.compute_interactions`

### 4. Compare suite is now isolated in its own folder

为了让项目结构更清晰，compare 相关文件已集中到：

- [`Code/gwxgb/compare/`](Code/gwxgb/compare/)

这样 `Code/gwxgb/` 只保留 GeoXGBoost 主流程，而 compare 套件作为独立子模块存在。

### 5. This repository is script-first

当前仓库的核心接口是脚本和 YAML，不是稳定的 Python package API。  
如果你要二次开发，建议把它当成“研究脚本仓库”而不是“库”来使用。

## Contributing

协作约定见：

- [`CONTRIBUTING.md`](CONTRIBUTING.md)

如果你准备继续扩展这个仓库，推荐遵守以下原则：

- 路径尽量写相对路径
- 输入放在 `Data/`
- 输出放在 `Output/`
- 配置驱动优先于硬编码路径
- 新增工作流优先增加独立配置文件
- 入口、输出或约束变更时，同步更新 `AGENTS.md`、`docs/repo_index.yaml` 与 `docs/LAST_STATE.md`

## Acknowledgements

This repository builds on the following core tools and ideas:

- `XGBoost`
- `SHAP`
- `GeoXGBoost`
- `scikit-learn`
- `pandas` / `numpy` / `matplotlib`

仓库中的 `Ref/` 目录也用于存放项目相关的论文和参考材料。

## Third-Party Notices

本仓库在运行时依赖多个第三方开源包，包括但不限于：

- `SHAP`
- `XGBoost`
- `GeoXGBoost`
- `scikit-learn`
- `pandas` / `numpy` / `matplotlib` / `scipy`

此外，仓库还保留了一个第三方 demo 子目录：

- [`test/DemoGXGBoost/`](test/DemoGXGBoost/)

第三方依赖与第三方材料的许可说明已集中写入：

- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)

如果你打算复用仓库中的 demo 数据、示例脚本、教程材料或未来新增的第三方附件，请先检查对应子目录中的许可文件与来源说明。

## License

本仓库的原创代码与原创文档采用 [`MIT License`](LICENSE) 发布。

选择 `MIT` 作为本项目许可证的原因是：

- 对科研论文配套代码更常见、简洁、易理解
- 允许他人复现、引用、修改和再利用，复用门槛低
- 与当前使用的主要依赖许可证兼容，包括 `MIT`、`BSD` 和 `Apache-2.0`

说明：

- 根目录 [`LICENSE`](LICENSE) 适用于本仓库的原创内容，除非某个子目录或文件另有单独说明
- 第三方代码、第三方 demo、数据、教程和其他材料仍受其各自许可证约束
- 第三方许可信息见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
