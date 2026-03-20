# Code 目录：程序 / 配置（cfg）输入输出整理

本目录是 **uv 项目根目录**（`Code/pyproject.toml`、`Code/uv.lock`、`Code/.venv/`）。

## 1) 环境与运行入口

在仓库根目录执行也可以，但推荐在 `Code/` 下运行：

```powershell
cd .\Code
uv sync
uv run python .\xgb\xgb_shap.py -c .\xgb\config.yaml
uv run python .\xgb\feature_corr_heatmap.py -c .\xgb\config.yaml
uv run python .\xgb\normalize_feature_table.py -c .\xgb\config.yaml
uv run python .\xgb\xgb_gridcv_tune.py -c .\xgb\config.yaml
uv run python .\gwxgb\gwxgb_shap.py -c .\gwxgb\gwxgb_config.yaml
```

说明：脚本会把 YAML 中的相对路径按 **YAML 文件所在目录** 解析，所以不强依赖 `cd` 到哪里运行。

输出目录约定：`output.output_dir` 作为**基准目录**；当 `output.timestamp_subdir: 1`（默认）时，脚本会在其下创建 `run_prefix + data_stem + timestamp` 子目录（`data_stem` 默认从 `data.path` 推断），并将**本次运行**的所有产出（含 log）写入该子目录。

## 2) 脚本清单与 I/O

### A) `xgb/xgb_shap.py`（全局 XGBoost + SHAP）

- **配置**：`xgb/config.yaml`
- **输入**
  - 数据：`data.path`（CSV）
  - 目标列：`data.target`
  - 特征列：`data.features`（可省略；省略则自动用除 target 外的所有列）
  - 模型参数：`model.params`（传给 `xgboost.XGBRegressor`）
  - 评估（可选）：`cv.use_cv`、`cv.n_splits`、`model.test_size`
  - SHAP（可选）：`shap.use_summary`、`shap.use_dependence`、`shap.compute_interactions` 等
- **输出**（写入 `output.output_dir/<run_prefix>_<data_stem>_<timestamp>/`；可通过 `output.timestamp_subdir: 0` 关闭）
  - SHAP summary：`output.summary_file`（默认 `shap_summary.png`）
  - Mean(|SHAP|)：`output.mean_abs_shap_file`（默认 `mean_abs_shap.png`）
  - Dependence：`output.dependence_prefix + <feature>.png`
  - 交互对 CSV：`output.interaction_pairs_file`
  - 固定基准交互图（可选）：`output.interaction_prefix + <base>_x_<other>.png`
  - 模型文件（可选）：`output.model_file`（若配置则保存 Booster）
  - 日志：`output.log_file`（默认 `run_log.txt`）
  - 说明：默认开启 `output.capture_prints: 1`，将控制台 stdout/stderr 一并写入日志文件，便于复盘。

### B) `xgb/xgb_nestedcv_tune.py`（嵌套交叉验证自动调参）

- **配置**：`xgb/config.yaml`（主要读取 `tuning.*` 与 `model.params`）
- **输入**：同 `xgb_shap.py` 的数据与特征配置
- **输出**：写入 `output.output_dir/<run_prefix>_<data_stem>_<timestamp>/`（日志 `run_log.txt`，并同时输出到控制台）
- **可选**：支持在调参阶段启用 early stopping：`tuning.early_stopping.*`

### C) `xgb/verify_fix.py`（CSV 百分号/字符串数值转换验证）

- **输入**：脚本内置读取 `xgb_shap.load_dataset`
- **输出**：仅控制台输出（dtype 与转换检查）

### D) `xgb/feature_corr_heatmap.py`（特征相关性矩阵 + 热力图）

- **配置**：`xgb/config.yaml`
- **输入**：同 `xgb_shap.py`（`data.path` / `data.features` / `data.target`）
- **输出**（写入 `output.output_dir/<run_prefix>_<data_stem>_<timestamp>/correlation/`）
  - 相关性矩阵 CSV：`feature_correlation_<method>.csv`
  - 热力图 PNG：`feature_correlation_<method>.png`
  - 日志：`output.log_file`（默认 `run_log.txt`）

### E) `xgb/normalize_feature_table.py`（数据预处理：绝对值特征 min-max 归一化）

- **配置**：`xgb/config.yaml` 中的 `preprocess_normalize.*`
- **输入**
  - 数据：`data.path`
  - 特征范围：默认读取 `data.features`；也可用 `preprocess_normalize.include_features` 单独指定
  - 排除名单：`preprocess_normalize.exclude_features`（适合填写已经是比例/比率/人均/单位面积等相对量的特征）
- **输出**（写入 `output.output_dir/<run_prefix>_<data_stem>_<timestamp>/<preprocess_normalize.output_subdir>/`）
  - 归一化后的整表：`normalized_feature_table.csv`
  - 归一化清单：`normalized_feature_manifest.csv`
  - 日志：`output.log_file`（默认 `run_log.txt`）
- **说明**
  - 当前脚本采用按列 `min-max` 归一化：`(x - min) / (max - min)`
  - 保留原表所有列，只替换被归一化的特征列；目标列 `data.target` 不参与归一化
  - 若某个特征是常数列，则该列输出为 0，并在 manifest 中标记为 `constant_zero`

### F) `xgb/xgb_gridcv_tune.py`（单层 GridSearchCV：候选组合排行榜）

- **配置**：`xgb/config.yaml`（读取 `tuning.param_grid` / `tuning.scoring` / `tuning.cv_splits` / `tuning.grid_verbose` / `tuning.early_stopping.*`）
- **输入**：同 `xgb_shap.py`
- **输出**（写入 `output.output_dir/<run_prefix>_<data_stem>_<timestamp>/gridcv/`）
  - 候选组合全量排行榜 CSV：`gridcv_candidates.csv`
  - 日志：`output.log_file`（默认 `run_log.txt`，默认会打印候选排行；可用 `--top` 限制打印条数）
  - 注意：该脚本适合“选参”；best_score 往往偏乐观，不能等价为最终泛化性能

### G) `xgb/xgb_shap_robustness.py`（归因优先的 SHAP 稳健性分析）

- **配置**：`xgb/config.yaml` 中的 `attribution.*`
- **输入**
  - 基础数据：仍读取 `data.path` / `data.target` / `data.features`
  - 额外特征：若 `attribution.categorical_features` 或 `attribution.continuous_features` 中声明了不在 `data.features` 的列，但这些列存在于原始 CSV，则脚本会自动补入本次归因分析（例如 `气候区`）
  - 候选口径：`attribution.candidate_specs`
  - 稳定性评估：`attribution.cv_repeats` / `attribution.cv_splits` / `attribution.cv_seeds` / `attribution.top_k`
  - 正式口径筛选约束：`attribution.baseline_spec` / `attribution.performance_r2_drop_tolerance` / `attribution.performance_rmse_rise_tolerance` / `attribution.spearman_tie_tolerance`
- **输出**（写入 `output.output_dir/<run_prefix>_<data_stem>_<timestamp>/<attribution.output_subdir>/`）
  - 每个 repeat/fold 的验证集指标：`attribution_fold_metrics.csv`
  - 每个 repeat/fold 的验证集 SHAP 重要性：`attribution_fold_shap_importance.csv`
  - 各方案按特征汇总的稳定性结果：`feature_stability_by_spec.csv`
  - 各方案总体稳健性汇总：`robustness_summary.csv`
  - 变换与编码清单：`transform_manifest.csv`
  - 正式推荐口径的特征相对重要性：`official_relative_importance.csv`
  - 推荐说明：`recommendation.txt`
  - 图像输出（默认位于 `<attribution.output_subdir>/<attribution.plots_dir>/`）
    - 每个方案的 SHAP summary 图：`shap_summary_<中文预处理名>.png`
    - 每个方案的 Top 特征相对重要性：`top_features_<中文预处理名>.png`
    - 各方案相对重要性热力图：`spec_comparison_importance_share_heatmap.png`
    - 各方案中位排名热力图：`spec_comparison_rank_heatmap.png`
    - 正式推荐口径的相对重要性 + 稳定性图：`official_relative_importance_and_stability.png`
  - 日志：`output.log_file`（默认 `run_log.txt`）
- **说明**
  - SHAP 只在 **验证 fold** 上计算并汇总，不使用训练 fold 作为正式归因口径
  - One-hot 后的类别变量 SHAP 会按原始特征名回聚合，再参与排序与稳定性评估
  - 跨方案画图默认使用 `median_importance_share`，避免 `raw_y` 与 `log_y` 方案之间的 SHAP 数值单位不可直接比较
  - `shap_summary_<中文预处理名>.png` 基于该方案所有验证 fold 的 SHAP 值与特征矩阵拼接生成；若方案包含类别编码，图中会显示 one-hot 后的列
  - `encoded_cat__zscore_x__raw_y` 与 `encoded_cat__minmax_x__raw_y` 仅作为诊断口径，不建议直接当正式结果

### H) `gwxgb/gwxgb_shap.py`（GeoXGBoost + 带宽优化 + 全局 SHAP）

- **配置**：`gwxgb/gwxgb_config.yaml`
- **输入**
  - 数据：`data.path`（CSV）
  - 坐标列：`data.coords`（长度必须为 2，例如 `["纬度","经度"]`）
  - 目标/特征：`data.target`、`data.features`
  - 全局模型：`model.params`
  - 网格搜索（可选）：`grid_search.enabled=True` + `grid_search.param_grid`（默认关闭，避免运行时间过长）
  - 带宽与空间权重：`gw.*`（供 `geoxgboost.optimize_bw` / `geoxgboost.gxgb`）
  - SHAP：对**全局基线模型**输出图/交互对（不对每个本地模型做 SHAP）
- **输出**（写入 `output.output_dir/<run_prefix>_<data_stem>_<timestamp>/`；可通过 `output.timestamp_subdir: 0` 关闭）
  - 全局 SHAP summary：`output.summary_file`（默认 `gw_shap_summary.png`）
  - Mean(|SHAP|)：`output.mean_abs_shap_file`（默认 `gw_mean_abs_shap.png`）
  - Dependence：`output.dependence_prefix + <feature>.png`
  - 交互对 CSV：`output.interaction_pairs_file`
  - 日志：`output.log_file`（默认 `run_log.txt`）
  - 说明：默认开启 `output.capture_prints: 1`，将控制台 stdout/stderr 一并写入日志文件，便于复盘。
  - GeoXGBoost 带宽优化结果：`BW_results.csv`
  - GeoXGBoost 本地模型结果：`LW_GXGB.xlsx`
  - （可选）局部模型 SHAP（开启 `local_shap.enabled=1`）
    - 每个局部模型的 SHAP summary 图：`<run_output_dir>/<local_shap.output_subdir>/<local_shap.plots_dir>/shap_local_*.png`
    - 汇总 CSV（所有局部模型、每个邻域样本的 SHAP 值）：`<run_output_dir>/<local_shap.output_subdir>/<local_shap.csv_file>`
    - 特征重要性汇总表（每个局部模型一行；每个特征一列；值为 mean(|SHAP|)）：`<run_output_dir>/<local_shap.output_subdir>/<local_shap.feature_importance_file>`

## 3) 现有输出（已归档）

- 历史输出 `Code/BW_results.csv`、`Code/LW_GXGB.xlsx` 已从 `Code/` 移出并归档到：
  - `Output/output_gwxgb/_legacy/20260114_from_Code/`
- 2026-02-26 运行时在仓库根目录生成的同名输出，已从仓库根目录移出并归档到：
  - `Output/output_gwxgb/_legacy/20260226_from_repo_root/`
- 当前脚本默认输出位置为：`Output/output_gwxgb/<gwxgb_数据名_时间戳>/`（不覆盖历史运行）

### I) `gwxgb/compare/gwxgb_compare_*_demo.py`（全局 SHAP vs 局部 SHAP 对比 demo）

- **推荐统一入口**
  - `gwxgb/compare/gwxgb_compare_all.py`
  - 配置：`gwxgb/compare/gwxgb_compare_all_config.yaml`
  - 用途：一次运行同时生成 `center_local`、`local_importance`、`pooled_local` 三类对比结果
  - 输出结构：
    - `<run_output_dir>/center_local/`
    - `<run_output_dir>/local_importance/`
    - `<run_output_dir>/pooled_local/`
    - `<run_output_dir>/comparison_metrics_overview.csv`
    - `<run_output_dir>/comparison_metrics_overview.txt`
- **配置**
  - `gwxgb/compare/gwxgb_compare_center_local_config.yaml`
  - `gwxgb/compare/gwxgb_compare_local_importance_config.yaml`
  - `gwxgb/compare/gwxgb_compare_pooled_local_config.yaml`
  - 3 个 compare cfg 默认都通过 `base_config: "../gwxgb_config.yaml"` 继承主 `gwxgb` 配置
- **用途**
  - `center_local`：每个中心样本只保留一条“该中心局部模型下的 SHAP”，再与全局 SHAP 对比
    - 适用问题：如果每个地理位置只解释自己一次，局部解释与全局解释是否一致
    - 解读重点：这是最适合做主比较的口径，因为一地一条解释，不会重复计数
  - `local_importance`：每个局部模型先汇总为一条 `mean(|SHAP|)` 重要性向量，再看其分布与全局 SHAP 的差异
    - 适用问题：不同位置的局部模型，其重要性结构和方向是否稳定，全局模型是否把空间差异平均掉
    - 解读重点：强度看 `mean(|SHAP|)`，方向看 `mean(SHAP)`；summary 图红蓝颜色表示各局部模型中心点的原始特征值高低
  - `pooled_local`：把所有局部模型邻域样本的 SHAP 行直接拼接后，与全局 SHAP 对比，用于演示“局部 SHAP 全部拼池”的思路及其重复计数偏差
    - 适用问题：如果把所有局部解释直接合并，整体形态会怎样
    - 解读重点：该口径会重复计数同一样本，应结合 `pooled_sample_reuse_counts.csv` 判断偏差程度
- **默认行为**
  - 数据、特征、模型参数、带宽、`local_shap` 口径默认与 `gwxgb/gwxgb_config.yaml` 完全一致
  - compare cfg 仅额外覆盖 `output.output_dir`、`output.run_prefix`、`output.capture_prints` 与 `compare.*`
  - `compare.row_limit: 0` 表示直接使用主配置对应的全量数据
  - 统一入口 `gwxgb_compare_all_config.yaml` 额外提供 `run_center_local`、`run_local_importance`、`run_pooled_local` 三个开关
- **输出**
  - 根目录
    - `global_shap_summary.png`
    - `global_shap_bar.png`
    - `comparison_metrics_overview.csv`
    - `comparison_metrics_overview.txt`
  - `center_local`
    - `global_vs_center_local_shap.csv`
    - `center_local_shap_rows.csv`
    - `center_local_shap_summary.png`
    - `center_local_shap_bar.png`
    - `center_local_comparison_metrics.txt`
  - `local_importance`
    - `global_vs_local_importance_summary.csv`
    - `local_model_importance_wide.csv`
    - `local_model_importance_long.csv`
    - `local_model_importance_summary.png`
      - 红蓝颜色编码使用各局部模型中心点的原始特征值
    - `local_model_importance_bar.png`
    - `local_model_signed_shap_wide.csv`
    - `global_vs_local_signed_shap_summary.csv`
    - `local_model_positive_negative_summary.csv`
    - `local_model_signed_summary.png`
      - 红蓝颜色编码使用各局部模型中心点的原始特征值
    - `local_importance_comparison_metrics.txt`
  - `pooled_local`
    - `global_vs_pooled_local_shap.csv`
    - `pooled_local_shap_rows.csv`
    - `pooled_sample_reuse_counts.csv`
    - `pooled_local_shap_summary.png`
    - `pooled_local_shap_bar.png`
    - `pooled_local_comparison_metrics.txt`
