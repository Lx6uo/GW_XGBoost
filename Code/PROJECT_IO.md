# Code 目录：程序 / 配置（cfg）输入输出整理

本目录是 **uv 项目根目录**（`Code/pyproject.toml`、`Code/uv.lock`、`Code/.venv/`）。

## 1) 环境与运行入口

在仓库根目录执行也可以，但推荐在 `Code/` 下运行：

```powershell
cd .\Code
uv sync
uv run python .\xgb\xgb_shap.py -c .\xgb\config.yaml
uv run python .\xgb\feature_corr_heatmap.py -c .\xgb\config.yaml
uv run python .\xgb\spearman_corr_heatmap_batch.py -c .\xgb\config.yaml
uv run python .\xgb\normalize_feature_table.py -c .\xgb\config.yaml
uv run python .\xgb\xgb_gridcv_tune.py -c .\xgb\config.yaml
uv run python .\gwxgb\gwxgb_shap.py -c .\gwxgb\gwxgb_config.yaml
uv run python .\gwxgb\run_curated_output_batch.py
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
  - Top 交互作用单独图：`output.interaction_prefix + top_<rank>_<feature1>_x_<feature2>.png`
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

### H) `xgb/xgb_shap_interaction_matrix.py`（SHAP interaction matrix 复刻图）

- **配置**：`xgb/config.yaml` 中的 `interaction_matrix.*`
- **输入**
  - 数据：仍读取 `data.path` / `data.target` / `data.features`
  - 模型基础参数：`model.params`
  - 专用训练/选参：`interaction_matrix.test_size`、`interaction_matrix.use_grid_search`、`interaction_matrix.param_grid`
  - 绘图样式：`interaction_matrix.scheme_index`、`interaction_matrix.style_index`
    - 当前默认 `scheme_index: 2` 为更有热力感、区分度也更稳的 `plasma`
    - 备选：`1=turbo`、`3=viridis`、`4=cividis`
    - `scheme_index` 当前只控制下三角蜂群与左侧 raw feature value colorbar；上三角热块固定按 `|SHAP interaction|` 的绝对值做橙色顺序渐变，并在右侧单独显示同高度的 `Low/High` 色标
    - `interaction_matrix.colormap_trim_low/high` 控制上三角热块端点裁剪
    - `interaction_matrix.scatter_colormap_trim_low/high` 控制下三角热力散点端点裁剪
- **输出**（写入 `output.output_dir/<run_prefix>_<data_stem>_<timestamp>/<interaction_matrix.output_subdir>/`）
  - PNG：`<interaction_matrix.output_stem>_<scheme_index>_<style_index>.png`
  - 日志：`output.log_file`（默认 `run_log.txt`）

### D-2) `xgb/spearman_corr_heatmap_batch.py`（4 个年份数据的批量 Spearman 热力图）

- **配置**：`xgb/config.yaml`
- **输入**
  - 基准数据路径：`data.path`
  - 特征列：`data.features`
  - 目标列：`data.target`
  - 年份列表：命令行 `--years`（默认 `2005 2010 2015 2020`）
- **行为**
  - 根据 `data.path` 自动识别文件名前缀，并在同目录下查找 `*_2005.csv`、`*_2010.csv`、`*_2015.csv`、`*_2020.csv`
  - 对每个年份按 `data.features` 计算 Spearman 相关性矩阵
  - 批量输出 CSV 与热力图 PNG，并汇总一份 manifest
- **输出**（写入 `output.output_dir/spearman_batch_<timestamp>/`）
  - 每个年份一个子目录：`<year>/spearman_correlation_<year>.csv`
  - 每个年份一个热力图：`<year>/spearman_correlation_<year>.png`
  - 批量输出清单：`spearman_heatmap_manifest.csv`
  - 日志：`output.log_file`（默认 `run_log.txt`）
- **说明**
  - 该脚本会单独执行一次 `train_test_split + GridSearchCV`，再在测试集上计算 `SHAP interaction values`
  - 图形结构为“上三角热块 + 下三角蜂群”，用于复刻参考图排版
  - 上三角色深使用 `abs(mean(interaction))`；数字标注使用带符号的 `mean(interaction)`
  - 下三角颜色编码使用对应列特征的原始取值

### I) `gwxgb/gwxgb_shap.py`（GeoXGBoost + 带宽优化 + 全局 SHAP）

- **配置**：`gwxgb/gwxgb_config.yaml`
- **输入**
  - 数据：`data.path`（CSV）
  - 坐标列：`data.coords`（长度必须为 2，例如 `["纬度","经度"]`）
  - 目标/特征：`data.target`、`data.features`
  - 全局模型：`model.params`
  - 网格搜索（可选）：`grid_search.enabled=True` + `grid_search.param_grid`（默认关闭，避免运行时间过长）
  - 带宽与空间权重：`gw.*`（供 `geoxgboost.optimize_bw` / `geoxgboost.gxgb`）
  - 距离口径：`gw.distance_metric` 默认为 `euclidean`，沿用原始坐标列的欧氏距离；如需地表大圆距离可设为 `haversine`，此时 `gw.coord_order` 支持 `auto / lat_lon / lon_lat`
  - SHAP：对**全局基线模型**输出图/交互对（不对每个本地模型做 SHAP）
- **输出**（写入 `output.output_dir/<run_prefix>_<data_stem>_<timestamp>/`；可通过 `output.timestamp_subdir: 0` 关闭）
  - 全局 SHAP summary：`output.summary_file`（默认 `gw_shap_summary.png`）
  - Mean(|SHAP|)：`output.mean_abs_shap_file`（默认 `gw_mean_abs_shap.png`）
  - Dependence：`output.dependence_prefix + <feature>.png`
  - 交互对 CSV：`output.interaction_pairs_file`
  - Top 交互作用单独图：`output.interaction_prefix + top_<rank>_<feature1>_x_<feature2>.png`
  - 基准特征交互图：`output.interaction_prefix + <base>_x_<other>.png`
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

### J) `gwxgb/compare/gwxgb_compare_*_demo.py`（全局 SHAP vs 局部 SHAP 对比 demo）

- **推荐统一入口**
  - `gwxgb/compare/gwxgb_compare_all.py`
  - 配置：`gwxgb/compare/gwxgb_compare_all_config.yaml`
  - 用途：一次运行同时生成 `center_local`、`local_importance`、`pooled_local`、`sample_aggregated_local` 四类对比结果
  - 输出结构：
    - `<run_output_dir>/center_local/`
    - `<run_output_dir>/local_importance/`
    - `<run_output_dir>/pooled_local/`
    - `<run_output_dir>/sample_aggregated_local/`
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
  - `sample_aggregated_local`：先保留所有 pooled 局部 SHAP 行，再按原始样本聚合成“一样本一条平均 SHAP 向量”
    - 适用问题：如果一个样本被多个局部模型重复解释，平均来看它在局部世界里是如何被解释的
    - 解读重点：该口径保留了“多次进入邻域”的信息，但避免了 `pooled_local` 的重复计数放大，更适合与全局 SHAP 直接比较
- **默认行为**
  - 数据、特征、模型参数、带宽、`local_shap` 口径默认与 `gwxgb/gwxgb_config.yaml` 完全一致
  - compare cfg 仅额外覆盖 `output.output_dir`、`output.run_prefix`、`output.capture_prints` 与 `compare.*`
  - `compare.row_limit: 0` 表示直接使用主配置对应的全量数据
  - 统一入口 `gwxgb_compare_all_config.yaml` 额外提供 `run_center_local`、`run_local_importance`、`run_pooled_local`、`run_sample_aggregated_local` 四个开关
- **输出**
  - 根目录
    - `global_shap_summary.png`
    - `global_shap_bar.png`
    - `output.interaction_pairs_file`（默认继承为 `gw_shap_interactions.csv`）
    - `output.interaction_prefix + top_<rank>_<feature1>_x_<feature2>.png`
    - `output.interaction_prefix + <base>_x_<other>.png`
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
  - `sample_aggregated_local`
    - `global_vs_sample_aggregated_local_shap.csv`
    - `sample_aggregated_local_shap_rows.csv`
    - `sample_aggregated_local_summary.png`
    - `sample_aggregated_local_shap_sum.png`（保留 y 轴特征名；上轴叠加 `Mean |SHAP|` 柱）
    - `sample_aggregated_local_bar.png`
    - `sample_aggregated_local_shap_mean_value.csv`
    - `sample_aggregated_local_dependence_<feature>.png`
    - `sample_aggregated_local_comparison_metrics.txt`

### K) `gwxgb/run_curated_output_batch.py`（5 套数据的批量整理输出）

- **配置来源**
  - 主体会读取并重写：
    - `gwxgb/gwxgb_config.yaml`
    - `gwxgb/compare/gwxgb_compare_all_config.yaml`
    - `xgb/config.yaml`
- **命令行参数**
  - `--run-root`：显式指定总输出目录；不填则自动创建 `Output/YYMMDD_n/`
  - `--datasets`：可选 `2005 2010 2015 2020 full`
  - `--data-root`：若提供，则从该目录下读取 5 个标准命名 CSV
  - `--force-rebuild`：忽略已有批处理缓存并强制重建；默认关闭
- **默认输入**
  - `Data/年份终市级指标数据（5年截面）/终市级指标数据_with_latlon_2005.csv`
  - `Data/年份终市级指标数据（5年截面）/终市级指标数据_with_latlon_2010.csv`
  - `Data/年份终市级指标数据（5年截面）/终市级指标数据_with_latlon_2015.csv`
  - `Data/年份终市级指标数据（5年截面）/终市级指标数据_with_latlon_2020.csv`
  - `Data/终市级指标数据_with_latlon.csv`
- **行为**
  - 每个数据集会生成一个目录：`2005/`、`2010/`、`2015/`、`2020/`、`2002_2020_full/`
  - 默认优先复用最新的完整批处理数据集目录；若命中完整缓存，则不重新训练 `gwxgb` 主流程，并会基于缓存表刷新 `sample_aggregated_local`、相关性热力图与 interaction matrix
  - interaction matrix 刷新会优先使用 `shap_interaction_matrix/` 中缓存的 `interaction_values.npy` 与 `interaction_test_features.csv` 直接重画；只有旧目录缺少这两份缓存时，才会补跑一次 interaction matrix 阶段并写回缓存
  - 对 `2005/2010/2015/2020`，脚本会优先复用最新的 `Output/output_gwxgb/gwxgb_终市级指标数据_with_latlon_<year>_*` 结果
  - 即使复用既有 `gwxgb` 结果，也会重新生成：
    - `sample_aggregated_local` 输出
    - SHAP interaction matrix 图
    - Spearman/Pearson 相关矩阵 CSV 与热力图 PNG
  - 每个数据集根目录都会额外生成 `model_metrics_and_hyperparams.csv`，统一汇总：
    - `LW_GXGB.xlsx` 中的 `gwxgb` 指标
    - 最终 `XGBoost` 超参数与地理加权参数（如 `Bandwidth` / `Kernel` / `Spatial Weights`）
    - 以及“同数据、同超参数”的 `XGBoost baseline` KFold CV 均值/标准差与 OOF 指标
  - 同时还会额外生成两张更适合人工阅读的表：
    - `model_summary_overview.csv`：按数据集一行，直接对照 `GeoXGBoost` 与 `XGBoost baseline`
    - `model_summary_details.csv`：按条目纵向展开，便于筛选/排序
  - 同时还会并存生成论文式 holdout benchmark：
    - `model_performance_matrix.csv`：统一 `test_size=0.2`、`random_state=42`，在同一测试集上对照 `XGBoost-global` 与 `GW-XGBoost-local`
    - `model_performance_details.csv`：记录样本划分、XGBoost 参数、地理加权设定与局部诊断汇总
    - `gwxgb_local_diagnostics.csv`：逐测试样本记录局部邻域规模、空间权重摘要、局部 top feature 与预测误差
  - 当同一个 `run_root` 内已同时具备 `2005/2010/2015/2020/2002_2020_full` 的 `sample_aggregated_local` 结果时，脚本末尾会额外生成跨数据集 SHAP mean value 桑基图目录 `shap_mean_value_sankey/`
  - 若 `2002_2020_full` 缺少 `sample_aggregated_local_shap_mean_value.csv`，但保留了 `sample_aggregated_local_shap_rows.csv`，脚本会先自动重建该 mean value 表，再画桑基图
- **输出结构**（写入 `Output/YYMMDD_n/<dataset>/` 或 `--run-root/<dataset>/`）
  - `model_metrics_and_hyperparams.csv`
    - 每个数据集 1 行
    - 列前缀 `gwxgb_*` 对应 GeoXGBoost 主流程
    - 列前缀 `xgb_baseline_*` 对应“同数据、同超参数”的 XGBoost baseline
  - `model_summary_overview.csv`
    - 每个数据集 1 行
    - 并排展示 `GeoXGBoost` OOB 指标、`XGBoost baseline` 的 KFold CV 均值/标准差、OOF 指标和两者差值
  - `model_summary_details.csv`
    - 纵向明细表
    - 每行 1 个指标或 1 个参数，附带阶段、分组、备注与来源文件
    - 仅保留 `GeoXGBoost 主模型` 与 `XGBoost baseline` 两类主模型对照，不纳入 interaction / correlation 等分析阶段性能
  - `model_performance_matrix.csv`
    - 每个数据集 2 行
    - 分别对应 `XGBoost-global` 与 `GW-XGBoost-local`
    - 基于统一 holdout 测试集整体计算 `R2 / RMSE / MAE`
  - `model_performance_details.csv`
    - 纵向明细表
    - 记录样本划分、地理加权设定、XGBoost 参数、两类模型主性能、局部诊断汇总
  - `gwxgb_local_diagnostics.csv`
    - 每个测试样本 1 行
    - 记录局部训练样本数、距离/权重摘要、local top feature、预测值与误差
  - `gwxgb_results_and_global_interactions/`
    - `BW_results.csv`
    - `LW_GXGB.xlsx`
    - `gw_shap_interactions.csv`
    - `gw_shap_interaction_top_*.png`
  - `holdout_model_benchmark/`
    - `BW_results.csv`（仅当 holdout benchmark 启用带宽优化时生成）
  - `local_shap_tables/`
    - `local_shap_values.csv`
    - `local_feature_importance_wide.csv`
  - `sample_aggregated_local_shap/`
    - `sample_aggregated_local_shap_rows.csv`
    - `sample_aggregated_local_summary.png`
    - `sample_aggregated_local_shap_sum.png`
    - `sample_aggregated_local_bar.png`
    - `sample_aggregated_local_shap_mean_value.csv`
    - `sample_aggregated_local_dependence_<feature>.png`
  - `shap_interaction_matrix/`
    - `<interaction_matrix.output_stem>_<scheme_index>_<style_index>.png`
    - `interaction_values.npy`
    - `interaction_test_features.csv`
  - `correlation_heatmap/`
    - `spearman_correlation_<dataset>.csv`
    - `spearman_correlation_<dataset>.png`
    - `pearson_correlation_<dataset>.csv`
    - `pearson_correlation_<dataset>.png`
    - `correlation_heatmap_manifest.csv`
  - `reused_source_logs/`
    - `reuse_run_log.txt`（仅在复用既有输出时生成）
- **run_root 级附加输出**（写入 `Output/YYMMDD_n/` 或 `--run-root/`）
  - `batch_model_metrics_and_hyperparams.csv`
    - 汇总本次批处理中所有已生成数据集的 `model_metrics_and_hyperparams.csv`
  - `batch_model_summary_overview.csv`
    - 汇总所有数据集的 `model_summary_overview.csv`
  - `batch_model_summary_details.csv`
    - 汇总所有数据集的 `model_summary_details.csv`
  - `batch_model_performance_matrix.csv`
    - 汇总所有数据集的 `model_performance_matrix.csv`
  - `batch_model_performance_details.csv`
    - 汇总所有数据集的 `model_performance_details.csv`
  - `batch_gwxgb_local_diagnostics.csv`
    - 汇总所有数据集的 `gwxgb_local_diagnostics.csv`
  - `shap_mean_value_sankey/`
    - `sample_aggregated_local_shap_mean_value_sankey.png`
    - `sample_aggregated_local_shap_mean_value_sankey.svg`
    - `sample_aggregated_local_shap_mean_value_sankey_gt9.png`
    - `sample_aggregated_local_shap_mean_value_sankey_gt9.svg`
    - `shap_mean_value_sankey_data.csv`
    - `shap_mean_value_sankey_gt9_data.csv`

### L) `gwxgb/run_holdout_bw_sweep.py`（统一 holdout 口径下的 BW 扫描寻优）

- **配置来源**
  - 主体会读取并重写：
    - `gwxgb/gwxgb_config.yaml`
- **命令行参数**
  - `--run-root`：显式指定总输出目录；不填则自动创建 `Output/YYMMDD_n/`
  - `--data-root`：若提供，则从该目录下读取 5 个标准命名 CSV
  - `--datasets`：可选 `2005 2010 2015 2020 full`
  - `--year-bw-min / --year-bw-max / --year-bw-step`：年份截面扫描区间，默认 `140 / 220 / 10`
    - 若 `--year-bw-max <= 0`，或显式上限超过当前 holdout 训练样本数，则会自动按训练样本数截断
  - `--full-bw-min / --full-bw-max / --full-bw-step`：`full` 扫描区间，默认 `2500 / 4000 / 150`
  - `--jobs`：按 `bw` 任务启用多进程并行；默认 `1`
  - `--no-progress`：关闭终端进度条
- **默认输入**
  - `Data/年份终市级指标数据（5年截面）/终市级指标数据_with_latlon_2005.csv`
  - `Data/年份终市级指标数据（5年截面）/终市级指标数据_with_latlon_2010.csv`
  - `Data/年份终市级指标数据（5年截面）/终市级指标数据_with_latlon_2015.csv`
  - `Data/年份终市级指标数据（5年截面）/终市级指标数据_with_latlon_2020.csv`
  - `Data/终市级指标数据_with_latlon.csv`
- **行为**
  - 对每套数据固定同一组 `test_size=0.2`、`random_state=42` 的 holdout 划分
  - 同一测试集上先训练一次 `XGBoost-global`，将其 `R2 / RMSE / MAE` 作为整段扫描的 baseline
  - 对每个候选 `bw`：
    - 复用当前 `gwxgb_config.yaml` 的核函数、空间权重、距离口径与 `XGBoost` 超参数
    - 只替换 `bw`，逐测试样本训练 `GW-XGBoost-local`
    - 合并全部测试点局部预测后，统一计算整体 `R2 / RMSE / MAE`
  - 当 `--jobs > 1` 时，脚本会把每个 `bw` 作为独立任务并行执行；每个局部 `XGBRegressor` 仍保持 `n_jobs=1`，避免过度抢占线程
  - 在受限 Windows 环境中若多进程初始化被 `WinError 5` 拒绝，脚本会自动回退到线程并行
  - 终端默认显示两个进度条：
    - `BW tasks`：全部候选 `bw` 任务的总体进度
    - `Datasets`：整套数据集完成的进度
  - 每套数据都会额外输出一张三联曲线图：
    - `R2` 曲线
    - `RMSE` 曲线
    - `MAE` 曲线
    - 并叠加 `XGBoost-global` 的虚线 baseline 和每个指标的最优点标注
- **输出结构**（写入 `Output/YYMMDD_n/<dataset>/` 或 `--run-root/<dataset>/`）
  - `bw_sweep_results.csv`
    - 每个候选 `bw` 1 行
    - 包含 `GW-XGBoost-local` 的 `R2 / RMSE / MAE`
    - 同时附带 `XGBoost-global` baseline 与三项差值
    - 也包含局部样本数、平均权重、距离口径和距离摘要
  - `bw_sweep_best_summary.csv`
    - 每个数据集 3 行
    - 分别对应按 `r2`、`rmse`、`mae` 选出的最佳 `bw`
  - `bw_sweep_local_diagnostics.csv`
    - 每个测试样本、每个 `bw` 1 行
    - 记录局部邻域规模、权重摘要、距离口径、预测值与误差
  - `bw_sweep_metrics.png`
    - 每套数据 1 张图
    - 纵向 3 个子图，展示 `R2 / RMSE / MAE` 随 `bw` 的变化
- **run_root 级附加输出**（写入 `Output/YYMMDD_n/` 或 `--run-root/`）
  - `batch_bw_sweep_results.csv`
    - 汇总所有数据集、所有候选 `bw` 的扫描结果
  - `batch_bw_sweep_best_summary.csv`
    - 汇总所有数据集在 `r2 / rmse / mae` 三个准则下的最优 `bw`
  - `batch_bw_sweep_overview.png`
    - run_root 级五套数据集合图
    - 按数据集分行、按 `R2 / RMSE / MAE` 分列
    - 每个子图都包含 `GW-XGBoost-local` 曲线、`XGBoost-global` baseline 横线，以及 baseline 数值标注
    - 图下方统一附加评估口径、评估方法和固定超参数说明；`bw` 作为扫描变量保留在横轴与最优点标注中

### M) `gwxgb/run_cv_bw_sweep.py`（统一 5 折 CV 口径下的 BW 扫描寻优）

- **配置来源**
  - 主体会读取并重写：
    - `gwxgb/gwxgb_config.yaml`
- **命令行参数**
  - `--run-root`：显式指定总输出目录；不填则自动创建 `Output/YYMMDD_n/`
  - `--data-root`：若提供，则从该目录下读取 5 个标准命名 CSV
  - `--datasets`：可选 `2005 2010 2015 2020 full`
  - `--year-bw-min / --year-bw-max / --year-bw-step`：年份截面扫描区间，默认 `190 / 280 / 5`
  - `--full-bw-min / --full-bw-max / --full-bw-step`：`full` 扫描区间，默认 `3300 / 3900 / 30`
  - `--cv-splits`：CV 折数，默认 `5`
  - `--cv-random-state`：`KFold(shuffle=True)` 的随机种子，默认 `42`
  - `--jobs`：按 `bw` 任务启用多进程并行；默认 `1`
  - `--no-progress`：关闭终端进度条
- **行为**
  - 对每套数据使用同一组 `KFold(n_splits=5, shuffle=True, random_state=42)` 划分
  - 对每个候选 `bw`：
    - `XGBoost-global` 在每折训练一个全局模型，预测该折验证集
    - `GW-XGBoost-local` 在每折对验证样本逐点训练局部加权模型
    - 主性能列 `r2 / rmse / mae` 为 5 折验证集指标均值
    - 同时输出合并所有 out-of-fold 预测后的 `gw_oof_*` 与 `xgb_oof_*` 整体指标
  - 复用当前 `gwxgb_config.yaml` 的核函数、空间权重、距离口径与 `XGBoost` 超参数，只替换 `bw`
  - 当 `--jobs > 1` 时，脚本会把每个 `bw` 作为独立任务并行执行；若多进程被受限 Windows 环境拒绝，会自动回退到线程并行
- **输出结构**（写入 `Output/YYMMDD_n/<dataset>/` 或 `--run-root/<dataset>/`）
  - `cv_bw_sweep_results.csv`
    - 每个候选 `bw` 1 行
    - 包含 `GW-XGBoost-local` 与 `XGBoost-global` 的 5 折 CV mean/std、OOF 整体指标和差值
  - `cv_bw_sweep_best_summary.csv`
    - 每个数据集 3 行
    - 分别对应按 CV mean `r2`、`rmse`、`mae` 选出的最佳 `bw`
  - `cv_bw_sweep_local_diagnostics.csv`
    - 每个验证样本、每个 `bw` 1 行
    - 记录 fold、局部邻域规模、权重摘要、距离口径、预测值与误差
  - `cv_bw_sweep_fold_details.csv`
    - 每个 `bw`、每个 fold、每个模型 1 行
    - 记录该折验证集 `R2 / RMSE / MAE`
  - `cv_bw_sweep_metrics.png`
    - 每套数据 1 张图，展示 CV mean `R2 / RMSE / MAE` 随 `bw` 的变化并叠加 baseline
- **run_root 级附加输出**
  - `batch_cv_bw_sweep_results.csv`
  - `batch_cv_bw_sweep_best_summary.csv`
  - `batch_cv_bw_sweep_fold_details.csv`
  - `batch_cv_bw_sweep_overview.png`
    - run_root 级集合图，图下方统一附加 CV 评估口径、评估方法和固定超参数说明

### N) `gwxgb/add_cv_ols_baseline.py`（对既有 CV BW 输出追加 OLS baseline）

- **配置来源**
  - 主体会读取并重写：
    - `gwxgb/gwxgb_config.yaml`
    - 已有 `run_cv_bw_sweep.py` 输出目录
- **命令行参数**
  - `--run-root`：已有 CV BW sweep 输出根目录，例如 `Output/260422_cv_euclidean_all`
  - `--data-root`：若提供，则从该目录下读取 5 个标准命名 CSV
  - `--datasets`：可选 `2005 2010 2015 2020 full`
  - `--cv-splits`：CV 折数，默认 `5`
  - `--cv-random-state`：`KFold(shuffle=True)` 的随机种子，默认 `42`
- **行为**
  - 使用与 CV BW sweep 相同的 `KFold(n_splits=5, shuffle=True, random_state=42)`
  - 对每个数据集单独训练 `sklearn.linear_model.LinearRegression(fit_intercept=True)`
  - 只计算 OLS baseline，不重跑 `GW-XGBoost-local` 或 `XGBoost-global`
  - 将 OLS 的 CV mean/std 和 OOF 整体指标追加到既有 `cv_bw_sweep_results.csv` 与 `cv_bw_sweep_best_summary.csv`
  - 重新绘制每个数据集的 `cv_bw_sweep_metrics.png` 和 run-root 总览图，在原 `XGBoost-global` baseline 外增加 `OLS` baseline
- **输出结构**
  - 每个数据集目录：
    - `cv_ols_baseline.csv`
    - `cv_ols_fold_details.csv`
    - 更新后的 `cv_bw_sweep_results.csv`
    - 更新后的 `cv_bw_sweep_best_summary.csv`
    - 更新后的 `cv_bw_sweep_metrics.png`
  - run-root：
    - `batch_cv_ols_baseline.csv`
    - `batch_cv_ols_fold_details.csv`
    - 更新后的 `batch_cv_bw_sweep_results.csv`
    - 更新后的 `batch_cv_bw_sweep_best_summary.csv`
    - 更新后的 `batch_cv_bw_sweep_overview.png`
