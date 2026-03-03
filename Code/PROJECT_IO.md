# Code 目录：程序 / 配置（cfg）输入输出整理

本目录是 **uv 项目根目录**（`Code/pyproject.toml`、`Code/uv.lock`、`Code/.venv/`）。

## 1) 环境与运行入口

在仓库根目录执行也可以，但推荐在 `Code/` 下运行：

```powershell
cd .\Code
uv sync
uv run python .\xgb\xgb_shap.py -c .\xgb\config.yaml
uv run python .\gwxgb\gwxgb_shap.py -c .\gwxgb\gwxgb_config.yaml
```

说明：脚本会把 YAML 中的相对路径按 **YAML 文件所在目录** 解析，所以不强依赖 `cd` 到哪里运行。

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
- **输出**（写入 `output.output_dir`）
  - SHAP summary：`output.summary_file`（默认 `shap_summary.png`）
  - Mean(|SHAP|)：`output.mean_abs_shap_file`（默认 `mean_abs_shap.png`）
  - Dependence：`output.dependence_prefix + <feature>.png`
  - 交互对 CSV：`output.interaction_pairs_file`
  - 固定基准交互图（可选）：`output.interaction_prefix + <base>_x_<other>.png`
  - 模型文件（可选）：`output.model_file`（若配置则保存 Booster）

### B) `xgb/xgb_nestedcv_tune.py`（嵌套交叉验证自动调参）

- **配置**：`xgb/config.yaml`（主要读取 `tuning.*` 与 `model.params`）
- **输入**：同 `xgb_shap.py` 的数据与特征配置
- **输出**：**仅控制台输出**（每折指标、best_params，以及一段可写回 `config.yaml` 的 `model.params` 建议片段）

### C) `xgb/verify_fix.py`（CSV 百分号/字符串数值转换验证）

- **输入**：脚本内置读取 `xgb_shap.load_dataset`
- **输出**：仅控制台输出（dtype 与转换检查）

### D) `gwxgb/gwxgb_shap.py`（GeoXGBoost + 带宽优化 + 全局 SHAP）

- **配置**：`gwxgb/gwxgb_config.yaml`
- **输入**
  - 数据：`data.path`（CSV）
  - 坐标列：`data.coords`（长度必须为 2，例如 `["纬度","经度"]`）
  - 目标/特征：`data.target`、`data.features`
  - 全局模型：`model.params`
  - 网格搜索（可选）：`grid_search.enabled=True` + `grid_search.param_grid`（默认关闭，避免运行时间过长）
  - 带宽与空间权重：`gw.*`（供 `geoxgboost.optimize_bw` / `geoxgboost.gxgb`）
  - SHAP：对**全局基线模型**输出图/交互对（不对每个本地模型做 SHAP）
- **输出**（写入 `output.output_dir`）
  - 全局 SHAP summary：`output.summary_file`（默认 `gw_shap_summary.png`）
  - Mean(|SHAP|)：`output.mean_abs_shap_file`（默认 `gw_mean_abs_shap.png`）
  - Dependence：`output.dependence_prefix + <feature>.png`
  - 交互对 CSV：`output.interaction_pairs_file`
  - 日志：`output.log_file`（默认 `run_log.txt`）
  - GeoXGBoost 带宽优化结果：`BW_results.csv`
  - GeoXGBoost 本地模型结果：`LW_GXGB.xlsx`

## 3) 现有输出（已归档）

- 历史输出 `Code/BW_results.csv`、`Code/LW_GXGB.xlsx` 已从 `Code/` 移出并归档到：
  - `Output/output_gwxgb/_legacy/20260114_from_Code/`
- 2026-02-26 运行时在仓库根目录生成的同名输出，已从仓库根目录移出并归档到：
  - `Output/output_gwxgb/_legacy/20260226_from_repo_root/`
- 当前脚本运行的默认输出位置为：`Output/output_gwxgb/`（会覆盖同名文件）
