# ISML 项目整理（程序 & 配置 I/O）

本仓库按当前目录结构分为：`Code/`（脚本与配置）、`Data/`（输入数据）、`Output/`（运行产出）、`Ref/`（参考文献）、`test/`（demo/示例）。

协作约定见 `CONTRIBUTING.md`。

## 1) 目录结构（约定的输入/输出位置）

- `Code/`：主要代码与配置
  - `Code/xgb/`：全局 XGBoost（回归）+ 评估 + SHAP
  - `Code/gwxgb/`：GeoXGBoost（带宽优化 + 本地模型）+ **全局模型** SHAP（用于解释）
- `Data/`：输入数据（CSV/XLSX/边界数据 GeoBoundaries）
- `Output/`：输出（图像 PNG / 交互对 CSV / 日志等）
- `Ref/`：论文与参考材料（不参与程序运行）
- `test/`：示例数据与官方 Demo（用于参考/对照）

## 2) 快速运行（推荐从仓库根目录）

### 2.1 安装依赖

- `uv`（推荐，使用锁文件）：在 `Code/` 下执行 `uv sync`，然后用 `uv run` 运行脚本，例如：

```powershell
cd .\Code
uv sync
uv run python .\xgb\xgb_shap.py -c .\xgb\config.yaml
```

- `pip`（通用）：使用仓库根目录的 `requirements.txt`：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r .\requirements.txt
python .\Code\xgb\xgb_shap.py -c .\Code\xgb\config.yaml
```

### 2.2 运行脚本

```powershell
python .\Code\xgb\xgb_shap.py -c .\Code\xgb\config.yaml
python .\Code\xgb\xgb_nestedcv_tune.py -c .\Code\xgb\config.yaml
python .\Code\gwxgb\gwxgb_shap.py -c .\Code\gwxgb\gwxgb_config.yaml
```

说明：YAML 中 `data.path` / `output.output_dir` 等相对路径，会按 **YAML 文件所在目录** 解析。

## 3) 脚本清单与输入/输出

### A. `Code/xgb/xgb_shap.py`（全局 XGBoost + SHAP）

- **配置文件**：默认读取 `Code/xgb/config.yaml`（可用 `-c/--config` 指定其它 YAML）。
- **输入**
  - 数据：`data.path`（CSV）
  - 列：`data.target`（因变量列名），`data.features`（自变量列名列表；缺省则用除 `target` 外的所有列）
  - 训练：`model.params`（传给 `xgboost.XGBRegressor`）
  - 评估（可选）：
    - `cv.use_cv=1` 时执行 K 折交叉验证并打印每折/均值指标
    - `model.test_size` 有效时做 hold-out（训练/测试划分）并打印指标（含回归 + 基于中位数阈值的“二分类”指标）
  - SHAP：`shap.use_summary`、`shap.use_dependence`、`shap.compute_interactions`、`shap.dependence_top_n`、`shap.interaction_*`
- **输出（写入 `output.output_dir`）**
  - SHAP summary 图：`output.summary_file`（默认 `shap_summary.png`）
  - Mean(|SHAP|) 条形图：`output.mean_abs_shap_file`（代码缺省为 `mean_abs_shap.png`；配置可不写）
  - Dependence 图：`output.dependence_prefix + <feature>.png`
  - 固定基准交互图：`output.interaction_prefix + <base>_x_<other>.png`
  - Top 交互特征对：`output.interaction_pairs_file`（CSV）
  - 可选模型文件：若配置 `output.model_file`，则保存 `XGBoost Booster.save_model()` 的结果

### B. `Code/xgb/xgb_nestedcv_tune.py`（嵌套交叉验证自动调参）

- **配置文件**：默认读取 `Code/xgb/config.yaml`（同样支持 `-c`）。
- **输入**
  - 数据：同 `xgb_shap.py`（`data.*`）
  - 初始模型参数：`model.params`
  - 调参范围：`tuning.use_nested_cv=1` 且配置 `tuning.param{1,2,3}` 与对应 `*_values`
  - 折数：`tuning.outer_splits`、`tuning.inner_splits`
- **输出**
  - **仅控制台输出**：每个 outer fold 的指标与 best_params；最后输出一段可直接写回 `config.yaml` 的 `model.params` 建议片段（不写文件）。

### C. `Code/xgb/verify_fix.py`（数据类型/百分号转换验证）

- **输入**：脚本内写死的 `config`（调用 `xgb_shap.load_dataset`），主要用于检查 CSV 中带 `%` 的列是否被自动转为数值。
- **输出**：仅控制台输出（各列 dtype 与验证通过/失败信息）。

### D. `Code/gwxgb/gwxgb_shap.py`（GeoXGBoost + 带宽优化 + 全局 SHAP）

- **配置文件**：默认读取 `Code/gwxgb/gwxgb_config.yaml`（支持 `-c`）。
- **输入**
  - 数据：`data.path`（CSV），并需提供 `data.coords` 两列（经纬度）
  - 列：`data.target`、`data.features`、`data.coords`
  - 全局模型参数：`model.params`
  - 全局网格搜索（可选）：`grid_search.enabled=1` 且提供 `grid_search.param_grid`
  - 带宽优化与本地模型：`gw.*`（`bw`、`kernel`、`optimize_bw`、`bw_min/max/step`、`spatial_weights` 等）
  - SHAP：复用 `xgb_shap.py` 的 SHAP 逻辑，仅对**全局 XGBoost 基线模型**输出图与交互对
- **输出（写入 `output.output_dir`）**
  - 全局 SHAP summary：`output.summary_file`（默认 `gw_shap_summary.png`）
  - 全局 Mean(|SHAP|) 图：`output.mean_abs_shap_file`（代码缺省 `mean_abs_shap.png`；配置可不写）
  - Dependence：`output.dependence_prefix + <feature>.png`
  - 交互对 CSV：`output.interaction_pairs_file`
  - 日志：`output.log_file`（默认 `run_log.txt`，同时输出到控制台与文件）
  - GeoXGBoost 带宽优化结果：`BW_results.csv`（由 `geoxgboost.optimize_bw` 写出）
  - GeoXGBoost 本地模型结果：`LW_GXGB.xlsx`（由 `geoxgboost.gxgb` 写出）

### E. Demo（`test/`）

- `test/DemoData/GXGB_call_demo.py` 与 `test/DemoGXGBoost/DemoData/GXGB_call_demo.py`：
  - 用于演示 geoxgboost 的 nestedCV / global_xgb / optimize_bw / gxgb 流程
  - 输入：DemoData 下的 `Coords.csv`、`Data.csv`（以及预测数据 CSV，默认注释掉）
  - 输出：示例会保存 `shap_global.png`（默认写到当前工作目录），其余主要在内存/控制台

## 4) 配置文件（cfg）字段速查

### `Code/xgb/config.yaml`

- `data`: `path` / `target` / `features` / `sep` / `encoding`
- `model`: `random_state` / `test_size` / `params`（XGBRegressor 参数）
- `cv`: `use_cv` / `n_splits` / `random_state`
- `shap`: summary/dependence/interaction 相关开关与参数
- `output`: `output_dir` / `summary_file` / `dependence_prefix` / `interaction_prefix` / `interaction_pairs_file`（可选 `model_file`）
- `tuning`: 嵌套 CV 调参用（仅 `xgb_nestedcv_tune.py` 读取）

### `Code/gwxgb/gwxgb_config.yaml`

- `data`: 额外多了 `coords: [<lat_col>, <lon_col>]`
- `grid_search`: 全局模型 GridSearchCV（可选）
- `gw`: 带宽与空间权重相关（供 `geoxgboost.optimize_bw` / `geoxgboost.gxgb`）
- `shap` / `output`: 与 `xgb` 类似，但文件名前缀不同

## 5) Review 备注

1. **路径相对性**：`xgb_shap.py` 已将 `data.path` / `output.output_dir` 的相对路径按“配置文件所在目录”解析（不再依赖运行时 `cd`）。
2. **YAML 中的历史路径**：`Code/xgb/config.yaml`、`Code/gwxgb/gwxgb_config.yaml` 已按当前 `Data/` 与 `Output/` 目录结构修正默认路径。
3. **Demo 绝对路径**：`test/` 下两份 `GXGB_call_demo.py` 已改为按脚本目录读取 `Coords.csv` / `Data.csv`，避免机器路径不一致导致无法运行。
