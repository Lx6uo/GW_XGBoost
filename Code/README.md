本目录为主要代码与配置（XGBoost / GeoXGBoost），同时也是 uv 项目根目录（`Code/pyproject.toml`、`Code/uv.lock`、`Code/.venv/`）。

- 运行前：在 `Code/` 下执行 `uv sync`
- 脚本与 cfg 的输入/输出速查：`Code/PROJECT_IO.md`
- 数据预处理（绝对值特征归一化）：`uv run python .\xgb\normalize_feature_table.py -c .\xgb\config.yaml`
- 归因稳健性分析：`uv run python .\xgb\xgb_shap_robustness.py -c .\xgb\config.yaml`
- 全局 SHAP vs 局部 SHAP 对比统一入口：`uv run python .\gwxgb\compare\gwxgb_compare_all.py -c .\gwxgb\compare\gwxgb_compare_all_config.yaml`
- compare 系列图片默认输出为 SHAP 原生 `summary_plot` / `bar`；其中 `local_importance` 额外输出 signed SHAP 方向 summary 与正负贡献汇总 CSV，且其 summary 图的红蓝颜色按“各局部模型中心点的原始特征值”编码
- 三种 compare 口径：
  - `center_local`：每个地理位置只保留一条中心样本局部解释，适合作为与全局 SHAP 的主比较
  - `local_importance`：每个局部模型先汇总强度与方向，适合看空间异质性是否被全局模型平均掉
  - `pooled_local`：把所有局部邻域样本解释直接拼池，适合演示整体形态，但要注意重复计数偏差
- 全局 SHAP vs 局部 SHAP 对比 demo（中心样本口径）：`uv run python .\gwxgb\compare\gwxgb_compare_center_local_demo.py -c .\gwxgb\compare\gwxgb_compare_center_local_config.yaml`
- 全局 SHAP vs 局部 SHAP 对比 demo（局部重要性口径）：`uv run python .\gwxgb\compare\gwxgb_compare_local_importance_demo.py -c .\gwxgb\compare\gwxgb_compare_local_importance_config.yaml`
- 全局 SHAP vs 局部 SHAP 对比 demo（局部拼池口径）：`uv run python .\gwxgb\compare\gwxgb_compare_pooled_local_demo.py -c .\gwxgb\compare\gwxgb_compare_pooled_local_config.yaml`
  - 上述 3 个 compare cfg 默认都会继承 `.\gwxgb\gwxgb_config.yaml`，因此主 `gwxgb` 配置一改，compare 系列会自动同步
- 项目整体说明：仓库根目录 `README.md`
