本目录为主要代码与配置（XGBoost / GeoXGBoost），同时也是 uv 项目根目录（`Code/pyproject.toml`、`Code/uv.lock`、`Code/.venv/`）。

- 运行前：在 `Code/` 下执行 `uv sync`
- 脚本与 cfg 的输入/输出速查：`Code/PROJECT_IO.md`
- 数据预处理（绝对值特征归一化）：`uv run python .\xgb\normalize_feature_table.py -c .\xgb\config.yaml`
- 归因稳健性分析：`uv run python .\xgb\xgb_shap_robustness.py -c .\xgb\config.yaml`
- 项目整体说明：仓库根目录 `README.md`
