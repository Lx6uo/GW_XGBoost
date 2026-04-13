本目录集中放置 `gwxgb` 的全局 SHAP vs 局部 SHAP 对比脚本与配置。

- 统一入口：`gwxgb_compare_all.py`
- 三个单独口径入口：
  - `gwxgb_compare_center_local_demo.py`
  - `gwxgb_compare_local_importance_demo.py`
  - `gwxgb_compare_pooled_local_demo.py`
- 共享逻辑：`gwxgb_compare_demo_common.py`
- 配置：`gwxgb_compare_*_config.yaml`

这些配置默认通过 `base_config: "../gwxgb_config.yaml"` 继承主 `gwxgb` 配置。

统一入口 `gwxgb_compare_all.py` 当前会同时输出：

- `center_local`
- `local_importance`
- `pooled_local`
- `sample_aggregated_local`

其中 `sample_aggregated_local` 表示：先保留所有 pooled 局部 SHAP 行，再按原始样本聚合成“一样本一条平均 SHAP 向量”，用于减少 `pooled_local` 的重复计数放大。
该口径现在也会按主 `gwxgb` 配置中的 `shap.use_dependence` / `dependence_*` 规则，额外输出单特征 dependence 图：`sample_aggregated_local_dependence_<feature>.png`。

compare 根输出除 `global_shap_summary.png` / `global_shap_bar.png` 外，也会继承主 `gwxgb` 的全局交互输出：

- `output.interaction_pairs_file`（默认 `gw_shap_interactions.csv`）
- `output.interaction_prefix + top_<rank>_<feature1>_x_<feature2>.png`
- `output.interaction_prefix + <base>_x_<other>.png`
