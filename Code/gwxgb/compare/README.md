本目录集中放置 `gwxgb` 的全局 SHAP vs 局部 SHAP 对比脚本与配置。

- 统一入口：`gwxgb_compare_all.py`
- 三个单独口径入口：
  - `gwxgb_compare_center_local_demo.py`
  - `gwxgb_compare_local_importance_demo.py`
  - `gwxgb_compare_pooled_local_demo.py`
- 共享逻辑：`gwxgb_compare_demo_common.py`
- 配置：`gwxgb_compare_*_config.yaml`

这些配置默认通过 `base_config: "../gwxgb_config.yaml"` 继承主 `gwxgb` 配置。
