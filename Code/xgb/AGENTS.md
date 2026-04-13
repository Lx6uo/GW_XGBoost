# XGB Workflow Guide

Use this guide when touching files under `Code/xgb/`.

## Read Order

1. `../../AGENTS.md`
2. `../../docs/AGENT_CONTEXT.md`
3. `../../docs/repo_index.yaml`
4. `config.yaml`
5. Only the specific `Code/xgb/*.py` files relevant to the task

## Main Entrypoints

- `xgb_shap.py`: global XGBoost + SHAP; also exports helper functions reused by `gwxgb`
- `xgb_nestedcv_tune.py`: nested CV tuning
- `xgb_gridcv_tune.py`: single-layer GridSearchCV ranking
- `xgb_shap_robustness.py`: attribution-oriented robustness analysis
- `xgb_shap_interaction_matrix.py`: SHAP interaction matrix figure
- `normalize_feature_table.py`: preprocessing
- `feature_corr_heatmap.py` and `spearman_corr_heatmap_batch.py`: diagnostics

## Guardrails

- `xgb_shap.py` is shared infrastructure, not only a script entrypoint. Changes there can affect `gwxgb_shap.py` and `run_curated_output_batch.py`.
- Default config values are dataset-specific and use Chinese column names.
- Interaction and robustness workflows are heavier than the basic global SHAP run.
- Keep output file naming stable unless the user explicitly wants a rename; many docs assume current names.

## Common Validation Commands

From `Code/`:

- `uv run python .\xgb\xgb_shap.py -c .\xgb\config.yaml`
- `uv run python .\xgb\xgb_shap_robustness.py -c .\xgb\config.yaml`
- `uv run python .\xgb\xgb_shap_interaction_matrix.py -c .\xgb\config.yaml`
- `uv run python .\xgb\spearman_corr_heatmap_batch.py -c .\xgb\config.yaml`

## Required Doc Sync

If you add or change an `xgb` workflow, update:

- `../../README.md`
- `../README.md`
- `../PROJECT_IO.md`
- `../../docs/repo_index.yaml`
- `../../docs/LAST_STATE.md`
