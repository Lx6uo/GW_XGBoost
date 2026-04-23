# GWXGB Workflow Guide

Use this guide when touching files under `Code/gwxgb/`, except the compare subfolder which has its own guide.

## Read Order

1. `../../AGENTS.md`
2. `../../docs/AGENT_CONTEXT.md`
3. `../../docs/repo_index.yaml`
4. `gwxgb_config.yaml`
5. Relevant script:
   - `gwxgb_shap.py`
   - `run_curated_output_batch.py`
   - or `compare/AGENTS.md` for compare work

## Main Entrypoints

- `gwxgb_shap.py`: GeoXGBoost main workflow, bandwidth optimization, global SHAP, optional local SHAP
- `run_curated_output_batch.py`: curated multi-dataset export runner for `2005`, `2010`, `2015`, `2020`, and `full`
- `run_holdout_bw_sweep.py`: holdout BW sweep benchmark for `GW-XGBoost-local` vs `XGBoost-global`
- `run_cv_bw_sweep.py`: 5-fold CV BW sweep benchmark for `GW-XGBoost-local` vs `XGBoost-global`
- `add_cv_ols_baseline.py`: append OLS baseline metrics to an existing CV BW sweep run root without rerunning GW-XGBoost

## Current Behavior

- Default `gwxgb_config.yaml` enables `local_shap.enabled: 1`.
- Default `gwxgb_config.yaml` uses `gw.distance_metric: "euclidean"` so local sampling and spatial weights use the original raw-coordinate distance; set `haversine` only when intentionally using real surface distance in km.
- Compare configs under `compare/` inherit from `gwxgb_config.yaml`.
- `run_curated_output_batch.py` writes stage-based outputs under `Output/YYMMDD_n/<dataset>/`.
- For yearly datasets, that curated runner first tries to reuse the latest existing `Output/output_gwxgb/gwxgb_终市级指标数据_with_latlon_<year>_*` output.

## Guardrails

- Changes to `gwxgb_shap.py` can cascade into compare workflows and curated batch exports.
- Keep `BW_results.csv`, `LW_GXGB.xlsx`, local SHAP CSV names, and interaction output names stable unless a coordinated rename is intended.
- `local_shap` and bandwidth optimization are runtime-heavy; avoid changing defaults casually.

## Common Validation Commands

From `Code/`:

- `uv run python .\gwxgb\gwxgb_shap.py -c .\gwxgb\gwxgb_config.yaml`
- `uv run python .\gwxgb\run_curated_output_batch.py --datasets 2005`

## Required Doc Sync

If you change entrypoints, outputs, or reuse rules here, update:

- `../../README.md`
- `../README.md`
- `../PROJECT_IO.md`
- `../../docs/repo_index.yaml`
- `../../docs/LAST_STATE.md`
