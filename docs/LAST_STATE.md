# LAST_STATE

## Last Updated

- 2026-04-16

## Stable Facts

- `Code/` is the `uv` project root.
- The unified compare runner currently outputs 4 modes: `center_local`, `local_importance`, `pooled_local`, `sample_aggregated_local`.
- `sample_aggregated_local` now has its own native plots and per-feature dependence plots.
- `Code/gwxgb/run_curated_output_batch.py` produces curated batch outputs under `Output/YYMMDD_n/`.
- `run_curated_output_batch.py` now prefers reusing the latest complete prior curated dataset directory, so repeated batch runs do not need to retrain models by default.
- when a complete curated dataset directory is reused, `sample_aggregated_local` plots and correlation heatmaps are refreshed from cached tables without retraining.
- `run_curated_output_batch.py` now emits per-dataset Spearman and Pearson correlation heatmaps plus a manifest under each curated dataset directory.
- `run_curated_output_batch.py` now also emits per-dataset raw `model_metrics_and_hyperparams.csv` files plus readable `model_summary_overview.csv` / `model_summary_details.csv` tables, focused on GeoXGBoost vs same-param XGBoost baseline comparison with baseline CV mean/std and OOF metrics, excluding analysis-stage performance, and matching run-root summaries.
- `run_curated_output_batch.py` now also emits a separate holdout benchmark suite: `model_performance_matrix.csv` / `model_performance_details.csv` / `gwxgb_local_diagnostics.csv` and run-root aggregates, comparing `XGBoost-global` vs `GW-XGBoost-local` on the same `test_size=0.2`, `random_state=42` split.
- `Code/gwxgb/run_holdout_bw_sweep.py` now scans `GW-XGBoost-local` bandwidth on the same holdout split, caps yearly bw upper bounds to available holdout train samples when needed, can parallelize per-bw tasks via `--jobs`, auto-falls back from process workers to thread workers in restricted Windows environments, shows terminal progress bars by default, and emits per-dataset sweep tables, local diagnostics, per-dataset 3-metric BW curve plots, plus a run-root combined `batch_bw_sweep_overview.png` figure with XGBoost baseline lines, numeric labels, and a footer describing the evaluation protocol and fixed hyperparameters.
- `Code/gwxgb/gwxgb_config.yaml` now defaults `gw.distance_metric: "euclidean"` with `gw.coord_order: "auto"`, so GW local sampling, spatial weights, holdout benchmarks, and local SHAP/compare distance diagnostics use the original coordinate-unit Euclidean distance. Set `gw.distance_metric: "haversine"` to use real surface great-circle distance in km.
- `Code/gwxgb/run_cv_bw_sweep.py` now provides a 5-fold CV BW sweep alongside the holdout sweep. It defaults yearly cross-sections to `bw=190~280, step=5`, uses fold-mean `R2 / RMSE / MAE` as primary metrics, preserves merged OOF metrics as supplemental columns, compares against same-fold `XGBoost-global`, and writes per-dataset `cv_bw_*` outputs plus `batch_cv_bw_sweep_overview.png`.
- `Code/gwxgb/add_cv_ols_baseline.py` now appends an OLS baseline to an existing CV BW sweep run root without rerunning GW-XGBoost. It trains `LinearRegression(fit_intercept=True)` on the same KFold splits, writes `cv_ols_*` / `batch_cv_ols_*` outputs, updates the existing CV sweep tables, and redraws the overview plots with OLS baseline lines.
- `run_curated_output_batch.py` now also emits run-root-level SHAP mean value Sankey charts when all 5 curated datasets are available in one run root.
- `run_curated_output_batch.py` now caches interaction-matrix redraw inputs (`interaction_values.npy`, `interaction_test_features.csv`) and reuses them on later batch refreshes when available.
- `xgb_shap_interaction_matrix.py` now defaults its lower-triangle scatter to a heat-style `plasma` palette (`scheme_index: 2`) and keeps `turbo` / `viridis` / `cividis` as alternatives.
- `xgb_shap_interaction_matrix.py` now colors upper-triangle heat blocks by `|SHAP interaction|` with an orange sequential palette and same-height right-side colorbar, while the lower-triangle heat-style scatter and left-side raw-feature colorbar follow `scheme_index`.
- `xgb_shap_interaction_matrix.py` now trims upper and lower colormap endpoints separately so the upper heat blocks and lower heat-style scatter can keep different contrast levels.
- `sample_aggregated_local_shap_sum.png` now keeps visible feature names on the y-axis while overlaying the `Mean |SHAP value|` top-axis bars.
- Compare configs default `output.capture_prints: 0`; most other main scripts default `1`.

## Changes Landed In This Turn

- Added root `AGENTS.md` and workflow-specific directory guides under `Code/`.
- Added `docs/AGENT_CONTEXT.md` and `docs/repo_index.yaml` so new sessions can route without re-reading the full repo.
- Added a repo-specific skill at `.agents/skills/isml-workflow/`.
- Synced `README.md`, `Code/README.md`, `Code/PROJECT_IO.md`, and `CONTRIBUTING.md` with the current 4-mode compare suite and curated batch runner.
- Extended `run_curated_output_batch.py` to generate cross-dataset SHAP mean value Sankey charts and synced the batch-output docs.
- Extended `run_curated_output_batch.py` to reuse complete prior curated dataset outputs by default and added `--force-rebuild`.
- Extended `run_curated_output_batch.py` to generate per-dataset Spearman/Pearson correlation heatmaps and synced the batch-output docs.
- Extended `run_curated_output_batch.py` to export both raw and readable per-dataset/run-root model metric/hyperparameter summary tables, and added same-param XGBoost baseline CV comparison.
- Extended `run_holdout_bw_sweep.py` with evaluation-protocol and hyperparameter footer text on the run-root BW overview figure, alongside baseline numeric annotations, yearly bw upper-bound auto-capping to holdout train size, per-bw task parallelism (`--jobs`), automatic process-to-thread fallback on restricted Windows setups, and visible terminal progress bars (`BW tasks` / `Datasets`), and synced the sweep workflow docs.
- Extended `run_curated_output_batch.py` to cache and refresh interaction-matrix redraw inputs across reused curated batch outputs.
- Updated the SHAP interaction matrix lower-scatter default color scheme to `plasma` and synced the relevant docs.
- Refined the interaction matrix upper-triangle coloring and restored feature labels in `sample_aggregated_local_shap_sum.png`.
- Rebalanced the interaction-matrix colormap trimming back toward a broader color range, and synced the docs.
- Swapped the interaction-matrix legends so the upper heatmap now uses a right-side `|SHAP interaction|` colorbar and the lower scatter keeps a left-side raw-feature colorbar.
- Changed the lower-triangle scatter to a heat-style palette with separate endpoint trimming.
- Unified both interaction-matrix colorbar heights and switched the upper heat blocks to an orange sequential palette.

## Next-Time Maintenance Rule

When workflows, outputs, or entrypoints change, update this file in the same turn with one short bullet so the next session has a recent checkpoint.
