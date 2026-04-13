# AGENT_CONTEXT

## One-Screen Summary

ISML is a script-first Python repository for:

- global `XGBoost` modeling
- `GeoXGBoost` local modeling with bandwidth control
- SHAP explanation, dependence, interactions, and robustness analysis
- global-vs-local SHAP comparison across 4 compare modes

`Code/` is the working code root and `uv` project root. Most tasks only need one workflow family, so route early and read narrowly.

## Trust Order

When docs disagree, prefer this order:

1. Runtime code in `Code/**/*.py`
2. Active configs in `Code/**/*.yaml`
3. `Code/PROJECT_IO.md`
4. `README.md` and subdirectory READMEs
5. `docs/LAST_STATE.md`

## Fast Routing

| Task | Read first | Then open |
| --- | --- | --- |
| Global XGBoost / preprocessing / tuning / robustness | `Code/xgb/AGENTS.md` | relevant `Code/xgb/*.py` and `Code/xgb/config.yaml` |
| GeoXGBoost main pipeline / local SHAP | `Code/gwxgb/AGENTS.md` | `Code/gwxgb/gwxgb_shap.py`, `Code/gwxgb/gwxgb_config.yaml` |
| Compare suite | `Code/gwxgb/compare/AGENTS.md` | `Code/gwxgb/compare/gwxgb_compare_all.py`, `gwxgb_compare_demo_common.py`, compare configs |
| Curated batch export | `Code/gwxgb/AGENTS.md` | `Code/gwxgb/run_curated_output_batch.py` |
| Repo documentation drift | `AGENTS.md` | `README.md`, `Code/README.md`, `Code/PROJECT_IO.md`, `CONTRIBUTING.md`, `docs/LAST_STATE.md` |

## Current Repository State

As of `2026-04-13`:

- Unified compare output has 4 modes: `center_local`, `local_importance`, `pooled_local`, `sample_aggregated_local`.
- `sample_aggregated_local` also emits native plots and `sample_aggregated_local_dependence_<feature>.png`.
- `Code/gwxgb/run_curated_output_batch.py` builds a curated run root under `Output/YYMMDD_n/`.
- That curated runner tries to reuse the latest yearly `gwxgb` outputs for `2005`, `2010`, `2015`, and `2020`, then regenerates `sample_aggregated_local` outputs and the SHAP interaction matrix.
- Compare configs default `output.capture_prints: 0` to avoid interpreter-shutdown logging issues; most other main scripts capture stdout/stderr into logs.

## Stable Guardrails

- YAML relative paths resolve relative to the YAML file location.
- Default configs are dataset-specific and use Chinese column names.
- `local_shap.enabled` is on in the default `gwxgb` config and can be expensive.
- `Code/xgb/xgb_shap.py` is not only an entry script; it also contains helper functions used by `gwxgb`.
- Generated files belong in `Output/`, not `Data/`.

## Minimum Docs To Keep In Sync

If behavior changes, update these in the same turn:

- `README.md`
- `Code/README.md`
- `Code/PROJECT_IO.md`
- `docs/repo_index.yaml`
- `docs/LAST_STATE.md`
