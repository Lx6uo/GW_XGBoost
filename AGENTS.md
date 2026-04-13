# ISML Agent Guide

## Start Here

New sessions should avoid re-reading the whole repository. Read files in this order:

1. `AGENTS.md`
2. `docs/AGENT_CONTEXT.md`
3. `docs/repo_index.yaml`
4. One relevant sub-guide only:
   - `Code/xgb/AGENTS.md`
   - `Code/gwxgb/AGENTS.md`
   - `Code/gwxgb/compare/AGENTS.md`
5. Only then open the exact `.py` / `.yaml` files needed for the task.

## Hard Rules

- `Code/` is the code and config root. `Data/` is input only. `Output/` is generated artifacts only.
- Prefer relative paths in YAML. Runtime resolves them relative to the YAML file, not the current shell directory.
- Do not edit `Code/.venv/`, `Code/.uv-cache/`, `Code/uv.lock`, or archived files under `Output/` unless the user explicitly asks.
- Do not move generated files back into `Data/`.
- When workflow behavior, entrypoints, config keys, or output names change, update the relevant docs in the same turn.

## Task Routing

- Global XGBoost, preprocessing, tuning, robustness: `Code/xgb/AGENTS.md`
- GeoXGBoost, bandwidth, local SHAP export, curated batch export: `Code/gwxgb/AGENTS.md`
- Global-vs-local SHAP compare suite: `Code/gwxgb/compare/AGENTS.md`
- Repo-wide orientation and recent state: `docs/AGENT_CONTEXT.md` and `docs/LAST_STATE.md`

## Current Facts Worth Trusting

- The unified compare runner currently outputs 4 modes: `center_local`, `local_importance`, `pooled_local`, `sample_aggregated_local`.
- `Code/gwxgb/run_curated_output_batch.py` is the curated batch runner for `2005`, `2010`, `2015`, `2020`, and `full`.
- Compare configs default `output.capture_prints: 0`; most other main scripts default `1`.
- `Code/xgb/xgb_shap.py` and `Code/xgb/xgb_shap_interaction_matrix.py` expose helper functions imported by `Code/gwxgb/`.

## Required Doc Sync

When you change:

- entrypoints or workflow behavior: update `README.md`, `Code/README.md`, `Code/PROJECT_IO.md`
- repo conventions or task routing: update `AGENTS.md` and `docs/repo_index.yaml`
- current state or newly added workflows: update `docs/LAST_STATE.md`
