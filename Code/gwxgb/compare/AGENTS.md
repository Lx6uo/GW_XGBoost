# Compare Workflow Guide

Use this guide when touching files under `Code/gwxgb/compare/`.

## Read Order

1. `../../../AGENTS.md`
2. `../../../docs/AGENT_CONTEXT.md`
3. `../../../docs/repo_index.yaml`
4. `README.md`
5. Relevant compare config and script files

## Main Files

- `gwxgb_compare_all.py`: unified runner
- `gwxgb_compare_demo_common.py`: shared compare logic
- `gwxgb_compare_all_config.yaml`: unified config
- `gwxgb_compare_*_config.yaml`: mode-specific configs

## Current Compare Modes

- `center_local`
- `local_importance`
- `pooled_local`
- `sample_aggregated_local`

## Current Behavior

- Compare configs inherit `../gwxgb_config.yaml` through `base_config`.
- Unified compare output defaults `output.capture_prints: 0`.
- `sample_aggregated_local` emits CSV rows, native plots, and `sample_aggregated_local_dependence_<feature>.png`.
- `local_importance` keeps both strength and direction outputs.

## Guardrails

- Do not document only 3 compare modes; the current repo state is 4.
- If output names or directories change, update both compare docs and top-level docs in the same turn.
- Changes in `gwxgb_compare_demo_common.py` can affect every compare entrypoint.

## Common Validation Commands

From `Code/`:

- `uv run python .\gwxgb\compare\gwxgb_compare_all.py -c .\gwxgb\compare\gwxgb_compare_all_config.yaml`
- `uv run python .\gwxgb\compare\gwxgb_compare_center_local_demo.py -c .\gwxgb\compare\gwxgb_compare_center_local_config.yaml`

## Required Doc Sync

If compare behavior changes, update:

- `../../../README.md`
- `../../README.md`
- `../../PROJECT_IO.md`
- `README.md`
- `../../../docs/repo_index.yaml`
- `../../../docs/LAST_STATE.md`
