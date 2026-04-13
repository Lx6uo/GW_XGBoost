# LAST_STATE

## Last Updated

- 2026-04-13

## Stable Facts

- `Code/` is the `uv` project root.
- The unified compare runner currently outputs 4 modes: `center_local`, `local_importance`, `pooled_local`, `sample_aggregated_local`.
- `sample_aggregated_local` now has its own native plots and per-feature dependence plots.
- `Code/gwxgb/run_curated_output_batch.py` produces curated batch outputs under `Output/YYMMDD_n/`.
- Compare configs default `output.capture_prints: 0`; most other main scripts default `1`.

## Changes Landed In This Turn

- Added root `AGENTS.md` and workflow-specific directory guides under `Code/`.
- Added `docs/AGENT_CONTEXT.md` and `docs/repo_index.yaml` so new sessions can route without re-reading the full repo.
- Added a repo-specific skill at `.agents/skills/isml-workflow/`.
- Synced `README.md`, `Code/README.md`, `Code/PROJECT_IO.md`, and `CONTRIBUTING.md` with the current 4-mode compare suite and curated batch runner.

## Next-Time Maintenance Rule

When workflows, outputs, or entrypoints change, update this file in the same turn with one short bullet so the next session has a recent checkpoint.
