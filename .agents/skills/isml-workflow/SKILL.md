---
name: isml-workflow
description: Use when working in the ISML repository for XGBoost, GeoXGBoost, SHAP, compare-suite, curated batch export, or repo-documentation sync tasks. Routes the task to the smallest relevant workflow guide so Codex does not need to reread the full repository.
---

# ISML Workflow

When a task is inside this repository:

1. Read `AGENTS.md`.
2. Read `docs/AGENT_CONTEXT.md` and `docs/repo_index.yaml`.
3. Route to one guide only:
   - `Code/xgb/AGENTS.md`
   - `Code/gwxgb/AGENTS.md`
   - `Code/gwxgb/compare/AGENTS.md`
4. Open only the exact Python or YAML files needed for the task.
5. Respect repo guardrails:
   - `Code/` is code and config
   - `Data/` is input only
   - `Output/` is generated artifacts only
   - avoid editing `Code/.venv/`, `Code/.uv-cache/`, and `Code/uv.lock` unless explicitly asked
6. If workflow behavior or outputs change, update `README.md`, `Code/README.md`, `Code/PROJECT_IO.md`, `docs/repo_index.yaml`, and `docs/LAST_STATE.md` in the same turn.

Current repo facts that should guide routing:

- Unified compare output has 4 modes, not 3.
- `run_curated_output_batch.py` is the curated batch export entrypoint.
- Compare configs default `output.capture_prints: 0`.
- `xgb_shap.py` contains shared helpers imported by `gwxgb`.
