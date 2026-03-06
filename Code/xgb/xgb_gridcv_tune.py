from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

from xgb_early_stopping import XGBRegressorWithEarlyStopping
from xgb_shap import ensure_run_output_dir, load_config, load_dataset, setup_logging


def _to_hashable(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _params_to_json(params: Dict[str, Any]) -> str:
    safe: Dict[str, Any] = {str(k): _to_hashable(v) for k, v in params.items()}
    return json.dumps(safe, ensure_ascii=False, sort_keys=True)


def _normalize_scoring_name(scoring: Any) -> str:
    name = str(scoring).strip().lower()
    aliases = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
    }
    return aliases.get(name, name)


def _primary_refit_key(scoring_norm: str) -> str | None:
    mapping = {
        "r2": "r2",
        "neg_root_mean_squared_error": "rmse",
        "neg_mean_absolute_error": "mae",
        "neg_mean_squared_error": "mse",
    }
    return mapping.get(scoring_norm)


def _log_early_stopping_summary(*, estimator: Any, label: str) -> int | None:
    best_iteration = getattr(estimator, "best_iteration", None)
    best_score = getattr(estimator, "best_score", None)
    num_boosted_rounds: int | None = None
    try:
        num_boosted_rounds = int(estimator.get_booster().num_boosted_rounds())
    except Exception:
        num_boosted_rounds = None

    if best_iteration is None:
        logging.warning(
            f"{label}: early stopping 已启用，但 best_iteration 不可用（可能未触发 early stopping / 或样本过少）。"
        )
        return None

    try:
        best_it = int(best_iteration)
    except Exception:
        logging.warning(f"{label}: best_iteration={best_iteration} 无法转换为 int。")
        return None

    recommend_n_estimators = best_it + 1
    parts: List[str] = [f"best_iteration={best_it} -> recommend_n_estimators={recommend_n_estimators}"]
    if best_score is not None:
        parts.append(f"best_score={best_score}")
    if num_boosted_rounds is not None:
        parts.append(f"num_boosted_rounds={num_boosted_rounds}")
    logging.info(f"{label}: early stopping 摘要: " + ", ".join(parts))
    return recommend_n_estimators


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回命名空间。"""
    parser = argparse.ArgumentParser(
        description=(
            "单层 GridSearchCV 超参数搜索：对每个参数组合做 KFold 交叉验证，"
            "并按分数排序输出每一种组合的表现（同时落盘 CSV）。\n\n"
            "注意：该方式适合“选参”；best_score 往往偏乐观，不能等价为最终泛化性能。"
        )
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default=None,
        help="YAML 配置文件路径；若不指定则默认使用脚本同目录下的 config.yaml",
    )
    parser.add_argument(
        "--top",
        type=int,
        required=False,
        default=0,
        help="仅打印 Top-N 个候选（0 表示打印全部；默认 0）。",
    )
    parser.add_argument(
        "--pick",
        type=str,
        choices=["primary", "min_r2_gap"],
        default="primary",
        help=(
            "如何选择推荐参数："
            "primary=按 refit 主指标最优；"
            "min_r2_gap=按 train/test R2 差值最小（abs）"
        ),
    )
    return parser.parse_args()


def build_param_grid(config: Dict[str, Any]) -> Dict[str, Any]:
    """根据 config.tuning 构建超参数搜索网格。"""
    tune_cfg = config.get("tuning") or {}

    # 新格式：允许在 YAML 中直接写完整的 param_grid（推荐）
    param_grid_raw = tune_cfg.get("param_grid")
    if isinstance(param_grid_raw, dict) and param_grid_raw:
        param_grid: Dict[str, Any] = {}
        for k, v in param_grid_raw.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple)) and len(v) > 0:
                param_grid[str(k)] = list(v)
            else:
                raise TypeError(
                    f"tuning.param_grid.{k} 必须是非空列表，例如 [1, 2, 3]。"
                )
        if not param_grid:
            raise ValueError("tuning.param_grid 为空，请至少配置一个参数。")
        return param_grid

    raise ValueError("请在配置中提供 tuning.param_grid（例如 n_estimators/max_depth/...）。")


def _log_yaml_snippet(
    *,
    title: str,
    base_params: Dict[str, Any],
    best_params: Dict[str, Any],
    random_state: int,
) -> None:
    params_tuned = dict(base_params)
    params_tuned.update(best_params)

    lines: List[str] = []
    lines.append(title)
    lines.append("")
    lines.append("model:")
    lines.append(f"  random_state: {int(random_state)}")
    lines.append("  params:")
    for k, v in params_tuned.items():
        if isinstance(v, float):
            lines.append(f"    {k}: {v:.6g}")
        else:
            lines.append(f"    {k}: {v}")
    logging.info("\n".join(lines))


def _candidate_count(param_grid: Dict[str, Any]) -> int:
    n = 1
    for v in param_grid.values():
        n *= len(v)
    return int(n)


def main() -> None:
    run_start = datetime.datetime.now()
    args = parse_args()

    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).with_name("config.yaml")

    config = load_config(config_path)
    output_dir = ensure_run_output_dir(config, prefix="tune_")
    setup_logging(config, output_dir)
    logging.info(f"使用配置文件: {config_path}")
    logging.info(f"配置内容: {config}")

    df, X, y = load_dataset(config)
    logging.info(
        f"已加载数据 `{config['data']['path']}`，共 {df.shape[0]} 行，{df.shape[1]} 列。"
    )

    tune_cfg = config.get("tuning") or {}
    param_grid = build_param_grid(config)
    n_candidates = _candidate_count(param_grid)
    logging.info(f"参数组合数: {n_candidates}（提示：组合数越大运行越久）")

    model_cfg = config.get("model") or {}
    base_params: Dict[str, Any] = dict(model_cfg.get("params") or {})
    random_state = int(model_cfg.get("random_state", 42))

    scoring_raw = tune_cfg.get("scoring", "r2")
    scoring_norm = _normalize_scoring_name(scoring_raw)
    refit_key = _primary_refit_key(scoring_norm)
    grid_verbose = int(tune_cfg.get("grid_verbose", tune_cfg.get("verbose", 1)))

    # 单层 CV 的折数：优先使用 tuning.cv_splits；否则复用 outer_splits；最后 fallback=5
    cv_splits = int(tune_cfg.get("cv_splits", tune_cfg.get("outer_splits", 5)))
    if cv_splits < 2 or cv_splits > len(X):
        raise ValueError(f"tuning.cv_splits 配置无效：cv_splits={cv_splits}。")

    # 模型与 CV
    model_params = dict(base_params)
    model_params.setdefault("random_state", random_state)
    model_params.setdefault("n_jobs", 1)
    es_cfg = tune_cfg.get("early_stopping") or {}
    es_enabled = isinstance(es_cfg, dict) and int(es_cfg.get("enabled", 0)) == 1
    if es_enabled:
        rounds = int(es_cfg.get("rounds", 50))
        eval_fraction = float(es_cfg.get("eval_fraction", 0.2))
        eval_metric = str(es_cfg.get("eval_metric", "rmse"))
        min_train_samples = int(es_cfg.get("min_train_samples", 20))

        model_params["early_stopping_rounds"] = rounds
        model_params.setdefault("eval_metric", eval_metric)
        estimator = XGBRegressorWithEarlyStopping(
            es_eval_fraction=eval_fraction,
            es_shuffle=True,
            es_random_state=random_state,
            es_min_train_samples=min_train_samples,
            **model_params,
        )
        logging.info(
            "已启用 early stopping（用于调参阶段）："
            f"rounds={rounds}, eval_fraction={eval_fraction}, "
            f"eval_metric={model_params.get('eval_metric')}, min_train_samples={min_train_samples}"
        )
    else:
        estimator = XGBRegressor(**model_params)

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # 优先采用多指标输出（便于同时看 rmse/mae/r2）；refit 指向用户选择的主指标
    if refit_key is not None:
        scoring: Dict[str, str] = {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "r2": "r2",
        }
        refit = refit_key
        logging.warning(
            "单层 GridSearchCV：best_score 用于选参会偏乐观；如需更可信泛化评估请用 nested CV 或独立测试集。"
        )
        logging.info(f"scoring(primary)={scoring_norm} -> refit={refit}")
    else:
        # 即便 primary scoring 不是内置别名，也尽量同时输出 r2，方便观察“过拟合差距”
        scoring = {"score": scoring_norm, "r2": "r2"}
        refit = "score"
        logging.warning(
            "单层 GridSearchCV：best_score 用于选参会偏乐观；如需更可信泛化评估请用 nested CV 或独立测试集。"
        )
        logging.info(
            f"scoring(primary)={scoring_norm}（非内置别名；同时输出 r2 便于观察差距）"
        )

    logging.info(
        "开始单层 GridSearchCV："
        f"cv={cv_splits}, candidates={n_candidates}, total_fits={n_candidates * cv_splits}, "
        f"n_jobs=1, verbose={grid_verbose}"
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit,
        cv=cv,
        n_jobs=1,
        verbose=grid_verbose,
        return_train_score=True,
    )
    search.fit(X, y)

    results = pd.DataFrame(search.cv_results_)

    # 将 params 展示为 JSON，便于落盘/排序/比对
    results["params_json"] = results["params"].apply(
        lambda p: _params_to_json(p if isinstance(p, dict) else {})
    )

    # 生成可读的指标列（rmse/mae/mse 转为正值）
    def _maybe_add_pos(metric_key: str) -> None:
        mean_col = f"mean_test_{metric_key}"
        std_col = f"std_test_{metric_key}"
        tr_mean_col = f"mean_train_{metric_key}"
        tr_std_col = f"std_train_{metric_key}"
        if mean_col in results.columns:
            results[f"test_{metric_key}_mean"] = -results[mean_col]
        if std_col in results.columns:
            results[f"test_{metric_key}_std"] = results[std_col]
        if tr_mean_col in results.columns:
            results[f"train_{metric_key}_mean"] = -results[tr_mean_col]
        if tr_std_col in results.columns:
            results[f"train_{metric_key}_std"] = results[tr_std_col]

    for k in ("rmse", "mae", "mse"):
        _maybe_add_pos(k)

    # r2 为正向指标，不做取负
    if "mean_test_r2" in results.columns:
        results["test_r2_mean"] = results["mean_test_r2"]
    if "std_test_r2" in results.columns:
        results["test_r2_std"] = results["std_test_r2"]
    if "mean_train_r2" in results.columns:
        results["train_r2_mean"] = results["mean_train_r2"]
    if "std_train_r2" in results.columns:
        results["train_r2_std"] = results["std_train_r2"]

    if "train_r2_mean" in results.columns and "test_r2_mean" in results.columns:
        results["r2_gap"] = results["train_r2_mean"] - results["test_r2_mean"]
        results["r2_gap_abs"] = results["r2_gap"].abs()

    rank_col = f"rank_test_{refit}" if f"rank_test_{refit}" in results.columns else "rank_test_score"
    if rank_col not in results.columns:
        raise RuntimeError(f"未找到 rank 列（期望 {rank_col}）。")

    results_sorted = results.sort_values(rank_col, ascending=True).reset_index(drop=True)

    out_dir = output_dir / "gridcv"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_csv = out_dir / "gridcv_candidates.csv"
    # 只输出一组稳定字段（避免过多 sklearn 内部列）
    field_candidates: List[str] = [
        rank_col,
        "params_json",
        "mean_fit_time",
        "std_fit_time",
    ]
    for col in (
        "test_rmse_mean",
        "test_rmse_std",
        "test_mae_mean",
        "test_mae_std",
        "test_mse_mean",
        "test_mse_std",
        "test_r2_mean",
        "test_r2_std",
        "train_rmse_mean",
        "train_mae_mean",
        "train_mse_mean",
        "train_r2_mean",
        "r2_gap",
        "r2_gap_abs",
    ):
        if col in results_sorted.columns:
            field_candidates.append(col)

    # 落盘 CSV（全量候选）
    results_sorted[field_candidates].to_csv(
        candidates_csv, index=False, encoding="utf-8-sig"
    )
    logging.info(f"已保存候选排行榜（全量）：{candidates_csv.resolve()}")

    # 逐条打印（可能很长）
    top_n = int(args.top)
    if top_n < 0:
        top_n = 0
    n_print = len(results_sorted) if top_n == 0 else min(len(results_sorted), top_n)
    logging.info(f"开始输出候选排行：共 {len(results_sorted)}，本次打印 {n_print} 条。")

    for i in range(n_print):
        row = results_sorted.iloc[i]
        parts: List[str] = []
        parts.append(f"rank={int(row[rank_col])}")
        for k in ("rmse", "mae", "mse", "r2"):
            test_mean = f"test_{k}_mean"
            test_std = f"test_{k}_std"
            if test_mean in results_sorted.columns:
                if test_std in results_sorted.columns:
                    parts.append(f"{k}={float(row[test_mean]):.6g}±{float(row[test_std]):.3g}")
                else:
                    parts.append(f"{k}={float(row[test_mean]):.6g}")
        if "r2_gap" in results_sorted.columns:
            parts.append(f"r2_gap={float(row['r2_gap']):.6g}")
        parts.append(f"params={row['params_json']}")
        logging.info(" | ".join(parts))

    best_params_primary = search.best_params_
    best_score_primary = float(search.best_score_)
    if refit in {"rmse", "mae", "mse"}:
        best_score_display = -best_score_primary
        logging.info(
            f"primary best_score({refit})={best_score_display:.6g}（由 sklearn 的 neg_* scorer 转换为正值）"
        )
    else:
        logging.info(f"primary best_score({refit})={best_score_primary:.6g}")
    logging.info(f"primary best_params={best_params_primary}")

    pick_mode = str(args.pick).strip().lower()
    picked_params: Dict[str, Any] = dict(best_params_primary)
    picked_tag = f"primary({refit})"

    if pick_mode == "min_r2_gap":
        if "r2_gap_abs" not in results_sorted.columns or "params" not in results_sorted.columns:
            logging.warning("无法按 r2_gap 选择：缺少 r2 训练/测试列。将回退到 primary 选择。")
        else:
            tmp = results_sorted.copy()
            tmp = tmp[np.isfinite(tmp["r2_gap_abs"].to_numpy(dtype=float))]
            if len(tmp) == 0:
                logging.warning("无法按 r2_gap 选择：所有 gap 均为 NaN/Inf。将回退到 primary 选择。")
            else:
                sort_cols: List[str] = ["r2_gap_abs"]
                ascending: List[bool] = [True]
                if "test_r2_mean" in tmp.columns:
                    sort_cols.append("test_r2_mean")
                    ascending.append(False)
                tmp_sorted = tmp.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
                row0 = tmp_sorted.iloc[0]
                params0 = row0["params"]
                if isinstance(params0, dict):
                    picked_params = dict(params0)
                    picked_tag = "min_r2_gap(abs)"
                    logging.info(
                        "按 r2_gap_abs 选择："
                        f"gap_abs={float(row0['r2_gap_abs']):.6g}, "
                        f"gap={float(row0['r2_gap']):.6g}, "
                        f"test_r2={float(row0.get('test_r2_mean', float('nan'))):.6g}, "
                        f"train_r2={float(row0.get('train_r2_mean', float('nan'))):.6g}"
                    )
                else:
                    logging.warning("无法按 r2_gap 选择：params 列不是 dict。将回退到 primary 选择。")

    _log_yaml_snippet(
        title=f"建议写回（单层 GridSearchCV；选择方式={picked_tag}）的 `model.params`：",
        base_params=base_params,
        best_params=picked_params,
        random_state=random_state,
    )

    if es_enabled:
        es_recommend = _log_early_stopping_summary(
            estimator=search.best_estimator_,
            label="GridSearchCV refit(best_estimator_)",
        )
        if picked_tag != f"primary({refit})":
            logging.warning(
                "注意：你选择的不是 GridSearchCV 的 refit 最优参数，因此上述 best_iteration 对应的是 primary(refit) 选择；"
                "如需为当前 picked 参数单独确定 best_iteration，请用该参数在全量数据上单独 fit 一次（同样启用 early stopping）。"
            )
        if es_recommend is not None:
            logging.info(
                "如需将 early stopping 的最佳迭代数固化为常量，可将 config.yaml 的 `model.params.n_estimators` 设为 "
                f"{es_recommend}（best_iteration+1）。"
            )

    run_end = datetime.datetime.now()
    logging.info(
        f"本次运行结束: {run_end}（耗时 {(run_end - run_start).total_seconds():.2f} 秒）"
    )


if __name__ == "__main__":
    main()
