from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
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


def _params_key(params: Dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((str(k), _to_hashable(v)) for k, v in params.items()))


def _normalize_scoring_name(scoring: Any) -> str:
    name = str(scoring).strip().lower()
    aliases = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
    }
    return aliases.get(name, name)


def _params_to_json(params: Dict[str, Any]) -> str:
    safe: Dict[str, Any] = {str(k): _to_hashable(v) for k, v in params.items()}
    return json.dumps(safe, ensure_ascii=False, sort_keys=True)


def _outer_score_from_metrics(
    scoring: str, *, mse: float, rmse: float, mae: float, r2: float
) -> float:
    scoring_norm = _normalize_scoring_name(scoring)
    if scoring_norm == "r2":
        return r2
    if scoring_norm == "neg_root_mean_squared_error":
        return -rmse
    if scoring_norm == "neg_mean_absolute_error":
        return -mae
    if scoring_norm == "neg_mean_squared_error":
        return -mse
    # fallback：未知 scoring 时按 r2 处理
    return r2


def _early_stopping_best_n_estimators(estimator: Any) -> int | None:
    best_iteration = getattr(estimator, "best_iteration", None)
    if best_iteration is None:
        return None
    try:
        return int(best_iteration) + 1
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回命名空间。"""
    parser = argparse.ArgumentParser(
        description=(
            "使用外层 KFold + 内层 GridSearch 的嵌套交叉验证，对 XGBoost 超参数进行自动调优，"
            "并输出适合写回 config.yaml 的参数配置。"
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

    # 旧格式：最多同时调 3 个超参数
    param1 = tune_cfg.get("param1")
    param1_vals = tune_cfg.get("param1_values") or []
    param2 = tune_cfg.get("param2")
    param2_vals = tune_cfg.get("param2_values") or []
    param3 = tune_cfg.get("param3")
    param3_vals = tune_cfg.get("param3_values") or []

    if not param1 or not param1_vals:
        raise ValueError(
            "请在配置中提供 tuning.param_grid（推荐），或提供 tuning.param1/param1_values（旧格式）。"
        )

    param_grid: Dict[str, Any] = {param1: param1_vals}
    if param2 and param2_vals:
        param_grid[param2] = param2_vals
    if param3 and param3_vals:
        param_grid[param3] = param3_vals

    return param_grid


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

    tune_cfg = config.get("tuning") or {}
    if int(tune_cfg.get("use_nested_cv", 0)) == 0:
        logging.info(
            "tuning.use_nested_cv 为 0，当前不会执行嵌套交叉验证。"
            "如需启用，请在 config.yaml 中将其改为 1。"
        )
        return

    df, X, y = load_dataset(config)
    logging.info(
        f"已加载数据 `{config['data']['path']}`，"
        f"共 {df.shape[0]} 行，{df.shape[1]} 列。"
    )

    # 初始超参数来自 config.model.params
    model_cfg = config.get("model") or {}
    params: Dict[str, Any] = dict(model_cfg.get("params") or {})
    logging.info(f"初始超参数（来自 model.params）: {params}")

    param_grid = build_param_grid(config)
    n_candidates = 1
    for v in param_grid.values():
        n_candidates *= len(v)
    logging.info(f"参数组合数: {n_candidates}（提示：组合数越大运行越久）")

    outer_splits = int(tune_cfg.get("outer_splits", 5))
    inner_splits = int(tune_cfg.get("inner_splits", 3))
    random_state = int(model_cfg.get("random_state", 42))
    scoring = str(tune_cfg.get("scoring", "r2"))
    scoring_norm = _normalize_scoring_name(scoring)
    if scoring_norm != scoring:
        logging.info(f"scoring 归一化: {scoring} -> {scoring_norm}")
    else:
        logging.info(f"scoring: {scoring_norm}")

    es_cfg = tune_cfg.get("early_stopping") or {}
    es_enabled = isinstance(es_cfg, dict) and int(es_cfg.get("enabled", 0)) == 1
    es_rounds = int(es_cfg.get("rounds", 50)) if es_enabled else 0
    es_eval_fraction = float(es_cfg.get("eval_fraction", 0.2)) if es_enabled else 0.0
    es_eval_metric = str(es_cfg.get("eval_metric", "rmse")) if es_enabled else ""
    es_min_train_samples = int(es_cfg.get("min_train_samples", 20)) if es_enabled else 0
    if es_enabled:
        logging.info(
            "已启用 early stopping（用于调参阶段）："
            f"rounds={es_rounds}, eval_fraction={es_eval_fraction}, "
            f"eval_metric={es_eval_metric}, min_train_samples={es_min_train_samples}"
        )

    n_samples = len(X)
    if n_samples == 0:
        logging.warning("数据为空，跳过嵌套交叉验证。")
        return
    if outer_splits < 2 or outer_splits > n_samples:
        logging.warning(
            f"tuning.outer_splits 配置无效（outer_splits={outer_splits}, n_samples={n_samples}），将跳过。"
        )
        return
    # 外层 KFold 最坏情况下训练集大小（避免 inner_splits 大于训练集样本数）
    max_test_size = int(np.ceil(n_samples / outer_splits))
    min_train_size = n_samples - max_test_size
    if inner_splits < 2 or inner_splits > min_train_size:
        logging.warning(
            f"tuning.inner_splits 配置无效（inner_splits={inner_splits}, min_train_size={min_train_size}），将跳过。"
        )
        return

    logging.info("开始嵌套交叉验证调参（外层 KFold + 内层 GridSearchCV）...")

    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)

    # outer_metrics: (RMSE, MAE, NRMSE, R2, outer_score) for each outer fold
    outer_metrics: List[Tuple[float, float, float, float, float]] = []
    outer_best_params: List[Dict[str, Any]] = []
    inner_best_scores: List[float] = []
    outer_best_n_estimators: List[int | None] = []
    grid_verbose = int(tune_cfg.get("grid_verbose", tune_cfg.get("verbose", 0)))

    fold_idx = 0
    for train_idx, test_idx in outer_cv.split(X):
        fold_idx += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_params = dict(params)
        if "random_state" not in model_params:
            model_params["random_state"] = random_state
        if "n_jobs" not in model_params:
            model_params["n_jobs"] = 1
        if es_enabled:
            model_params["early_stopping_rounds"] = es_rounds
            model_params.setdefault("eval_metric", es_eval_metric)
            base_model = XGBRegressorWithEarlyStopping(
                es_eval_fraction=es_eval_fraction,
                es_shuffle=True,
                es_random_state=random_state,
                es_min_train_samples=es_min_train_samples,
                **model_params,
            )
        else:
            base_model = XGBRegressor(**model_params)
        inner_cv = KFold(
            n_splits=inner_splits, shuffle=True, random_state=random_state
        )

        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=scoring_norm,  # 内层按配置指标选择超参（支持 rmse/mae/mse 别名）
            cv=inner_cv,
            n_jobs=1,
            verbose=grid_verbose,
        )
        logging.info(
            f"Outer fold {fold_idx}/{outer_splits} - "
            "开始内层 GridSearchCV："
            f"candidates={n_candidates}, inner_cv={inner_splits}, total_fits={n_candidates * inner_splits}, "
            f"scoring={scoring_norm}, n_jobs=1, verbose={grid_verbose}"
        )
        search.fit(X_train, y_train)
        inner_best_scores.append(float(search.best_score_))

        best_model = search.best_estimator_
        best_n_estimators = (
            _early_stopping_best_n_estimators(best_model) if es_enabled else None
        )
        outer_best_n_estimators.append(best_n_estimators)
        y_pred = best_model.predict(X_test)

        y_true_arr = y_test.to_numpy(dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        mse = float(np.mean((y_true_arr - y_pred_arr) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
        y_min, y_max = float(np.min(y_true_arr)), float(np.max(y_true_arr))
        rng = y_max - y_min
        nrmse = float("nan") if rng == 0.0 else rmse / rng
        r2 = float(r2_score(y_true_arr, y_pred_arr))
        outer_score = float(
            _outer_score_from_metrics(scoring_norm, mse=mse, rmse=rmse, mae=mae, r2=r2)
        )

        outer_metrics.append((rmse, mae, nrmse, r2, outer_score))
        outer_best_params.append(search.best_params_)

        es_msg = ""
        if es_enabled:
            best_it = getattr(best_model, "best_iteration", None)
            if best_it is None or best_n_estimators is None:
                es_msg = ", early_stopping(best_iteration)=NA"
            else:
                es_msg = f", early_stopping(best_iteration={int(best_it)}, n_estimators={int(best_n_estimators)})"

        logging.info(
            f"Outer fold {fold_idx}/{outer_splits} - "
            f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, NRMSE: {nrmse:.4f}, R^2: {r2:.4f}, "
            f"inner_best_score({scoring_norm}): {float(search.best_score_):.6g}, "
            f"best_params: {search.best_params_}"
            f"{es_msg}"
        )

    rmses, maes, nrmse_s, r2s, outer_scores = zip(*outer_metrics)
    logging.info(
        "嵌套交叉验证结果（外层 {} 折） - RMSE mean: {:.4f} (std: {:.4f}), "
        "MAE mean: {:.4f} (std: {:.4f}), NRMSE mean: {:.4f} (std: {:.4f}), "
        "R^2 mean: {:.4f} (std: {:.4f})".format(
            outer_splits,
            float(np.mean(rmses)),
            float(np.std(rmses)),
            float(np.mean(maes)),
            float(np.std(maes)),
            float(np.mean(nrmse_s)),
            float(np.std(nrmse_s)),
            float(np.mean(r2s)),
            float(np.std(r2s)),
        )
    )

    # 保存 outer fold 明细（便于复盘与对比）
    outer_csv = output_dir / "nestedcv_outer_folds.csv"
    with outer_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "rmse",
                "mae",
                "nrmse",
                "r2",
                "score",
                "inner_best_score",
                "best_params_json",
                "best_iteration",
                "best_n_estimators",
            ],
        )
        writer.writeheader()
        for idx, (
            (rmse, mae, nrmse, r2, score),
            inner_best,
            best_params,
            best_n_estimators,
        ) in enumerate(
            zip(
                outer_metrics,
                inner_best_scores,
                outer_best_params,
                outer_best_n_estimators,
            ),
            start=1,
        ):
            best_iteration = None
            if es_enabled and best_n_estimators is not None:
                best_iteration = int(best_n_estimators) - 1
            writer.writerow(
                {
                    "fold": idx,
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "nrmse": float(nrmse),
                    "r2": float(r2),
                    "score": float(score),
                    "inner_best_score": float(inner_best),
                    "best_params_json": _params_to_json(best_params),
                    "best_iteration": best_iteration,
                    "best_n_estimators": best_n_estimators,
                }
            )
    logging.info(f"已保存 outer fold 明细: {outer_csv.resolve()}")

    if es_enabled:
        valid = [n for n in outer_best_n_estimators if isinstance(n, int)]
        if len(valid) == 0:
            logging.warning("early stopping 已启用，但未能从 outer folds 获得 best_iteration/best_n_estimators。")
        else:
            arr = np.asarray(valid, dtype=float)
            rec_median = int(np.median(arr))
            rec_mean = float(np.mean(arr))
            rec_min = int(np.min(arr))
            rec_max = int(np.max(arr))
            logging.info(
                "early stopping（outer folds）推荐 n_estimators 统计: "
                f"median={rec_median}, mean={rec_mean:.3g}, min={rec_min}, max={rec_max}, values={valid}"
            )
            logging.info(
                "如需将最佳迭代数固化为常量，可将 config.yaml 的 `model.params.n_estimators` 设为 "
                f"{rec_median}（建议取 outer folds median）。"
            )

    # 选择策略 1：外层单折最佳（用于参考；可能偏乐观）
    best_outer_idx = int(np.argmax(np.asarray(outer_scores)))
    best_outer_params = outer_best_params[best_outer_idx]
    logging.info(
        "外层单折最佳（按 scoring={}）: fold={}/{}, score={:.6g}, best_params={}".format(
            scoring_norm,
            best_outer_idx + 1,
            outer_splits,
            float(outer_scores[best_outer_idx]),
            best_outer_params,
        )
    )

    # 选择策略 2：外层“多数投票 + 平均表现”汇总（更稳健）
    agg: Dict[tuple[tuple[str, Any], ...], Dict[str, Any]] = {}
    for fold_params, (rmse, mae, nrmse, r2, score) in zip(
        outer_best_params, outer_metrics
    ):
        key = _params_key(fold_params)
        rec = agg.get(key)
        if rec is None:
            rec = {
                "params": fold_params,
                "count": 0,
                "rmse": [],
                "mae": [],
                "r2": [],
                "score": [],
            }
            agg[key] = rec
        rec["count"] += 1
        rec["rmse"].append(float(rmse))
        rec["mae"].append(float(mae))
        rec["r2"].append(float(r2))
        rec["score"].append(float(score))

    summary_rows: List[Dict[str, Any]] = []
    for rec in agg.values():
        summary_rows.append(
            {
                "count": int(rec["count"]),
                "score_mean": float(np.mean(rec["score"])),
                "score_std": float(np.std(rec["score"])),
                "rmse_mean": float(np.mean(rec["rmse"])),
                "mae_mean": float(np.mean(rec["mae"])),
                "r2_mean": float(np.mean(rec["r2"])),
                "params": rec["params"],
            }
        )

    summary_csv = output_dir / "nestedcv_param_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "count",
                "score_mean",
                "score_std",
                "rmse_mean",
                "mae_mean",
                "r2_mean",
                "params_json",
            ],
        )
        writer.writeheader()
        for row in sorted(
            summary_rows, key=lambda r: (r["count"], r["score_mean"]), reverse=True
        ):
            writer.writerow(
                {
                    "count": int(row["count"]),
                    "score_mean": float(row["score_mean"]),
                    "score_std": float(row["score_std"]),
                    "rmse_mean": float(row["rmse_mean"]),
                    "mae_mean": float(row["mae_mean"]),
                    "r2_mean": float(row["r2_mean"]),
                    "params_json": _params_to_json(row["params"]),
                }
            )
    logging.info(f"已保存参数汇总表: {summary_csv.resolve()}")

    # mode：出现次数最多；并用 score_mean 作为 tie-break（score 越大越好）
    mode_pick = max(summary_rows, key=lambda r: (r["count"], r["score_mean"]))
    mean_pick = max(summary_rows, key=lambda r: r["score_mean"])

    logging.info(
        "外层汇总（多数投票）推荐: count={}, score_mean={:.6g}, rmse_mean={:.6g}, r2_mean={:.6g}, params={}".format(
            int(mode_pick["count"]),
            float(mode_pick["score_mean"]),
            float(mode_pick["rmse_mean"]),
            float(mode_pick["r2_mean"]),
            mode_pick["params"],
        )
    )
    if mean_pick["params"] != mode_pick["params"]:
        logging.info(
            "外层汇总（平均分数最高）推荐: count={}, score_mean={:.6g}, rmse_mean={:.6g}, r2_mean={:.6g}, params={}".format(
                int(mean_pick["count"]),
                float(mean_pick["score_mean"]),
                float(mean_pick["rmse_mean"]),
                float(mean_pick["r2_mean"]),
                mean_pick["params"],
            )
        )

    def _log_yaml_snippet(title: str, best_params: Dict[str, Any]) -> None:
        params_tuned = dict(params)
        params_tuned.update(best_params)
        lines: List[str] = []
        lines.append(title)
        lines.append("")
        lines.append("model:")
        lines.append("  random_state: {}".format(model_cfg.get("random_state", 42)))
        lines.append("  params:")
        for k, v in params_tuned.items():
            if isinstance(v, float):
                lines.append(f"    {k}: {v:.6g}")
            else:
                lines.append(f"    {k}: {v}")
        logging.info("\n".join(lines))

    _log_yaml_snippet("建议写回（外层汇总-多数投票）的 `model.params`：", mode_pick["params"])

    # 选择策略 3：在全量数据上再跑一次 GridSearchCV，得到最终 best_params（用于最终训练/落盘）
    final_enabled = int(tune_cfg.get("final_grid_search", 0)) == 1
    if final_enabled:
        final_cv = int(tune_cfg.get("final_cv", inner_splits))
        if final_cv < 2 or final_cv > n_samples:
            logging.warning(
                f"\nfinal_grid_search: final_cv 配置无效（final_cv={final_cv}, n_samples={n_samples}），跳过全量搜索。"
            )
            return

        logging.info(
            f"开始在全量数据上执行最终 GridSearchCV（scoring={scoring_norm}, cv={final_cv}）..."
        )
        final_model_params = dict(params)
        final_model_params.setdefault("random_state", random_state)
        final_model_params.setdefault("n_jobs", 1)
        final_model = XGBRegressor(**final_model_params)
        final_kfold = KFold(n_splits=final_cv, shuffle=True, random_state=random_state)
        final_search = GridSearchCV(
            estimator=final_model,
            param_grid=param_grid,
            scoring=scoring_norm,
            cv=final_kfold,
            n_jobs=1,
            verbose=grid_verbose,
        )
        final_search.fit(X, y)
        logging.info(
            f"全量 GridSearchCV 完成: best_score={final_search.best_score_:.6g}, "
            f"best_params={final_search.best_params_}"
        )
        _log_yaml_snippet(
            "建议写回（全量 GridSearchCV 最优）的 `model.params`：",
            final_search.best_params_,
        )

    run_end = datetime.datetime.now()
    logging.info(
        f"本次运行结束: {run_end}（耗时 {(run_end - run_start).total_seconds():.2f} 秒）"
    )


if __name__ == "__main__":
    main()
