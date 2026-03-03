from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

from xgb_shap import load_config, load_dataset


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
    args = parse_args()
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).with_name("config.yaml")
        print(f"未指定配置文件路径，将使用默认配置文件: {config_path}")

    config = load_config(config_path)

    tune_cfg = config.get("tuning") or {}
    if int(tune_cfg.get("use_nested_cv", 0)) == 0:
        print(
            "tuning.use_nested_cv 为 0，当前不会执行嵌套交叉验证。"
            "如需启用，请在 config.yaml 中将其改为 1。"
        )
        return

    print(f"使用配置文件: {config_path}")
    df, X, y = load_dataset(config)
    print(
        f"已加载数据 `{config['data']['path']}`，"
        f"共 {df.shape[0]} 行，{df.shape[1]} 列。"
    )

    # 初始超参数来自 config.model.params
    model_cfg = config.get("model") or {}
    params: Dict[str, Any] = dict(model_cfg.get("params") or {})
    print("初始超参数（来自 model.params）:")
    print(params)

    param_grid = build_param_grid(config)
    n_candidates = 1
    for v in param_grid.values():
        n_candidates *= len(v)
    print(f"参数组合数: {n_candidates}（提示：组合数越大运行越久）")

    outer_splits = int(tune_cfg.get("outer_splits", 5))
    inner_splits = int(tune_cfg.get("inner_splits", 3))
    random_state = int(model_cfg.get("random_state", 42))
    scoring = str(tune_cfg.get("scoring", "r2"))
    scoring_norm = _normalize_scoring_name(scoring)

    n_samples = len(X)
    if n_samples == 0:
        print("数据为空，跳过嵌套交叉验证。")
        return
    if outer_splits < 2 or outer_splits > n_samples:
        print(
            f"tuning.outer_splits 配置无效（outer_splits={outer_splits}, n_samples={n_samples}），将跳过。"
        )
        return
    # 外层 KFold 最坏情况下训练集大小（避免 inner_splits 大于训练集样本数）
    max_test_size = int(np.ceil(n_samples / outer_splits))
    min_train_size = n_samples - max_test_size
    if inner_splits < 2 or inner_splits > min_train_size:
        print(
            f"tuning.inner_splits 配置无效（inner_splits={inner_splits}, min_train_size={min_train_size}），将跳过。"
        )
        return

    print("\n开始嵌套交叉验证调参（外层 KFold + 内层 GridSearchCV）...")

    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)

    # outer_metrics: (RMSE, MAE, NRMSE, R2, outer_score) for each outer fold
    outer_metrics: List[Tuple[float, float, float, float, float]] = []
    outer_best_params: List[Dict[str, Any]] = []

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
        base_model = XGBRegressor(**model_params)
        inner_cv = KFold(
            n_splits=inner_splits, shuffle=True, random_state=random_state
        )

        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=scoring,  # 内层按配置指标选择超参
            cv=inner_cv,
            n_jobs=1,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
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

        print(
            f"Outer fold {fold_idx}/{outer_splits} - "
            f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, NRMSE: {nrmse:.4f}, R^2: {r2:.4f}, "
            f"best_params: {search.best_params_}"
        )

    rmses, maes, nrmse_s, r2s, outer_scores = zip(*outer_metrics)
    print(
        f"\n嵌套交叉验证结果（外层 {outer_splits} 折） - "
        f"RMSE mean: {np.mean(rmses):.4f} (std: {np.std(rmses):.4f}), "
        f"MAE mean: {np.mean(maes):.4f} (std: {np.std(maes):.4f}), "
        f"NRMSE mean: {np.mean(nrmse_s):.4f} (std: {np.std(nrmse_s):.4f}), "
        f"R^2 mean: {np.mean(r2s):.4f} (std: {np.std(r2s):.4f})"
    )

    # 选择策略 1：外层单折最佳（用于参考；可能偏乐观）
    best_outer_idx = int(np.argmax(np.asarray(outer_scores)))
    best_outer_params = outer_best_params[best_outer_idx]
    print(
        f"\n外层单折最佳（按 scoring={scoring_norm}）: "
        f"fold={best_outer_idx + 1}/{outer_splits}, score={outer_scores[best_outer_idx]:.6g}, "
        f"best_params={best_outer_params}"
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

    # mode：出现次数最多；并用 score_mean 作为 tie-break（score 越大越好）
    mode_pick = max(summary_rows, key=lambda r: (r["count"], r["score_mean"]))
    mean_pick = max(summary_rows, key=lambda r: r["score_mean"])

    print(
        "\n外层汇总（多数投票）推荐:"
        f" count={mode_pick['count']}, score_mean={mode_pick['score_mean']:.6g}, "
        f"rmse_mean={mode_pick['rmse_mean']:.6g}, r2_mean={mode_pick['r2_mean']:.6g}, "
        f"params={mode_pick['params']}"
    )
    if mean_pick["params"] != mode_pick["params"]:
        print(
            "\n外层汇总（平均分数最高）推荐:"
            f" count={mean_pick['count']}, score_mean={mean_pick['score_mean']:.6g}, "
            f"rmse_mean={mean_pick['rmse_mean']:.6g}, r2_mean={mean_pick['r2_mean']:.6g}, "
            f"params={mean_pick['params']}"
        )

    def _print_yaml_snippet(title: str, best_params: Dict[str, Any]) -> None:
        params_tuned = dict(params)
        params_tuned.update(best_params)
        print(f"\n{title}\n")
        print("model:")
        print("  random_state: {}".format(model_cfg.get("random_state", 42)))
        print("  params:")
        for k, v in params_tuned.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6g}")
            else:
                print(f"    {k}: {v}")

    _print_yaml_snippet("建议写回（外层汇总-多数投票）的 `model.params`：", mode_pick["params"])

    # 选择策略 3：在全量数据上再跑一次 GridSearchCV，得到最终 best_params（用于最终训练/落盘）
    final_enabled = int(tune_cfg.get("final_grid_search", 0)) == 1
    if final_enabled:
        final_cv = int(tune_cfg.get("final_cv", inner_splits))
        if final_cv < 2 or final_cv > n_samples:
            print(
                f"\nfinal_grid_search: final_cv 配置无效（final_cv={final_cv}, n_samples={n_samples}），跳过全量搜索。"
            )
            return

        print(
            f"\n开始在全量数据上执行最终 GridSearchCV（scoring={scoring_norm}, cv={final_cv}）..."
        )
        final_model_params = dict(params)
        final_model_params.setdefault("random_state", random_state)
        final_model_params.setdefault("n_jobs", 1)
        final_model = XGBRegressor(**final_model_params)
        final_kfold = KFold(n_splits=final_cv, shuffle=True, random_state=random_state)
        final_search = GridSearchCV(
            estimator=final_model,
            param_grid=param_grid,
            scoring=scoring,
            cv=final_kfold,
            n_jobs=1,
        )
        final_search.fit(X, y)
        print(
            f"全量 GridSearchCV 完成: best_score={final_search.best_score_:.6g}, "
            f"best_params={final_search.best_params_}"
        )
        _print_yaml_snippet(
            "建议写回（全量 GridSearchCV 最优）的 `model.params`：",
            final_search.best_params_,
        )


if __name__ == "__main__":
    main()
