from __future__ import annotations

import argparse
import atexit
import datetime
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import shap
import xgboost as xgb
import yaml
import numpy as np

rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Arial Unicode MS",
]
rcParams["axes.unicode_minus"] = False


def _get_config_base_dir(config: Dict[str, Any]) -> Path:
    config_dir = config.get("_config_dir")
    if isinstance(config_dir, str) and config_dir:
        return Path(config_dir)
    return Path.cwd()


def _resolve_path(path_value: Any, base_dir: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _parse_version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for raw in str(version).split("."):
        digits = ""
        for ch in raw:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _compute_xgboost_native_shap(
    model: xgb.XGBModel, X: pd.DataFrame, compute_interactions: bool
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """使用 xgboost 原生 pred_contribs / pred_interactions 计算 SHAP。"""
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    dmatrix = xgb.DMatrix(X)

    shap_contribs = booster.predict(dmatrix, pred_contribs=True)
    contribs_arr = np.asarray(shap_contribs, dtype=float)
    if contribs_arr.ndim == 3:
        # 多分类/多目标场景下可能返回 (n, k, p+1)，此处保持与 shap.TreeExplainer 的旧逻辑一致：取最后一组
        contribs_arr = contribs_arr[:, -1, :]
    shap_values_array = contribs_arr[:, :-1]

    interaction_values_array: Optional[np.ndarray] = None
    if compute_interactions:
        interactions = booster.predict(dmatrix, pred_interactions=True)
        inter_arr = np.asarray(interactions, dtype=float)
        if inter_arr.ndim == 4:
            inter_arr = inter_arr[:, -1, :, :]
        interaction_values_array = inter_arr[:, :-1, :-1]

    return shap_values_array, interaction_values_array


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回命名空间。"""
    parser = argparse.ArgumentParser(
        description=(
            "基于 YAML 配置训练 XGBoost 模型，"
            "并生成 SHAP 重要性和交互作用图。"
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


def load_config(path: Path) -> Dict[str, Any]:
    """加载 YAML 配置文件; 参数: path 为配置文件路径。"""
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise TypeError("配置文件必须解析为 YAML dict。")
    config_path = path.resolve()
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def load_dataset(config: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """根据配置加载数据集; 参数: config 为整体配置字典。"""
    data_cfg = config["data"]
    base_dir = _get_config_base_dir(config)
    data_path = _resolve_path(data_cfg["path"], base_dir)
    data_cfg["path"] = str(data_path)
    df = pd.read_csv(
        data_path,
        sep=data_cfg.get("sep", ","),
        encoding=data_cfg.get("encoding", "utf-8"),
    )

    # 自动尝试将字符串列（含百分号/空格）转换为数值列
    for col in df.columns:
        if not (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ):
            continue
        try:
            series_cleaned = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
                .replace({"nan": np.nan, "None": np.nan, "": np.nan})
            )
            df[col] = pd.to_numeric(series_cleaned)
            logging.info(f"已自动将列 `{col}` 中的字符串数值转换为数字。")
        except (ValueError, TypeError):
            # 转换失败，说明可能不是纯数字/百分号列（例如真正的类别文本），跳过
            continue

    target = data_cfg["target"]
    features = data_cfg.get("features") or [c for c in df.columns if c != target]
    X = df[features]
    y = df[target]
    return df, X, y


def build_and_train_model(
    config: Dict[str, Any], X: pd.DataFrame, y: pd.Series
) -> xgb.XGBModel:
    """构建并训练 XGBoost 模型; 参数: config 配置, X 特征, y 标签。"""
    model_cfg = config.get("model") or {}
    params: Dict[str, Any] = dict(model_cfg.get("params") or {})

    # 兼容旧配置：允许将随机种子写在 `model.random_state`（而不是 `model.params.random_state`）
    if "random_state" not in params and model_cfg.get("random_state") is not None:
        params["random_state"] = int(model_cfg["random_state"])

    if "n_jobs" not in params and model_cfg.get("n_jobs") is not None:
        params["n_jobs"] = int(model_cfg["n_jobs"])
    model: xgb.XGBModel = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def cross_validate_model(config: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> None:
    """对全量数据执行 K 折交叉验证，仅打印评估指标。"""
    cv_cfg = config.get("cv") or {}
    if not int(cv_cfg.get("use_cv", 0)):
        return
    n_splits = int(cv_cfg.get("n_splits", 5))
    n_samples = len(X)
    if n_samples == 0:
        logging.info("交叉验证: 数据为空，跳过。")
        return
    if n_splits < 2:
        logging.info("交叉验证: `cv.n_splits` 必须 >= 2，当前将跳过。")
        return
    if n_splits > n_samples:
        logging.info(
            f"交叉验证: `cv.n_splits`({n_splits}) 大于样本数({n_samples})，将跳过。"
        )
        return
    model_cfg = config.get("model") or {}
    random_state = int(cv_cfg.get("random_state", model_cfg.get("random_state", 42)))
    indices = np.random.RandomState(random_state).permutation(n_samples)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop

    metrics = []
    logging.info(f"开始 {n_splits} 折交叉验证:")
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])

        X_train_cv = X.iloc[train_idx].reset_index(drop=True)
        y_train_cv = y.iloc[train_idx].reset_index(drop=True)
        X_val_cv = X.iloc[val_idx].reset_index(drop=True)
        y_val_cv = y.iloc[val_idx].reset_index(drop=True)
        model_cv = build_and_train_model(config, X_train_cv, y_train_cv)
        y_pred_val = model_cv.predict(X_val_cv)

        y_true_arr = np.asarray(y_val_cv, dtype=float)
        y_pred_arr = np.asarray(y_pred_val, dtype=float)
        mse = float(np.mean((y_true_arr - y_pred_arr) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
        y_min, y_max = float(np.min(y_true_arr)), float(np.max(y_true_arr))
        rng = y_max - y_min
        nrmse = float("nan") if rng == 0.0 else rmse / rng
        ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
        ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
        r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
        metrics.append((rmse, mae, nrmse, r2))
        logging.info(
            f"折 {i + 1}/{n_splits} - 验证集 "
            f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, NRMSE: {nrmse:.4f}, R^2: {r2:.4f}"
        )

    rmses, maes, nrmse_s, r2s = zip(*metrics)
    logging.info(
        f"{n_splits} 折交叉验证结果 - "
        f"RMSE 平均: {np.mean(rmses):.4f}, RMSE 标准差: {np.std(rmses):.4f}, "
        f"MAE 平均: {np.mean(maes):.4f}, MAE 标准差: {np.std(maes):.4f}, "
        f"NRMSE 平均: {np.mean(nrmse_s):.4f}, NRMSE 标准差: {np.std(nrmse_s):.4f}, "
        f"R^2 平均: {np.mean(r2s):.4f}, R^2 标准差: {np.std(r2s):.4f}"
    )


def compute_shap_and_interactions(
    model: xgb.XGBModel, X: pd.DataFrame, config: Dict[str, Any]
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """计算样本的 SHAP 值和特征交互值。"""
    shap_cfg = config.get("shap") or {}
    compute_interactions = int(shap_cfg.get("compute_interactions", 0)) == 1
    interaction_values_array: Optional[np.ndarray] = None

    engine = str(shap_cfg.get("engine", "auto")).strip().lower()
    if engine not in {"auto", "shap", "xgboost"}:
        engine = "auto"

    # xgboost>=3 会将 base_score 写成字符串形式的数组，如 "[8.4462683E2]"；
    # shap<0.50 的 TreeExplainer 在解析该字段时可能报 `could not convert string to float`。
    # 为避免在默认环境里频繁触发异常/回退，这里在 auto 模式下直接使用 xgboost 原生 SHAP。
    if engine in {"xgboost", "auto"}:
        shap_v = _parse_version_tuple(getattr(shap, "__version__", "0"))
        xgb_v = _parse_version_tuple(getattr(xgb, "__version__", "0"))
        if engine == "xgboost" or (xgb_v >= (3, 0, 0) and shap_v < (0, 50, 0)):
            return _compute_xgboost_native_shap(model, X, compute_interactions)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values_array = np.asarray(shap_values[-1])
        else:
            shap_values_array = np.asarray(shap_values)

        if compute_interactions:
            interaction_values = explainer.shap_interaction_values(X)
            if isinstance(interaction_values, list):
                interaction_values_array = np.asarray(interaction_values[-1])
            else:
                interaction_values_array = np.asarray(interaction_values)

        return shap_values_array, interaction_values_array

    except Exception as exc:
        # shap.TreeExplainer 失败时，回退到 xgboost 原生的 pred_contribs / pred_interactions。
        logging.warning(
            "警告: shap.TreeExplainer 初始化失败，将回退到 xgboost 原生 SHAP 计算。"
            f"原因: {exc}"
        )
        return _compute_xgboost_native_shap(model, X, compute_interactions)


def ensure_output_dir(config: Dict[str, Any]) -> Path:
    """确保输出目录存在并返回路径; 参数: config 配置。"""
    output_cfg = config.get("output")
    if not isinstance(output_cfg, dict):
        output_cfg = {}
        config["output"] = output_cfg

    base_dir = _get_config_base_dir(config)
    output_dir = output_cfg.get("output_dir", "outputs")
    output_path = _resolve_path(output_dir, base_dir)
    output_cfg["output_dir"] = str(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _sanitize_subdir_name(name: str) -> str:
    # Windows/macOS/Linux 通用：保留字母数字 + 少量安全字符，其余替换为 '-'
    safe_chars = set("-_.")
    cleaned = "".join(ch if (ch.isalnum() or ch in safe_chars) else "-" for ch in str(name))
    cleaned = cleaned.strip(" .-_")
    return cleaned or "run"


def _sanitize_subdir_part(part: str) -> str:
    """用于子目录命名的“片段”清洗：空片段保持为空（不回退到 'run'）。"""
    safe_chars = set("-_.")
    cleaned = "".join(
        ch if (ch.isalnum() or ch in safe_chars) else "-" for ch in str(part)
    )
    return cleaned.strip(" .-_")


def _dataset_stem_from_config(config: Dict[str, Any]) -> str:
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        return ""
    path_value = data_cfg.get("path")
    if not path_value:
        return ""
    try:
        return Path(str(path_value)).stem
    except Exception:
        return ""


def ensure_run_output_dir(config: Dict[str, Any], *, prefix: str = "") -> Path:
    """按运行时间创建独立输出目录，并将 config.output.output_dir 指向该目录。

    - base 输出目录来自 `output.output_dir`
    - run 子目录默认名：`<run_prefix>_<data_stem>_<timestamp>`（默认会尝试从 `data.path` 推断 data_stem）
    - 可通过以下配置覆盖：
      - output.timestamp_subdir: 0/1（默认 1）
      - output.timestamp_format: str（默认 "%Y%m%d_%H%M%S"）
      - output.run_prefix: str（默认使用函数参数 prefix）
      - output.include_data_name: 0/1（默认 1）
      - output.data_name_max_len: int（默认 40；0 表示不截断）
    """
    if isinstance(config.get("_run_output_dir"), str) and config["_run_output_dir"]:
        return Path(config["_run_output_dir"])

    output_cfg = config.get("output")
    if not isinstance(output_cfg, dict):
        output_cfg = {}
        config["output"] = output_cfg

    enabled = int(output_cfg.get("timestamp_subdir", 1)) == 1
    base_output_dir = ensure_output_dir(config)
    config["_base_output_dir"] = str(base_output_dir)

    if not enabled:
        config["_run_output_dir"] = str(base_output_dir)
        return base_output_dir

    timestamp_format = str(output_cfg.get("timestamp_format", "%Y%m%d_%H%M%S"))
    run_prefix = str(output_cfg.get("run_prefix") or prefix or "")
    ts = datetime.datetime.now().strftime(timestamp_format)

    include_data_name = int(output_cfg.get("include_data_name", 1)) == 1
    data_stem = _dataset_stem_from_config(config) if include_data_name else ""
    data_part = _sanitize_subdir_part(data_stem)
    max_len = int(output_cfg.get("data_name_max_len", 40))
    if max_len > 0 and len(data_part) > max_len:
        data_part = data_part[:max_len]

    prefix_part = _sanitize_subdir_part(run_prefix).rstrip(" .-_")
    parts = [p for p in (prefix_part, data_part, ts) if p]
    subdir = _sanitize_subdir_name("_".join(parts))

    candidate = base_output_dir / subdir
    if candidate.exists():
        for i in range(1, 1000):
            alt = base_output_dir / f"{subdir}_{i:03d}"
            if not alt.exists():
                candidate = alt
                break

    output_cfg["output_dir"] = str(candidate)
    run_output_dir = ensure_output_dir(config)
    config["_run_output_dir"] = str(run_output_dir)
    return run_output_dir


class _TeeStream:
    def __init__(self, primary: Any, secondary: Any) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._secondary.write(data)
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)


def setup_logging(config: Dict[str, Any], output_dir: Path) -> None:
    """配置日志记录，将日志同时输出到控制台和文件。"""
    output_cfg = config.get("output") or {}
    log_file = output_cfg.get("log_file", "run_log.txt")
    log_path = output_dir / str(log_file)

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(orig_stdout),
        ],
        force=True,
    )
    logging.info("=" * 50)
    logging.info(f"本次运行开始: {datetime.datetime.now()}")
    logging.info(f"命令行: {' '.join(sys.argv)}")
    logging.info(f"输出目录: {output_dir.resolve()}")
    config_path = config.get("_config_path")
    if config_path:
        logging.info(f"配置文件: {config_path}")

    # 捕获所有 print / 第三方库 stdout/stderr 输出到同一个日志文件，便于复盘长时间运行
    capture_prints = int(output_cfg.get("capture_prints", 1)) == 1
    if capture_prints:
        log_stream = log_path.open("a", encoding="utf-8")
        atexit.register(log_stream.close)
        sys.stdout = _TeeStream(orig_stdout, log_stream)
        sys.stderr = _TeeStream(orig_stderr, log_stream)
        logging.info("已启用 stdout/stderr 捕获到日志文件。")


def save_model_if_requested(
    model: xgb.XGBModel, config: Dict[str, Any], output_dir: Path
) -> None:
    """按配置决定是否保存模型文件; 参数: model 模型, config 配置, output_dir 输出目录。"""
    output_cfg = config.get("output") or {}
    model_file = output_cfg.get("model_file")
    if not model_file:
        return
    model_path = output_dir / model_file
    booster = model.get_booster()
    booster.save_model(model_path)
    logging.info(f"模型已保存到: {model_path}")


def evaluate_full_model(
    model: xgb.XGBModel,
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    """在全量数据上评估模型（in-sample；仅用于参考，不代表泛化能力）。"""
    y_true = np.asarray(y, dtype=float)
    y_pred = model.predict(X)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mse = float(np.mean((y_true - y_pred_arr) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred_arr)))
    y_min, y_max = float(np.min(y_true)), float(np.max(y_true))
    rng = y_max - y_min
    nrmse = float("nan") if rng == 0.0 else rmse / rng
    ss_res = float(np.sum((y_true - y_pred_arr) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

    logging.info(
        "训练内（全量数据 / in-sample）评估指标: "
        f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, NRMSE: {nrmse:.4f}, R^2: {r2:.4f}"
    )

    # 这里只打印全量数据上的指标，不再生成图像文件。


def holdout_evaluate_model(
    config: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """按配置的 test_size 划分训练/测试集并评估模型。"""
    model_cfg = config.get("model") or {}
    test_size = float(model_cfg.get("test_size", 0.3))
    if test_size <= 0.0 or test_size >= 1.0:
        logging.info("Hold-out: 跳过，因为 `model.test_size` 配置无效。")
        return

    n_samples = len(X)
    if n_samples < 3:
        logging.info("Hold-out: 样本数过少，跳过 7:3 划分评估。")
        return

    random_state = int(model_cfg.get("random_state", 42))
    indices = np.random.RandomState(random_state).permutation(n_samples)
    n_test = int(n_samples * test_size)
    n_test = max(1, min(n_samples - 1, n_test))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    model = build_and_train_model(config, X_train, y_train)

    y_train_pred = np.asarray(model.predict(X_train), dtype=float)
    y_test_pred = np.asarray(model.predict(X_test), dtype=float)
    y_train_true = np.asarray(y_train, dtype=float)
    y_test_true = np.asarray(y_test, dtype=float)

    # 使用训练集 y 的中位数作为二分类阈值，用于计算 Accuracy/Precision/Recall/F1
    cls_threshold = float(np.median(y_train_true))

    def _reg_metrics(
        y_true_arr: np.ndarray, y_pred_arr: np.ndarray
    ) -> tuple[float, float, float, float]:
        mse = float(np.mean((y_true_arr - y_pred_arr) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
        y_min, y_max = float(np.min(y_true_arr)), float(np.max(y_true_arr))
        rng = y_max - y_min
        nrmse = float("nan") if rng == 0.0 else rmse / rng
        ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
        ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
        r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
        return rmse, mae, nrmse, r2

    def _cls_metrics(
        y_true_arr: np.ndarray, y_pred_arr: np.ndarray
    ) -> tuple[float, float, float, float]:
        y_true_bin = y_true_arr >= cls_threshold
        y_pred_bin = y_pred_arr >= cls_threshold

        tp = float(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
        tn = float(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
        fp = float(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
        fn = float(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0.0 else float("nan")
        prec = tp / (tp + fp) if (tp + fp) > 0.0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0.0 else float("nan")
        if np.isnan(prec) or np.isnan(rec):
            f1 = float("nan")
        elif (prec + rec) == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * prec * rec / (prec + rec)
        return acc, prec, rec, f1

    rmse_tr, mae_tr, nrmse_tr, r2_tr = _reg_metrics(y_train_true, y_train_pred)
    rmse_te, mae_te, nrmse_te, r2_te = _reg_metrics(y_test_true, y_test_pred)

    acc_tr, prec_tr, rec_tr, f1_tr = _cls_metrics(y_train_true, y_train_pred)
    acc_te, prec_te, rec_te, f1_te = _cls_metrics(y_test_true, y_test_pred)

    train_ratio = (n_samples - n_test) / n_samples
    test_ratio = n_test / n_samples
    r2_gap = r2_tr - r2_te if (np.isfinite(r2_tr) and np.isfinite(r2_te)) else float("nan")
    if np.isfinite(r2_gap) and r2_gap >= 0.05:
        logging.warning(
            "检测到较大的 Train/Test 泛化差距（可能过拟合或数据分布差异）："
            f"R^2 gap = {r2_gap:.4f}（Train={r2_tr:.4f}, Test={r2_te:.4f}）。"
        )
    logging.info(
        "Hold-out 评估（Train 为训练集 in-sample 指标，仅供参考；请以 Test/CV 为准；"
        f"训练集比例约为 {train_ratio:.2f}，测试集比例约为 {test_ratio:.2f}）：\n"
        f"二分类阈值（基于训练集 y 中位数）: {cls_threshold:.4f}\n"
        f"Train - RMSE: {rmse_tr:.4f}, MAE: {mae_tr:.4f}, "
        f"NRMSE: {nrmse_tr:.4f}, R^2: {r2_tr:.4f}\n"
        f"Test  - RMSE: {rmse_te:.4f}, MAE: {mae_te:.4f}, "
        f"NRMSE: {nrmse_te:.4f}, R^2: {r2_te:.4f}\n"
        f"Train - Accuracy: {acc_tr:.4f}, Precision: {prec_tr:.4f}, "
        f"Recall: {rec_tr:.4f}, F1: {f1_tr:.4f}\n"
        f"Test  - Accuracy: {acc_te:.4f}, Precision: {prec_te:.4f}, "
        f"Recall: {rec_te:.4f}, F1: {f1_te:.4f}"
    )


def plot_shap_summary(
    shap_values: np.ndarray, X: pd.DataFrame, config: Dict[str, Any], output_dir: Path
) -> None:
    """绘制并保存 SHAP summary 图和 mean |SHAP| bar 图。"""
    shap_cfg = config.get("shap") or {}
    use_summary = int(shap_cfg.get("use_summary", 1)) == 1
    if not use_summary:
        return

    output_cfg = config.get("output") or {}
    summary_file = output_cfg.get("summary_file", "shap_summary.png")
    summary_path = output_dir / summary_file

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"SHAP summary plot saved to: {summary_path}")

    # mean |SHAP| bar 图
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    indices = np.argsort(mean_abs_shap)[::-1]
    feat_names = X.columns[indices].tolist()
    values = mean_abs_shap[indices]

    plt.figure(figsize=(6, max(2, 0.3 * len(feat_names))))
    y_pos = np.arange(len(feat_names))
    plt.barh(y_pos, values)
    plt.yticks(y_pos, feat_names)
    plt.gca().invert_yaxis()
    plt.xlabel("mean |SHAP value|")
    plt.tight_layout()
    bar_file = output_cfg.get("mean_abs_shap_file", "mean_abs_shap.png")
    bar_path = output_dir / bar_file
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Mean |SHAP| bar plot saved to: {bar_path}")


def select_dependence_features(
    shap_values: np.ndarray, X: pd.DataFrame, shap_cfg: Dict[str, Any]
) -> List[str]:
    """选择需要绘制 SHAP 依赖图的特征列表。"""
    features_cfg: Optional[List[str]] = shap_cfg.get("dependence_features")
    if features_cfg:
        return [f for f in features_cfg if f in X.columns]

    dependence_all = int(shap_cfg.get("dependence_all", 0)) == 1
    if dependence_all:
        return [str(c) for c in X.columns]

    top_n = int(shap_cfg.get("dependence_top_n", 5))
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    return [X.columns[i] for i in indices]


def plot_shap_dependence(
    shap_values: np.ndarray, X: pd.DataFrame, config: Dict[str, Any], output_dir: Path
) -> None:
    """绘制并保存各特征的 SHAP dependence 图。"""
    shap_cfg = config.get("shap") or {}
    use_dependence = int(shap_cfg.get("use_dependence", 0)) == 1
    if not use_dependence:
        return

    output_cfg = config.get("output") or {}
    prefix = output_cfg.get("dependence_prefix", "shap_dependence_")

    features = select_dependence_features(shap_values, X, shap_cfg)
    if not features:
        logging.info("未选中任何特征来绘制 SHAP 依赖图。")
        return

    for feature in features:
        try:
            plt.figure()
            shap.dependence_plot(feature, shap_values, X, show=False)

            # 计算并叠加一条平滑曲线（若无法转换为数值则跳过）
            idx = X.columns.get_loc(feature)
            y = np.asarray(shap_values[:, idx], dtype=float)
            try:
                x_num = (
                    pd.to_numeric(X[feature], errors="coerce")
                    .to_numpy(dtype=float, copy=False)
                )
                mask = np.isfinite(x_num) & np.isfinite(y)
                if int(mask.sum()) >= 5:
                    x_use = x_num[mask]
                    y_use = y[mask]
                    order = np.argsort(x_use)
                    x_sorted = x_use[order]
                    y_sorted = y_use[order]
                    n = len(x_sorted)
                    window = max(11, n // 5)
                    y_smooth = (
                        pd.Series(y_sorted)
                        .rolling(window, center=True, min_periods=1)
                        .mean()
                        .to_numpy()
                    )
                    plt.plot(x_sorted, y_smooth, color="red")
            except Exception:
                pass

            plt.xlabel(feature)
            plt.ylabel("SHAP value")
            safe_feature = feature.replace("/", "_").replace("\\", "_")
            dep_path = output_dir / f"{prefix}{safe_feature}.png"
            plt.tight_layout()
            plt.savefig(dep_path, dpi=300, bbox_inches="tight")
            plt.close()
            logging.info(f"SHAP dependence plot for `{feature}` saved to: {dep_path}")
        except Exception as exc:
            plt.close("all")
            logging.warning(
                f"SHAP dependence plot for `{feature}` failed and was skipped. Reason: {exc}"
            )


def summarize_and_save_interactions(
    interaction_values: Optional[np.ndarray],
    X: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    """汇总特征交互强度并保存为 CSV 文件。"""
    if interaction_values is None:
        return

    shap_cfg = config.get("shap") or {}
    compute_interactions = int(shap_cfg.get("compute_interactions", 0)) == 1
    if not compute_interactions:
        return

    mean_abs_interactions = np.mean(np.abs(interaction_values), axis=0)
    n_features = mean_abs_interactions.shape[0]

    rows = [
        {
            "feature_1": X.columns[i],
            "feature_2": X.columns[j],
            "mean_abs_interaction": float(mean_abs_interactions[i, j]),
        }
        for i in range(n_features)
        for j in range(i + 1, n_features)
    ]
    df_pairs = pd.DataFrame(rows)
    df_pairs.sort_values("mean_abs_interaction", ascending=False, inplace=True)

    top_n = int(shap_cfg.get("interaction_top_n_pairs", 20))
    df_top = df_pairs.head(top_n)

    top_array = df_top[["feature_1", "feature_2", "mean_abs_interaction"]].values.tolist()
    logging.info(
        "Top SHAP interaction pairs "
        "[feature_1, feature_2, mean_abs_interaction]: "
        f"{top_array}"
    )

    output_cfg = config.get("output") or {}
    filename = output_cfg.get("interaction_pairs_file", "shap_interactions.csv")
    path = output_dir / filename
    df_top.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info(
        f"Top {len(df_top)} SHAP interaction pairs "
        f"saved to: {path}"
    )
    
def plot_fixed_base_interactions(
    interaction_values: Optional[np.ndarray],
    X: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    """绘制基准特征与指定特征的 SHAP 交互图（含平滑曲线）。"""
    if interaction_values is None:
        return

    shap_cfg = config.get("shap") or {}
    base_feature: Optional[str] = shap_cfg.get("interaction_base")
    with_features_cfg: Optional[List[str]] = shap_cfg.get("interaction_with")
    if not base_feature or not isinstance(with_features_cfg, list):
        return

    if base_feature not in X.columns:
        return

    other_features = [
        f for f in with_features_cfg if isinstance(f, str) and f in X.columns
    ]
    if not other_features:
        return

    base_idx = X.columns.get_loc(base_feature)

    output_cfg = config.get("output") or {}
    prefix = output_cfg.get("interaction_prefix", "shap_interaction_")

    x_base = X[base_feature].to_numpy(dtype=float)

    for other in other_features:
        other_idx = X.columns.get_loc(other)
        inter_pair = interaction_values[:, base_idx, other_idx]
        x_other = X[other].to_numpy(dtype=float)

        plt.figure()
        plt.scatter(x_base, inter_pair, c=x_other, cmap="plasma", s=10)

        # 使用滑动窗口在排序后的 x 轴上做滚动平均，得到更平滑的曲线
        order = np.argsort(x_base)
        x_sorted = x_base[order]
        y_sorted = inter_pair[order]
        n = len(x_sorted)
        if n >= 5:
            window = max(11, n // 5)
            y_smooth = (
                pd.Series(y_sorted)
                .rolling(window, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )
            plt.plot(x_sorted, y_smooth, color="red")

        plt.xlabel(base_feature)
        plt.ylabel(f"SHAP interaction values for {base_feature} and {other}")
        cbar = plt.colorbar()
        cbar.set_label(f"{other} (feature value)")

        safe_base = base_feature.replace("/", "_").replace("\\", "_")
        safe_other = other.replace("/", "_").replace("\\", "_")
        path = output_dir / f"{prefix}{safe_base}_x_{safe_other}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(
            f"SHAP interaction plot for `{base_feature}` and `{other}` saved to: {path}"
        )


def main() -> None:
    """脚本主入口，执行训练、评估和 SHAP 分析流程。"""
    run_start = datetime.datetime.now()
    args = parse_args()
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).with_name("config.yaml")

    config = load_config(config_path)

    output_dir = ensure_run_output_dir(config, prefix="xgb_")
    setup_logging(config, output_dir)
    logging.info(f"配置内容: {config}")

    logging.info(f"使用配置文件: {config_path}")
    df, X, y = load_dataset(config)
    logging.info(
        f"已加载数据 `{config['data']['path']}`，"
        f"共 {df.shape[0]} 行，{df.shape[1]} 列。"
    )

    cross_validate_model(config, X, y)
    holdout_evaluate_model(config, X, y)

    model_full = build_and_train_model(config, X, y)
    shap_values, interaction_values = compute_shap_and_interactions(
        model_full, X, config
    )
    logging.info("已在全量数据上训练模型并计算 SHAP 值。")

    evaluate_full_model(model_full, X, y, config, output_dir)
    summarize_and_save_interactions(interaction_values, X, config, output_dir)
    plot_shap_summary(shap_values, X, config, output_dir)
    plot_shap_dependence(shap_values, X, config, output_dir)
    plot_fixed_base_interactions(interaction_values, X, config, output_dir)

    save_model_if_requested(model_full, config, output_dir)

    logging.info(f"所有输出文件已保存到目录: {output_dir.resolve()}")
    run_end = datetime.datetime.now()
    logging.info(
        f"本次运行结束: {run_end}（耗时 {(run_end - run_start).total_seconds():.2f} 秒）"
    )


if __name__ == "__main__":
    main()
