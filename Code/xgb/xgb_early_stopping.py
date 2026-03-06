from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


class XGBRegressorWithEarlyStopping(XGBRegressor):
    """在每次 fit() 内部从训练数据再切出一份 eval_set，用于 early stopping。

    设计目标：让 GridSearchCV / nested CV 在每个 fold 的训练阶段都能使用 early stopping，
    且 eval_set 不使用 fold 的测试集（避免用“评分集”来做 early stopping）。
    """

    def __init__(
        self,
        *,
        objective: str = "reg:squarederror",
        es_eval_fraction: float = 0.2,
        es_shuffle: bool = True,
        es_random_state: Optional[int] = None,
        es_min_train_samples: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(objective=objective, **kwargs)
        self.es_eval_fraction = float(es_eval_fraction)
        self.es_shuffle = bool(es_shuffle)
        self.es_random_state = es_random_state
        self.es_min_train_samples = int(es_min_train_samples)

    def fit(  # type: ignore[override]
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Optional[Any] = None,
        base_margin: Optional[Any] = None,
        eval_set: Optional[Sequence[Tuple[Any, Any]]] = None,
        verbose: bool | int | None = False,
        xgb_model: Any = None,
        sample_weight_eval_set: Optional[Sequence[Any]] = None,
        base_margin_eval_set: Optional[Sequence[Any]] = None,
        feature_weights: Optional[Any] = None,
    ) -> "XGBRegressorWithEarlyStopping":
        # 如果用户显式传了 eval_set，则尊重用户的设置（不再二次切分）
        if eval_set is not None:
            return super().fit(
                X,
                y,
                sample_weight=sample_weight,
                base_margin=base_margin,
                eval_set=eval_set,
                verbose=verbose,
                xgb_model=xgb_model,
                sample_weight_eval_set=sample_weight_eval_set,
                base_margin_eval_set=base_margin_eval_set,
                feature_weights=feature_weights,
            )

        rounds = getattr(self, "early_stopping_rounds", None)
        if rounds is None or int(rounds) <= 0:
            return super().fit(
                X,
                y,
                sample_weight=sample_weight,
                base_margin=base_margin,
                eval_set=None,
                verbose=verbose,
                xgb_model=xgb_model,
                sample_weight_eval_set=None,
                base_margin_eval_set=None,
                feature_weights=feature_weights,
            )

        try:
            n_samples = len(X)
        except Exception:
            n_samples = 0

        if n_samples < 2:
            logging.warning(
                "early stopping 已启用，但训练样本数过少，跳过 eval_set 切分并关闭 early stopping（本次 fit）。"
            )
            orig = self.early_stopping_rounds
            try:
                self.early_stopping_rounds = None
                return super().fit(
                    X,
                    y,
                    sample_weight=sample_weight,
                    base_margin=base_margin,
                    eval_set=None,
                    verbose=verbose,
                    xgb_model=xgb_model,
                    sample_weight_eval_set=None,
                    base_margin_eval_set=None,
                    feature_weights=feature_weights,
                )
            finally:
                self.early_stopping_rounds = orig

        eval_fraction = float(self.es_eval_fraction)
        if not (0.0 < eval_fraction < 1.0):
            raise ValueError(
                f"es_eval_fraction 必须在 (0, 1) 之间，当前 {self.es_eval_fraction}。"
            )

        # 训练样本太少时，固定留 1 个做 eval
        test_size: float | int = eval_fraction
        if n_samples < int(self.es_min_train_samples):
            test_size = 1
            logging.info(
                f"训练样本数较少（n={n_samples}），early stopping eval_set 将固定留 1 条样本。"
            )

        if sample_weight is not None:
            X_train, X_eval, y_train, y_eval, w_train, w_eval = train_test_split(
                X,
                y,
                sample_weight,
                test_size=test_size,
                shuffle=self.es_shuffle,
                random_state=self.es_random_state,
            )
            return super().fit(
                X_train,
                y_train,
                sample_weight=w_train,
                base_margin=None,
                eval_set=[(X_eval, y_eval)],
                verbose=verbose,
                xgb_model=xgb_model,
                sample_weight_eval_set=[w_eval],
                base_margin_eval_set=None,
                feature_weights=feature_weights,
            )

        X_train, X_eval, y_train, y_eval = train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=self.es_shuffle,
            random_state=self.es_random_state,
        )
        return super().fit(
            X_train,
            y_train,
            sample_weight=None,
            base_margin=None,
            eval_set=[(X_eval, y_eval)],
            verbose=verbose,
            xgb_model=xgb_model,
            sample_weight_eval_set=None,
            base_margin_eval_set=None,
            feature_weights=feature_weights,
        )

