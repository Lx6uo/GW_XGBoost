# XGBoost + SHAP 调优策略与方案（针对 `xgb_shap.py`）

本方案只通过修改 `Code/config.yaml` 来调优，不改代码逻辑。核心目标：**在交叉验证指标（RMSE / MAE / NRMSE / R²）上尽量提升泛化性能，同时保持 SHAP 解释稳定可靠**。

---

## 1. 先看现有指标，判断问题类型

每次运行：

```bash
cd Code
python xgb_shap.py
```

重点观察两块输出：

1. **K 折交叉验证**（`cross_validate_model` 打印）  
   - 每折：`RMSE, MAE, NRMSE, R^2`  
   - 汇总：各指标的平均值和标准差。

2. **全量模型评估**（`evaluate_full_model` 打印）  
   - 全量数据的：`RMSE, MAE, NRMSE, R^2`。

根据两者对比判断：

- 若 **全量模型 R² 很高、RMSE 很小，而 K 折指标明显更差** → 模型可能过拟合。
- 若 **全量模型和 K 折都表现一般** → 模型容量可能不足，或特征信息不够。

后续调参以“让 K 折指标尽量好”为主，不要只看全量训练误差。

---

## 2. `config.yaml` 中关键配置说明

### 2.1 模型参数（`model.params`）

```yaml
model:
  random_state: 42
  params:
    n_estimators: 300
    max_depth: 4
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
    reg_alpha: 0
    reg_lambda: 1
    #（可添加）min_child_weight: 1
```

关键含义：

- `max_depth`：单棵树最大深度，越大越容易过拟合。
- `n_estimators`：弱学习器棵数，多数情况和 `learning_rate` 联动调节。
- `learning_rate`：学习率，越小越稳，但需要更多树。
- `subsample`：样本采样率（行采样）。
- `colsample_bytree`：特征采样率（列采样）。
- `reg_alpha`：L1 正则，有助于稀疏化、抑制过拟合。
- `reg_lambda`：L2 正则，控制权重大小。
- `min_child_weight`（建议添加）：每个叶节点最小样本权重和，越大越“保守”。

### 2.2 交叉验证配置（`cv`）

```yaml
cv:
  use_cv: 1
  n_splits: 5
  random_state: 42
```

- `n_splits`：折数，5–10 比较常用；折数越大，估计越稳定但训练越慢。
- `random_state`：折分随机种子，保持固定方便不同参数之间的公平对比。

---

## 3. 推荐的调优步骤

### 步骤 0：建立一个“基线参数组合”

在 `config.yaml` 中先设定一个偏保守的基线：

```yaml
model:
  random_state: 42
  params:
    n_estimators: 600
    max_depth: 3
    learning_rate: 0.03
    min_child_weight: 5
    subsample: 0.7
    colsample_bytree: 0.7
    reg_alpha: 0.1
    reg_lambda: 1.0
```

跑一遍 `python xgb_shap.py`，记录：

- K 折：`RMSE / MAE / NRMSE / R²` 的均值 ± 标准差；
- 全量：`RMSE / MAE / NRMSE / R²`。

后续所有调参都和这组基线对比。

---

### 步骤 1：先锁定模型复杂度（`max_depth` + `min_child_weight`）

在 `learning_rate`、`n_estimators` 固定时，尝试几组组合，例如：

1. `max_depth: 2, min_child_weight: 5`
2. `max_depth: 3, min_child_weight: 5`（基线）
3. `max_depth: 4, min_child_weight: 5`
4. 若数据量较大，可以试 `max_depth: 3, min_child_weight: 10`

比较各组合的 **K 折 RMSE / NRMSE / R²**：

- 若增大 `max_depth` 后 R² 提升不明显但 NRMSE 明显变差 → 说明更深的树在过拟合，退回浅一档。
- 若所有深度都表现一般，可以考虑先增加 `n_estimators` 再重复这一轮。

选出一组在 K 折上综合最好（或最稳）的 `max_depth` 和 `min_child_weight` 后再向下调。

---

### 步骤 2：调 `learning_rate` 与 `n_estimators`

在已选好的复杂度下，做“学习率–树数”的权衡：

可以尝试的组合（示例）：

- A：`learning_rate: 0.05, n_estimators: 400`
- B：`learning_rate: 0.03, n_estimators: 600`（偏平滑）
- C：`learning_rate: 0.02, n_estimators: 800`（更平滑，训练略慢）

对比 K 折指标：

- 若减小学习率 + 增加树数后，**K 折 RMSE / NRMSE 有提升且标准差下降**，说明模型更稳定，可以采用更小的学习率。
- 若提升不明显但训练时间明显增加，可以退回前一档。

一般倾向选一组：

- 训练时间可接受；
- K 折均值好，标准差相对小。

---

### 步骤 3：使用采样和正则化“收紧”模型

在上面基础上，可以微调：

1. **行/列采样：**
   - 试几档 `subsample` / `colsample_bytree`：
     - 0.6 / 0.6  
     - 0.7 / 0.7（基线）  
     - 0.8 / 0.8
   - 若过拟合迹象明显（全量 R² 很高，但 K 折 R² 较低）→ 更倾向小采样率（如 0.6–0.7）。

2. **正则化：**
   - 从无正则 → 逐步加大：
     ```yaml
     reg_alpha: 0.0 → 0.1 → 0.3
     reg_lambda: 1.0 → 2.0
     ```
   - 观察 K 折 RMSE / R² 是否更稳定；若 R² 降得很厉害而 RMSE 改善不明显，说明正则过强，可以略减。

通过这一步，让模型在 K 折上表现**稳定且不过分“尖锐”**（即不同折之间指标差异不大）。

---

## 4. 如何使用指标判断“够好”

