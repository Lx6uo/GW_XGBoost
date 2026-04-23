## Step 1 Import libraries and data ## 第一步：导入库和数据
import geoxgboost as gx
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
import shap as sp
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

_THIS_DIR = Path(__file__).resolve().parent

# Import data 导入数据
# Coordinates of centroid 质心坐标（空间位置）
Coords= pd.read_csv(
    _THIS_DIR / "Coords.csv"
)
# Data including GISid, X(independent variables) and  y (dependent variable)
# 包含GISid、X（自变量）和y（因变量）的数据
Data = pd.read_csv(
    _THIS_DIR / "Data.csv"
)

# Remove GISid and y from original data to keep only independent variables
# 去除GISid和y，仅保留自变量特征X
X= Data.iloc [:, 1 : -1]
# Dependent y 因变量y
y= Data.iloc [:, -1]
# Get variables' names. Used for labeling Dataframes 提取变量名，用于后续结果标注
VarNames = X.columns[:]

## Step 2 Hyper parameter tuning. Define initial hyperparameters for inner loop  
## 第二步：超参数调优，定义内层循环的初始超参数
params= {
    'n_estimators':100,     #default is 100   树的数量，默认100
    'learning_rate':0.1,    #default is 0.3   学习率，默认0.3
    'max_depth':6,          #default is 6     树的最大深度，默认6
    'min_child_weight':1,   #default is 1     叶子节点最小样本权重和，默认1
    'gamma':0,              #default is 0     节点分裂所需的最小损失下降，默认0
    'subsample':0.8,        #default is 1     行采样比例（样本子采样），默认1
    'colsample_bytree':0.8, #default is 1     列采样比例（特征子采样），默认1
    'reg_alpha':0,          #default is 0     L1 正则化系数，默认0
    'reg_lambda':1,         #default is 1     L2 正则化系数，默认1
    }

# Define search space for hyperparameters of inner loop. A maximum of 3 hyperparameters can be tuned at the same time
# 定义内层交叉验证的超参数搜索空间（最多同时调三个超参数）
# Set hyperparameters to None to avoid overlapping if the function runs again
# 初始化为None，避免函数多次运行时参数残留
Param1=None; Param2=None; Param3=None
# 存放每个待调优超参数的候选取值列表
Param1_Values = []; Param2_Values = []; Param3_Values = []

# Set hyperparameters and values according to the problem. Select and deselect for one or more hyperparameters
# 根据问题需求选择要调优的超参数及其取值
Param1='n_estimators'
Param1_Values = [100, 200, 300, 500]
Param2='learning_rate'
Param2_Values = [0.1, 0.05,0.01]
Param3='max_depth'
Param3_Values = [2,3,5,6]

# Create grid 构建超参数网格
param_grid= gx.create_param_grid(
    Param1,
    Param1_Values,
    Param2,
    Param2_Values,
    Param3,
    Param3_Values,
)

## Step 3 Nested CV to tune hyperparameters 第三步：使用嵌套交叉验证调优超参数
params, Output_NestedCV= gx.nestedCV(
    X,
    y,
    param_grid,
    Param1,
    Param2,
    Param3,
    params,
)

## Step 4 GlobalXGBoost model 第四步：拟合全局XGBoost模型
Output_GlobalXGBoost=gx.global_xgb(X,y,params)

## Step 4b SnengHAP explainability for global XGBoost 第四步扩展：对全局XGBoost模型做SHAP可解释性分析
# Train an XGBoost model with the tuned hyperparameters 使用调优后的超参数训练XGBoost模型
xgb_model = XGBRegressor(**params)
xgb_model.fit(X, y)

# Create SHAP explainer and compute SHAP values 创建SHAP解释器并计算SHAP值
explainer = sp.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)

# Global summary plot (saved to file) 绘制全局SHAP重要性汇总图并保存为图片文件
sp.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig(_THIS_DIR / "shap_global.png", dpi=300)
plt.close()

## Step 5 Optimize Bandwidth 第五步：优化带宽（空间权重窗口大小）
bw= gx.optimize_bw(
    X,
    y,
    Coords,
    params,
    bw_min=77,
    bw_max=83,
    step=1,
    Kernel='Adaptive',
    spatial_weights=True,
)

## Step 6 GXGB (Geographical-XGBoost) 第六步：拟合地理加权XGBoost模型（GXGB）
Output_GXGB_LocalModel= gx.gxgb(
    X,
    y,
    Coords,
    params,
    bw=bw,
    Kernel='Adaptive',
    spatial_weights=True,
    alpha_wt_type='varying',
    alpha_wt=0.5,
)

# ## Step 6b SHAP explainability for all local GXGB models 第六步扩展：对所有局部GXGB模型进行SHAP解释
# # Reconstruct local neighborhoods (same logic as in gx.gxgb) and compute SHAP for each best local model
# # 复现gx.gxgb内部的局部邻域构建逻辑，对每个最佳局部模型计算SHAP值
# best_local_models = Output_GXGB_LocalModel["bestLocalModel"]
# DistanceMatrix_ij = pd.DataFrame(distance_matrix(Coords, Coords))
# # 计算坐标间距离矩阵
# num_rows = len(X)
# # 样本数量

# for i in range(num_rows):
#     # 打印当前正在解释的局部模型序号
#     print(f"Calculating SHAP for local model {i + 1} of {num_rows}")
#     neighbours = pd.DataFrame(DistanceMatrix_ij.iloc[:, i])
#     neighbours.columns = ["Distance"]
#     # 将特征、自变量和距离拼接
#     data_all = pd.concat([X, y, neighbours], axis=1)
#     # 按距当前点距离排序
#     data_sorted = data_all.sort_values(by=["Distance"])

#     # For Kernel='Adaptive' the local training/prediction window uses knn = bw
#     # 自适应核下，局部训练窗口使用前knn = bw 个近邻
#     knn = bw
#     # 选择前knn+1个样本构成局部数据
#     local_data_full = data_sorted.iloc[: knn + 1, :]
#     # drop y and Distance
#     # 去掉y和距离列，保留局部特征X
#     local_X_full = local_data_full.iloc[:, :-2]

#     # 取第i个位置的最佳局部模型
#     local_model = best_local_models[i]
#     # 为该局部模型创建SHAP解释器
#     local_explainer = sp.TreeExplainer(local_model)
#     # 计算该局部模型的SHAP值
#     local_shap_values = local_explainer.shap_values(local_X_full)

#     # 绘制局部SHAP重要性图
#     sp.summary_plot(local_shap_values, local_X_full, show=False)
#     plt.tight_layout()
#     plt.savefig(f"shap_local_model_{i + 1}.png", dpi=300)
#     plt.close()

# ## Step 7 Predict (unseen data) 第七步：对新（未见）数据进行预测
# # Input data to predict 导入待预测数据
# DataPredict = pd.read_csv(
#     _THIS_DIR / "PredictData.csv"
# )
# CoordsPredict= pd.read_csv(
#     _THIS_DIR / "PredictCoords.csv"
# )

# # predict 使用已训练的GXGB模型对新位置数据进行预测
# Output_PredictGXGBoost= gx.predict_gxgb(
#     DataPredict,
#     CoordsPredict,
#     Coords,
#     Output_GXGB_LocalModel,
#     alpha_wt = 0.5,
#     alpha_wt_type = 'varying',
# )
