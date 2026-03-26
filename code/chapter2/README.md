# lgbm_shap_warning.py 步骤说明

本文档描述 `lgbm_shap_warning.py` 的完整流程，方便随时查看。

---

## 整体功能

基于 `monitoring data.xlsx` 构建「时空位移预测 + 预警 + 可解释性分析」流水线，核心为 **LightGBM 回归 + LightGBM 分类 + SHAP 解释**。

---

## 步骤 1：读取监测数据

- 调用 `load_monitoring_data()` 从 `monitoring data.xlsx` 读取完整 DataFrame
- 包含列：`Date`、`MJ9/mm`、`MJ1/mm`、`MJ3/mm`、`Rainfall/mm`、`GWT/m`、`RWL/m`、`aveT/℃`、`minT/℃`、`maxT/℃`、`DP`、`RH` 等

---

## 步骤 2：构造多监测点时序样本（下一天位移预测）

函数：`build_supervised_samples`

- 时间窗长度：`window = 5`
- 按 `Date` 排序
- 对每个监测点（MJ9、MJ1、MJ3）：
  - **输入特征**：
    - 该点过去 5 天位移 \(u_{t-5} \sim u_{t-1}\)
    - 过去 5 天的环境因子（雨量、库水位、地下水位、温度、湿度等，每列 5 天滞后）
    - 监测点 one-hot 编码
  - **回归目标**：当天位移 \(u_t\)
  - **预警标签**：位移增量 `delta = u_t - u_{t-1}`，取 `delta` 的 90% 分位数为阈值
    - `y_cls = 1`：delta ≥ 阈值（预警）
    - `y_cls = 0`：否则（正常）
- 将所有监测点样本拼成统一的 `X`、`y_reg`、`y_cls`、`feat_names`

---

## 步骤 3：数据划分与标准化

- 按样本顺序划分：前 75% 为训练集，后 25% 为测试集
- 使用 `StandardScaler` 对 `X` 进行标准化，得到 `X_train_std`、`X_test_std`

---

## 步骤 4：LightGBM 回归模型（时空位移预测）

函数：`train_lgbm_regressor`

- 目标：预测下一天位移
- 参数：`objective='regression'`，`metric='l2'`，`num_leaves=4`，`max_depth=3` 等
- 输出：训练集和测试集的 **MSE**

---

## 步骤 5：LightGBM 分类模型（预警判定）

函数：`train_lgbm_classifier`

- 目标：二分类——是否进入预警状态
- 特征：与回归模型相同
- 标签：步骤 2 中构造的 `y_cls`
- 参数：`objective='binary'`，`metric=['binary_logloss','auc']`
- 输出：训练集和测试集的 **ACC**、**AUC**

---

## 步骤 6：SHAP 可解释性分析

### 6.1 回归模型 SHAP

函数：`analyze_shap_reg`

- 使用 `shap.TreeExplainer` 计算训练集 SHAP 值
- 生成 summary 图：各特征对位移预测的贡献排序
- 保存至：`outputs/shap_summary_regression.png`

### 6.2 分类模型 SHAP

函数：`analyze_shap_cls`

- 对正类（进入预警状态）计算 SHAP 值
- 生成 summary 图：促使进入预警的关键因子
- 保存至：`outputs/shap_summary_classification.png`

---

## 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/shap_summary_regression.png` | 回归模型 SHAP 总结图 |
| `outputs/shap_summary_classification.png` | 分类模型 SHAP 总结图 |

---

## 运行方式

在项目根目录下执行：

```bash
cd code/chapter2
uv run python lgbm_shap_warning.py
```
