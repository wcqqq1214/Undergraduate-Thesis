# 第三章：MVIF + 分位数LSTM

基于MVIF趋势项提取和Pinball Loss的分位数LSTM预测模型

## 项目结构

```
chapter3/
├── config.py              # 配置文件
├── main.py               # 主程序入口
├── requirements.txt      # 依赖包
├── data/
│   ├── __init__.py
│   └── data_loader.py    # 数据加载和预处理
├── models/
│   ├── __init__.py
│   ├── mvif.py          # MVIF趋势项提取
│   └── quantile_lstm.py # 分位数LSTM模型
├── utils/
│   ├── __init__.py
│   ├── metrics.py       # 评估指标
│   └── visualization.py # 可视化
└── outputs/
    ├── figures/         # 图表输出
    ├── tables/          # LaTeX表格
    └── models/          # 模型保存
```

## 快速开始

### 1. 安装依赖

```bash
cd /home/wcqqq21/thesis/my_code/chapter3
uv pip install -r requirements.txt
```

### 2. 运行主程序

```bash
uv run main.py
```

## 主要功能

### 数据处理
- 自动加载Excel数据
- 缺失值处理（ffill + bfill）
- 数据标准化（X和y都标准化）
- 数据集划分（70% train / 10% val / 20% test）

### MVIF趋势项提取
- 非线性最小二乘拟合
- 趋势项和周期项分离
- Savitzky-Golay平滑

### 分位数LSTM
- Pinball Loss损失函数
- 多分位数预测（5%, 50%, 95%）
- 早停机制
- LayerNorm正则化

### 评估指标
- 点预测：MAE, RMSE, R², MAPE
- 区间预测：PICP, PINAW, CWC

### 可视化
- MVIF分解图
- 训练曲线
- 预测结果对比
- 预测区间可视化
- LaTeX表格生成

## 配置说明

所有配置参数在 `config.py` 中：

- `TARGET_POINT`: 目标监测点（默认'MJ9'）
- `LOOKBACK_DAYS`: 时间窗口（默认5天）
- `LSTM_CONFIG`: LSTM超参数
- `QUANTILES`: 分位数列表（默认[0.05, 0.5, 0.95]）

## 输出文件

运行完成后，输出文件位于 `outputs/` 目录：

- `figures/mvif_decomposition.pdf`: MVIF分解图
- `figures/training_curves.pdf`: 训练曲线
- `figures/test_predictions.pdf`: 预测结果
- `figures/prediction_intervals.pdf`: 预测区间
- `tables/model_performance.tex`: LaTeX表格
- `models/`: 训练好的模型
- `training.log`: 完整日志

## 测试模块

每个模块都可以独立测试：

```bash
# 测试数据加载
uv run data/data_loader.py

# 测试MVIF
uv run models/mvif.py

# 测试LSTM
uv run models/quantile_lstm.py

# 测试指标
uv run utils/metrics.py

# 测试可视化
uv run utils/visualization.py
```

## 作者

wcqqq21

## 日期

2026-03-25
