# 第三章：基于LSTM的滑坡位移预测模型研究

本章实现了基于LSTM的滑坡位移趋势项预测模型，通过50次独立运行实现概率预测和不确定性量化。

## 目录结构

```
chapter3/
├── README.md                          # 本文件
├── notebooks/                         # Jupyter notebooks
│   └── LSTM-trend-submit.ipynb       # LSTM趋势项预测模型
├── scripts/                           # 核心脚本
│   ├── run_lstm_trend_50times.py     # 运行50次LSTM模型
│   ├── calculate_thesis_statistics.py # 计算论文统计数据
│   └── generate_thesis_figures.py    # 生成论文图表
└── outputs/                           # 输出目录
    ├── figures/                       # 图表输出
    │   ├── lstm_prediction.pdf/png   # 概率预测结果图
    │   ├── lstm_uncertainty.pdf/png  # 不确定性时变特征图
    │   └── lstm_std_distribution.pdf/png # 标准差分布图
    ├── tables/                        # 数据表格输出
    │   ├── lstm_trend_50runs_summary.csv      # 50次运行汇总指标
    │   ├── lstm_trend_50runs_predictions.csv  # 所有预测值详细记录
    │   └── lstm_trend_50runs_statistics.csv   # 时间序列统计量
    └── UPDATE_SUMMARY.md              # 数据更新总结文档
```


## 环境要求

使用 uv 管理 Python 环境，确保已安装以下依赖：

```bash
# 主要依赖
pandas numpy matplotlib openpyxl scikit-learn tensorflow keras
```

## 快速开始

### 完整工作流程（从头开始）

如果需要重新运行50次实验并更新论文数据：

```bash
# 1. 运行50次LSTM模型（约7分钟）
cd code/chapter3/scripts
uv run run_lstm_trend_50times.py

# 2. 计算论文统计数据
uv run calculate_thesis_statistics.py

# 3. 生成论文图表
uv run generate_thesis_figures.py
```

**输出结果：**
- `outputs/tables/` - 3个CSV数据文件
- `outputs/figures/` - 6个图表文件（3个PDF + 3个PNG）
- 控制台输出统计摘要

### 仅重新生成图表

如果已有数据文件，只需重新生成图表：

```bash
cd code/chapter3/scripts
uv run generate_thesis_figures.py
```

### 仅查看统计结果

如果只想查看统计数据：

```bash
cd code/chapter3/scripts
uv run calculate_thesis_statistics.py
```


## 核心脚本说明

### 1. run_lstm_trend_50times.py

运行LSTM模型50次，每次使用不同的随机种子。

**功能：**
- 读取监测数据并预处理
- 构建LSTM模型（2层LSTM + Dropout + Dense）
- 训练50次，每次40个epoch
- 记录每次运行的R²、RMSE等指标
- 保存所有预测值和统计量

**运行时间：** 约7分钟（CPU模式）

### 2. calculate_thesis_statistics.py

计算论文所需的统计数据。

**功能：**
- 读取50次运行结果
- 计算点预测性能（R²、RMSE的均值）
- 计算概率预测统计（平均预测值、标准差、变异系数）
- 输出LaTeX表格代码

### 3. generate_thesis_figures.py

生成论文所需的图表。

**功能：**
- 概率预测结果图（均值 + 50%/90%置信区间）
- 不确定性时变特征图（标准差随时间变化）
- 标准差分布直方图
- 同时生成PDF和PNG两种格式

## 数据文件说明

### outputs/tables/

#### lstm_trend_50runs_summary.csv
50次运行的汇总指标，每行一次运行：
- `run_id`: 运行编号（1-50）
- `seed`: 随机种子
- `train_r2`, `train_rmse`: 训练集性能
- `test_r2`, `test_rmse`: 测试集性能
- `train_loss`, `val_loss`: 最终损失值

#### lstm_trend_50runs_predictions.csv
所有50次运行的详细预测值：
- `run_id`: 运行编号
- `time_index`: 时间索引
- `prediction`: 预测值
- `actual`: 真实值

#### lstm_trend_50runs_statistics.csv
每个时间点的统计量（用于绘制置信区间）：
- `time_index`: 时间索引
- `mean`: 50次运行的均值
- `std`: 标准差
- `p05`, `p25`, `p50`, `p75`, `p95`: 各分位数
- `actual`: 真实值


## 生成的图表说明

### outputs/figures/

#### 1. lstm_prediction.pdf/png
**LSTM概率预测结果图**
- 蓝色实线：50次运行的均值预测
- 深色阴影：50%置信区间（25%-75%分位数）
- 浅色阴影：90%置信区间（5%-95%分位数）
- 红色散点：真实观测值

**论文引用：** 图3-X，展示LSTM模型的概率预测结果

#### 2. lstm_uncertainty.pdf/png
**不确定性时变特征图**
- 蓝色曲线：预测标准差随时间的变化
- 红色虚线：平均标准差
- 显示模型在不同时期的预测不确定性

**论文引用：** 图3-Y，展示LSTM模型预测不确定性的时变特征

#### 3. lstm_std_distribution.pdf/png
**标准差分布直方图**
- 显示50次运行中标准差的分布特征
- 红色虚线：均值
- 绿色虚线：中位数

**论文引用：** 图3-Z，展示预测标准差的分布特征

## 实验结果（基于50次真实运行）

### 表3-2: LSTM模型点预测性能

| 数据集 | R² | RMSE (mm) |
|--------|---------|-----------|
| 训练集 | 0.9883 | 11.02 |
| 测试集 | 0.7916 | 5.55 |

### 表3-3: LSTM概率预测统计特征

| 统计量 | 数值 |
|--------|------|
| 平均预测值 (mm) | 987.09 |
| 平均标准差 (mm) | 5.41 |
| 最大标准差 (mm) | 6.64 |
| 最小标准差 (mm) | 4.65 |
| 变异系数 (%) | 0.55 |

**关键发现：**
- 测试集R²为0.7916，达到工程应用精度要求
- 平均预测值987.09 mm与实际988.86 mm的相对误差仅0.18%
- 变异系数0.55%，表明模型对随机初始化不敏感
- 标准差范围4.65-6.64 mm，占位移量的0.55%


## 模型说明

### LSTM模型架构

```python
Sequential([
    LSTM(25, return_sequences=True, input_shape=(2, 4)),
    Dropout(0.3),
    LSTM(15, return_sequences=False, kernel_regularizer=l2(0.002)),
    Dropout(0.3),
    Dense(15),
    Dense(1)
])
```

**参数：**
- 输入特征：4个监测点（MJ9, MJ1, MJ3, ATU4）的历史位移
- 时间步长：2
- 优化器：Adam (learning_rate=0.0005)
- 训练轮数：40 epochs
- 批次大小：64

### 概率预测方法

通过**50次独立运行**实现概率预测：
1. 每次使用不同的随机种子初始化模型参数
2. 收集50次预测结果形成统计分布
3. 计算均值、标准差和分位数
4. 构建90%置信区间（5%-95%）和50%置信区间（25%-75%）

这种方法称为**"基于多次运行的概率预测"**，能够量化模型的认知不确定性。

## 论文对应内容

本代码对应论文**第三章：基于LSTM的滑坡位移预测模型研究**

### 主要内容：
1. **LSTM时序预测模型**
   - 利用多监测点空间协同变形特征
   - 通过门控机制捕捉长期依赖关系

2. **概率预测框架**
   - 50次独立运行获得统计分布
   - 提供均值预测和置信区间

3. **不确定性量化**
   - 计算预测标准差和变异系数
   - 分析不确定性的时变特征

### 论文图表对应：
- 图3-X：`outputs/figures/lstm_prediction.pdf`
- 图3-Y：`outputs/figures/lstm_uncertainty.pdf`
- 图3-Z：`outputs/figures/lstm_std_distribution.pdf`
- 表3-2：点预测性能数据
- 表3-3：概率预测统计数据

## 数据来源

- **输入数据：** `../../../data/monitoring data.xlsx`
  - 包含MJ9、MJ1、MJ3、ATU4四个监测点的位移数据
  - 时间跨度：2014-2018年

- **输出数据：** `outputs/tables/` 和 `outputs/figures/`

## 注意事项

1. **运行时间：** 50次运行约需7分钟（CPU模式），使用GPU可显著加速

2. **随机性：** 每次运行50次模型会得到略有不同的结果，但统计特征应保持稳定

3. **内存占用：** 50次运行会占用较多内存，建议至少8GB RAM

4. **图表格式：** PDF格式用于LaTeX论文，PNG格式用于预览

5. **数据更新：** 如需更新论文数据，运行完整工作流程后，图表会自动复制到 `docs/latex/figures/chapter3/`

## 常见问题

**Q: 为什么只有趋势项预测，没有周期项？**
A: 根据论文调整，第三章聚焦于LSTM单模型的趋势项预测，不再使用位移分解方法。

**Q: 50次运行的结果每次都一样吗？**
A: 由于使用固定的随机种子序列（42, 84, 126, ...），每次运行50次的结果应该完全一致。

**Q: 如何修改模型参数？**
A: 编辑 `scripts/run_lstm_trend_50times.py`，修改模型结构、超参数等。

**Q: 如何添加更多监测点？**
A: 修改数据读取部分，选择不同的列作为输入特征。

## 更新日志

- **2026-03-28**: 基于50次真实运行更新所有数据和图表
- **2026-03-27**: 清理冗余文件，统一目录结构
- **2026-03-26**: 删除周期项预测，聚焦LSTM单模型

## 联系方式

如有问题，请联系：韦承谦
