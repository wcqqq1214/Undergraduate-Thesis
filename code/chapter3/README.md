# 第三章：基于LSTM和GRU的滑坡位移预测模型研究

本章实现了基于LSTM和GRU的滑坡位移预测模型，采用多项式分解方法将位移分为趋势项和周期项，通过50次独立运行实现概率预测和不确定性量化。

## 目录结构

```
chapter3/
├── README.md                          # 本文件
├── LSTM-Periodic-submit.ipynb         # LSTM周期项预测模型
├── LSTM-trend-submit.ipynb            # LSTM趋势项预测模型
├── GRU-Periodic-submit.ipynb          # GRU周期项预测模型
├── GRU-Trend-submit.ipynb             # GRU趋势项预测模型
├── visualize_results.py               # 结果可视化脚本
└── outputs/                           # 输出目录（自动创建）
    ├── lstm_gru_comparison.pdf        # LSTM和GRU对比图
    ├── lstm_gru_combined.pdf          # LSTM和GRU综合对比
    ├── uncertainty_comparison.pdf     # 不确定性对比
    └── statistics_summary.csv         # 统计摘要表格
```

## 环境要求

确保已安装以下依赖：

```bash
uv pip install pandas numpy matplotlib openpyxl scikit-learn torch
```

## 快速开始

### 方法1：直接生成可视化结果（推荐）

如果已有 `result.xlsx` 文件，直接运行可视化脚本：

```bash
# 进入chapter3目录
cd code/chapter3

# 运行可视化脚本
uv run python visualize_results.py
```

**输出结果：**
- `outputs/lstm_gru_comparison.pdf` - LSTM和GRU分别的预测结果（上下对比）
- `outputs/uncertainty_comparison.pdf` - LSTM和GRU的不确定性对比（标准差）
- `outputs/lstm_gru_combined.pdf` - LSTM和GRU在同一图中的对比
- `outputs/statistics_summary.csv` - 统计摘要表格

### 方法2：重新训练模型（可选）

如果需要重新训练模型：

```bash
# 启动Jupyter
uv run jupyter notebook

# 然后依次运行以下notebooks：
# 1. LSTM-trend-submit.ipynb      - 训练LSTM趋势项模型
# 2. LSTM-Periodic-submit.ipynb   - 训练LSTM周期项模型
# 3. GRU-Trend-submit.ipynb       - 训练GRU趋势项模型
# 4. GRU-Periodic-submit.ipynb    - 训练GRU周期项模型
```

## 模型说明

### 模型选择

根据论文第三章标题**"基于LSTM和GRU的滑坡位移预测模型研究"**，选择了：
- **LSTM-50runs** - LSTM模型50次运行结果
- **GRU-50runs** - GRU模型50次运行结果

**不使用ST-GNN系列模型**，因为：
- ST-GNN是时空图神经网络，用于多监测点的空间关系建模
- 第三章聚焦**单监测点**的位移预测
- ST-GNN应该留给涉及多监测点空间关系的章节

### LSTM模型

长短期记忆网络（LSTM）是一种特殊的循环神经网络，能够学习长期依赖关系。本章使用LSTM分别对趋势项和周期项进行预测：

- **趋势项**：使用多项式拟合提取长期蠕变趋势
- **周期项**：使用LSTM学习外部触发因子（降雨、库水位等）与位移的非线性关系

### GRU模型

门控循环单元（GRU）是LSTM的简化版本，参数更少，训练更快。作为对比模型，验证LSTM的必要性。

### 概率预测方法

通过**50次独立运行**，获得预测的统计分布，提供：
- 均值预测
- 标准差（不确定性）
- 置信区间（5%, 25%, 50%, 75%, 95%分位数）

这种方法称为**"基于多次运行的概率预测"**，通过多次独立训练量化模型的预测不确定性。

## 生成的图表说明

### 1. lstm_gru_comparison.pdf
- 上图：LSTM模型预测结果
- 下图：GRU模型预测结果
- 显示均值预测、50%置信区间、90%置信区间

### 2. uncertainty_comparison.pdf
- 对比LSTM和GRU的预测标准差（不确定性）
- 标准差越大，表示预测不确定性越高

### 3. lstm_gru_combined.pdf
- LSTM和GRU在同一图中对比
- 便于直接比较两个模型的性能

### 4. statistics_summary.csv
统计摘要表格，包含：
- 平均预测值
- 平均标准差
- 最大/最小标准差

**当前结果：**
```
  模型 平均预测值 (mm) 平均标准差 (mm) 最大标准差 (mm) 最小标准差 (mm)
LSTM     578.64     407.96     466.65     308.27
 GRU     581.47     409.91     472.92     296.88
```

## 论文中如何使用这些图表

1. **图3-X：LSTM与GRU模型预测结果对比**
   - 使用 `lstm_gru_comparison.pdf`
   - 说明：展示两个模型的预测结果和置信区间

2. **图3-Y：LSTM与GRU模型不确定性对比**
   - 使用 `uncertainty_comparison.pdf`
   - 说明：对比两个模型的预测不确定性

3. **表3-Z：LSTM与GRU模型性能统计**
   - 使用 `statistics_summary.csv` 中的数据
   - 说明：定量对比两个模型的性能指标

## 数据来源

- 输入数据：`../../data/monitoring data.xlsx`
- 模型结果：`../../data/result.xlsx`
  - Sheet: `LSTM-50runs` - LSTM模型50次运行结果
  - Sheet: `GRU-50runs` - GRU模型50次运行结果

## 注意事项

1. **中文字体警告**：运行时会出现中文字体缺失的警告，但不影响图表生成。图表中的中文会显示为方框，但数据和布局都是正确的。

2. **数据路径**：确保 `../../data/result.xlsx` 文件存在且包含 `LSTM-50runs` 和 `GRU-50runs` 两个sheet。

3. **图表分辨率**：所有图表都以300 DPI保存，适合论文使用。

4. **单监测点预测**：本章聚焦于**单监测点**的位移预测，不涉及多监测点的空间关系建模。

## 常见问题

**Q: 为什么不使用ST-GNN模型？**
A: ST-GNN是时空图神经网络，用于多监测点的空间关系建模。第三章聚焦单点预测，因此使用LSTM/GRU。

**Q: 如何修改图表样式？**
A: 编辑 `visualize_results.py`，修改 matplotlib 参数。

**Q: 如何添加真实值对比？**
A: 需要从 `monitoring data.xlsx` 读取真实位移数据，然后在可视化脚本中添加对比曲线。

**Q: 50次运行的目的是什么？**
A: 通过多次独立运行，获得预测的统计分布，从而计算置信区间，量化预测的不确定性。

## 下一步工作

1. **完善论文第三章**：
   - 补充实验结果分析
   - 添加LSTM和GRU的性能对比讨论
   - 解释为什么选择这两个模型

2. **添加更多可视化**（可选）：
   - 训练损失曲线
   - 不同监测点的预测结果
   - 误差分布直方图

3. **模型评估指标**：
   - 计算RMSE、MAE、R²等指标
   - 对比LSTM和GRU的性能

## 论文对应章节

本代码对应论文第三章：**基于LSTM和GRU的滑坡位移预测模型研究**

主要内容：
1. 位移分解方法（趋势项+周期项）
2. LSTM时序预测模型
3. GRU对比模型
4. 概率预测与不确定性量化

## 联系方式

如有问题，请联系：韦承谦
