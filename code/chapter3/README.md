# 第三章：基于位移运动学特征的滑坡位移预测模型研究

本章实现了基于LSTM和GRU的滑坡位移预测模型，采用多项式分解方法将位移分为趋势项和周期项。

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
```

## 环境要求

确保已安装以下依赖：

```bash
uv pip install pandas numpy matplotlib openpyxl scikit-learn torch
```

## 运行步骤

### 1. 训练模型（可选）

如果需要重新训练模型，可以运行Jupyter notebooks：

```bash
# 启动Jupyter
uv run jupyter notebook

# 然后依次运行以下notebooks：
# 1. LSTM-trend-submit.ipynb      - 训练LSTM趋势项模型
# 2. LSTM-Periodic-submit.ipynb   - 训练LSTM周期项模型
# 3. GRU-Trend-submit.ipynb       - 训练GRU趋势项模型
# 4. GRU-Periodic-submit.ipynb    - 训练GRU周期项模型
```

### 2. 生成可视化结果

使用已有的 `result.xlsx` 文件生成图表：

```bash
# 进入chapter3目录
cd code/chapter3

# 运行可视化脚本
uv run python visualize_results.py
```

### 3. 查看输出

脚本会在 `outputs/` 目录下生成以下文件：

- `lstm_gru_comparison.png` - LSTM和GRU分别的预测结果（上下对比）
- `uncertainty_comparison.png` - LSTM和GRU的不确定性对比（标准差）
- `lstm_gru_combined.png` - LSTM和GRU在同一图中的对比
- `statistics_summary.csv` - 统计摘要表格

## 模型说明

### LSTM模型

长短期记忆网络（LSTM）是一种特殊的循环神经网络，能够学习长期依赖关系。本章使用LSTM分别对趋势项和周期项进行预测：

- **趋势项**：使用多项式拟合提取长期蠕变趋势
- **周期项**：使用LSTM学习外部触发因子（降雨、库水位等）与位移的非线性关系

### GRU模型

门控循环单元（GRU）是LSTM的简化版本，参数更少，训练更快。作为对比模型，验证LSTM的必要性。

### 概率预测

通过50次独立运行，获得预测的统计分布，提供：
- 均值预测
- 标准差（不确定性）
- 置信区间（5%, 25%, 50%, 75%, 95%分位数）

## 数据来源

- 输入数据：`../../data/monitoring data.xlsx`
- 模型结果：`../../data/result.xlsx`
  - Sheet: `LSTM-50runs` - LSTM模型50次运行结果
  - Sheet: `GRU-50runs` - GRU模型50次运行结果

## 论文对应章节

本代码对应论文第三章：**基于位移运动学特征的滑坡位移预测模型研究**

主要内容：
1. 位移分解方法（趋势项+周期项）
2. LSTM时序预测模型
3. GRU对比模型
4. 概率预测与不确定性量化

## 注意事项

1. 本章聚焦于**单监测点**的位移预测
2. 不涉及多监测点的空间关系建模（那是ST-GNN的工作）
3. 确保 `result.xlsx` 文件路径正确
4. 生成的图表为高分辨率（300 DPI），适合论文使用

## 常见问题

**Q: 为什么不使用ST-GNN模型？**
A: ST-GNN是时空图神经网络，用于多监测点的空间关系建模。第三章聚焦单点预测，因此使用LSTM/GRU。

**Q: 如何修改图表样式？**
A: 编辑 `visualize_results.py`，修改 matplotlib 参数。

**Q: 如何添加真实值对比？**
A: 需要从 `monitoring data.xlsx` 读取真实位移数据，然后在可视化脚本中添加对比曲线。

## 联系方式

如有问题，请联系：韦承谦
