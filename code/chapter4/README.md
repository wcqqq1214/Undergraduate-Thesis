# 第四章代码：基于概率预测的滑坡预警方法研究

## 📁 目录结构

```
chapter4/
├── src/                                    # 源代码
│   ├── 01_calculate_exceed_probability.py  # 模块1: 计算越限概率
│   ├── 02_determine_warning_levels.py      # 模块2: 确定预警等级
│   ├── 03_traditional_velocity_warning.py  # 模块3: 传统速率预警
│   ├── 04_evaluate_performance.py          # 模块4: 性能评估
│   └── 05_calculate_lead_time.py           # 模块5: 计算预警提前时间
├── outputs/                                # 输出结果
│   ├── tables/                             # 表格数据
│   └── figures/                            # 图表
├── run_all.py                              # 主运行脚本
└── README.md                               # 本文件
```

## 🚀 快速开始

### 方法1：运行所有模块（推荐）

```bash
cd /home/wcqqq21/Undergraduate-Thesis/code/chapter4
python run_all.py
```

### 方法2：单独运行某个模块

```bash
cd /home/wcqqq21/Undergraduate-Thesis/code/chapter4/src
python 01_calculate_exceed_probability.py
python 02_determine_warning_levels.py
python 03_traditional_velocity_warning.py
python 04_evaluate_performance.py
python 05_calculate_lead_time.py
```

## 📊 模块说明

### 模块1：计算越限概率
- **输入**: `chapter3/outputs/tables/lstm_trend_50runs_predictions.csv`
- **输出**:
  - `exceed_probability.csv` - 每个时间步的越限概率
  - `daily_increments_50runs.npy` - 50次预测的日增量
- **功能**: 基于50次LSTM预测结果，计算日位移增量超过0.3mm的概率

### 模块2：确定预警等级
- **输入**: `exceed_probability.csv`
- **输出**:
  - `warning_levels.csv` - 预警等级（绿/蓝/黄/橙/红）
- **功能**: 根据越限概率确定五级预警等级

### 模块3：传统速率预警
- **输入**: `data/monitoring data.xlsx`
- **输出**:
  - `traditional_warning_levels.csv` - 传统方法的预警等级
  - `actual_displacement_MJ1.csv` - MJ1实际位移数据
- **功能**: 实现传统速率预警方法作为对比基准

### 模块4：性能评估
- **输入**:
  - `warning_levels.csv` (概率预警)
  - `traditional_warning_levels.csv` (传统预警)
  - `actual_displacement_MJ1.csv` (实际位移)
- **输出**:
  - `confusion_matrix.csv` - 混淆矩阵（表4-3）
  - `performance_comparison.csv` - 性能指标对比（表4-4）
- **功能**: 计算准确率、召回率、精确率、F1分数等指标

### 模块5：计算预警提前时间
- **输入**:
  - `warning_levels.csv`
  - `traditional_warning_levels.csv`
  - `actual_displacement_MJ1.csv`
- **输出**:
  - `lead_time_comparison.csv` - 提前时间对比（表4-2）
  - `probability_warning_events.json` - 预警事件详情
- **功能**: 统计预警触发时刻与实际越限时刻的时间差

## 📈 生成的表格

运行完成后，会在`outputs/tables/`目录下生成以下文件：

### 论文表格（paper_tables/）
| 文件名 | 对应论文表格 | 说明 |
|--------|-------------|------|
| `lead_time_comparison.csv` | 表4-2 | 预警提前时间统计 |
| `confusion_matrix.csv` | 表4-3 | 混淆矩阵对比 |
| `performance_comparison.csv` | 表4-4 | 性能指标对比 |

### 中间数据（intermediate_data/）
- `exceed_probability.csv` - 越限概率时间序列
- `warning_levels.csv` - 概率预警等级
- `traditional_warning_levels.csv` - 传统预警等级
- `actual_displacement_MJ1.csv` - MJ1实际位移数据
- `daily_increments_50runs.npy` - 50次运行的日增量数组

### 统计信息（statistics/）
- `probability_warning_metrics.json` - 概率预警性能指标详情
- `traditional_warning_metrics.json` - 传统预警性能指标详情
- `probability_warning_events.json` - 预警事件详细记录
- 其他统计JSON文件

## 🎨 图表生成

表格数据生成后，运行绘图脚本生成图表：

```bash
python src/06_plot_warning_timeseries.py    # 图4-1: 预警时间序列
python src/07_plot_detailed_periods.py      # 图4-2: 详细时段分析
```

生成的图表会保存为PNG和PDF两种格式：
- `outputs/figures/warning_timeseries.{png,pdf}` - 图4-1
- `outputs/figures/detailed_period_YYYYMMDD.{png,pdf}` - 图4-2

PDF格式的图表会自动复制到 `docs/latex/figures/chapter4/` 供论文使用。

## ⚙️ 依赖环境

```bash
pip install numpy pandas openpyxl matplotlib seaborn
```

## 🔍 数据说明

### 输入数据要求

1. **LSTM预测结果** (`lstm_trend_50runs_predictions.csv`)
   - 格式: `run_id, time_index, prediction, actual`
   - 50次独立运行的预测结果
   - prediction为累计位移（mm）

2. **实际监测数据** (`monitoring data.xlsx`)
   - 包含MJ1/mm列（累计位移）
   - 包含Date列（日期）
   - 时间粒度：天

### 关键参数

- **预警阈值**: 0.3 mm/天（日位移增量）
- **预警等级划分**:
  - 绿色: P < 5%
  - 蓝色: 5% ≤ P < 20%
  - 黄色: 20% ≤ P < 50%
  - 橙色: 50% ≤ P < 80%
  - 红色: P ≥ 80%

## 📝 注意事项

1. **数据对齐**: 确保LSTM预测结果和实际监测数据的时间范围一致
2. **差分操作**: 所有累计位移都需要先差分得到日增量
3. **Ground Truth**: 使用实际日位移增量 > 0.3mm 作为真实标签
4. **预警判定**: 黄色及以上（level ≥ 2）视为预警触发

## 🐛 故障排除

### 问题1: 找不到输入文件
```
FileNotFoundError: lstm_trend_50runs_predictions.csv
```
**解决**: 确保已运行第三章代码，生成了预测结果文件

### 问题2: 数据形状不匹配
```
ValueError: operands could not be broadcast together
```
**解决**: 检查LSTM预测结果和实际监测数据的时间范围是否一致

### 问题3: 越限概率全为0
```
mean_exceed_prob: 0.00%
```
**解决**: 检查阈值设置是否合理，或者预测结果是否正确

## 📧 联系方式

如有问题，请检查代码注释或查看论文第四章说明。
