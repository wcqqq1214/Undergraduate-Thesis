"""
根据50次运行结果计算论文所需的统计数据
"""

import pandas as pd
import numpy as np

# 读取汇总数据
summary_df = pd.read_csv('../outputs/tables/lstm_trend_50runs_summary.csv')

# 读取时间序列统计数据
stats_df = pd.read_csv('../outputs/tables/lstm_trend_50runs_statistics.csv')

print("=" * 80)
print("表3-2: LSTM模型点预测性能（基于50次运行均值）")
print("=" * 80)

# 计算点预测性能（使用50次运行的平均值）
train_r2_mean = summary_df['train_r2'].mean()
train_rmse_mean = summary_df['train_rmse'].mean()
test_r2_mean = summary_df['test_r2'].mean()
test_rmse_mean = summary_df['test_rmse'].mean()

print(f"\n训练集:")
print(f"  R² = {train_r2_mean:.4f}")
print(f"  RMSE = {train_rmse_mean:.2f} mm")

print(f"\n测试集:")
print(f"  R² = {test_r2_mean:.4f}")
print(f"  RMSE = {test_rmse_mean:.2f} mm")

print("\n" + "=" * 80)
print("表3-3: LSTM概率预测统计特征")
print("=" * 80)

# 计算概率预测统计特征
mean_prediction = stats_df['mean'].mean()  # 所有时间点的平均预测值
mean_std = stats_df['std'].mean()  # 所有时间点的平均标准差
max_std = stats_df['std'].max()  # 最大标准差
min_std = stats_df['std'].min()  # 最小标准差
cv = (mean_std / mean_prediction) * 100  # 变异系数

print(f"\n平均预测值: {mean_prediction:.2f} mm")
print(f"平均标准差: {mean_std:.2f} mm")
print(f"最大标准差: {max_std:.2f} mm")
print(f"最小标准差: {min_std:.2f} mm")
print(f"变异系数: {cv:.2f}%")

# 计算实际平均位移
actual_mean = stats_df['actual'].mean()
print(f"\n实际平均位移: {actual_mean:.2f} mm")
print(f"相对误差: {abs(mean_prediction - actual_mean) / actual_mean * 100:.2f}%")

print("\n" + "=" * 80)
print("额外统计信息（用于论文描述）")
print("=" * 80)

# 计算R²和RMSE的标准差（用于描述模型稳定性）
train_r2_std = summary_df['train_r2'].std()
train_rmse_std = summary_df['train_rmse'].std()
test_r2_std = summary_df['test_r2'].std()
test_rmse_std = summary_df['test_rmse'].std()

print(f"\n训练集 R² 标准差: {train_r2_std:.4f}")
print(f"训练集 RMSE 标准差: {train_rmse_std:.2f} mm")
print(f"测试集 R² 标准差: {test_r2_std:.4f}")
print(f"测试集 RMSE 标准差: {test_rmse_std:.2f} mm")

# 计算R²和RMSE的范围
print(f"\n测试集 R² 范围: [{summary_df['test_r2'].min():.4f}, {summary_df['test_r2'].max():.4f}]")
print(f"测试集 RMSE 范围: [{summary_df['test_rmse'].min():.2f}, {summary_df['test_rmse'].max():.2f}] mm")

print("\n" + "=" * 80)
print("LaTeX表格代码")
print("=" * 80)

print("\n% 表3-2: LSTM模型点预测性能")
print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{LSTM模型点预测性能（基于50次运行均值）}")
print("\\label{tab:lstm-point-prediction}")
print("\\begin{tabular}{lcc}")
print("\\toprule")
print("数据集 & $R^2$ & RMSE (mm) \\\\")
print("\\midrule")
print(f"训练集 & {train_r2_mean:.4f} & {train_rmse_mean:.2f} \\\\")
print(f"测试集 & {test_r2_mean:.4f} & {test_rmse_mean:.2f} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print("\n% 表3-3: LSTM概率预测统计特征")
print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{LSTM概率预测统计特征}")
print("\\label{tab:lstm-probability-stats}")
print("\\begin{tabular}{lc}")
print("\\toprule")
print("统计量 & 数值 \\\\")
print("\\midrule")
print(f"平均预测值 (mm) & {mean_prediction:.2f} \\\\")
print(f"平均标准差 (mm) & {mean_std:.2f} \\\\")
print(f"最大标准差 (mm) & {max_std:.2f} \\\\")
print(f"最小标准差 (mm) & {min_std:.2f} \\\\")
print(f"变异系数 (\\%) & {cv:.2f} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print("\n完成！")
