"""
生成LSTM概率预测图表（用于论文）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# 设置字体：中文使用SimSun，英文使用Times New Roman
# 注意：SimSun在前，这样中文会用SimSun，英文会fallback到Times New Roman
plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 读取统计数据
stats_df = pd.read_csv('../outputs/tables/lstm_trend_50runs_statistics.csv')

# 读取原始数据以获取日期
data = pd.read_excel('../../../data/monitoring data.xlsx', sheet_name=0)
data['Date'] = pd.to_datetime(data['Date'])

# 计算测试集的起始索引
train_size = int(len(data) * 0.8)
time_steps = 2
test_start_idx = train_size + time_steps

# 获取测试集对应的日期
test_dates = data['Date'].iloc[test_start_idx:test_start_idx + len(stats_df)]

# 创建图表
fig, ax = plt.subplots(figsize=(12, 6))

# 绘制90%置信区间（浅色）
ax.fill_between(test_dates, stats_df['p05'], stats_df['p95'],
                alpha=0.2, color='blue', label='90% 置信区间 (5%-95%)')

# 绘制50%置信区间（深色）
ax.fill_between(test_dates, stats_df['p25'], stats_df['p75'],
                alpha=0.4, color='blue', label='50% 置信区间 (25%-75%)')

# 绘制均值预测（蓝色实线）
ax.plot(test_dates, stats_df['mean'], 'b-', linewidth=2, label='均值预测 (50次运行)')

# 绘制真实值（红色散点）
ax.scatter(test_dates, stats_df['actual'], c='red', s=20, alpha=0.6,
          label='实际观测值', zorder=5)

# 设置标签和标题
ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('位移 (mm)', fontsize=12)
ax.set_title('LSTM 概率预测结果', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# 格式化x轴日期
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)

plt.tight_layout()

# 保存图表
output_path = '../outputs/figures/lstm_prediction.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"概率预测图已保存到: {output_path}")

# 同时保存PNG版本
output_path_png = '../outputs/figures/lstm_prediction.png'
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
print(f"PNG版本已保存到: {output_path_png}")

plt.close()

# 生成不确定性分析图
fig, ax = plt.subplots(figsize=(12, 5))

# 绘制标准差随时间的变化
ax.plot(test_dates, stats_df['std'], 'b-', linewidth=2)
ax.fill_between(test_dates, 0, stats_df['std'], alpha=0.3, color='blue')

# 添加平均标准差线
mean_std = stats_df['std'].mean()
ax.axhline(y=mean_std, color='r', linestyle='--', linewidth=1.5,
          label=f'平均标准差 = {mean_std:.2f} mm')

# 设置标签和标题
ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('标准差 (mm)', fontsize=12)
ax.set_title('预测不确定性随时间变化', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# 格式化x轴日期
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)

plt.tight_layout()

# 保存图表
output_path = '../outputs/figures/lstm_uncertainty.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"不确定性分析图已保存到: {output_path}")

# 同时保存PNG版本
output_path_png = '../outputs/figures/lstm_uncertainty.png'
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
print(f"PNG版本已保存到: {output_path_png}")

plt.close()

# 生成标准差分布直方图
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制直方图
n, bins, patches = ax.hist(stats_df['std'], bins=30, color='skyblue',
                           edgecolor='black', alpha=0.7)

# 添加统计信息
ax.axvline(stats_df['std'].mean(), color='red', linestyle='--',
          linewidth=2, label=f'均值 = {stats_df["std"].mean():.2f} mm')
ax.axvline(stats_df['std'].median(), color='green', linestyle='--',
          linewidth=2, label=f'中位数 = {stats_df["std"].median():.2f} mm')

# 设置标签和标题
ax.set_xlabel('标准差 (mm)', fontsize=12)
ax.set_ylabel('频数', fontsize=12)
ax.set_title('预测标准差分布', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# 保存图表
output_path = '../outputs/figures/lstm_std_distribution.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"标准差分布图已保存到: {output_path}")

# 同时保存PNG版本
output_path_png = '../outputs/figures/lstm_std_distribution.png'
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
print(f"PNG版本已保存到: {output_path_png}")

plt.close()

# ── 新增：点预测对比图（训练集 + 测试集）────────────────────────────────────────
train_stats_df = pd.read_csv('../outputs/tables/lstm_trend_50runs_train_statistics.csv')

# 训练集日期
train_dates = data['Date'].iloc[time_steps:time_steps + len(train_stats_df)]

fig, ax = plt.subplots(figsize=(12, 5))

# 实测值：蓝色实线（训练集 + 测试集连续）
all_actual_dates = list(train_dates) + list(test_dates)
all_actual_values = list(train_stats_df['actual']) + list(stats_df['actual'])
ax.plot(all_actual_dates, all_actual_values, color='#1f77b4', linewidth=1.5,
        label='实测位移')

# 预测均值：红色虚线（训练集 + 测试集连续）
all_pred_dates = list(train_dates) + list(test_dates)
all_pred_values = list(train_stats_df['mean']) + list(stats_df['mean'])
ax.plot(all_pred_dates, all_pred_values, color='#d62728', linewidth=1.5,
        linestyle='--', label='LSTM预测值（50次运行均值）')

# 训练/测试分割线
split_date = test_dates.iloc[0]
ax.axvline(x=split_date, color='gray', linestyle=':', linewidth=1.2)
ax.text(split_date, ax.get_ylim()[0], ' 测试集', fontsize=9, color='gray', va='bottom')

ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('累计位移 (mm)', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

plt.tight_layout()

output_path = '../outputs/figures/lstm_fitting.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"拟合对比图已保存到: {output_path}")

output_path_png = '../outputs/figures/lstm_fitting.png'
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
print(f"PNG版本已保存到: {output_path_png}")

plt.close()
# ─────────────────────────────────────────────────────────────────────────────

print("\n所有图表生成完成！")
