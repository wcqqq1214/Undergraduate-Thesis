"""
生成LSTM概率预测图表（用于论文，MJ9/MJ1/MJ3 三点三子图布局）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 字体：中文 SimSun，英文 Times New Roman
plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 三个目标点 + 各自的 time_steps（与 run_lstm_trend_50times.py 的 TARGET_CONFIG 保持一致）
TARGETS = ['MJ9', 'MJ1', 'MJ3']
TIME_STEPS = {'MJ9': 7, 'MJ1': 2, 'MJ3': 2}

TABLES_DIR = '../outputs/tables'
FIG_DIR = '../outputs/figures'

# 读取原始数据以获取日期
data = pd.read_excel('../../../data/monitoring data.xlsx', sheet_name=0)
data['Date'] = pd.to_datetime(data['Date'])
train_size = int(len(data) * 0.8)


def load_point(tag):
    """读取某个目标点的统计结果及其训练/测试集对应日期"""
    ts = TIME_STEPS[tag]
    stats = pd.read_csv(f'{TABLES_DIR}/lstm_trend_50runs_statistics_{tag}.csv')
    train_stats = pd.read_csv(
        f'{TABLES_DIR}/lstm_trend_50runs_train_statistics_{tag}.csv')
    test_start = train_size + ts
    test_dates = data['Date'].iloc[test_start:test_start + len(stats)].reset_index(drop=True)
    train_dates = data['Date'].iloc[ts:ts + len(train_stats)].reset_index(drop=True)
    return stats, train_stats, test_dates, train_dates


def save_fig(fig, name):
    fig.savefig(f'{FIG_DIR}/{name}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{FIG_DIR}/{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存: {name}.pdf / {name}.png')


# 预先加载三个点的数据，避免重复 IO
POINT_DATA = {tag: load_point(tag) for tag in TARGETS}


# ── 图1：点预测拟合对比（训练+测试） 3 行 × 1 列 ───────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
for ax, tag in zip(axes, TARGETS):
    stats, train_stats, test_dates, train_dates = POINT_DATA[tag]
    all_dates = list(train_dates) + list(test_dates)
    all_actual = list(train_stats['actual']) + list(stats['actual'])
    all_pred = list(train_stats['mean']) + list(stats['mean'])
    ax.plot(all_dates, all_actual, color='#1f77b4', linewidth=1.5, label='实测位移')
    ax.plot(all_dates, all_pred, color='#d62728', linewidth=1.5, linestyle='--',
            label='LSTM预测值（50次运行均值）')
    split_date = test_dates.iloc[0]
    ax.axvline(x=split_date, color='gray', linestyle=':', linewidth=1.2)
    ax.text(split_date, ax.get_ylim()[0], ' 测试集', fontsize=9, color='gray', va='bottom')
    ax.set_title(f'{tag} 监测点', fontsize=12, fontweight='bold', loc='left')
    ax.set_ylabel('累计位移 (mm)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
axes[0].legend(loc='upper left', fontsize=10)
axes[-1].set_xlabel('日期', fontsize=11)
plt.tight_layout()
save_fig(fig, 'lstm_fitting')


# ── 图2：概率预测结果（均值+置信区间） 3 行 × 1 列 ───────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
for ax, tag in zip(axes, TARGETS):
    stats, _, test_dates, _ = POINT_DATA[tag]
    ax.fill_between(test_dates, stats['p05'], stats['p95'],
                    alpha=0.2, color='blue', label='90% 置信区间 (5%-95%)')
    ax.fill_between(test_dates, stats['p25'], stats['p75'],
                    alpha=0.4, color='blue', label='50% 置信区间 (25%-75%)')
    ax.plot(test_dates, stats['mean'], 'b-', linewidth=2, label='均值预测 (50次运行)')
    ax.scatter(test_dates, stats['actual'], c='red', s=14, alpha=0.6,
               label='实际观测值', zorder=5)
    ax.set_title(f'{tag} 监测点', fontsize=12, fontweight='bold', loc='left')
    ax.set_ylabel('位移 (mm)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
axes[0].legend(loc='upper left', fontsize=9)
axes[-1].set_xlabel('日期', fontsize=11)
plt.tight_layout()
save_fig(fig, 'lstm_prediction')


# ── 图3：预测不确定性随时间变化 3 行 × 1 列 ─────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
for ax, tag in zip(axes, TARGETS):
    stats, _, test_dates, _ = POINT_DATA[tag]
    ax.plot(test_dates, stats['std'], 'b-', linewidth=2)
    ax.fill_between(test_dates, 0, stats['std'], alpha=0.3, color='blue')
    mean_std = stats['std'].mean()
    ax.axhline(y=mean_std, color='r', linestyle='--', linewidth=1.5,
               label=f'平均标准差 = {mean_std:.2f} mm')
    ax.set_title(f'{tag} 监测点', fontsize=12, fontweight='bold', loc='left')
    ax.set_ylabel('标准差 (mm)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
axes[-1].set_xlabel('日期', fontsize=11)
plt.tight_layout()
save_fig(fig, 'lstm_uncertainty')


# ── 图4：标准差分布直方图 1 行 × 3 列 ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, tag in zip(axes, TARGETS):
    stats, _, _, _ = POINT_DATA[tag]
    ax.hist(stats['std'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(stats['std'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'均值 = {stats["std"].mean():.2f} mm')
    ax.axvline(stats['std'].median(), color='green', linestyle='--', linewidth=2,
               label=f'中位数 = {stats["std"].median():.2f} mm')
    ax.set_title(f'{tag} 监测点', fontsize=12, fontweight='bold', loc='left')
    ax.set_xlabel('标准差 (mm)', fontsize=11)
    ax.set_ylabel('频数', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
save_fig(fig, 'lstm_std_distribution')

print('\n所有图表生成完成！')
