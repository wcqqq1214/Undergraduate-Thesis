"""
图4-3: MJ1监测点阶跃变形期预警细节（2018年6-8月）
展示预测位移的不同分位数、安全阈值、实际监测位移、预警等级
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置字体
plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 路径配置
BASE_DIR = Path(__file__).parent.parent
TABLES_DIR = BASE_DIR / 'outputs' / 'tables'
INTERMEDIATE_DIR = TABLES_DIR / 'intermediate_data'
FIGURES_DIR = BASE_DIR / 'outputs' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 预警等级颜色映射
WARNING_COLORS = {
    0: '#2ecc71',  # 绿色
    1: '#3498db',  # 蓝色
    2: '#f1c40f',  # 黄色
    3: '#e67e22',  # 橙色
    4: '#e74c3c'   # 红色
}

def load_data():
    """加载数据"""
    # 加载50次运行的日增量预测
    daily_increments = np.load(INTERMEDIATE_DIR / 'daily_increments_50runs.npy')

    # 加载实际位移
    actual_disp = pd.read_csv(INTERMEDIATE_DIR / 'actual_displacement_MJ1.csv')
    actual_disp['date'] = pd.to_datetime(actual_disp['date'])

    # 加载预警等级
    warning_levels = pd.read_csv(INTERMEDIATE_DIR / 'warning_levels.csv')
    warning_levels['date'] = actual_disp['date'].iloc[:len(warning_levels)]

    return daily_increments, actual_disp, warning_levels

def calculate_quantile_predictions(daily_increments, actual_disp):
    """
    从日增量预测计算累积位移的分位数
    """
    # 累积求和得到位移预测
    cumulative_disp = np.cumsum(daily_increments, axis=0)

    # 加上初始位移
    initial_disp = actual_disp['displacement'].iloc[0]
    cumulative_disp = cumulative_disp + initial_disp

    # 计算分位数
    q50 = np.percentile(cumulative_disp, 50, axis=1)
    q75 = np.percentile(cumulative_disp, 75, axis=1)
    q95 = np.percentile(cumulative_disp, 95, axis=1)

    return q50, q75, q95

def plot_active_period():
    """绘制阶跃变形期预警细节"""
    # 加载数据
    daily_increments, actual_disp, warning_levels = load_data()

    # 计算分位数预测
    q50, q75, q95 = calculate_quantile_predictions(daily_increments, actual_disp)

    # 筛选2018年6-8月数据
    start_date = pd.Timestamp('2018-06-01')
    end_date = pd.Timestamp('2018-08-31')
    mask = (actual_disp['date'] >= start_date) & (actual_disp['date'] <= end_date)

    period_disp = actual_disp[mask].reset_index(drop=True)
    period_warning = warning_levels[mask].reset_index(drop=True)

    # 获取对应的分位数预测
    start_idx = actual_disp[actual_disp['date'] == start_date].index[0]
    end_idx = actual_disp[actual_disp['date'] == end_date].index[0]

    period_q50 = q50[start_idx:end_idx+1]
    period_q75 = q75[start_idx:end_idx+1]
    period_q95 = q95[start_idx:end_idx+1]

    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 子图1: 位移预测与实际位移
    ax1 = axes[0]

    # 绘制分位数预测
    ax1.plot(period_disp['date'], period_q50, 'b-', linewidth=2, label='预测位移 (50%分位数)', alpha=0.8)
    ax1.plot(period_disp['date'], period_q75, 'g--', linewidth=1.5, label='预测位移 (75%分位数)', alpha=0.7)
    ax1.plot(period_disp['date'], period_q95, 'r:', linewidth=1.5, label='预测位移 (95%分位数)', alpha=0.7)

    # 绘制实际位移
    ax1.plot(period_disp['date'], period_disp['displacement'], 'ko-',
             linewidth=2, markersize=4, label='实际监测位移', alpha=0.6)

    # 安全阈值（假设为实际位移的平均值 + 标准差）
    threshold = period_disp['displacement'].mean() + period_disp['displacement'].std()
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                label=f'安全阈值 ({threshold:.1f} mm)', alpha=0.8)

    ax1.set_ylabel('累积位移 (mm)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('MJ1监测点阶跃变形期预警细节 (2018年6月-8月)',
                  fontsize=14, fontweight='bold', pad=15)

    # 子图2: 预警等级
    ax2 = axes[1]

    # 绘制预警等级背景色
    for i in range(len(period_warning)):
        level = period_warning.loc[i, 'warning_level']
        ax2.axvspan(period_disp.loc[i, 'date'],
                   period_disp.loc[min(i+1, len(period_disp)-1), 'date'],
                   facecolor=WARNING_COLORS[level], alpha=0.3)

    # 绘制预警等级散点
    for level in range(5):
        mask_level = period_warning['warning_level'] == level
        if mask_level.any():
            level_names = ['绿色', '蓝色', '黄色', '橙色', '红色']
            ax2.scatter(period_disp.loc[mask_level, 'date'],
                       period_warning.loc[mask_level, 'warning_level'],
                       c=WARNING_COLORS[level], s=80, alpha=0.9,
                       label=f'{level_names[level]}预警', marker='o', edgecolors='black', linewidths=0.5)

    ax2.set_ylabel('预警等级', fontsize=12, fontweight='bold')
    ax2.set_xlabel('日期', fontsize=12, fontweight='bold')
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(['0-绿色', '1-蓝色', '2-黄色', '3-橙色', '4-红色'])
    ax2.legend(loc='upper left', fontsize=9, ncol=5, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # 旋转x轴标签
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # 保存图片
    output_png = FIGURES_DIR / 'mj1_active_period.png'
    output_pdf = FIGURES_DIR / 'mj1_active_period.pdf'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ 图4-3已保存: {output_png}")
    print(f"✓ 图4-3已保存: {output_pdf}")

    plt.close()

if __name__ == '__main__':
    print("=" * 60)
    print("生成图4-3: MJ1监测点阶跃变形期预警细节")
    print("=" * 60)

    plot_active_period()

    print("\n✓ 图表生成完成!")
