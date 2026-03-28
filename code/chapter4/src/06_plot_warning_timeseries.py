"""
图4-1: 预警时间序列图
显示概率预警、传统预警和实际位移随时间的变化
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置字体：中文使用SimSun，英文使用Times New Roman
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman', 'DejaVu Sans']
plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun']
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
    """加载所有需要的数据"""
    # 概率预警
    prob_warning = pd.read_csv(INTERMEDIATE_DIR / 'warning_levels.csv')

    # 传统预警
    trad_warning = pd.read_csv(INTERMEDIATE_DIR / 'traditional_warning_levels.csv')
    trad_warning['date'] = pd.to_datetime(trad_warning['date'])

    # 实际位移
    actual_disp = pd.read_csv(INTERMEDIATE_DIR / 'actual_displacement_MJ1.csv')
    actual_disp['date'] = pd.to_datetime(actual_disp['date'])

    # 对齐时间索引
    prob_warning['date'] = actual_disp['date'].iloc[:len(prob_warning)]

    return prob_warning, trad_warning, actual_disp

def plot_warning_timeseries():
    """绘制完整的预警时间序列"""
    prob_warning, trad_warning, actual_disp = load_data()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 子图1: 概率预警等级
    ax1 = axes[0]
    for level in range(5):
        mask = prob_warning['warning_level'] == level
        if mask.any():
            ax1.scatter(prob_warning.loc[mask, 'date'],
                       prob_warning.loc[mask, 'warning_level'],
                       c=WARNING_COLORS[level], s=20, alpha=0.7,
                       label=f'Level {level}')
    ax1.set_ylabel('概率预警等级', fontsize=12)
    ax1.set_yticks([0, 1, 2, 3, 4])
    ax1.set_yticklabels(['绿', '蓝', '黄', '橙', '红'])
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('基于概率预测的滑坡预警时间序列', fontsize=14, fontweight='bold')

    # 子图2: 传统预警等级
    ax2 = axes[1]
    for level in range(5):
        mask = trad_warning['warning_level'] == level
        if mask.any():
            ax2.scatter(trad_warning.loc[mask, 'date'],
                       trad_warning.loc[mask, 'warning_level'],
                       c=WARNING_COLORS[level], s=20, alpha=0.7,
                       label=f'Level {level}')
    ax2.set_ylabel('传统预警等级', fontsize=12)
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(['绿', '蓝', '黄', '橙', '红'])
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 子图3: 实际位移
    ax3 = axes[2]
    ax3.plot(actual_disp['date'], actual_disp['displacement'],
             'k-', linewidth=1.5, label='实际累计位移')
    ax3.set_ylabel('累计位移 (mm)', fontsize=12)
    ax3.set_xlabel('日期', fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片 (PNG和PDF格式)
    output_png = FIGURES_DIR / 'warning_timeseries.png'
    output_pdf = FIGURES_DIR / 'warning_timeseries.pdf'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ 图4-1已保存: {output_png}")
    print(f"✓ 图4-1已保存: {output_pdf}")

    plt.close()

if __name__ == '__main__':
    print("=" * 60)
    print("生成图4-1: 预警时间序列图")
    print("=" * 60)

    plot_warning_timeseries()

    print("\n✓ 图表生成完成!")
