"""
图4-1: 预警时间序列图
显示概率预警、传统预警和实际位移随时间的变化
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置字体：中文使用SimSun，英文使用Times New Roman
# 注意：SimSun在前，这样中文会用SimSun，英文会fallback到Times New Roman
plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 路径配置
BASE_DIR = Path(__file__).parent.parent
TABLES_DIR = BASE_DIR / 'outputs' / 'tables'
FIGURES_DIR = BASE_DIR / 'outputs' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 预警等级颜色映射（V0 体系 4 级，去除蓝色）
WARNING_COLORS = {
    0: '#2ecc71',  # 绿色（安全）
    1: '#f1c40f',  # 黄色（警示）
    2: '#e67e22',  # 橙色（警戒）
    3: '#e74c3c',  # 红色（警报）
}
WARNING_LABELS = {
    0: '安全 (V<V0)',
    1: '警示 (V0<V<5V0)',
    2: '警戒 (5V0<V<10V0)',
    3: '警报 (V>10V0)',
}
N_LEVELS = 4

def load_data(target='MJ1'):
    """加载所有需要的数据"""
    intermediate_dir = TABLES_DIR / 'intermediate_data' / target

    prob_warning = pd.read_csv(intermediate_dir / 'warning_levels.csv')

    trad_warning = pd.read_csv(intermediate_dir / 'traditional_warning_levels.csv')
    trad_warning['date'] = pd.to_datetime(trad_warning['date'])
    trad_warning = trad_warning[trad_warning['warning_level'] >= 0].reset_index(drop=True)

    actual_disp = pd.read_csv(intermediate_dir / f'actual_displacement_{target}.csv')
    actual_disp['date'] = pd.to_datetime(actual_disp['date'])

    # 概率预警的 time_index 对应"测试集第 time_index 天 + LSTM 滚动窗口偏移"，
    # 这里按序贴附测试集末尾的日期
    n_prob = len(prob_warning)
    prob_warning['date'] = actual_disp['date'].iloc[-n_prob:].reset_index(drop=True)

    return prob_warning, trad_warning, actual_disp


def plot_warning_timeseries(target='MJ1'):
    prob_warning, trad_warning, actual_disp = load_data(target)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax1 = axes[0]
    for level in range(N_LEVELS):
        mask = prob_warning['warning_level'] == level
        if mask.any():
            ax1.scatter(prob_warning.loc[mask, 'date'],
                        prob_warning.loc[mask, 'warning_level'],
                        c=WARNING_COLORS[level], s=20, alpha=0.7,
                        label=WARNING_LABELS[level])
    ax1.set_ylabel('概率预警等级', fontsize=12)
    ax1.set_yticks(list(range(N_LEVELS)))
    ax1.set_yticklabels(['安全', '警示', '警戒', '警报'])
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{target} 监测点：基于概率预测的滑坡预警时间序列',
                  fontsize=14, fontweight='bold')

    ax2 = axes[1]
    for level in range(N_LEVELS):
        mask = trad_warning['warning_level'] == level
        if mask.any():
            ax2.scatter(trad_warning.loc[mask, 'date'],
                        trad_warning.loc[mask, 'warning_level'],
                        c=WARNING_COLORS[level], s=20, alpha=0.7,
                        label=WARNING_LABELS[level])
    ax2.set_ylabel('传统预警等级', fontsize=12)
    ax2.set_yticks(list(range(N_LEVELS)))
    ax2.set_yticklabels(['安全', '警示', '警戒', '警报'])
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(actual_disp['date'], actual_disp['displacement'],
             'k-', linewidth=1.5, label='实际累计位移')
    ax3.set_ylabel('累计位移 (mm)', fontsize=12)
    ax3.set_xlabel('日期', fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_png = FIGURES_DIR / f'warning_timeseries_{target}.png'
    output_pdf = FIGURES_DIR / f'warning_timeseries_{target}.pdf'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ 已保存: {output_png}")
    print(f"✓ 已保存: {output_pdf}")
    plt.close()

def main(target='MJ1'):
    print("=" * 60)
    print(f"生成图4-1: 预警时间序列图 [{target}]")
    print("=" * 60)
    plot_warning_timeseries(target)

if __name__ == '__main__':
    import sys
    tgt = sys.argv[1] if len(sys.argv) > 1 else 'MJ1'
    main(tgt)
