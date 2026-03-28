"""
图4-2, 4-3: 详细时段分析图
放大显示特定时间段的预警情况
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# 设置字体：中文使用SimSun，英文使用Times New Roman
# 注意：SimSun在前，这样中文会用SimSun，英文会fallback到Times New Roman
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

    # 计算日增量
    actual_disp['daily_increment'] = actual_disp['displacement'].diff()

    return prob_warning, trad_warning, actual_disp

def find_critical_periods(prob_warning, actual_disp, n_periods=2):
    """
    自动识别关键时段
    选择预警等级较高或位移变化较大的时段
    """
    # 找到预警等级 >= 2 (黄色及以上) 的时段
    warning_periods = prob_warning[prob_warning['warning_level'] >= 2]

    if len(warning_periods) == 0:
        # 如果没有高等级预警，选择位移增量最大的时段
        actual_disp_sorted = actual_disp.sort_values('daily_increment', ascending=False)
        critical_dates = actual_disp_sorted.head(n_periods * 30)['date'].tolist()
    else:
        # 选择预警等级最高的时段
        critical_dates = warning_periods.nlargest(n_periods * 30, 'warning_level')['date'].tolist()

    # 将日期分组为连续时段
    periods = []
    if critical_dates:
        critical_dates = sorted(critical_dates)
        start = critical_dates[0]
        end = critical_dates[0]

        for i in range(1, len(critical_dates)):
            if (critical_dates[i] - end).days <= 3:  # 3天内视为连续
                end = critical_dates[i]
            else:
                # 扩展时段前后各15天
                periods.append((start - pd.Timedelta(days=15), end + pd.Timedelta(days=15)))
                start = critical_dates[i]
                end = critical_dates[i]

        periods.append((start - pd.Timedelta(days=15), end + pd.Timedelta(days=15)))

    # 如果没有找到关键时段，选择整个时间序列的中间部分
    if not periods:
        mid_date = actual_disp['date'].iloc[len(actual_disp) // 2]
        periods = [
            (mid_date - pd.Timedelta(days=30), mid_date),
            (mid_date, mid_date + pd.Timedelta(days=30))
        ]

    return periods[:n_periods]

def plot_detailed_period(prob_warning, trad_warning, actual_disp, start_date, end_date, fig_num):
    """绘制单个详细时段"""
    # 筛选时间范围
    mask_prob = (prob_warning['date'] >= start_date) & (prob_warning['date'] <= end_date)
    mask_trad = (trad_warning['date'] >= start_date) & (trad_warning['date'] <= end_date)
    mask_disp = (actual_disp['date'] >= start_date) & (actual_disp['date'] <= end_date)

    prob_period = prob_warning[mask_prob]
    trad_period = trad_warning[mask_trad]
    disp_period = actual_disp[mask_disp]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # 子图1: 越限概率
    ax1 = axes[0]
    ax1.plot(prob_period['date'], prob_period['exceed_probability'] * 100,
             'b-', linewidth=2, label='越限概率')
    ax1.axhline(y=50, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='橙色预警阈值')
    ax1.axhline(y=20, color='gold', linestyle='--', linewidth=1, alpha=0.7, label='黄色预警阈值')
    ax1.set_ylabel('越限概率 (%)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'详细时段分析 ({start_date.strftime("%Y-%m-%d")} 至 {end_date.strftime("%Y-%m-%d")})',
                  fontsize=13, fontweight='bold')

    # 子图2: 概率预警等级
    ax2 = axes[1]
    for level in range(5):
        mask = prob_period['warning_level'] == level
        if mask.any():
            ax2.scatter(prob_period.loc[mask, 'date'],
                       prob_period.loc[mask, 'warning_level'],
                       c=WARNING_COLORS[level], s=50, alpha=0.8,
                       label=f'Level {level}', marker='o')
    ax2.set_ylabel('概率预警等级', fontsize=11)
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(['绿', '蓝', '黄', '橙', '红'])
    ax2.legend(loc='upper left', fontsize=8, ncol=5)
    ax2.grid(True, alpha=0.3)

    # 子图3: 传统预警等级
    ax3 = axes[2]
    for level in range(5):
        mask = trad_period['warning_level'] == level
        if mask.any():
            ax3.scatter(trad_period.loc[mask, 'date'],
                       trad_period.loc[mask, 'warning_level'],
                       c=WARNING_COLORS[level], s=50, alpha=0.8,
                       label=f'Level {level}', marker='s')
    ax3.set_ylabel('传统预警等级', fontsize=11)
    ax3.set_yticks([0, 1, 2, 3, 4])
    ax3.set_yticklabels(['绿', '蓝', '黄', '橙', '红'])
    ax3.legend(loc='upper left', fontsize=8, ncol=5)
    ax3.grid(True, alpha=0.3)

    # 子图4: 日位移增量
    ax4 = axes[3]
    ax4.bar(disp_period['date'], disp_period['daily_increment'],
            width=0.8, color='gray', alpha=0.6, label='日位移增量')
    ax4.axhline(y=0.3, color='red', linestyle='--', linewidth=2, label='预警阈值 (0.3 mm/天)')
    ax4.set_ylabel('日位移增量 (mm)', fontsize=11)
    ax4.set_xlabel('日期', fontsize=11)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片 (PNG和PDF格式)
    output_png = FIGURES_DIR / f'detailed_period_{start_date.strftime("%Y%m%d")}.png'
    output_pdf = FIGURES_DIR / f'detailed_period_{start_date.strftime("%Y%m%d")}.pdf'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ 图4-{fig_num}已保存: {output_png}")
    print(f"✓ 图4-{fig_num}已保存: {output_pdf}")

    plt.close()

if __name__ == '__main__':
    print("=" * 60)
    print("生成图4-2, 4-3: 详细时段分析图")
    print("=" * 60)

    # 加载数据
    prob_warning, trad_warning, actual_disp = load_data()

    # 自动识别关键时段
    print("\n正在识别关键时段...")
    critical_periods = find_critical_periods(prob_warning, actual_disp, n_periods=2)

    # 绘制每个时段
    for i, (start, end) in enumerate(critical_periods, start=2):
        print(f"\n绘制时段 {i-1}: {start.strftime('%Y-%m-%d')} 至 {end.strftime('%Y-%m-%d')}")
        plot_detailed_period(prob_warning, trad_warning, actual_disp, start, end, i)

    print("\n✓ 所有图表生成完成!")
