"""
图 4-2: 详细时段分析图 (V0 体系)

三个子图：
  1) 越限概率 P(V > V0)
  2) 每日 4 级预警（概率 / 传统并列）
  3) 实测月速率 (mm/M) + V0/5V0/10V0 阈值线
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

BASE_DIR = Path(__file__).parent.parent
TABLES_DIR = BASE_DIR / 'outputs' / 'tables'
FIGURES_DIR = BASE_DIR / 'outputs' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

WARNING_COLORS = {0: '#2ecc71', 1: '#f1c40f', 2: '#e67e22', 3: '#e74c3c'}
WARNING_LABELS = {0: '安全', 1: '警示', 2: '警戒', 3: '警报'}


def load_data(target):
    intermediate_dir = TABLES_DIR / 'intermediate_data' / target

    v0_conf = json.loads((intermediate_dir / 'v0.json').read_text())
    v0 = v0_conf['v0_mm_per_month']
    v0_orange = v0_conf['v0_orange_threshold']
    v0_red = v0_conf['v0_red_threshold']

    prob_warning = pd.read_csv(intermediate_dir / 'warning_levels.csv')
    trad_warning = pd.read_csv(intermediate_dir / 'traditional_warning_levels.csv')
    trad_warning['date'] = pd.to_datetime(trad_warning['date'])
    trad_warning = trad_warning[trad_warning['warning_level'] >= 0].reset_index(drop=True)

    actual_disp = pd.read_csv(intermediate_dir / f'actual_displacement_{target}.csv')
    actual_disp['date'] = pd.to_datetime(actual_disp['date'])

    n_prob = len(prob_warning)
    prob_warning['date'] = actual_disp['date'].iloc[-n_prob:].reset_index(drop=True)

    # 全序列的实测月速率
    daily_inc = pd.concat([pd.Series([0.0]), actual_disp['displacement'].diff().iloc[1:]])
    actual_disp['monthly_rate'] = daily_inc.rolling(30, min_periods=30).sum().to_numpy()

    return prob_warning, trad_warning, actual_disp, v0, v0_orange, v0_red


def find_critical_periods(prob_warning, n_periods=2, half_window_days=15):
    warning = prob_warning[prob_warning['warning_level'] >= 1]
    if warning.empty:
        mid = prob_warning['date'].iloc[len(prob_warning) // 2]
        return [(mid - pd.Timedelta(days=half_window_days),
                 mid + pd.Timedelta(days=half_window_days))]
    # 取等级最高的若干天的日期，扩展前后窗口合并
    top = warning.nlargest(n_periods * 20, 'warning_level').sort_values('date')
    periods = []
    dates = list(top['date'])
    start = end = dates[0]
    for d in dates[1:]:
        if (d - end).days <= 3:
            end = d
        else:
            periods.append((start - pd.Timedelta(days=half_window_days),
                            end + pd.Timedelta(days=half_window_days)))
            start = end = d
    periods.append((start - pd.Timedelta(days=half_window_days),
                    end + pd.Timedelta(days=half_window_days)))
    return periods[:n_periods]


def plot_detailed_period(data, start_date, end_date, target):
    prob_warning, trad_warning, actual_disp, v0, v0_orange, v0_red = data
    mp = (prob_warning['date'] >= start_date) & (prob_warning['date'] <= end_date)
    mt = (trad_warning['date'] >= start_date) & (trad_warning['date'] <= end_date)
    md = (actual_disp['date'] >= start_date) & (actual_disp['date'] <= end_date)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax1 = axes[0]
    prob_period = prob_warning[mp]
    ax1.plot(prob_period['date'], prob_period['exceed_probability'] * 100,
             'b-', linewidth=2, label='越限概率 P(V > V0)')
    ax1.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.7,
                label='50% 越限概率参考线')
    ax1.set_ylabel('越限概率 (%)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{target} 监测点详细时段分析'
                  f' ({start_date.strftime("%Y-%m-%d")} ~ {end_date.strftime("%Y-%m-%d")})',
                  fontsize=13, fontweight='bold')

    ax2 = axes[1]
    for level in range(4):
        m_lv = prob_period['warning_level'] == level
        if m_lv.any():
            ax2.scatter(prob_period.loc[m_lv, 'date'],
                        [level + 0.1] * int(m_lv.sum()),
                        c=WARNING_COLORS[level], s=60, alpha=0.8,
                        label=f'概率-{WARNING_LABELS[level]}', marker='o')
    trad_period = trad_warning[mt]
    for level in range(4):
        m_lv = trad_period['warning_level'] == level
        if m_lv.any():
            ax2.scatter(trad_period.loc[m_lv, 'date'],
                        [level - 0.1] * int(m_lv.sum()),
                        c=WARNING_COLORS[level], s=60, alpha=0.8,
                        label=f'传统-{WARNING_LABELS[level]}', marker='s')
    ax2.set_ylabel('预警等级 (概率↑ / 传统↓)', fontsize=11)
    ax2.set_yticks(list(range(4)))
    ax2.set_yticklabels(['安全', '警示', '警戒', '警报'])
    ax2.legend(loc='upper left', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    disp_period = actual_disp[md]
    ax3.plot(disp_period['date'], disp_period['monthly_rate'],
             color='#444', linewidth=1.5, label='实测月速率')
    ax3.axhline(y=v0, color=WARNING_COLORS[1], linestyle='--', linewidth=1.4,
                label=f'V0 = {v0:.2f}')
    ax3.axhline(y=v0_orange, color=WARNING_COLORS[2], linestyle='--', linewidth=1.4,
                label=f'5V0 = {v0_orange:.2f}')
    ax3.axhline(y=v0_red, color=WARNING_COLORS[3], linestyle='--', linewidth=1.4,
                label=f'10V0 = {v0_red:.2f}')
    ax3.set_ylabel('月速率 (mm/M)', fontsize=11)
    ax3.set_xlabel('日期', fontsize=11)
    ax3.legend(loc='upper left', fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    tag = start_date.strftime("%Y%m%d")
    out_png = FIGURES_DIR / f'detailed_period_{target}_{tag}.png'
    out_pdf = FIGURES_DIR / f'detailed_period_{target}_{tag}.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f'✓ 已保存: {out_png}')
    plt.close()


def main(target='MJ1'):
    print('=' * 60)
    print(f'生成详细时段分析图 [{target}]')
    print('=' * 60)
    data = load_data(target)
    periods = find_critical_periods(data[0], n_periods=2)
    for start, end in periods:
        print(f'  {start.strftime("%Y-%m-%d")} ~ {end.strftime("%Y-%m-%d")}')
        plot_detailed_period(data, start, end, target)


if __name__ == '__main__':
    import sys
    tgt = sys.argv[1] if len(sys.argv) > 1 else 'MJ1'
    main(tgt)
