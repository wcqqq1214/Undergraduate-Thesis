"""
图 4-3: 阶跃变形期预警细节 (V0 体系)

两张子图：
  1) LSTM 预测的月速率：50 次集成的 50%/75%/95% 分位数 + 实测月速率 + V0/5V0/10V0
  2) 每日预警等级（概率 / 传统 并排）
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
CHAPTER3_TABLES = Path('/home/wcqqq21/Undergraduate-Thesis/code/chapter3/outputs/tables')
FIGURES_DIR = BASE_DIR / 'outputs' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

WARNING_COLORS = {0: '#2ecc71', 1: '#f1c40f', 2: '#e67e22', 3: '#e74c3c'}
LEVEL_NAMES = ['绿色', '黄色', '橙色', '红色']
MONTH_WINDOW_DAYS = 30


def load_inputs(target):
    intermediate_dir = TABLES_DIR / 'intermediate_data' / target

    v0_conf = json.loads((intermediate_dir / 'v0.json').read_text())
    v0 = v0_conf['v0_mm_per_month']
    v0_orange = v0_conf['v0_orange_threshold']
    v0_red = v0_conf['v0_red_threshold']

    pred_df = pd.read_csv(CHAPTER3_TABLES / f'lstm_trend_50runs_predictions_{target}.csv')
    pred_pivot = pred_df.pivot(index='time_index', columns='run_id', values='prediction').sort_index()
    predictions = pred_pivot.to_numpy()  # (T, n_runs)

    # 测试集月速率 = 30 天滚动差分
    n_days, n_runs = predictions.shape
    monthly_rate_pred = predictions[MONTH_WINDOW_DAYS:] - predictions[:-MONTH_WINDOW_DAYS]

    actual_disp = pd.read_csv(intermediate_dir / f'actual_displacement_{target}.csv')
    actual_disp['date'] = pd.to_datetime(actual_disp['date'])

    test_dates_full = actual_disp['date'].iloc[-n_days:].reset_index(drop=True)
    test_dates = test_dates_full.iloc[MONTH_WINDOW_DAYS:].reset_index(drop=True)

    daily_inc = pd.concat([pd.Series([0.0]),
                           actual_disp['displacement'].diff().iloc[1:]])
    actual_monthly = daily_inc.rolling(MONTH_WINDOW_DAYS,
                                       min_periods=MONTH_WINDOW_DAYS).sum().to_numpy()
    # 只取测试集 & 滚动后
    actual_monthly_test = actual_monthly[-len(test_dates):]

    prob_warning = pd.read_csv(intermediate_dir / 'warning_levels.csv')
    prob_warning['date'] = test_dates

    trad_warning = pd.read_csv(intermediate_dir / 'traditional_warning_levels.csv')
    trad_warning['date'] = pd.to_datetime(trad_warning['date'])
    trad_warning = trad_warning[trad_warning['warning_level'] >= 0].reset_index(drop=True)

    return {
        'v0': v0, 'v0_orange': v0_orange, 'v0_red': v0_red,
        'test_dates': test_dates,
        'monthly_rate_pred': monthly_rate_pred,
        'actual_monthly': actual_monthly_test,
        'prob_warning': prob_warning,
        'trad_warning': trad_warning,
    }


def plot_active_period(target='MJ1', start_date=None, end_date=None):
    if start_date is None:
        start_date = pd.Timestamp('2020-03-18')
    if end_date is None:
        end_date = pd.Timestamp('2020-04-30')

    data = load_inputs(target)
    test_dates = data['test_dates']
    mr_pred = data['monthly_rate_pred']
    actual_mr = data['actual_monthly']

    mask = (test_dates >= start_date) & (test_dates <= end_date)
    if not mask.any():
        print(f'  [{target}] 时段 {start_date.date()} ~ {end_date.date()} 内无预测数据，跳过')
        return
    idx = np.where(mask)[0]
    dates = test_dates[mask].reset_index(drop=True)

    q50 = np.percentile(mr_pred[idx], 50, axis=1)
    q75 = np.percentile(mr_pred[idx], 75, axis=1)
    q95 = np.percentile(mr_pred[idx], 95, axis=1)
    actual_slice = actual_mr[idx]

    prob_slice = data['prob_warning'][mask].reset_index(drop=True)
    mt = (data['trad_warning']['date'] >= start_date) & (data['trad_warning']['date'] <= end_date)
    trad_slice = data['trad_warning'][mt].reset_index(drop=True)

    v0 = data['v0']; v0o = data['v0_orange']; v0r = data['v0_red']

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax1 = axes[0]
    ax1.plot(dates, q50, 'b-', linewidth=2.5, label='LSTM 50% 分位月速率')
    ax1.plot(dates, q75, 'g--', linewidth=2, label='LSTM 75% 分位月速率')
    ax1.plot(dates, q95, 'r:', linewidth=2, label='LSTM 95% 分位月速率')
    ax1.plot(dates, actual_slice, 'ko-', linewidth=2, markersize=4,
             label='实测月速率', alpha=0.75)
    ax1.axhline(y=v0, color=WARNING_COLORS[1], linestyle='--', linewidth=1.8,
                label=f'V0 = {v0:.2f}')
    ax1.axhline(y=v0o, color=WARNING_COLORS[2], linestyle='--', linewidth=1.8,
                label=f'5V0 = {v0o:.2f}')
    ax1.axhline(y=v0r, color=WARNING_COLORS[3], linestyle='--', linewidth=1.8,
                label=f'10V0 = {v0r:.2f}')
    ax1.set_ylabel('月速率 (mm/M)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_title(
        f'{target} 监测点阶跃变形期预警细节 '
        f'({start_date.strftime("%Y-%m-%d")} ~ {end_date.strftime("%Y-%m-%d")})',
        fontsize=14, fontweight='bold', pad=15)

    ax2 = axes[1]
    for level in range(4):
        m_lv = prob_slice['warning_level'] == level
        if m_lv.any():
            ax2.scatter(prob_slice.loc[m_lv, 'date'],
                        [level + 0.1] * int(m_lv.sum()),
                        c=WARNING_COLORS[level], s=90, alpha=0.95,
                        label=f'概率-{LEVEL_NAMES[level]}', marker='o',
                        edgecolors='black', linewidths=0.8)
    for level in range(4):
        m_lv = trad_slice['warning_level'] == level
        if m_lv.any():
            ax2.scatter(trad_slice.loc[m_lv, 'date'],
                        [level - 0.1] * int(m_lv.sum()),
                        c=WARNING_COLORS[level], s=90, alpha=0.95,
                        label=f'传统-{LEVEL_NAMES[level]}', marker='s',
                        edgecolors='black', linewidths=0.8)
    ax2.set_ylabel('预警等级 (概率↑ / 传统↓)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('日期', fontsize=13, fontweight='bold')
    ax2.set_yticks(list(range(4)))
    ax2.set_yticklabels(['安全', '警示', '警戒', '警报'], fontsize=11)
    ax2.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(-0.6, 3.6)

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()

    out_pdf = FIGURES_DIR / f'{target.lower()}_active_period.pdf'
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f'✓ 已保存: {out_pdf}')

    print(f'\n统计 [{target}] {start_date.date()} ~ {end_date.date()}')
    print(f'  实测月速率范围 {actual_slice.min():.2f} ~ {actual_slice.max():.2f} mm/M')
    print(f'  预测月速率 50% 分位 {q50.min():.2f} ~ {q50.max():.2f}, '
          f'95% 分位 {q95.min():.2f} ~ {q95.max():.2f}')
    print(f'  V0={v0:.2f}  5V0={v0o:.2f}  10V0={v0r:.2f}')
    for lv in range(4):
        cnt = int((prob_slice['warning_level'] == lv).sum())
        print(f'  概率预警 {LEVEL_NAMES[lv]}: {cnt} 天')

    plt.close()


def main(target='MJ1'):
    print('=' * 60)
    print(f'生成阶跃变形期预警细节图 [{target}]')
    print('=' * 60)
    plot_active_period(target)


if __name__ == '__main__':
    import sys
    tgt = sys.argv[1] if len(sys.argv) > 1 else 'MJ1'
    main(tgt)
