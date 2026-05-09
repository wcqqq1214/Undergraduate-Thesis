"""
模块 9 (A2 方案): 训练期 in-sample 阶跃段预警演示

背景：按第三章 80/20 时间序列划分，测试集 (2019-06 ~ 2020-06) 恰好落入藕塘滑坡
全周期中最平静的阶段，测试期各点最大月速率均远低于 V0 阈值，故测试集下概率预警
系统虽能正确"全绿无误报"但缺乏真实越限事件可供展示召回能力。

为直观展示方法在阶跃变形段的行为，本模块对 LSTM **训练集 in-sample 预测**
(chapter3 产出的 lstm_trend_50runs_train_predictions_{tgt}.csv) 进行同一套
V0 体系判级，重点关注 2017 和 2018 两个已知的阶跃变形段。

**严格声明**：此处 LSTM 的预测值并非 out-of-sample，故所得指标不纳入严格的泛化性
评价，仅作为方法在阶跃变形段的定性演示。
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

BASE_DIR = Path(__file__).parent.parent
TABLES_DIR = BASE_DIR / 'outputs' / 'tables'
CHAPTER3_TABLES = Path('/home/wcqqq21/Undergraduate-Thesis/code/chapter3/outputs/tables')
DATA_FILE = BASE_DIR.parent.parent / 'data' / 'monitoring data.xlsx'
FIGURES_DIR = BASE_DIR / 'outputs' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MONTH_WINDOW_DAYS = 30
WARNING_COLORS = {0: '#2ecc71', 1: '#f1c40f', 2: '#e67e22', 3: '#e74c3c'}
LEVEL_NAMES = ['绿色', '黄色', '橙色', '红色']
TIME_STEPS_MAP = {'MJ9': 7, 'MJ1': 2, 'MJ3': 2}

# 演示时段（仅保留 2017 年阶跃段；2018 年经核查实测月速率未触及 V0，属于平静年）
DEMO_PERIODS = [
    ('2017阶跃段', pd.Timestamp('2017-06-01'), pd.Timestamp('2017-12-31')),
]


def load_in_sample(target):
    intermediate_dir = TABLES_DIR / 'intermediate_data' / target
    v0_conf = json.loads((intermediate_dir / 'v0.json').read_text())

    train_pred = pd.read_csv(CHAPTER3_TABLES / f'lstm_trend_50runs_train_predictions_{target}.csv')
    predictions = train_pred.pivot(index='time_index', columns='run_id',
                                    values='prediction').sort_index().to_numpy()

    data = pd.read_excel(DATA_FILE)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date').sort_index()
    disp = data[f'{target}/mm'].astype(float)

    ts = TIME_STEPS_MAP[target]
    train_size = int(len(disp) * 0.80)
    # chapter3 训练集预测对应 disp.iloc[ts : train_size]
    train_dates_full = disp.index[ts: ts + len(predictions)]

    # in-sample 月速率
    in_monthly = predictions[MONTH_WINDOW_DAYS:] - predictions[:-MONTH_WINDOW_DAYS]
    monthly_dates = train_dates_full[MONTH_WINDOW_DAYS:]

    # 实测月速率
    daily_inc = disp.diff().fillna(0)
    actual_monthly_full = daily_inc.rolling(MONTH_WINDOW_DAYS,
                                            min_periods=MONTH_WINDOW_DAYS).sum()
    # 对齐到 monthly_dates
    actual_monthly = actual_monthly_full.loc[monthly_dates].to_numpy()

    return v0_conf, monthly_dates, in_monthly, actual_monthly


def compute_probs_and_levels(monthly_pred, v0, v0_orange, v0_red):
    p_exceed_v0 = (monthly_pred > v0).mean(axis=1)
    p_exceed_5v0 = (monthly_pred > v0_orange).mean(axis=1)
    p_exceed_10v0 = (monthly_pred > v0_red).mean(axis=1)

    p_green = 1.0 - p_exceed_v0
    p_yellow = p_exceed_v0 - p_exceed_5v0
    p_orange = p_exceed_5v0 - p_exceed_10v0
    p_red = p_exceed_10v0
    probs = np.stack([p_green, p_yellow, p_orange, p_red], axis=1)
    levels = probs.argmax(axis=1)
    return probs, levels


def plot_demo_period(target, period_name, start, end, dates, in_monthly, actual_monthly,
                     probs, levels, v0, v0_orange, v0_red):
    mask = (dates >= start) & (dates <= end)
    if not mask.any():
        print(f'  [{target}] {period_name} 无样本，跳过')
        return

    idx = np.where(mask)[0]
    d = dates[mask]
    q50 = np.percentile(in_monthly[idx], 50, axis=1)
    q75 = np.percentile(in_monthly[idx], 75, axis=1)
    q95 = np.percentile(in_monthly[idx], 95, axis=1)
    actual = actual_monthly[idx]
    lv = levels[idx]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    ax1 = axes[0]
    ax1.plot(d, q50, 'b-', linewidth=2.5, label='LSTM 50% 分位月速率')
    ax1.fill_between(d, q50, q75, color='blue', alpha=0.15, label='50-75% 分位')
    ax1.fill_between(d, q75, q95, color='blue', alpha=0.08, label='75-95% 分位')
    ax1.plot(d, actual, 'k-', linewidth=1.5, label='实测月速率')
    ax1.axhline(y=v0, color=WARNING_COLORS[1], linestyle='--', linewidth=1.5,
                label=f'V0 = {v0:.2f}')
    ax1.axhline(y=v0_orange, color=WARNING_COLORS[2], linestyle='--', linewidth=1.5,
                label=f'5V0 = {v0_orange:.2f}')
    ax1.axhline(y=v0_red, color=WARNING_COLORS[3], linestyle='--', linewidth=1.5,
                label=f'10V0 = {v0_red:.2f}')
    ax1.set_ylabel('月速率 (mm/M)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        f'{target} 监测点 {period_name} 训练期演示 '
        f'({start.strftime("%Y-%m-%d")} ~ {end.strftime("%Y-%m-%d")}, in-sample)',
        fontsize=13, fontweight='bold')

    ax2 = axes[1]
    for level in range(4):
        m_lv = lv == level
        if m_lv.any():
            ax2.scatter(d[m_lv], [level] * int(m_lv.sum()),
                        c=WARNING_COLORS[level], s=70, alpha=0.9,
                        label=LEVEL_NAMES[level], edgecolors='black', linewidths=0.6)
    ax2.set_ylabel('预警等级', fontsize=12, fontweight='bold')
    ax2.set_xlabel('日期', fontsize=12, fontweight='bold')
    ax2.set_yticks(list(range(4)))
    ax2.set_yticklabels(['安全', '警示', '警戒', '警报'])
    ax2.set_ylim(-0.6, 3.6)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
    plt.tight_layout()

    tag = start.strftime('%Y%m')
    out_pdf = FIGURES_DIR / f'{target.lower()}_train_demo_{tag}.pdf'
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()

    cnts = [(lv == i).sum() for i in range(4)]
    print(f'  [{target}/{period_name}] 预警分布: 绿={cnts[0]} 黄={cnts[1]} 橙={cnts[2]} 红={cnts[3]}  '
          f'→ {out_pdf.name}')


def main(target='MJ1'):
    print('=' * 60)
    print(f'模块 9: 训练期 in-sample 阶跃段预警演示 [{target}]')
    print('=' * 60)

    v0_conf, dates, in_monthly, actual_monthly = load_in_sample(target)
    v0 = v0_conf['v0_mm_per_month']
    v0_orange = v0_conf['v0_orange_threshold']
    v0_red = v0_conf['v0_red_threshold']

    probs, levels = compute_probs_and_levels(in_monthly, v0, v0_orange, v0_red)

    # 保存 in-sample 预警时序
    out_dir = TABLES_DIR / 'intermediate_data' / target
    demo_df = pd.DataFrame({
        'date': dates,
        'p_green': probs[:, 0],
        'p_yellow': probs[:, 1],
        'p_orange': probs[:, 2],
        'p_red': probs[:, 3],
        'warning_level': levels,
        'rate_p50': np.percentile(in_monthly, 50, axis=1),
        'rate_p95': np.percentile(in_monthly, 95, axis=1),
        'actual_monthly_rate': actual_monthly,
    })
    demo_file = out_dir / 'train_in_sample_demo.csv'
    demo_df.to_csv(demo_file, index=False)
    print(f'\n已保存: {demo_file}')
    print(f'总样本 {len(demo_df)} 天, 预警等级分布:')
    for i in range(4):
        cnt = int((levels == i).sum())
        pct = cnt / len(levels) * 100 if len(levels) else 0
        print(f'  {LEVEL_NAMES[i]}: {cnt} 天 ({pct:.1f}%)')

    print('\n绘制阶跃段演示图...')
    for name, start, end in DEMO_PERIODS:
        plot_demo_period(target, name, start, end, dates, in_monthly, actual_monthly,
                         probs, levels, v0, v0_orange, v0_red)

    print(f'\n模块 9 完成! [{target}]')


if __name__ == '__main__':
    import sys
    tgt = sys.argv[1] if len(sys.argv) > 1 else 'MJ1'
    main(tgt)
