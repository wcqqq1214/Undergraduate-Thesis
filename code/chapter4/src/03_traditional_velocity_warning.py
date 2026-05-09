"""
模块 3: 传统速率预警方法

与 V0 概率预警进行公平对比：
  - 使用同一 V0/5V0/10V0 阈值体系（4 级：绿/黄/橙/红）
  - V 使用实测累计位移的 30 天滚动月速率（不使用 LSTM 预测）

这样传统方法与概率方法的差异仅在于"速率来源"：
  - 传统：实测（后验，无前瞻性）
  - 概率：LSTM 集成预测（前瞻 + 不确定性）
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


MONTH_WINDOW_DAYS = 30
LEVEL_COLORS_EN = ['green', 'yellow', 'orange', 'red']
LEVEL_COLORS_ZH = ['绿色', '黄色', '橙色', '红色']


def classify(v_series, v0, v0_orange, v0_red):
    levels = np.zeros(len(v_series), dtype=int)
    levels[(v_series >= v0) & (v_series < v0_orange)] = 1
    levels[(v_series >= v0_orange) & (v_series < v0_red)] = 2
    levels[v_series >= v0_red] = 3
    return levels


def traditional_velocity_warning(displacement_series, v0, v0_orange, v0_red,
                                 window=MONTH_WINDOW_DAYS):
    """基于实测累计位移做 30 天滚动月速率，用 V0 阈值体系分级"""
    daily_inc = np.diff(displacement_series)  # (T-1,)
    # 30 天滚动求和 → 月速率
    monthly = pd.Series(daily_inc).rolling(window=window, min_periods=window).sum().to_numpy()
    # 前 window-1 个值是 NaN，保留结构方便后续按时间轴对齐
    valid_mask = ~np.isnan(monthly)
    levels = np.full(len(monthly), -1, dtype=int)
    levels[valid_mask] = classify(monthly[valid_mask], v0, v0_orange, v0_red)
    colors = [LEVEL_COLORS_EN[lv] if lv >= 0 else 'n/a' for lv in levels]

    level_counts = {zh: int(((levels == i) & valid_mask).sum())
                    for i, zh in enumerate(LEVEL_COLORS_ZH)}
    valid_days = int(valid_mask.sum())

    stats = {
        'total_days': len(daily_inc),
        'valid_days': valid_days,
        'level_distribution': level_counts,
        'warning_days_yellow_or_above': int(((levels >= 1) & valid_mask).sum()),
        'high_risk_days_orange_or_above': int(((levels >= 2) & valid_mask).sum()),
        'mean_monthly_rate': float(np.nanmean(monthly)),
        'max_monthly_rate': float(np.nanmax(monthly)),
        'min_monthly_rate': float(np.nanmin(monthly)),
    }

    print('传统速率预警统计 (V0 体系, 月口径):')
    print(f'  有效天数: {valid_days} (前 {window - 1} 天因滚动窗口无值)')
    print(f'  月速率范围: [{stats["min_monthly_rate"]:.3f}, {stats["max_monthly_rate"]:.3f}] mm/M')
    for zh in LEVEL_COLORS_ZH:
        pct = level_counts[zh] / valid_days * 100 if valid_days else 0
        print(f'  {zh}: {level_counts[zh]} 天 ({pct:.1f}%)')
    print(f'  预警天数 (黄色及以上): {stats["warning_days_yellow_or_above"]}')
    print(f'  高风险 (橙色及以上): {stats["high_risk_days_orange_or_above"]}')

    return levels, colors, monthly, stats


def load_actual_displacement(data_file, monitoring_point):
    df = pd.read_excel(data_file)
    displacement = df[monitoring_point].to_numpy()
    dates = pd.to_datetime(df['Date'])
    print(f'数据时间范围: {dates.min()} 至 {dates.max()}')
    print(f'位移范围: [{displacement.min():.2f}, {displacement.max():.2f}] mm')
    return displacement, dates


def main(target='MJ1'):
    base_dir = Path(__file__).parent.parent.parent.parent
    data_file = base_dir / 'data' / 'monitoring data.xlsx'
    chapter4_out = base_dir / 'code' / 'chapter4' / 'outputs' / 'tables'
    intermediate_dir = chapter4_out / 'intermediate_data' / target
    stats_dir = chapter4_out / 'statistics' / target
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(f'模块 3: 传统速率预警 (V0 体系) [{target}]')
    print('=' * 60)

    v0_conf = json.loads((intermediate_dir / 'v0.json').read_text())
    v0 = v0_conf['v0_mm_per_month']
    v0_orange = v0_conf['v0_orange_threshold']
    v0_red = v0_conf['v0_red_threshold']
    print(f'V0={v0:.3f}  5V0={v0_orange:.3f}  10V0={v0_red:.3f} mm/M')

    displacement, dates = load_actual_displacement(data_file, f'{target}/mm')

    levels, colors, monthly_rate, stats = traditional_velocity_warning(
        displacement, v0, v0_orange, v0_red)

    out_df = pd.DataFrame({
        'date': dates[1:],             # 差分后少一天
        'monthly_rate': monthly_rate,  # 30 天滚动月速率 (mm/M)，前 29 天为 NaN
        'warning_level': levels,
        'warning_color': colors,
    })
    out_file = intermediate_dir / 'traditional_warning_levels.csv'
    out_df.to_csv(out_file, index=False)
    print(f'\n传统预警结果已保存: {out_file}')

    (stats_dir / 'traditional_warning_statistics.json').write_text(
        json.dumps(stats, indent=2, ensure_ascii=False))

    # 保存实测位移（与原脚本兼容）
    pd.DataFrame({'date': dates, 'displacement': displacement}).to_csv(
        intermediate_dir / f'actual_displacement_{target}.csv', index=False)

    print('\n' + '=' * 60)
    print(f'模块 3 完成！[{target}]')
    print('=' * 60)


if __name__ == '__main__':
    import sys
    tgt = sys.argv[1] if len(sys.argv) > 1 else 'MJ1'
    main(tgt)
