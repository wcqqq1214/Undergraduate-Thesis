"""
模块2: 确定预警等级

新 V0 体系下，每天已经从 01 模块得到 4 等级概率 (p_green/p_yellow/p_orange/p_red)。
本模块取每天概率最高的等级作为最终预警等级（与 docx 5.2.3 综合预警方法一致）。

等级编码（去除蓝色，0-3 共 4 级）：
  0 - 绿色（安全）      V < V0
  1 - 黄色（警示）      V0 <= V < 5 V0
  2 - 橙色（警戒）      5 V0 <= V < 10 V0
  3 - 红色（警报）      V >= 10 V0
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


LEVEL_COLORS_EN = ['green', 'yellow', 'orange', 'red']
LEVEL_COLORS_ZH = ['绿色', '黄色', '橙色', '红色']


def determine_warning_level(level_probs_df):
    """取每天概率最高的等级作为预警等级。

    level_probs_df: columns 包含 p_green / p_yellow / p_orange / p_red
    返回 (warning_levels: ndarray[int 0..3], warning_colors: list[str], statistics: dict)
    """
    probs = level_probs_df[['p_green', 'p_yellow', 'p_orange', 'p_red']].to_numpy()
    levels = probs.argmax(axis=1)
    colors = [LEVEL_COLORS_EN[lv] for lv in levels]

    level_counts = {zh: int((levels == i).sum()) for i, zh in enumerate(LEVEL_COLORS_ZH)}
    total = int(len(levels))

    statistics = {
        'total_days': total,
        'level_distribution': level_counts,
        'warning_days_yellow_or_above': int((levels >= 1).sum()),
        'high_risk_days_orange_or_above': int((levels >= 2).sum()),
        'red_days': int((levels == 3).sum()),
    }

    print('预警等级统计 (取每日最高概率等级):')
    for i, zh in enumerate(LEVEL_COLORS_ZH):
        pct = level_counts[zh] / total * 100 if total else 0
        print(f'  {zh}: {level_counts[zh]} 天 ({pct:.1f}%)')
    print(f'  黄色及以上: {statistics["warning_days_yellow_or_above"]} 天')
    print(f'  橙色及以上: {statistics["high_risk_days_orange_or_above"]} 天')

    return levels, colors, statistics


def main(target='MJ1'):
    base_dir = Path(__file__).parent.parent.parent.parent
    chapter4_out = base_dir / 'code' / 'chapter4' / 'outputs' / 'tables'
    intermediate_dir = chapter4_out / 'intermediate_data' / target

    exceed_file = intermediate_dir / 'exceed_probability.csv'

    print('=' * 60)
    print(f'模块 2: 确定预警等级 (V0 体系, 4 级) [{target}]')
    print('=' * 60)

    df = pd.read_csv(exceed_file)
    levels, colors, stats = determine_warning_level(df)

    out_df = pd.DataFrame({
        'time_index': df['time_index'],
        'exceed_probability': df['exceed_probability'],
        'p_green': df['p_green'],
        'p_yellow': df['p_yellow'],
        'p_orange': df['p_orange'],
        'p_red': df['p_red'],
        'warning_level': levels,
        'warning_color': colors,
    })
    out_file = intermediate_dir / 'warning_levels.csv'
    out_df.to_csv(out_file, index=False)
    print(f'\n预警等级已保存: {out_file}')

    stats_dir = chapter4_out / 'statistics' / target
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / 'warning_levels_statistics.json').write_text(
        json.dumps(stats, indent=2, ensure_ascii=False))

    print('\n' + '=' * 60)
    print(f'模块 2 完成！[{target}]')
    print('=' * 60)


if __name__ == '__main__':
    import sys
    tgt = sys.argv[1] if len(sys.argv) > 1 else 'MJ1'
    main(tgt)
