"""
模块 5: 预警提前时间

正确的"提前时间"定义（事件粒度）：
  1. 识别"越限事件"：连续几天 实测月速率 > V0 合并为一次事件，事件锚点 = 首次越限日
  2. 识别"预警事件"：连续几天 warning_level >= 1 合并为一次事件，事件锚点 = 首触发日
  3. 对每个越限事件，在其锚点之前的窗口（本实现里允许到 0 天，即当日触发）找离其最近
     且未被使用的预警事件；提前时间 = 越限锚点 - 预警锚点 (≥ 0)
  4. 找不到预警事件 → 漏报

对 warning_level 各等级分别统计（仅计算 >= 该等级首触发日 vs 越限日）。

评估窗口 = 测试集 & 跳过前 30 天（与模块 4 对齐）。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


MONTH_WINDOW_DAYS = 30
TRAIN_RATIO = 0.80


def identify_events(binary_series):
    """连续 1 的段合并为事件，返回每个事件的首个 1 的索引"""
    events = []
    i = 0
    n = len(binary_series)
    while i < n:
        if binary_series[i]:
            events.append(i)
            while i < n and binary_series[i]:
                i += 1
        else:
            i += 1
    return events


def match_events(warning_starts, exceed_starts, max_lead=180):
    """
    每个越限事件匹配"之前且最近、未被占用"的预警事件。

    返回：
      matches: list of (warning_idx, exceed_idx, lead_time)
      missed_exceed: 未被匹配的越限事件首日列表
    """
    matches = []
    used = set()
    for ex in exceed_starts:
        best = None
        for wi, w in enumerate(warning_starts):
            if wi in used:
                continue
            lead = ex - w
            if 0 <= lead <= max_lead:
                if best is None or lead < best[2]:
                    best = (wi, w, lead)
        if best is not None:
            matches.append((best[1], ex, best[2]))
            used.add(best[0])
    missed = [ex for ex in exceed_starts if not any(m[1] == ex for m in matches)]
    return matches, missed


def summarise(matches, missed):
    if matches:
        leads = [m[2] for m in matches]
        return {
            'event_count': len(matches) + len(missed),
            'detected_count': len(matches),
            'missed_count': len(missed),
            'detection_rate': float(len(matches) / (len(matches) + len(missed)))
                if (matches or missed) else 0.0,
            'mean_lead_time': float(np.mean(leads)),
            'median_lead_time': float(np.median(leads)),
            'max_lead_time': int(np.max(leads)),
            'min_lead_time': int(np.min(leads)),
            'std_lead_time': float(np.std(leads, ddof=1)) if len(leads) > 1 else 0.0,
        }
    return {
        'event_count': len(missed),
        'detected_count': 0,
        'missed_count': len(missed),
        'detection_rate': 0.0,
        'mean_lead_time': 0.0,
        'median_lead_time': 0.0,
        'max_lead_time': 0,
        'min_lead_time': 0,
        'std_lead_time': 0.0,
    }


def per_level_stats(levels_array, exceed_starts):
    out = {}
    for lv, zh in [(1, '黄色预警'), (2, '橙色预警'), (3, '红色预警')]:
        binary = (levels_array >= lv).astype(int)
        warn_starts = identify_events(binary)
        matches, missed = match_events(warn_starts, exceed_starts)
        s = summarise(matches, missed)
        out[zh] = {
            'event_count': s['detected_count'],
            'mean_lead_time': s['mean_lead_time'],
            'max_lead_time': s['max_lead_time'],
        }
    return out


def main(target='MJ1'):
    base_dir = Path(__file__).parent.parent.parent.parent
    chapter4_out = base_dir / 'code' / 'chapter4' / 'outputs' / 'tables'
    intermediate_dir = chapter4_out / 'intermediate_data' / target
    paper_tables_dir = chapter4_out / 'paper_tables' / target
    stats_dir = chapter4_out / 'statistics' / target
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(f'模块 5: 预警提前时间 (V0 体系) [{target}]')
    print('=' * 60)

    v0 = json.loads((intermediate_dir / 'v0.json').read_text())['v0_mm_per_month']

    actual_df = pd.read_csv(intermediate_dir / f'actual_displacement_{target}.csv')
    actual_disp = actual_df['displacement'].to_numpy()
    daily_inc = np.concatenate([[0.0], np.diff(actual_disp)])
    monthly_rate = pd.Series(daily_inc).rolling(
        MONTH_WINDOW_DAYS, min_periods=MONTH_WINDOW_DAYS).sum().to_numpy()

    n_total = len(actual_disp)
    train_size = int(n_total * TRAIN_RATIO)
    eval_start = train_size + MONTH_WINDOW_DAYS

    exceed_binary = (monthly_rate[eval_start:] > v0).astype(int)
    exceed_starts = identify_events(exceed_binary)

    prob_levels_full = pd.read_csv(intermediate_dir / 'warning_levels.csv')['warning_level'].to_numpy()
    prob_levels = prob_levels_full[:len(exceed_binary)]

    trad_levels_full = pd.read_csv(intermediate_dir / 'traditional_warning_levels.csv')['warning_level'].to_numpy()
    trad_levels = trad_levels_full[eval_start - 1:eval_start - 1 + len(exceed_binary)]

    # 概率预警总体
    prob_binary = (prob_levels[:len(exceed_binary)] >= 1).astype(int)
    prob_warn_starts = identify_events(prob_binary)
    prob_matches, prob_missed = match_events(prob_warn_starts, exceed_starts)
    prob_overall = summarise(prob_matches, prob_missed)
    prob_level = per_level_stats(prob_levels[:len(exceed_binary)], exceed_starts)
    prob_overall['level_statistics'] = prob_level

    # 传统预警总体
    trad_binary = (trad_levels >= 1).astype(int)
    trad_warn_starts = identify_events(trad_binary)
    trad_matches, trad_missed = match_events(trad_warn_starts, exceed_starts)
    trad_overall = summarise(trad_matches, trad_missed)

    print(f'\n检测到越限事件: {len(exceed_starts)} 个')
    print(f'\n本文概率预警:')
    print(f'  detected={prob_overall["detected_count"]}  missed={prob_overall["missed_count"]}  '
          f'detection_rate={prob_overall["detection_rate"]*100:.1f}%  '
          f'mean_lead_time={prob_overall["mean_lead_time"]:.2f} 天 '
          f'(max={prob_overall["max_lead_time"]})')
    for zh, s in prob_level.items():
        if s['event_count']:
            print(f'    {zh}: {s["event_count"]} 次, 平均提前 {s["mean_lead_time"]:.2f} 天, '
                  f'最大 {s["max_lead_time"]}')

    print(f'\n传统速率预警:')
    print(f'  detected={trad_overall["detected_count"]}  missed={trad_overall["missed_count"]}  '
          f'detection_rate={trad_overall["detection_rate"]*100:.1f}%  '
          f'mean_lead_time={trad_overall["mean_lead_time"]:.2f} 天 '
          f'(max={trad_overall["max_lead_time"]})')

    (stats_dir / 'probability_lead_time_statistics.json').write_text(
        json.dumps(prob_overall, indent=2, ensure_ascii=False))
    (stats_dir / 'traditional_lead_time_statistics.json').write_text(
        json.dumps(trad_overall, indent=2, ensure_ascii=False))

    comparison = pd.DataFrame({
        '预警等级': ['黄色预警', '橙色预警', '红色预警', '总体(黄及以上)'],
        '事件数_概率': [
            prob_level['黄色预警']['event_count'],
            prob_level['橙色预警']['event_count'],
            prob_level['红色预警']['event_count'],
            prob_overall['detected_count'],
        ],
        '平均提前_概率(天)': [
            f'{prob_level["黄色预警"]["mean_lead_time"]:.2f}',
            f'{prob_level["橙色预警"]["mean_lead_time"]:.2f}',
            f'{prob_level["红色预警"]["mean_lead_time"]:.2f}',
            f'{prob_overall["mean_lead_time"]:.2f}',
        ],
        '最大提前_概率(天)': [
            prob_level['黄色预警']['max_lead_time'],
            prob_level['橙色预警']['max_lead_time'],
            prob_level['红色预警']['max_lead_time'],
            prob_overall['max_lead_time'],
        ],
        '平均提前_传统(天)': ['-', '-', '-', f'{trad_overall["mean_lead_time"]:.2f}'],
    })
    comparison.to_csv(paper_tables_dir / 'lead_time_comparison.csv', index=False)

    print('\n' + '=' * 60)
    print(f'模块 5 完成！[{target}]')
    print('=' * 60)


if __name__ == '__main__':
    import sys
    tgt = sys.argv[1] if len(sys.argv) > 1 else 'MJ1'
    main(tgt)
