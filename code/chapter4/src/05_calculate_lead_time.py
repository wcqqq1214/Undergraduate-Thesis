"""
模块5：计算预警提前时间
统计预警触发时刻与实际越限时刻的时间差
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


def calculate_lead_time(warning_series, actual_displacement, threshold=0.3, warning_threshold=2):
    """
    计算预警提前时间（天）

    逻辑:
        1. 找到所有实际日位移增量超过阈值的时刻
        2. 向前搜索最近的预警触发时刻（黄色及以上）
        3. 计算时间差（天数）

    参数:
        warning_series: (T-1,) - 预警等级序列
        actual_displacement: (T,) - 实际累计位移
        threshold: 0.3 mm/天 - 位移增量阈值
        warning_threshold: 2 - 预警等级阈值（2=黄色）

    返回:
        statistics: dict - 预警提前时间统计
    """
    # 计算实际日位移增量
    actual_daily_increment = np.diff(actual_displacement)

    # 确保长度一致
    min_len = min(len(warning_series), len(actual_daily_increment))
    warning_series = warning_series[:min_len]
    actual_daily_increment = actual_daily_increment[:min_len]

    lead_times = []
    warning_events = []

    # 找到所有越限事件（日增量 > 0.3mm）
    exceed_indices = np.where(actual_daily_increment > threshold)[0]

    print(f"检测到 {len(exceed_indices)} 个越限事件")

    for exceed_idx in exceed_indices:
        # 向前搜索预警触发时刻（黄色及以上，level >= 2）
        found = False
        for i in range(exceed_idx, -1, -1):
            if warning_series[i] >= warning_threshold:
                lead_time = exceed_idx - i
                lead_times.append(lead_time)
                warning_events.append({
                    'warning_index': int(i),
                    'exceed_index': int(exceed_idx),
                    'lead_time_days': int(lead_time),
                    'warning_level': int(warning_series[i]),
                    'actual_increment': float(actual_daily_increment[exceed_idx])
                })
                found = True
                break

        if not found:
            # 如果没有找到预警，记录为漏报
            warning_events.append({
                'warning_index': None,
                'exceed_index': int(exceed_idx),
                'lead_time_days': None,
                'warning_level': None,
                'actual_increment': float(actual_daily_increment[exceed_idx]),
                'missed': True
            })

    if len(lead_times) == 0:
        statistics = {
            'mean_lead_time': 0,
            'median_lead_time': 0,
            'max_lead_time': 0,
            'min_lead_time': 0,
            'std_lead_time': 0,
            'event_count': len(exceed_indices),
            'detected_count': 0,
            'missed_count': len(exceed_indices),
            'detection_rate': 0.0
        }
    else:
        statistics = {
            'mean_lead_time': float(np.mean(lead_times)),
            'median_lead_time': float(np.median(lead_times)),
            'max_lead_time': int(np.max(lead_times)),
            'min_lead_time': int(np.min(lead_times)),
            'std_lead_time': float(np.std(lead_times)),
            'event_count': len(exceed_indices),
            'detected_count': len(lead_times),
            'missed_count': len(exceed_indices) - len(lead_times),
            'detection_rate': float(len(lead_times) / len(exceed_indices))
        }

    return statistics, warning_events, lead_times


def calculate_lead_time_by_level(warning_series, actual_displacement, threshold=0.3):
    """
    按预警等级统计提前时间

    返回:
        level_statistics: dict - 各等级的提前时间统计
    """
    actual_daily_increment = np.diff(actual_displacement)
    min_len = min(len(warning_series), len(actual_daily_increment))
    warning_series = warning_series[:min_len]
    actual_daily_increment = actual_daily_increment[:min_len]

    exceed_indices = np.where(actual_daily_increment > threshold)[0]

    level_lead_times = {2: [], 3: [], 4: []}  # 黄色、橙色、红色
    level_names = {2: '黄色预警', 3: '橙色预警', 4: '红色预警'}

    for exceed_idx in exceed_indices:
        for i in range(exceed_idx, -1, -1):
            level = warning_series[i]
            if level >= 2:
                lead_time = exceed_idx - i
                level_lead_times[level].append(lead_time)
                break

    level_statistics = {}
    for level, lead_times in level_lead_times.items():
        if len(lead_times) > 0:
            level_statistics[level_names[level]] = {
                'event_count': len(lead_times),
                'mean_lead_time': float(np.mean(lead_times)),
                'max_lead_time': int(np.max(lead_times))
            }
        else:
            level_statistics[level_names[level]] = {
                'event_count': 0,
                'mean_lead_time': 0,
                'max_lead_time': 0
            }

    return level_statistics


def print_statistics(statistics, method_name):
    """打印统计信息"""
    print(f"\n{method_name} 预警提前时间统计:")
    print(f"  总越限事件数: {statistics['event_count']}")
    print(f"  成功检测数: {statistics['detected_count']}")
    print(f"  漏报数: {statistics['missed_count']}")
    print(f"  检测率: {statistics['detection_rate']:.1%}")
    if statistics['detected_count'] > 0:
        print(f"  平均提前时间: {statistics['mean_lead_time']:.1f} 天")
        print(f"  中位数提前时间: {statistics['median_lead_time']:.1f} 天")
        print(f"  最大提前时间: {statistics['max_lead_time']} 天")
        print(f"  最小提前时间: {statistics['min_lead_time']} 天")


def main():
    """主函数"""
    # 设置路径
    base_dir = Path(__file__).parent.parent.parent
    chapter4_output = base_dir / "chapter4" / "outputs" / "tables"
    intermediate_dir = chapter4_output / "intermediate_data"
    paper_tables_dir = chapter4_output / "paper_tables"
    statistics_dir = chapter4_output / "statistics"

    # 文件路径
    prob_warning_file = intermediate_dir / "warning_levels.csv"
    trad_warning_file = intermediate_dir / "traditional_warning_levels.csv"
    actual_disp_file = intermediate_dir / "actual_displacement_MJ1.csv"

    print("="*60)
    print("模块5: 计算预警提前时间")
    print("="*60)

    # 1. 加载数据
    print("\n加载数据...")
    prob_df = pd.read_csv(prob_warning_file)
    trad_df = pd.read_csv(trad_warning_file)
    actual_df = pd.read_csv(actual_disp_file)

    prob_warnings = prob_df['warning_level'].values
    trad_warnings = trad_df['warning_level'].values
    actual_displacement = actual_df['displacement'].values

    # 2. 计算概率预警的提前时间
    print("\n计算概率预警提前时间...")
    prob_stats, prob_events, prob_lead_times = calculate_lead_time(
        prob_warnings,
        actual_displacement,
        threshold=0.3
    )
    print_statistics(prob_stats, "概率预警方法")

    # 按等级统计
    prob_level_stats = calculate_lead_time_by_level(
        prob_warnings,
        actual_displacement,
        threshold=0.3
    )
    print("\n按预警等级统计:")
    for level_name, stats in prob_level_stats.items():
        if stats['event_count'] > 0:
            print(f"  {level_name}: {stats['event_count']}次, "
                  f"平均提前{stats['mean_lead_time']:.1f}天, "
                  f"最大提前{stats['max_lead_time']}天")

    # 3. 计算传统预警的提前时间
    print("\n计算传统预警提前时间...")
    trad_stats, trad_events, trad_lead_times = calculate_lead_time(
        trad_warnings,
        actual_displacement,
        threshold=0.3
    )
    print_statistics(trad_stats, "传统预警方法")

    # 4. 保存结果
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    statistics_dir.mkdir(parents=True, exist_ok=True)

    # 保存概率预警统计
    prob_stats_file = statistics_dir / "probability_lead_time_statistics.json"
    prob_stats['level_statistics'] = prob_level_stats
    with open(prob_stats_file, 'w') as f:
        json.dump(prob_stats, f, indent=2, ensure_ascii=False)
    print(f"\n概率预警提前时间统计已保存至: {prob_stats_file}")

    # 保存传统预警统计
    trad_stats_file = statistics_dir / "traditional_lead_time_statistics.json"
    with open(trad_stats_file, 'w') as f:
        json.dump(trad_stats, f, indent=2, ensure_ascii=False)
    print(f"传统预警提前时间统计已保存至: {trad_stats_file}")

    # 保存预警事件详情
    prob_events_file = statistics_dir / "probability_warning_events.json"
    with open(prob_events_file, 'w') as f:
        json.dump(prob_events, f, indent=2, ensure_ascii=False)
    print(f"概率预警事件详情已保存至: {prob_events_file}")

    # 创建对比表格（论文用）
    comparison_df = pd.DataFrame({
        '预警等级': ['黄色预警', '橙色预警', '红色预警', '总体'],
        '事件数量_概率': [
            prob_level_stats.get('黄色预警', {}).get('event_count', 0),
            prob_level_stats.get('橙色预警', {}).get('event_count', 0),
            prob_level_stats.get('红色预警', {}).get('event_count', 0),
            prob_stats['detected_count']
        ],
        '平均提前时间_概率': [
            prob_level_stats.get('黄色预警', {}).get('mean_lead_time', 0),
            prob_level_stats.get('橙色预警', {}).get('mean_lead_time', 0),
            prob_level_stats.get('红色预警', {}).get('mean_lead_time', 0),
            prob_stats['mean_lead_time']
        ],
        '最大提前时间_概率': [
            prob_level_stats.get('黄色预警', {}).get('max_lead_time', 0),
            prob_level_stats.get('橙色预警', {}).get('max_lead_time', 0),
            prob_level_stats.get('红色预警', {}).get('max_lead_time', 0),
            prob_stats['max_lead_time']
        ],
        '平均提前时间_传统': ['-', '-', '-', trad_stats['mean_lead_time']]
    })
    comparison_file = paper_tables_dir / "lead_time_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"提前时间对比表格已保存至: {comparison_file}")

    print("\n" + "="*60)
    print("模块5 完成！")
    print("="*60)


if __name__ == "__main__":
    main()
