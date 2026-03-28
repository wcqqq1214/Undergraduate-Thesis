"""
模块2：确定预警等级
根据越限概率确定五级预警等级（绿/蓝/黄/橙/红）
"""

import numpy as np
import pandas as pd
from pathlib import Path


def determine_warning_level(exceed_prob):
    """
    根据越限概率确定预警等级

    预警等级划分（基于日位移增量 > 0.3mm 的概率）:
        绿色(0): P < 5%
        蓝色(1): 5% <= P < 20%
        黄色(2): 20% <= P < 50%
        橙色(3): 50% <= P < 80%
        红色(4): P >= 80%

    参数:
        exceed_prob: (T,) - 越限概率数组

    返回:
        warning_levels: (T,) - 预警等级 (0-4)
        warning_colors: list - 预警颜色名称
        statistics: dict - 统计信息
    """
    warning_levels = np.zeros(len(exceed_prob), dtype=int)
    warning_colors = []

    # 定义预警等级阈值
    thresholds = [0.05, 0.20, 0.50, 0.80]
    color_names = ['green', 'blue', 'yellow', 'orange', 'red']
    color_chinese = ['绿色', '蓝色', '黄色', '橙色', '红色']

    for i, p in enumerate(exceed_prob):
        if p < thresholds[0]:
            level = 0
        elif p < thresholds[1]:
            level = 1
        elif p < thresholds[2]:
            level = 2
        elif p < thresholds[3]:
            level = 3
        else:
            level = 4

        warning_levels[i] = level
        warning_colors.append(color_names[level])

    # 统计各等级的天数
    level_counts = {}
    for level in range(5):
        count = (warning_levels == level).sum()
        level_counts[color_chinese[level]] = int(count)

    statistics = {
        'total_days': len(exceed_prob),
        'level_distribution': level_counts,
        'high_risk_days': int((warning_levels >= 3).sum()),  # 橙色+红色
        'warning_days': int((warning_levels >= 2).sum())  # 黄色及以上
    }

    print(f"预警等级统计:")
    print(f"  总天数: {statistics['total_days']}")
    for color, count in level_counts.items():
        percentage = count / statistics['total_days'] * 100
        print(f"  {color}: {count} 天 ({percentage:.1f}%)")
    print(f"  预警天数(黄色及以上): {statistics['warning_days']} 天")
    print(f"  高风险天数(橙色及以上): {statistics['high_risk_days']} 天")

    return warning_levels, warning_colors, statistics


def main():
    """主函数"""
    # 设置路径
    base_dir = Path(__file__).parent.parent.parent
    chapter4_output = base_dir / "chapter4" / "outputs" / "tables"
    intermediate_dir = chapter4_output / "intermediate_data"

    exceed_prob_file = intermediate_dir / "exceed_probability.csv"

    print("="*60)
    print("模块2: 确定预警等级")
    print("="*60)

    # 1. 加载越限概率
    print(f"\n加载越限概率数据: {exceed_prob_file}")
    df = pd.read_csv(exceed_prob_file)
    exceed_prob = df['exceed_probability'].values

    print(f"数据形状: {exceed_prob.shape}")
    print(f"越限概率范围: [{exceed_prob.min():.2%}, {exceed_prob.max():.2%}]")

    # 2. 确定预警等级
    print("\n计算预警等级...")
    warning_levels, warning_colors, statistics = determine_warning_level(exceed_prob)

    # 3. 保存结果
    statistics_dir = chapter4_output / "statistics"
    statistics_dir.mkdir(parents=True, exist_ok=True)

    output_file = intermediate_dir / "warning_levels.csv"
    result_df = pd.DataFrame({
        'time_index': df['time_index'],
        'exceed_probability': exceed_prob,
        'warning_level': warning_levels,
        'warning_color': warning_colors
    })
    result_df.to_csv(output_file, index=False)
    print(f"\n预警等级已保存至: {output_file}")

    # 4. 保存统计信息
    stats_file = statistics_dir / "warning_levels_statistics.json"
    import json
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    print(f"统计信息已保存至: {stats_file}")

    print("\n" + "="*60)
    print("模块2 完成！")
    print("="*60)


if __name__ == "__main__":
    main()
