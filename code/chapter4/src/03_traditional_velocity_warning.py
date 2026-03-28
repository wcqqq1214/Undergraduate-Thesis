"""
模块3：传统速率预警方法
实现基于日位移增量的传统速率预警方法作为对比基准
"""

import numpy as np
import pandas as pd
from pathlib import Path


def traditional_velocity_warning(displacement_series):
    """
    传统速率预警方法（基于日位移增量）

    速率阈值（mm/天）:
        绿色(0): V < 0.1
        蓝色(1): 0.1 <= V < 0.2
        黄色(2): 0.2 <= V < 0.3
        橙色(3): 0.3 <= V < 0.5
        红色(4): V >= 0.5

    参数:
        displacement_series: (T,) - 累计位移序列

    返回:
        warning_levels: (T-1,) - 预警等级
        warning_colors: list - 预警颜色
        daily_velocity: (T-1,) - 日位移增量
        statistics: dict - 统计信息
    """
    # 计算日位移增量（速率）
    daily_velocity = np.diff(displacement_series)  # (T-1,)

    warning_levels = np.zeros(len(daily_velocity), dtype=int)
    warning_colors = []

    # 定义速率阈值
    thresholds = [0.1, 0.2, 0.3, 0.5]
    color_names = ['green', 'blue', 'yellow', 'orange', 'red']
    color_chinese = ['绿色', '蓝色', '黄色', '橙色', '红色']

    for i, v in enumerate(daily_velocity):
        if v < thresholds[0]:
            level = 0
        elif v < thresholds[1]:
            level = 1
        elif v < thresholds[2]:
            level = 2
        elif v < thresholds[3]:
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
        'total_days': len(daily_velocity),
        'level_distribution': level_counts,
        'high_risk_days': int((warning_levels >= 3).sum()),
        'warning_days': int((warning_levels >= 2).sum()),
        'mean_velocity': float(np.mean(daily_velocity)),
        'max_velocity': float(np.max(daily_velocity)),
        'min_velocity': float(np.min(daily_velocity)),
        'std_velocity': float(np.std(daily_velocity))
    }

    print(f"传统速率预警统计:")
    print(f"  总天数: {statistics['total_days']}")
    print(f"  平均速率: {statistics['mean_velocity']:.3f} mm/天")
    print(f"  速率范围: [{statistics['min_velocity']:.3f}, {statistics['max_velocity']:.3f}] mm/天")
    for color, count in level_counts.items():
        percentage = count / statistics['total_days'] * 100
        print(f"  {color}: {count} 天 ({percentage:.1f}%)")
    print(f"  预警天数(黄色及以上): {statistics['warning_days']} 天")

    return warning_levels, warning_colors, daily_velocity, statistics


def load_actual_displacement(data_file, monitoring_point='MJ1/mm'):
    """
    加载实际监测位移数据

    参数:
        data_file: 监测数据文件路径
        monitoring_point: 监测点列名

    返回:
        displacement: (T,) - 累计位移序列
        dates: (T,) - 日期序列
    """
    print(f"加载实际监测数据: {data_file}")
    df = pd.read_excel(data_file)

    print(f"数据列: {df.columns.tolist()}")
    print(f"数据形状: {df.shape}")

    # 提取指定监测点的位移数据
    displacement = df[monitoring_point].values
    dates = pd.to_datetime(df['Date'])

    print(f"监测点: {monitoring_point}")
    print(f"数据时间范围: {dates.min()} 至 {dates.max()}")
    print(f"位移范围: [{displacement.min():.2f}, {displacement.max():.2f}] mm")

    return displacement, dates


def main():
    """主函数"""
    # 设置路径
    base_dir = Path(__file__).parent.parent.parent.parent
    data_dir = base_dir / "data"
    chapter4_output = base_dir / "code" / "chapter4" / "outputs" / "tables"

    monitoring_data_file = data_dir / "monitoring data.xlsx"

    print("="*60)
    print("模块3: 传统速率预警方法")
    print("="*60)

    # 1. 加载实际监测数据
    displacement, dates = load_actual_displacement(
        monitoring_data_file,
        monitoring_point='MJ1/mm'
    )

    # 2. 计算传统速率预警
    print("\n计算传统速率预警...")
    warning_levels, warning_colors, daily_velocity, statistics = \
        traditional_velocity_warning(displacement)

    # 3. 保存结果
    intermediate_dir = chapter4_output / "intermediate_data"
    statistics_dir = chapter4_output / "statistics"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    statistics_dir.mkdir(parents=True, exist_ok=True)

    output_file = intermediate_dir / "traditional_warning_levels.csv"
    result_df = pd.DataFrame({
        'date': dates[1:],  # 差分后少一个时间步
        'daily_velocity': daily_velocity,
        'warning_level': warning_levels,
        'warning_color': warning_colors
    })
    result_df.to_csv(output_file, index=False)
    print(f"\n传统预警结果已保存至: {output_file}")

    # 4. 保存统计信息
    stats_file = statistics_dir / "traditional_warning_statistics.json"
    import json
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    print(f"统计信息已保存至: {stats_file}")

    # 5. 保存实际位移数据（用于后续分析）
    displacement_file = intermediate_dir / "actual_displacement_MJ1.csv"
    pd.DataFrame({
        'date': dates,
        'displacement': displacement
    }).to_csv(displacement_file, index=False)
    print(f"实际位移数据已保存至: {displacement_file}")

    print("\n" + "="*60)
    print("模块3 完成！")
    print("="*60)


if __name__ == "__main__":
    main()
