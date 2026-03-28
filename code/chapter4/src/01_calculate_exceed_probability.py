"""
模块1：计算越限概率
基于第三章LSTM的50次预测结果，计算日位移增量的越限概率
"""

import numpy as np
import pandas as pd
from pathlib import Path


def calculate_exceed_probability(predictions_50runs, threshold=0.3):
    """
    计算日位移增量的越限概率

    参数:
        predictions_50runs: (50, T) - 50次LSTM预测的累计位移
        threshold: 0.3 mm/天 - 日位移增量阈值

    返回:
        exceed_prob: (T-1,) - 每个时间步的越限概率
        daily_increments_50runs: (50, T-1) - 50次预测的日增量
        statistics: dict - 统计信息
    """
    print(f"输入数据形状: {predictions_50runs.shape}")
    print(f"预警阈值: {threshold} mm/天")

    # 关键步骤：对累计位移求差分，得到日增量
    daily_increments_50runs = np.diff(predictions_50runs, axis=1)  # (50, T-1)
    print(f"日增量数据形状: {daily_increments_50runs.shape}")

    # 统计50次预测中，有多少次的日增量超过阈值
    exceed_count = (daily_increments_50runs > threshold).sum(axis=0)  # (T-1,)

    # 计算越限概率
    exceed_prob = exceed_count / 50.0  # (T-1,)

    # 统计信息
    statistics = {
        'mean_exceed_prob': float(np.mean(exceed_prob)),
        'max_exceed_prob': float(np.max(exceed_prob)),
        'min_exceed_prob': float(np.min(exceed_prob)),
        'exceed_prob_std': float(np.std(exceed_prob)),
        'high_risk_days': int((exceed_prob >= 0.5).sum()),  # 橙色及以上
        'total_days': len(exceed_prob)
    }

    print(f"\n越限概率统计:")
    print(f"  平均越限概率: {statistics['mean_exceed_prob']:.2%}")
    print(f"  最大越限概率: {statistics['max_exceed_prob']:.2%}")
    print(f"  高风险天数(P>=50%): {statistics['high_risk_days']}/{statistics['total_days']}")

    return exceed_prob, daily_increments_50runs, statistics


def load_lstm_predictions(predictions_file):
    """
    加载LSTM预测结果并重塑为(50, T)矩阵

    参数:
        predictions_file: 预测结果CSV文件路径

    返回:
        predictions_matrix: (50, T) - 50次预测的累计位移矩阵
        time_indices: (T,) - 时间索引
    """
    print(f"加载预测数据: {predictions_file}")
    df = pd.read_csv(predictions_file)

    # 检查数据格式
    print(f"数据列: {df.columns.tolist()}")
    print(f"数据形状: {df.shape}")
    print(f"前5行:\n{df.head()}")

    # 假设数据格式为: run_id, time_index, prediction, actual
    # 需要重塑为 (50, T) 矩阵
    n_runs = df['run_id'].nunique()
    n_timesteps = df['time_index'].nunique()

    print(f"\n检测到 {n_runs} 次运行, {n_timesteps} 个时间步")

    # 重塑数据
    predictions_matrix = df.pivot(
        index='run_id',
        columns='time_index',
        values='prediction'
    ).values  # (50, T)

    time_indices = df['time_index'].unique()

    print(f"重塑后矩阵形状: {predictions_matrix.shape}")

    return predictions_matrix, time_indices


def main():
    """主函数"""
    # 设置路径
    base_dir = Path(__file__).parent.parent.parent.parent
    chapter3_output = base_dir / "code" / "chapter3" / "outputs" / "tables"
    chapter4_output = base_dir / "code" / "chapter4" / "outputs" / "tables"

    predictions_file = chapter3_output / "lstm_trend_50runs_predictions.csv"

    print("="*60)
    print("模块1: 计算越限概率")
    print("="*60)

    # 1. 加载LSTM预测结果
    predictions_matrix, time_indices = load_lstm_predictions(predictions_file)

    # 2. 计算越限概率
    exceed_prob, daily_increments, statistics = calculate_exceed_probability(
        predictions_matrix,
        threshold=0.3
    )

    # 3. 保存结果
    intermediate_dir = chapter4_output / "intermediate_data"
    statistics_dir = chapter4_output / "statistics"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    statistics_dir.mkdir(parents=True, exist_ok=True)

    output_file = intermediate_dir / "exceed_probability.csv"
    result_df = pd.DataFrame({
        'time_index': time_indices[1:],  # 差分后少一个时间步
        'exceed_probability': exceed_prob
    })
    result_df.to_csv(output_file, index=False)
    print(f"\n越限概率已保存至: {output_file}")

    # 4. 保存统计信息
    stats_file = statistics_dir / "exceed_probability_statistics.json"
    import json
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"统计信息已保存至: {stats_file}")

    # 5. 保存日增量数据（用于后续分析）
    increments_file = intermediate_dir / "daily_increments_50runs.npy"
    np.save(increments_file, daily_increments)
    print(f"日增量数据已保存至: {increments_file}")

    print("\n" + "="*60)
    print("模块1 完成！")
    print("="*60)


if __name__ == "__main__":
    main()
