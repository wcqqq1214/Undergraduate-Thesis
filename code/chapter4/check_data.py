"""
数据检查脚本：在运行主程序前检查数据是否准备好
"""

import pandas as pd
import numpy as np
from pathlib import Path


def check_file_exists(file_path, description):
    """检查文件是否存在"""
    if file_path.exists():
        print(f"✓ {description}: {file_path}")
        return True
    else:
        print(f"✗ {description}: {file_path} (文件不存在)")
        return False


def check_lstm_predictions(file_path):
    """检查LSTM预测结果文件"""
    print("\n检查LSTM预测结果...")
    if not file_path.exists():
        print(f"✗ 文件不存在: {file_path}")
        return False

    try:
        df = pd.read_csv(file_path)
        print(f"✓ 文件加载成功")
        print(f"  数据形状: {df.shape}")
        print(f"  列名: {df.columns.tolist()}")

        # 检查必需的列
        required_cols = ['run_id', 'time_index', 'prediction', 'actual']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"✗ 缺少必需的列: {missing_cols}")
            return False

        # 检查数据范围
        n_runs = df['run_id'].nunique()
        n_timesteps = df['time_index'].nunique()
        print(f"  运行次数: {n_runs}")
        print(f"  时间步数: {n_timesteps}")
        print(f"  预测值范围: [{df['prediction'].min():.2f}, {df['prediction'].max():.2f}]")

        if n_runs != 50:
            print(f"⚠ 警告: 运行次数不是50次，而是{n_runs}次")

        return True
    except Exception as e:
        print(f"✗ 读取文件失败: {str(e)}")
        return False


def check_monitoring_data(file_path):
    """检查监测数据文件"""
    print("\n检查监测数据...")
    if not file_path.exists():
        print(f"✗ 文件不存在: {file_path}")
        return False

    try:
        df = pd.read_excel(file_path)
        print(f"✓ 文件加载成功")
        print(f"  数据形状: {df.shape}")
        print(f"  列名: {df.columns.tolist()}")

        # 检查必需的列
        if 'Date' not in df.columns:
            print(f"✗ 缺少Date列")
            return False
        if 'MJ1/mm' not in df.columns:
            print(f"✗ 缺少MJ1/mm列")
            return False

        # 检查数据范围
        dates = pd.to_datetime(df['Date'])
        displacement = df['MJ1/mm'].values
        print(f"  时间范围: {dates.min()} 至 {dates.max()}")
        print(f"  数据点数: {len(displacement)}")
        print(f"  位移范围: [{displacement.min():.2f}, {displacement.max():.2f}] mm")

        # 计算日增量统计
        daily_increment = np.diff(displacement)
        print(f"  日增量均值: {daily_increment.mean():.3f} mm/天")
        print(f"  日增量范围: [{daily_increment.min():.3f}, {daily_increment.max():.3f}] mm/天")
        print(f"  超过0.3mm的天数: {(daily_increment > 0.3).sum()} / {len(daily_increment)}")

        return True
    except Exception as e:
        print(f"✗ 读取文件失败: {str(e)}")
        return False


def main():
    """主函数"""
    print("="*70)
    print(" "*25 + "数据检查")
    print("="*70)

    base_dir = Path(__file__).parent.parent.parent
    chapter3_output = base_dir / "code" / "chapter3" / "outputs" / "tables"
    data_dir = base_dir / "data"

    # 检查文件
    files_ok = True

    # 1. LSTM预测结果
    lstm_file = chapter3_output / "lstm_trend_50runs_predictions.csv"
    if not check_lstm_predictions(lstm_file):
        files_ok = False

    # 2. 监测数据
    monitoring_file = data_dir / "monitoring data.xlsx"
    if not check_monitoring_data(monitoring_file):
        files_ok = False

    # 3. 其他可选文件
    print("\n检查其他文件...")
    stats_file = chapter3_output / "lstm_trend_50runs_statistics.csv"
    check_file_exists(stats_file, "LSTM统计数据")

    # 总结
    print("\n" + "="*70)
    if files_ok:
        print("✓ 所有必需的数据文件都已准备好")
        print("\n可以运行主程序:")
        print("  python run_all.py")
    else:
        print("✗ 部分数据文件缺失或格式不正确")
        print("\n请先:")
        print("  1. 运行第三章代码生成LSTM预测结果")
        print("  2. 确保监测数据文件存在且格式正确")
    print("="*70)


if __name__ == "__main__":
    main()
