#!/usr/bin/env python3
"""
第三章结果可视化脚本
从 result.xlsx 读取 LSTM-50runs 和 GRU-50runs 数据，生成对比图表
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置
RESULT_FILE = Path(__file__).parent.parent.parent / "data" / "result.xlsx"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data(sheet_name):
    """加载指定sheet的数据"""
    df = pd.read_excel(RESULT_FILE, sheet_name=sheet_name, header=None)
    # 第一行是NaN，第二行是列名，第三行开始是数据
    df.columns = df.iloc[1]
    df = df.iloc[2:].reset_index(drop=True)

    # 处理重复的列名（添加后缀使其唯一）
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_mask = cols == dup
        dup_count = dup_mask.sum()
        cols[dup_mask] = [f'{dup}_{i}' if i != 0 else dup
                          for i in range(dup_count)]
    df.columns = cols

    # 转换日期列
    df['Date'] = pd.to_datetime(df['Date'])

    return df


def calculate_statistics(df, prefix='Predict'):
    """计算50次运行的统计量"""
    predict_cols = [col for col in df.columns
                    if isinstance(col, str) and col.startswith(prefix) and '/mm' in col]

    # 转换为数值类型
    for col in predict_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    predictions = df[predict_cols].values

    stats = {
        'mean': np.nanmean(predictions, axis=1),
        'std': np.nanstd(predictions, axis=1),
        'q05': np.nanpercentile(predictions, 5, axis=1),
        'q25': np.nanpercentile(predictions, 25, axis=1),
        'q50': np.nanpercentile(predictions, 50, axis=1),
        'q75': np.nanpercentile(predictions, 75, axis=1),
        'q95': np.nanpercentile(predictions, 95, axis=1),
    }

    return stats


def plot_lstm_gru_comparison():
    """绘制LSTM和GRU的对比图"""
    print("加载LSTM和GRU数据...")
    lstm_df = load_data('LSTM-50runs')
    gru_df = load_data('GRU-50runs')

    print("计算统计量...")
    lstm_stats = calculate_statistics(lstm_df)
    gru_stats = calculate_statistics(gru_df)

    dates = lstm_df['Date']

    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # LSTM预测结果
    ax1 = axes[0]
    ax1.plot(dates, lstm_stats['mean'], 'b-', label='LSTM均值', linewidth=1.5)
    ax1.fill_between(dates, lstm_stats['q05'], lstm_stats['q95'],
                      alpha=0.2, color='blue', label='90%置信区间')
    ax1.fill_between(dates, lstm_stats['q25'], lstm_stats['q75'],
                      alpha=0.3, color='blue', label='50%置信区间')
    ax1.set_ylabel('累积位移 (mm)', fontsize=12)
    ax1.set_title('LSTM模型预测结果 (50次运行)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # GRU预测结果
    ax2 = axes[1]
    ax2.plot(dates, gru_stats['mean'], 'r-', label='GRU均值', linewidth=1.5)
    ax2.fill_between(dates, gru_stats['q05'], gru_stats['q95'],
                      alpha=0.2, color='red', label='90%置信区间')
    ax2.fill_between(dates, gru_stats['q25'], gru_stats['q75'],
                      alpha=0.3, color='red', label='50%置信区间')
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('累积位移 (mm)', fontsize=12)
    ax2.set_title('GRU模型预测结果 (50次运行)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    output_path = OUTPUT_DIR / "lstm_gru_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"保存图表: {output_path}")
    plt.close()


def plot_uncertainty_comparison():
    """绘制LSTM和GRU的不确定性对比"""
    print("生成不确定性对比图...")
    lstm_df = load_data('LSTM-50runs')
    gru_df = load_data('GRU-50runs')

    lstm_stats = calculate_statistics(lstm_df)
    gru_stats = calculate_statistics(gru_df)

    dates = lstm_df['Date']

    fig, ax = plt.subplots(figsize=(14, 6))

    # 绘制标准差对比
    ax.plot(dates, lstm_stats['std'], 'b-', label='LSTM标准差', linewidth=1.5)
    ax.plot(dates, gru_stats['std'], 'r-', label='GRU标准差', linewidth=1.5)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('预测标准差 (mm)', fontsize=12)
    ax.set_title('LSTM与GRU模型预测不确定性对比', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    output_path = OUTPUT_DIR / "uncertainty_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"保存图表: {output_path}")
    plt.close()


def plot_combined_comparison():
    """绘制LSTM和GRU在同一图中的对比"""
    print("生成综合对比图...")
    lstm_df = load_data('LSTM-50runs')
    gru_df = load_data('GRU-50runs')

    lstm_stats = calculate_statistics(lstm_df)
    gru_stats = calculate_statistics(gru_df)

    dates = lstm_df['Date']

    fig, ax = plt.subplots(figsize=(14, 7))

    # LSTM
    ax.plot(dates, lstm_stats['mean'], 'b-', label='LSTM均值', linewidth=2)
    ax.fill_between(dates, lstm_stats['q05'], lstm_stats['q95'],
                     alpha=0.15, color='blue', label='LSTM 90%置信区间')

    # GRU
    ax.plot(dates, gru_stats['mean'], 'r-', label='GRU均值', linewidth=2)
    ax.fill_between(dates, gru_stats['q05'], gru_stats['q95'],
                     alpha=0.15, color='red', label='GRU 90%置信区间')

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('累积位移 (mm)', fontsize=12)
    ax.set_title('LSTM与GRU模型预测结果对比', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    output_path = OUTPUT_DIR / "lstm_gru_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"保存图表: {output_path}")
    plt.close()


def generate_statistics_table():
    """生成统计表格"""
    print("生成统计表格...")
    lstm_df = load_data('LSTM-50runs')
    gru_df = load_data('GRU-50runs')

    lstm_stats = calculate_statistics(lstm_df)
    gru_stats = calculate_statistics(gru_df)

    stats_summary = {
        '模型': ['LSTM', 'GRU'],
        '平均预测值 (mm)': [
            f"{np.mean(lstm_stats['mean']):.2f}",
            f"{np.mean(gru_stats['mean']):.2f}"
        ],
        '平均标准差 (mm)': [
            f"{np.mean(lstm_stats['std']):.2f}",
            f"{np.mean(gru_stats['std']):.2f}"
        ],
        '最大标准差 (mm)': [
            f"{np.max(lstm_stats['std']):.2f}",
            f"{np.max(gru_stats['std']):.2f}"
        ],
        '最小标准差 (mm)': [
            f"{np.min(lstm_stats['std']):.2f}",
            f"{np.min(gru_stats['std']):.2f}"
        ]
    }

    df_stats = pd.DataFrame(stats_summary)
    output_path = OUTPUT_DIR / "statistics_summary.csv"
    df_stats.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"保存统计表格: {output_path}")
    print("\n统计摘要:")
    print(df_stats.to_string(index=False))


def main():
    """主函数"""
    print("=" * 60)
    print("第三章结果可视化")
    print("=" * 60)

    try:
        plot_lstm_gru_comparison()
        plot_uncertainty_comparison()
        plot_combined_comparison()
        generate_statistics_table()

        print("\n" + "=" * 60)
        print("所有图表已生成完成！")
        print(f"输出目录: {OUTPUT_DIR}")
        print("=" * 60)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
