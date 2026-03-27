#!/usr/bin/env python3
"""
第三章结果可视化脚本
从 result.xlsx 读取 LSTM-50runs 数据，生成图表
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
RESULT_FILE = Path(__file__).parent.parent.parent.parent / "data" / "result.xlsx"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


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


def plot_lstm_prediction():
    """绘制LSTM预测结果"""
    print("加载LSTM数据...")
    lstm_df = load_data('LSTM-50runs')

    print("计算统计量...")
    lstm_stats = calculate_statistics(lstm_df)

    dates = lstm_df['Date']

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 7))

    # LSTM预测结果
    ax.plot(dates, lstm_stats['mean'], 'b-', label='LSTM均值', linewidth=2)
    ax.fill_between(dates, lstm_stats['q05'], lstm_stats['q95'],
                      alpha=0.2, color='blue', label='90%置信区间')
    ax.fill_between(dates, lstm_stats['q25'], lstm_stats['q75'],
                      alpha=0.3, color='blue', label='50%置信区间')
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('累积位移 (mm)', fontsize=12)
    ax.set_title('LSTM模型预测结果 (50次运行)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    output_path = FIGURES_DIR / "lstm_prediction.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"保存图表: {output_path}")
    plt.close()


def plot_uncertainty():
    """绘制LSTM的不确定性"""
    print("生成不确定性图...")
    lstm_df = load_data('LSTM-50runs')

    lstm_stats = calculate_statistics(lstm_df)

    dates = lstm_df['Date']

    fig, ax = plt.subplots(figsize=(14, 6))

    # 绘制标准差
    ax.plot(dates, lstm_stats['std'], 'b-', label='LSTM标准差', linewidth=1.5)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('预测标准差 (mm)', fontsize=12)
    ax.set_title('LSTM模型预测不确定性', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    output_path = FIGURES_DIR / "lstm_uncertainty.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"保存图表: {output_path}")
    plt.close()


def generate_statistics_table():
    """生成统计表格"""
    print("生成统计表格...")
    lstm_df = load_data('LSTM-50runs')

    lstm_stats = calculate_statistics(lstm_df)

    stats_summary = {
        '模型': ['LSTM'],
        '平均预测值 (mm)': [f"{np.mean(lstm_stats['mean']):.2f}"],
        '平均标准差 (mm)': [f"{np.mean(lstm_stats['std']):.2f}"],
        '最大标准差 (mm)': [f"{np.max(lstm_stats['std']):.2f}"],
        '最小标准差 (mm)': [f"{np.min(lstm_stats['std']):.2f}"]
    }

    df_stats = pd.DataFrame(stats_summary)
    output_path = TABLES_DIR / "statistics_summary.csv"
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
        plot_lstm_prediction()
        plot_uncertainty()
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
