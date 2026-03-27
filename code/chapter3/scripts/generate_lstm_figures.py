#!/usr/bin/env python3
"""
生成LSTM预测结果的可视化图表
基于真实的50次运行数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 10

def load_data():
    """加载数据"""
    # 读取50次运行的统计数据
    stats_df = pd.read_csv('code/chapter3/outputs/tables/lstm_50runs_statistics.csv')
    stats_df['Date'] = pd.to_datetime(stats_df['Date'])

    # 读取监测数据获取真实值
    monitoring_df = pd.read_excel('data/monitoring data.xlsx')
    monitoring_df['Date'] = pd.to_datetime(monitoring_df['Date'])

    # 合并数据
    merged_df = pd.merge(stats_df, monitoring_df[['Date', 'MJ1/mm']], on='Date', how='left')

    return merged_df

def plot_prediction_with_confidence(df, output_path):
    """
    绘制预测结果及置信区间
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # 绘制置信区间
    ax.fill_between(df['Date'], df['Q25_Prediction'], df['Q75_Prediction'],
                     alpha=0.3, color='blue', label='50% Confidence Interval (Q25-Q75)')
    ax.fill_between(df['Date'], df['Min_Prediction'], df['Max_Prediction'],
                     alpha=0.15, color='blue', label='90% Confidence Interval (Min-Max)')

    # 绘制平均预测值
    ax.plot(df['Date'], df['Mean_Prediction'], 'b-', linewidth=1.5, label='Mean Prediction (50 runs)')

    # 绘制真实值
    ax.scatter(df['Date'], df['MJ1/mm'], c='red', s=10, alpha=0.6, label='Observed Values', zorder=5)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Displacement (mm)', fontsize=12)
    ax.set_title('LSTM Model Probabilistic Prediction Results (MJ1 Monitoring Point)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 格式化x轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"预测结果图已保存到: {output_path}")
    plt.close()

def plot_uncertainty_analysis(df, output_path):
    """
    绘制不确定性时变特征
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 上图：预测值和真实值
    ax1.plot(df['Date'], df['Mean_Prediction'], 'b-', linewidth=1.5, label='Mean Prediction')
    ax1.scatter(df['Date'], df['MJ1/mm'], c='red', s=10, alpha=0.6, label='Observed Values')
    ax1.set_ylabel('Cumulative Displacement (mm)', fontsize=12)
    ax1.set_title('LSTM Prediction vs Observed Values', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 下图：标准差时变特征
    ax2.plot(df['Date'], df['Std_Prediction'], 'g-', linewidth=1.5, label='Standard Deviation (50 runs)')
    ax2.fill_between(df['Date'], 0, df['Std_Prediction'], alpha=0.3, color='green')
    ax2.axhline(y=df['Std_Prediction'].mean(), color='r', linestyle='--',
                label=f'Mean Std = {df["Std_Prediction"].mean():.2f} mm')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Standard Deviation (mm)', fontsize=12)
    ax2.set_title('Temporal Variation of Prediction Uncertainty', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # 格式化x轴日期
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"不确定性分析图已保存到: {output_path}")
    plt.close()

def plot_uncertainty_histogram(df, output_path):
    """
    绘制标准差分布直方图
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df['Std_Prediction'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(df['Std_Prediction'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean = {df["Std_Prediction"].mean():.2f} mm')
    ax.axvline(df['Std_Prediction'].median(), color='green', linestyle='--', linewidth=2,
               label=f'Median = {df["Std_Prediction"].median():.2f} mm')

    ax.set_xlabel('Standard Deviation (mm)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Prediction Standard Deviation (50 runs)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加统计信息文本框
    stats_text = f'Min: {df["Std_Prediction"].min():.2f} mm\n'
    stats_text += f'Max: {df["Std_Prediction"].max():.2f} mm\n'
    stats_text += f'CV: {(df["Std_Prediction"].mean() / df["Mean_Prediction"].mean() * 100):.2f}%'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"标准差分布图已保存到: {output_path}")
    plt.close()

def main():
    print("=" * 80)
    print("生成LSTM预测结果可视化图表")
    print("=" * 80)

    # 加载数据
    print("\n加载数据...")
    df = load_data()
    print(f"数据点数量: {len(df)}")
    print(f"日期范围: {df['Date'].min()} 到 {df['Date'].max()}")

    # 生成图表
    print("\n生成图表...")

    # 1. 预测结果及置信区间
    plot_prediction_with_confidence(
        df,
        'code/chapter3/outputs/figures/lstm_prediction.png'
    )

    # 2. 不确定性时变特征
    plot_uncertainty_analysis(
        df,
        'code/chapter3/outputs/figures/lstm_uncertainty.png'
    )

    # 3. 标准差分布直方图
    plot_uncertainty_histogram(
        df,
        'code/chapter3/outputs/figures/lstm_std_distribution.png'
    )

    print("\n" + "=" * 80)
    print("所有图表生成完成")
    print("=" * 80)

    # 打印统计摘要
    print("\n统计摘要:")
    print(f"  平均预测值: {df['Mean_Prediction'].mean():.2f} mm")
    print(f"  平均标准差: {df['Std_Prediction'].mean():.2f} mm")
    print(f"  标准差范围: {df['Std_Prediction'].min():.2f} - {df['Std_Prediction'].max():.2f} mm")
    print(f"  变异系数: {(df['Std_Prediction'].mean() / df['Mean_Prediction'].mean() * 100):.2f}%")

if __name__ == '__main__':
    main()
