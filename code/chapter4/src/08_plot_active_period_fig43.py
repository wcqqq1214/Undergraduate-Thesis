"""
图4-3: MJ1监测点阶跃变形期预警细节（2020年3-4月）
展示预测位移的不同分位数、安全阈值、实际监测位移、预警等级
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置字体
plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 路径配置
BASE_DIR = Path(__file__).parent.parent
TABLES_DIR = BASE_DIR / 'outputs' / 'tables'
INTERMEDIATE_DIR = TABLES_DIR / 'intermediate_data'
FIGURES_DIR = BASE_DIR / 'outputs' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 预警等级颜色映射
WARNING_COLORS = {
    0: '#2ecc71',  # 绿色
    1: '#3498db',  # 蓝色
    2: '#f1c40f',  # 黄色
    3: '#e67e22',  # 橙色
    4: '#e74c3c'   # 红色
}

def load_data():
    """加载数据"""
    # 加载chapter3的完整预测结果
    pred_data = pd.read_csv('/home/wcqqq21/Undergraduate-Thesis/code/chapter3/outputs/tables/lstm_trend_50runs_predictions.csv')

    # 加载实际位移
    actual_disp = pd.read_csv(INTERMEDIATE_DIR / 'actual_displacement_MJ1.csv')
    actual_disp['date'] = pd.to_datetime(actual_disp['date'])

    # 加载预警等级
    warning_levels = pd.read_csv(INTERMEDIATE_DIR / 'warning_levels.csv')

    return pred_data, actual_disp, warning_levels

def calculate_quantile_predictions(pred_data):
    """
    从50次运行的预测结果计算分位数
    """
    # 重塑数据：从长格式转为宽格式
    n_runs = pred_data['run_id'].max()
    n_timesteps = pred_data['time_index'].max() + 1

    predictions = np.zeros((n_timesteps, n_runs))

    for run_id in range(1, n_runs + 1):
        run_data = pred_data[pred_data['run_id'] == run_id].sort_values('time_index')
        predictions[:, run_id - 1] = run_data['prediction'].values

    # 计算分位数
    q50 = np.percentile(predictions, 50, axis=1)
    q75 = np.percentile(predictions, 75, axis=1)
    q95 = np.percentile(predictions, 95, axis=1)

    # 获取实际值
    actual = pred_data[pred_data['run_id'] == 1].sort_values('time_index')['actual'].values

    return q50, q75, q95, actual

def plot_active_period():
    """绘制阶跃变形期预警细节"""
    # 加载数据
    pred_data, actual_disp, warning_levels = load_data()

    # 计算分位数预测
    q50, q75, q95, actual = calculate_quantile_predictions(pred_data)

    # 预测数据对应的日期范围（测试集）
    test_size = len(q50)
    test_dates = actual_disp['date'].iloc[-test_size:].reset_index(drop=True)

    # 筛选2020年3月18日-4月30日数据（阶跃变形期）
    start_date = pd.Timestamp('2020-03-18')
    end_date = pd.Timestamp('2020-04-30')

    mask = (test_dates >= start_date) & (test_dates <= end_date)
    period_dates = test_dates[mask]
    period_indices = np.where(mask)[0]

    period_q50 = q50[period_indices]
    period_q75 = q75[period_indices]
    period_q95 = q95[period_indices]
    period_actual = actual[period_indices]
    period_warning = warning_levels.iloc[period_indices].reset_index(drop=True)

    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 子图1: 位移预测与实际位移
    ax1 = axes[0]

    # 绘制分位数预测
    ax1.plot(period_dates, period_q50, 'b-', linewidth=2.5, label='预测位移 (50%分位数)', alpha=0.9)
    ax1.plot(period_dates, period_q75, 'g--', linewidth=2, label='预测位移 (75%分位数)', alpha=0.8)
    ax1.plot(period_dates, period_q95, 'r:', linewidth=2, label='预测位移 (95%分位数)', alpha=0.8)

    # 绘制实际位移
    ax1.plot(period_dates, period_actual, 'ko-',
             linewidth=2, markersize=5, label='实际监测位移', alpha=0.7)

    # 安全阈值（使用实际位移的均值 + 0.5倍标准差作为示例）
    threshold = period_actual.mean() + 0.5 * period_actual.std()
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2.5,
                label=f'安全阈值 ({threshold:.1f} mm)', alpha=0.9)

    ax1.set_ylabel('累积位移 (mm)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_title('MJ1监测点阶跃变形期预警细节 (2020年3月18日-4月30日)',
                  fontsize=14, fontweight='bold', pad=15)

    # 添加y轴范围
    y_min = min(period_actual.min(), period_q50.min()) - 5
    y_max = max(period_actual.max(), period_q95.max()) + 5
    ax1.set_ylim(y_min, y_max)

    # 子图2: 预警等级
    ax2 = axes[1]

    # 绘制预警等级背景色
    for i in range(len(period_warning)):
        level = period_warning.loc[i, 'warning_level']
        if i < len(period_dates) - 1:
            ax2.axvspan(period_dates.iloc[i],
                       period_dates.iloc[i+1],
                       facecolor=WARNING_COLORS[level], alpha=0.4)

    # 绘制预警等级散点
    level_names = ['绿色', '蓝色', '黄色', '橙色', '红色']
    for level in range(5):
        mask_level = period_warning['warning_level'] == level
        if mask_level.any():
            indices = period_warning[mask_level].index
            ax2.scatter(period_dates.iloc[indices],
                       period_warning.loc[mask_level, 'warning_level'],
                       c=WARNING_COLORS[level], s=100, alpha=0.95,
                       label=f'{level_names[level]}预警', marker='o',
                       edgecolors='black', linewidths=1)

    ax2.set_ylabel('预警等级', fontsize=13, fontweight='bold')
    ax2.set_xlabel('日期', fontsize=13, fontweight='bold')
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(['0-绿色', '1-蓝色', '2-黄色', '3-橙色', '4-红色'], fontsize=11)
    ax2.legend(loc='upper left', fontsize=10, ncol=5, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(-0.5, 4.5)

    # 旋转x轴标签
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # 保存图片
    output_png = FIGURES_DIR / 'mj1_active_period.png'
    output_pdf = FIGURES_DIR / 'mj1_active_period.pdf'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ 图4-3已保存: {output_png}")
    print(f"✓ 图4-3已保存: {output_pdf}")

    # 打印一些统计信息
    print(f"\n统计信息:")
    print(f"  时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    print(f"  实际位移范围: {period_actual.min():.2f} - {period_actual.max():.2f} mm")
    print(f"  预测位移范围 (50%): {period_q50.min():.2f} - {period_q50.max():.2f} mm")
    print(f"  安全阈值: {threshold:.2f} mm")
    print(f"  预警等级分布:")
    for level in range(5):
        count = (period_warning['warning_level'] == level).sum()
        print(f"    {level_names[level]}: {count} 天")

    plt.close()

if __name__ == '__main__':
    print("=" * 60)
    print("生成图4-3: MJ1监测点阶跃变形期预警细节")
    print("=" * 60)

    plot_active_period()

    print("\n✓ 图表生成完成!")
