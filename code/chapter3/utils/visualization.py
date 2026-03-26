"""
可视化模块
包含：趋势分解图、训练曲线、预测结果、预测区间、LaTeX表格生成
Author: wcqqq21
Date: 2026-03-25
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys

# 添加父目录到路径以导入config
sys.path.append(str(Path(__file__).parent.parent))
from config import *

logger = logging.getLogger(__name__)

# 配置matplotlib中文支持 - 必须在导入pyplot之前设置
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = FONT_CONFIG['size']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 现在导入pyplot
import matplotlib.pyplot as plt

# 验证字体设置
logger.info(f"当前字体设置: {matplotlib.rcParams['font.sans-serif']}")

# 设置样式
try:
    plt.style.use(FIGURE_CONFIG['style'])
    # 样式加载后重新设置中文字体（样式会覆盖字体设置）
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
    logger.info(f"样式加载后重新设置字体: {matplotlib.rcParams['font.sans-serif']}")
except:
    logger.warning(f"样式 {FIGURE_CONFIG['style']} 不可用，使用默认样式")


def plot_detrend_decomposition(t, y_original, y_trend, y_periodic, params, save_path=None):
    """
    绘制趋势分解结果（多项式或MVIF）

    Args:
        t: 时间数组 (天)
        y_original: 原始位移
        y_trend: 趋势项
        y_periodic: 周期项
        params: 参数字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 子图1: 原始数据 + 趋势项
    axes[0].plot(t, y_original, 'o', markersize=2, alpha=0.5, label='Original Observation')
    axes[0].plot(t, y_trend, 'r-', linewidth=2, label='Trend Component')
    axes[0].set_ylabel('Displacement (mm)')
    axes[0].set_title('Trend Extraction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 添加参数文本
    if 'equation' in params:
        param_text = f"Polynomial: {params['equation']}\nDegree: {params['degree']}"
    elif 'A' in params:
        param_text = f"A={params['A']:.2f} mm, B={params['B']:.4f}\n" \
                     f"C={params['C']:.6f} 1/day, tf={params['tf']:.2f} day"
    else:
        param_text = str(params)

    axes[0].text(0.02, 0.98, param_text, transform=axes[0].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    # 子图2: 周期项
    axes[1].plot(t, y_periodic, 'b-', linewidth=1, label='Periodic Component')
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1].set_ylabel('Displacement (mm)')
    axes[1].set_title('Periodic Component (Residual)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 子图3: 重构对比
    y_reconstructed = y_trend + y_periodic
    axes[2].plot(t, y_original, 'o', markersize=2, alpha=0.5, label='Original')
    axes[2].plot(t, y_reconstructed, 'g-', linewidth=1, label='Reconstructed (Trend+Periodic)')
    axes[2].set_xlabel('Time (days)')
    axes[2].set_ylabel('Displacement (mm)')
    axes[2].set_title('Reconstruction Verification')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_CONFIG['dpi'], format=FIGURE_CONFIG['format'], bbox_inches='tight')
        logger.info(f"MVIF分解图已保存: {save_path}")

    plt.close()


def plot_training_curves(train_history, val_history, quantiles, save_path=None):
    """
    绘制训练曲线

    Args:
        train_history: {quantile: [losses]}
        val_history: {quantile: [losses]}
        quantiles: 分位数列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, len(quantiles), figsize=(15, 4))

    if len(quantiles) == 1:
        axes = [axes]

    for i, q in enumerate(quantiles):
        axes[i].plot(train_history[q], label='训练损失', linewidth=2)
        axes[i].plot(val_history[q], label='验证损失', linewidth=2)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Pinball Loss')
        axes[i].set_title(f'分位数 τ={q}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_CONFIG['dpi'], format=FIGURE_CONFIG['format'], bbox_inches='tight')
        logger.info(f"训练曲线已保存: {save_path}")

    plt.close()


def plot_predictions(y_true, y_pred, dataset_name='测试集', save_path=None):
    """
    绘制预测结果对比

    Args:
        y_true: 真实值
        y_pred: 预测值
        dataset_name: 数据集名称
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1: 时间序列对比
    axes[0].plot(y_true, 'o-', markersize=3, label='真实值', alpha=0.7)
    axes[0].plot(y_pred, 's-', markersize=3, label='预测值', alpha=0.7)
    axes[0].set_xlabel('样本索引')
    axes[0].set_ylabel('位移 (mm)')
    axes[0].set_title(f'{dataset_name} - 预测对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2: 散点图
    axes[1].scatter(y_true, y_pred, alpha=0.5, s=20)

    # 添加y=x参考线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')

    axes[1].set_xlabel('真实值 (mm)')
    axes[1].set_ylabel('预测值 (mm)')
    axes[1].set_title(f'{dataset_name} - 散点图')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_CONFIG['dpi'], format=FIGURE_CONFIG['format'], bbox_inches='tight')
        logger.info(f"预测对比图已保存: {save_path}")

    plt.close()


def plot_prediction_intervals(y_true, y_pred, y_lower, y_upper, dataset_name='测试集', save_path=None):
    """
    绘制预测区间

    Args:
        y_true: 真实值
        y_pred: 预测值 (中位数)
        y_lower: 预测区间下界
        y_upper: 预测区间上界
        dataset_name: 数据集名称
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(y_true))

    # 绘制预测区间
    ax.fill_between(x, y_lower, y_upper, alpha=0.3, color='blue', label='90% 预测区间')

    # 绘制预测值和真实值
    ax.plot(x, y_pred, 'b-', linewidth=2, label='预测值 (中位数)')
    ax.plot(x, y_true, 'ro', markersize=4, alpha=0.6, label='真实值')

    # 标记超出区间的点
    out_of_bounds = np.logical_or(y_true < y_lower, y_true > y_upper)
    if np.any(out_of_bounds):
        ax.plot(x[out_of_bounds], y_true[out_of_bounds], 'rx', markersize=8,
               markeredgewidth=2, label='超出区间')

    ax.set_xlabel('样本索引')
    ax.set_ylabel('位移 (mm)')
    ax.set_title(f'{dataset_name} - 预测区间 (5%-95%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_CONFIG['dpi'], format=FIGURE_CONFIG['format'], bbox_inches='tight')
        logger.info(f"预测区间图已保存: {save_path}")

    plt.close()


def generate_latex_table(metrics_dict, caption="模型性能对比", label="tab:model_comparison"):
    """
    生成LaTeX表格代码

    Args:
        metrics_dict: {model_name: {metric_name: value}}
        caption: 表格标题
        label: 表格标签

    Returns:
        latex_code: LaTeX代码字符串
    """
    import pandas as pd

    # 转换为DataFrame
    df = pd.DataFrame(metrics_dict).T

    # 格式化数值
    for col in df.columns:
        if col in ['PICP', 'PINAW', 'CWC', 'R2']:
            df[col] = df[col].apply(lambda x: f"{x:.4f}")
        elif col == 'MAPE':
            df[col] = df[col].apply(lambda x: f"{x:.2f}\\%")
        else:
            df[col] = df[col].apply(lambda x: f"{x:.4f}")

    # 生成LaTeX代码
    latex_code = "\\begin{table}[htbp]\n"
    latex_code += "\\centering\n"
    latex_code += f"\\caption{{{caption}}}\n"
    latex_code += f"\\label{{{label}}}\n"
    latex_code += "\\begin{tabular}{l" + "c" * len(df.columns) + "}\n"
    latex_code += "\\toprule\n"

    # 表头
    latex_code += "模型 & " + " & ".join(df.columns) + " \\\\\n"
    latex_code += "\\midrule\n"

    # 数据行
    for model_name, row in df.iterrows():
        latex_code += f"{model_name} & " + " & ".join(row.values) + " \\\\\n"

    latex_code += "\\bottomrule\n"
    latex_code += "\\end{tabular}\n"
    latex_code += "\\end{table}\n"

    return latex_code


def save_latex_table(metrics_dict, save_path, caption="模型性能对比", label="tab:model_comparison"):
    """
    保存LaTeX表格到文件

    Args:
        metrics_dict: {model_name: {metric_name: value}}
        save_path: 保存路径
        caption: 表格标题
        label: 表格标签
    """
    latex_code = generate_latex_table(metrics_dict, caption, label)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)

    logger.info(f"LaTeX表格已保存: {save_path}")


def plot_all_results(results, save_dir=FIGURE_DIR):
    """
    绘制所有结果图

    Args:
        results: 结果字典，包含所有必要数据
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("开始生成所有可视化图表...")

    # 1. 趋势分解图（支持多项式或MVIF）
    if 'detrend' in results:
        plot_detrend_decomposition(
            t=results['detrend']['t'],
            y_original=results['detrend']['y_original'],
            y_trend=results['detrend']['y_trend'],
            y_periodic=results['detrend']['y_periodic'],
            params=results['detrend']['params'],
            save_path=save_dir / 'detrend_decomposition.pdf'
        )
    elif 'mvif' in results:  # 向后兼容
        plot_detrend_decomposition(
            t=results['mvif']['t'],
            y_original=results['mvif']['y_original'],
            y_trend=results['mvif']['y_trend'],
            y_periodic=results['mvif']['y_periodic'],
            params=results['mvif']['params'],
            save_path=save_dir / 'mvif_decomposition.pdf'
        )

    # 2. 训练曲线
    if 'training' in results:
        plot_training_curves(
            train_history=results['training']['train_history'],
            val_history=results['training']['val_history'],
            quantiles=results['training']['quantiles'],
            save_path=save_dir / 'training_curves.pdf'
        )

    # 3. 测试集预测对比
    if 'test' in results:
        plot_predictions(
            y_true=results['test']['y_true'],
            y_pred=results['test']['y_pred'],
            dataset_name='Test Set',
            save_path=save_dir / 'test_predictions.pdf'
        )

    # 4. 预测区间
    if 'test' in results and 'y_lower' in results['test']:
        plot_prediction_intervals(
            y_true=results['test']['y_true'],
            y_pred=results['test']['y_pred'],
            y_lower=results['test']['y_lower'],
            y_upper=results['test']['y_upper'],
            dataset_name='测试集',
            save_path=save_dir / 'prediction_intervals.pdf'
        )

    logger.info(f"所有图表已保存到: {save_dir}")


if __name__ == "__main__":
    # 测试代码
    logger.info("=" * 50)
    logger.info("测试可视化模块")
    logger.info("=" * 50)

    # 生成模拟数据
    np.random.seed(42)
    n = 200

    # MVIF分解测试
    t = np.linspace(0, 1000, n)
    y_trend = 1000 / (1 + np.exp(-0.001 * (5000 - t)))
    y_periodic = 10 * np.sin(2 * np.pi * t / 100)
    y_original = y_trend + y_periodic + np.random.randn(n) * 2

    params = {'A': 1000, 'B': 1.0, 'C': 0.001, 'tf': 5000}

    plot_mvif_decomposition(t, y_original, y_trend, y_periodic, params,
                           save_path=FIGURE_DIR / 'test_mvif.pdf')

    # 预测结果测试
    y_true = np.random.randn(100) * 10 + 50
    y_pred = y_true + np.random.randn(100) * 2
    y_lower = y_pred - 5
    y_upper = y_pred + 5

    plot_predictions(y_true, y_pred, save_path=FIGURE_DIR / 'test_predictions.pdf')
    plot_prediction_intervals(y_true, y_pred, y_lower, y_upper,
                             save_path=FIGURE_DIR / 'test_intervals.pdf')

    # LaTeX表格测试
    metrics_dict = {
        'MVIF-QLSTM': {'MAE': 1.23, 'RMSE': 1.56, 'R2': 0.95, 'PICP': 0.92, 'PINAW': 0.15},
        'LSTM': {'MAE': 1.45, 'RMSE': 1.78, 'R2': 0.93, 'PICP': 0.88, 'PINAW': 0.18}
    }

    latex_code = generate_latex_table(metrics_dict)
    logger.info(f"LaTeX表格:\n{latex_code}")

