"""
模型评估指标模块
包含：MAE, RMSE, R², MAPE, PICP, PINAW, CWC
Author: wcqqq21
Date: 2026-03-25
"""

import numpy as np
import logging
from pathlib import Path
import sys

# 添加父目录到路径以导入config
sys.path.append(str(Path(__file__).parent.parent))
from config import *

logger = logging.getLogger(__name__)


def mae(y_true, y_pred):
    """
    平均绝对误差 (Mean Absolute Error)

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        MAE值
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """
    均方根误差 (Root Mean Square Error)

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        RMSE值
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred):
    """
    决定系数 (R² Score)

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        R²值
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def mape(y_true, y_pred, epsilon=1e-10):
    """
    平均绝对百分比误差 (Mean Absolute Percentage Error)

    Args:
        y_true: 真实值
        y_pred: 预测值
        epsilon: 防止除零的小常数

    Returns:
        MAPE值 (%)
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def picp(y_true, y_lower, y_upper):
    """
    预测区间覆盖概率 (Prediction Interval Coverage Probability)

    PICP = (1/n) * Σ c_i
    其中 c_i = 1 if y_i ∈ [y_lower_i, y_upper_i], else 0

    Args:
        y_true: 真实值
        y_lower: 预测区间下界
        y_upper: 预测区间上界

    Returns:
        PICP值 (0-1之间)
    """
    coverage = np.logical_and(y_true >= y_lower, y_true <= y_upper)
    return np.mean(coverage)


def pinaw(y_lower, y_upper, y_range):
    """
    预测区间归一化平均宽度 (Prediction Interval Normalized Average Width)

    PINAW = (1/n) * Σ (y_upper_i - y_lower_i) / R
    其中 R = max(y) - min(y) 是目标变量的范围

    Args:
        y_lower: 预测区间下界
        y_upper: 预测区间上界
        y_range: 目标变量范围 (max - min)

    Returns:
        PINAW值
    """
    if y_range == 0:
        return 0.0

    widths = y_upper - y_lower
    return np.mean(widths) / y_range


def cwc(y_true, y_lower, y_upper, confidence_level=CONFIDENCE_LEVEL, eta=50):
    """
    覆盖宽度准则 (Coverage Width-based Criterion)

    CWC = PINAW * (1 + γ(PICP) * e^(-η(PICP - μ)))

    其中:
    - γ(PICP) = 1 if PICP < μ, else 0
    - μ 是期望的置信水平
    - η 是惩罚系数

    Args:
        y_true: 真实值
        y_lower: 预测区间下界
        y_upper: 预测区间上界
        confidence_level: 期望的置信水平 (默认0.95)
        eta: 惩罚系数 (默认50)

    Returns:
        CWC值
    """
    picp_val = picp(y_true, y_lower, y_upper)
    y_range = np.max(y_true) - np.min(y_true)
    pinaw_val = pinaw(y_lower, y_upper, y_range)

    # 计算惩罚项
    if picp_val < confidence_level:
        gamma = 1
        penalty = np.exp(-eta * (picp_val - confidence_level))
    else:
        gamma = 0
        penalty = 1

    cwc_val = pinaw_val * (1 + gamma * penalty)

    return cwc_val


def calculate_all_metrics(y_true, y_pred, y_lower=None, y_upper=None):
    """
    计算所有评估指标

    Args:
        y_true: 真实值
        y_pred: 预测值 (中位数)
        y_lower: 预测区间下界 (可选)
        y_upper: 预测区间上界 (可选)

    Returns:
        metrics: 指标字典
    """
    metrics = {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mape(y_true, y_pred)
    }

    # 如果提供了预测区间，计算区间指标
    if y_lower is not None and y_upper is not None:
        y_range = np.max(y_true) - np.min(y_true)

        metrics['PICP'] = picp(y_true, y_lower, y_upper)
        metrics['PINAW'] = pinaw(y_lower, y_upper, y_range)
        metrics['CWC'] = cwc(y_true, y_lower, y_upper)

    return metrics


def print_metrics(metrics, title="评估指标"):
    """
    打印评估指标

    Args:
        metrics: 指标字典
        title: 标题
    """
    logger.info("=" * 50)
    logger.info(title)
    logger.info("=" * 50)

    for name, value in metrics.items():
        if name in ['PICP', 'PINAW', 'CWC', 'R2']:
            logger.info(f"{name:10s}: {value:.4f}")
        elif name == 'MAPE':
            logger.info(f"{name:10s}: {value:.2f}%")
        else:
            logger.info(f"{name:10s}: {value:.4f}")

    logger.info("=" * 50)


def compare_models(results_dict, y_true):
    """
    比较多个模型的性能

    Args:
        results_dict: {model_name: {'y_pred': ..., 'y_lower': ..., 'y_upper': ...}}
        y_true: 真实值

    Returns:
        comparison: DataFrame格式的比较结果
    """
    import pandas as pd

    comparison = {}

    for model_name, results in results_dict.items():
        y_pred = results['y_pred']
        y_lower = results.get('y_lower', None)
        y_upper = results.get('y_upper', None)

        metrics = calculate_all_metrics(y_true, y_pred, y_lower, y_upper)
        comparison[model_name] = metrics

    df = pd.DataFrame(comparison).T

    logger.info("\n模型性能比较:")
    logger.info(df.to_string())

    return df


def calculate_quantile_metrics(y_true, predictions_dict):
    """
    计算分位数预测的所有指标

    Args:
        y_true: 真实值
        predictions_dict: {quantile: predictions}
                         例如 {0.05: [...], 0.5: [...], 0.95: [...]}

    Returns:
        metrics: 指标字典
    """
    # 提取中位数和预测区间
    y_pred = predictions_dict[0.5]  # 中位数
    y_lower = predictions_dict[0.05]  # 5%分位数
    y_upper = predictions_dict[0.95]  # 95%分位数

    # 计算所有指标
    metrics = calculate_all_metrics(y_true, y_pred, y_lower, y_upper)

    return metrics


if __name__ == "__main__":
    # 测试代码
    logger.info("=" * 50)
    logger.info("测试评估指标模块")
    logger.info("=" * 50)

    # 生成模拟数据
    np.random.seed(42)
    n = 100
    y_true = np.random.randn(n) * 10 + 50
    y_pred = y_true + np.random.randn(n) * 2  # 添加噪声
    y_lower = y_pred - 5
    y_upper = y_pred + 5

    # 计算指标
    metrics = calculate_all_metrics(y_true, y_pred, y_lower, y_upper)
    print_metrics(metrics, title="测试指标")

    # 测试分位数指标
    predictions_dict = {
        0.05: y_lower,
        0.5: y_pred,
        0.95: y_upper
    }
    quantile_metrics = calculate_quantile_metrics(y_true, predictions_dict)
    print_metrics(quantile_metrics, title="分位数预测指标")
