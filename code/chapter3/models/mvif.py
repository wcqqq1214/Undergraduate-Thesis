"""
MVIF (Modified Verhulst Inverse Function) 趋势项提取模块
用于从位移时间序列中提取长期趋势项
Author: wcqqq21
Date: 2026-03-25
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import logging
from pathlib import Path
import sys

# 添加父目录到路径以导入config
sys.path.append(str(Path(__file__).parent.parent))
from config import *

logger = logging.getLogger(__name__)


def mvif_function(t, A, B, C, tf):
    """
    MVIF模型函数

    Args:
        t: 时间数组 (天)
        A: 位移饱和值 (mm)
        B: 曲线位置参数
        C: 蠕变速率参数 (1/day)
        tf: 破坏时间 (day)

    Returns:
        位移预测值 (mm)
    """
    return A / (1 + B * np.exp(-C * (tf - t)))


class MVIFModel:
    """MVIF趋势项提取模型"""

    def __init__(self, point_name=TARGET_POINT):
        """
        初始化MVIF模型

        Args:
            point_name: 监测点名称（用于获取特定配置）
        """
        self.point_name = point_name
        self.params = None  # 拟合参数 (A, B, C, tf)
        self.trend = None   # 趋势项
        self.periodic = None  # 周期项

        # 获取监测点特定配置
        if point_name in POINT_SPECIFIC_CONFIG:
            self.config = POINT_SPECIFIC_CONFIG[point_name]
        else:
            # 使用默认配置
            self.config = {
                'sg_window': PERIODIC_SMOOTH_CONFIG['window_length'],
                'sg_polyorder': PERIODIC_SMOOTH_CONFIG['polyorder'],
                'mvif_init_scale': 1.0
            }

        logger.info(f"初始化MVIF模型: 监测点={point_name}")

    def fit(self, t, y):
        """
        拟合MVIF模型

        Args:
            t: 时间数组 (天)，从0开始
            y: 位移观测值 (mm)

        Returns:
            拟合参数 (A, B, C, tf)
        """
        try:
            logger.info("开始MVIF拟合...")

            # 初始参数（根据监测点调整）
            scale = self.config['mvif_init_scale']
            p0 = [
                MVIF_INIT_PARAMS['A'] * scale,
                MVIF_INIT_PARAMS['B'],
                MVIF_INIT_PARAMS['C'],
                MVIF_INIT_PARAMS['tf'] * scale
            ]

            # 参数边界（确保物理意义）
            bounds = (
                [0, 0, 0, t.max()],  # 下界
                [np.inf, np.inf, 1, np.inf]  # 上界 - 放宽C的上限
            )

            # 非线性最小二乘拟合
            self.params, pcov = curve_fit(
                mvif_function,
                t, y,
                p0=p0,
                bounds=bounds,
                method=MVIF_FIT_CONFIG['method'],
                max_nfev=MVIF_FIT_CONFIG['max_nfev'],
                ftol=MVIF_FIT_CONFIG['ftol'],
                xtol=MVIF_FIT_CONFIG['xtol']
            )

            A, B, C, tf = self.params

            # 计算拟合质量
            y_pred = mvif_function(t, *self.params)
            residuals = y - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean(residuals**2))

            logger.info(f"MVIF拟合完成:")
            logger.info(f"  A (饱和值) = {A:.2f} mm")
            logger.info(f"  B (位置参数) = {B:.4f}")
            logger.info(f"  C (蠕变速率) = {C:.6f} 1/day")
            logger.info(f"  tf (破坏时间) = {tf:.2f} day")
            logger.info(f"  R² = {r2:.4f}")
            logger.info(f"  RMSE = {rmse:.4f} mm")

            return self.params

        except Exception as e:
            logger.error(f"MVIF拟合失败: {e}")
            raise

    def extract_trend(self, t):
        """
        提取趋势项

        Args:
            t: 时间数组 (天)

        Returns:
            趋势项 (mm)
        """
        if self.params is None:
            raise ValueError("模型尚未拟合，请先调用fit()")

        self.trend = mvif_function(t, *self.params)
        logger.info(f"趋势项提取完成: 范围=[{self.trend.min():.2f}, {self.trend.max():.2f}] mm")

        return self.trend

    def extract_periodic(self, y, smooth=True):
        """
        提取周期项（残差）

        Args:
            y: 原始位移观测值 (mm)
            smooth: 是否对周期项进行平滑

        Returns:
            周期项 (mm)
        """
        if self.trend is None:
            raise ValueError("趋势项尚未提取，请先调用extract_trend()")

        # 计算残差
        self.periodic = y - self.trend

        # 可选：Savitzky-Golay平滑
        if smooth and PERIODIC_SMOOTH_CONFIG['enable']:
            window = self.config['sg_window']
            polyorder = self.config['sg_polyorder']

            # 确保窗口长度为奇数且不超过数据长度
            if window % 2 == 0:
                window += 1
            window = min(window, len(self.periodic))

            if window > polyorder:
                self.periodic = savgol_filter(
                    self.periodic,
                    window_length=window,
                    polyorder=polyorder
                )
                logger.info(f"周期项已平滑: window={window}, polyorder={polyorder}")

        logger.info(f"周期项提取完成: 范围=[{self.periodic.min():.2f}, {self.periodic.max():.2f}] mm")

        return self.periodic

    def decompose(self, t, y, smooth=True):
        """
        完整的分解流程：拟合 -> 提取趋势 -> 提取周期

        Args:
            t: 时间数组 (天)
            y: 位移观测值 (mm)
            smooth: 是否平滑周期项

        Returns:
            trend, periodic, params
        """
        self.fit(t, y)
        trend = self.extract_trend(t)
        periodic = self.extract_periodic(y, smooth=smooth)

        return trend, periodic, self.params

    def predict(self, t):
        """
        预测未来趋势

        Args:
            t: 时间数组 (天)，可以超出训练范围

        Returns:
            预测的趋势项 (mm)
        """
        if self.params is None:
            raise ValueError("模型尚未拟合，请先调用fit()")

        return mvif_function(t, *self.params)

    def get_params_dict(self):
        """获取参数字典（用于保存）"""
        if self.params is None:
            return None

        return {
            'A': float(self.params[0]),
            'B': float(self.params[1]),
            'C': float(self.params[2]),
            'tf': float(self.params[3]),
            'point_name': self.point_name
        }


def batch_mvif_decomposition(data_dict, smooth=True):
    """
    批量MVIF分解（用于多个监测点）

    Args:
        data_dict: {point_name: (t, y)} 字典
        smooth: 是否平滑周期项

    Returns:
        results: {point_name: {'trend': ..., 'periodic': ..., 'params': ...}}
    """
    results = {}

    for point_name, (t, y) in data_dict.items():
        logger.info(f"处理监测点: {point_name}")

        model = MVIFModel(point_name=point_name)
        trend, periodic, params = model.decompose(t, y, smooth=smooth)

        results[point_name] = {
            'trend': trend,
            'periodic': periodic,
            'params': model.get_params_dict(),
            'model': model
        }

    logger.info(f"批量MVIF分解完成: {len(results)} 个监测点")

    return results


if __name__ == "__main__":
    # 测试代码
    logger.info("=" * 50)
    logger.info("测试MVIF模块")
    logger.info("=" * 50)

    # 生成模拟数据
    t = np.linspace(0, 1000, 1000)
    A_true, B_true, C_true, tf_true = 1000, 1.0, 0.001, 5000
    y_true = mvif_function(t, A_true, B_true, C_true, tf_true)
    y_noisy = y_true + np.random.normal(0, 10, len(t))

    # 拟合模型
    model = MVIFModel(point_name='TEST')
    trend, periodic, params = model.decompose(t, y_noisy, smooth=True)

    logger.info(f"真实参数: A={A_true}, B={B_true}, C={C_true}, tf={tf_true}")
    logger.info(f"拟合参数: {params}")
    logger.info(f"趋势项: {trend[:5]}")
    logger.info(f"周期项: {periodic[:5]}")

