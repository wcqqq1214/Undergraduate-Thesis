"""
多项式去趋势模块（替代MVIF）
用于从位移时间序列中提取长期趋势项
Author: wcqqq21
Date: 2026-03-25
"""

import numpy as np
from scipy.signal import savgol_filter
import logging
from pathlib import Path
import sys

# 添加父目录到路径以导入config
sys.path.append(str(Path(__file__).parent.parent))
from config import *

logger = logging.getLogger(__name__)


class PolynomialDetrendModel:
    """多项式去趋势模型"""

    def __init__(self, point_name=TARGET_POINT, degree=2):
        """
        初始化多项式去趋势模型

        Args:
            point_name: 监测点名称（用于获取特定配置）
            degree: 多项式阶数（默认2，即二次多项式）
        """
        self.point_name = point_name
        self.degree = degree
        self.coeffs = None  # 多项式系数
        self.trend = None   # 趋势项
        self.periodic = None  # 周期项（残差）

        # 获取监测点特定配置
        if point_name in POINT_SPECIFIC_CONFIG:
            self.config = POINT_SPECIFIC_CONFIG[point_name]
        else:
            # 使用默认配置
            self.config = {
                'sg_window': PERIODIC_SMOOTH_CONFIG['window_length'],
                'sg_polyorder': PERIODIC_SMOOTH_CONFIG['polyorder']
            }

        logger.info(f"初始化多项式去趋势模型: 监测点={point_name}, 多项式阶数={degree}")

    def fit(self, t, y):
        """
        拟合多项式趋势

        Args:
            t: 时间数组（天），从0开始
            y: 位移观测值（mm）

        Returns:
            多项式系数
        """
        try:
            logger.info(f"开始多项式拟合（阶数={self.degree}）...")

            # 多项式拟合
            self.coeffs = np.polyfit(t, y, self.degree)

            # 计算拟合质量
            y_pred = np.polyval(self.coeffs, t)
            residuals = y - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean(residuals**2))

            logger.info(f"多项式拟合完成:")
            logger.info(f"  系数: {self.coeffs}")
            logger.info(f"  R² = {r2:.4f}")
            logger.info(f"  RMSE = {rmse:.4f} mm")

            return self.coeffs

        except Exception as e:
            logger.error(f"多项式拟合失败: {e}")
            raise

    def extract_trend(self, t):
        """
        提取趋势项

        Args:
            t: 时间数组（天）

        Returns:
            趋势项（mm）
        """
        if self.coeffs is None:
            raise ValueError("模型尚未拟合，请先调用fit()")

        self.trend = np.polyval(self.coeffs, t)
        logger.info(f"趋势项提取完成: 范围=[{self.trend.min():.2f}, {self.trend.max():.2f}] mm")

        return self.trend

    def extract_periodic(self, y, smooth=True):
        """
        提取周期项（残差）

        Args:
            y: 原始位移观测值（mm）
            smooth: 是否对周期项进行平滑

        Returns:
            周期项（mm）
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
            t: 时间数组（天）
            y: 位移观测值（mm）
            smooth: 是否平滑周期项

        Returns:
            trend, periodic, coeffs
        """
        self.fit(t, y)
        trend = self.extract_trend(t)
        periodic = self.extract_periodic(y, smooth=smooth)

        return trend, periodic, self.coeffs

    def predict(self, t):
        """
        预测未来趋势

        Args:
            t: 时间数组（天），可以超出训练范围

        Returns:
            预测的趋势项（mm）
        """
        if self.coeffs is None:
            raise ValueError("模型尚未拟合，请先调用fit()")

        return np.polyval(self.coeffs, t)

    def get_params_dict(self):
        """获取参数字典（用于保存）"""
        if self.coeffs is None:
            return None

        return {
            'coefficients': self.coeffs.tolist(),
            'degree': self.degree,
            'point_name': self.point_name,
            'equation': self._get_equation_string()
        }

    def _get_equation_string(self):
        """生成多项式方程字符串"""
        if self.coeffs is None:
            return None

        terms = []
        for i, coeff in enumerate(self.coeffs):
            power = len(self.coeffs) - 1 - i
            if power == 0:
                terms.append(f"{coeff:.4f}")
            elif power == 1:
                terms.append(f"{coeff:.4f}*t")
            else:
                terms.append(f"{coeff:.4f}*t^{power}")

        return " + ".join(terms)


def batch_polynomial_decomposition(data_dict, degree=2, smooth=True):
    """
    批量多项式分解（用于多个监测点）

    Args:
        data_dict: {point_name: (t, y)} 字典
        degree: 多项式阶数
        smooth: 是否平滑周期项

    Returns:
        results: {point_name: {'trend': ..., 'periodic': ..., 'coeffs': ...}}
    """
    results = {}

    for point_name, (t, y) in data_dict.items():
        logger.info(f"处理监测点: {point_name}")

        model = PolynomialDetrendModel(point_name=point_name, degree=degree)
        trend, periodic, coeffs = model.decompose(t, y, smooth=smooth)

        results[point_name] = {
            'trend': trend,
            'periodic': periodic,
            'coeffs': coeffs,
            'model': model
        }

    logger.info(f"批量多项式分解完成: {len(results)} 个监测点")

    return results


if __name__ == "__main__":
    # 测试代码
    logger.info("=" * 50)
    logger.info("测试多项式去趋势模块")
    logger.info("=" * 50)

    # 生成模拟数据（二次趋势 + 周期波动）
    t = np.linspace(0, 1000, 1000)
    y_trend = 0.0001 * t**2 + 0.1 * t + 100  # 二次趋势
    y_periodic = 10 * np.sin(2 * np.pi * t / 365)  # 年周期
    y = y_trend + y_periodic + np.random.normal(0, 2, len(t))

    # 拟合模型
    model = PolynomialDetrendModel(point_name='TEST', degree=2)
    trend, periodic, coeffs = model.decompose(t, y, smooth=True)

    logger.info(f"拟合系数: {coeffs}")
    logger.info(f"方程: {model._get_equation_string()}")
    logger.info(f"趋势项: {trend[:5]}")
    logger.info(f"周期项: {periodic[:5]}")
