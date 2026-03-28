"""
藕塘滑坡LightGBM预测模型 - 严格时序交叉验证版本

使用时序交叉验证 + SMOTE重采样评估模型性能：
- 回归任务：预测位移增量
- 分类任务：预警状态判定（固定阈值0.3mm）
- 避免数据泄漏，确保学术严谨性
- SHAP可解释性分析

运行方式：
    python3 lgbm_shap_warning.py

输出：
    - 控制台：交叉验证详细结果
    - outputs/shap_summary_regression.png / .pdf
    - outputs/shap_summary_classification.png / .pdf
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

import numpy as np
import pandas as pd
from typing import List, Tuple

import matplotlib.pyplot as plt

from read_monitoring_data import load_monitoring_data
from model_training import cross_validate_models
from shap_analysis import analyze_shap_reg, analyze_shap_cls

plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False


def build_supervised_samples(
    df: pd.DataFrame,
    window: int,
    point_cols: List[str],
    env_cols: List[str],
    warning_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    基于多监测点位移与环境因子构造监督学习样本。
    回归目标：下一天位移增量 (delta = u_t - u_{t-1})
    分类目标：增量是否超过固定阈值（预警）
    """
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    print(f"[Info] 使用固定预警阈值: {warning_threshold} mm")

    samples, targets_reg, targets_cls = [], [], []
    point_to_id = {p: i for i, p in enumerate(point_cols)}

    for t in range(window, len(df_sorted)):
        for point in point_cols:
            u = df_sorted[point].values
            past_env = np.concatenate([df_sorted[col].values[t - window:t] for col in env_cols])
            point_one_hot = np.zeros(len(point_cols), dtype=float)
            point_one_hot[point_to_id[point]] = 1.0

            samples.append(np.concatenate([u[t - window:t], past_env, point_one_hot]))
            delta = u[t] - u[t - 1]
            targets_reg.append(delta)
            targets_cls.append(int(delta >= warning_threshold))

    feat_names: List[str] = []
    for k in range(window, 0, -1):
        feat_names.append(f"disp(t-{k})")
    for col in env_cols:
        for k in range(window, 0, -1):
            feat_names.append(f"{col}(t-{k})")
    for p in point_cols:
        feat_names.append(f"point_is_{p}")

    return np.vstack(samples), np.array(targets_reg), np.array(targets_cls), feat_names


def main() -> None:
    df = load_monitoring_data()

    window = 5
    point_cols = ["MJ9/mm", "MJ1/mm", "MJ3/mm"]
    env_cols = ["Rainfall/mm", "GWT/m", "RWL/m", "aveT/℃", "minT/℃", "maxT/℃", "DP", "RH"]

    X, y_reg, y_cls, feat_names = build_supervised_samples(df, window, point_cols, env_cols)
    print(f"[Info] 总样本数: {len(X)}")
    print(f"[Info] 总体预警样本比例: {y_cls.sum()}/{len(y_cls)} = {y_cls.mean():.4f}")

    reg_model, cls_model, _ = cross_validate_models(X, y_reg, y_cls, feat_names, n_splits=5)

    print(f"\n=== 使用最后一折模型进行SHAP分析 ===")
    output_dir = Path(__file__).parent / "outputs"
    analyze_shap_reg(reg_model, X, feat_names, output_dir)
    analyze_shap_cls(cls_model, X, feat_names, output_dir)


if __name__ == "__main__":
    main()
