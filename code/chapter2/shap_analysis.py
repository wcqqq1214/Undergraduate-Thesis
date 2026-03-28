"""
SHAP 可解释性分析模块

提供回归与分类模型的 SHAP 蜂拥图绘制与学术级导出功能。
"""
from pathlib import Path
from typing import List

import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np


# 英文原始列名 → 中文规范名称映射
# 在此处添加新特征的中文名称
CN_COL_MAPPING = {
    'Level':             '库水位 (m)',
    'Displacement_Rate': '位移速率 (mm/d)',
    'Rain_7d':           '7日累积降雨量 (mm)',
    'Rain_3d':           '3日累积降雨量 (mm)',
    'Rain_15d':          '15日累积降雨量 (mm)',
    'Rain':              '当日降雨量 (mm)',
    'Level_Change':      '库水位日变化量 (m/d)',
    # 'Feature_X':       '中文名称 (单位)',
}

# 英文学术名映射（用于滞后特征等）
_ENV_MAPPING = {
    'Rainfall/mm': ('降雨量', '(mm)'),
    'GWT/m':       ('地下水位', '(m)'),
    'RWL/m':       ('库水位', '(m)'),
    'aveT/℃':      ('日均气温', '(°C)'),
    'minT/℃':      ('日最低气温', '(°C)'),
    'maxT/℃':      ('日最高气温', '(°C)'),
    'DP':          ('露点温度', '(°C)'),
    'RH':          ('相对湿度', '(%)'),
}


def _build_academic_names(feature_names: List[str]) -> List[str]:
    """将原始特征名转换为可读的学术名称（优先中文，其次英文学术名）"""
    names = []
    for name in feature_names:
        if name in CN_COL_MAPPING:
            names.append(CN_COL_MAPPING[name])
            continue

        # 位移滞后特征
        if name.startswith('disp(t-'):
            lag = name.split('(t-')[1].rstrip(')')
            names.append(f'位移 (t-{lag}) (mm)')
            continue

        # 环境因子滞后特征
        matched = False
        for key, (label, unit) in _ENV_MAPPING.items():
            if name.startswith(f'{key}(t-'):
                lag = name.split('(t-')[1].rstrip(')')
                names.append(f'{label} (t-{lag}) {unit}')
                matched = True
                break

        if not matched:
            # 监测点指示变量
            if name.startswith('point_is_'):
                names.append(f'监测点: {name.replace("point_is_", "")}')
            else:
                names.append(name)

    return names


def _sort_by_lag(shap_values: np.ndarray, feature_names: List[str], max_display: int):
    """
    按滞后阶数排序：先按 t-1, t-2, ... 分组，组内按均值绝对值降序。
    返回重排后的 (shap_values, feature_names)。
    """
    import re

    def lag_key(name: str) -> int:
        m = re.search(r't-(\d+)', name)
        return int(m.group(1)) if m else 999  # 无滞后的特征（如监测点）排最后

    n = shap_values.shape[1]
    # 先按全局重要性筛出 top max_display 个特征
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:max_display]

    # 在 top 特征中按滞后阶数排序（t-1 在最上方 → 反转后放最后，因为 SHAP 图从下往上画）
    top_idx_sorted = sorted(top_idx, key=lambda i: lag_key(feature_names[i]), reverse=True)

    return shap_values[:, top_idx_sorted], [feature_names[i] for i in top_idx_sorted]


def _setup_matplotlib() -> None:
    """配置中文字体与负号显示"""
    plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun']
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
    })


def _fix_colorbar_label() -> None:
    """将 SHAP 图右侧颜色条的英文标签替换为中文"""
    for ax in plt.gcf().axes:
        if ax.get_ylabel() == 'Feature value':
            ax.set_ylabel('特征值', fontsize=11)
            ax.set_yticklabels(['低', '高'], fontsize=11)
            break


def _save_figure(output_dir: Path, stem: str) -> None:
    """保存高清 PNG（600 dpi）和矢量 PDF"""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close()


def analyze_shap_reg(
    model: lgb.Booster,
    X: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
) -> None:
    """回归模型 SHAP 蜂拥图"""
    _setup_matplotlib()
    academic_names = _build_academic_names(feature_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    plt.figure(figsize=(10, 6))
    sv_sorted, names_sorted = _sort_by_lag(shap_values, academic_names, max_display=12)
    shap.summary_plot(sv_sorted, X[:, [academic_names.index(n) for n in names_sorted]],
                      feature_names=names_sorted, show=False)
    _fix_colorbar_label()
    plt.xlabel("SHAP值 (对模型输出的影响幅度)", fontsize=12)
    plt.tight_layout()
    _save_figure(output_dir, "shap_summary_regression")


def analyze_shap_cls(
    model: lgb.Booster,
    X: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
) -> None:
    """分类模型 SHAP 蜂拥图（正类/预警状态）"""
    _setup_matplotlib()
    academic_names = _build_academic_names(feature_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # 新版 SHAP 解析 LightGBM 二分类时返回 [负类, 正类]，取索引 1
    shap_values_pos = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values

    plt.figure(figsize=(10, 6))
    sv_sorted, names_sorted = _sort_by_lag(shap_values_pos, academic_names, max_display=12)
    shap.summary_plot(sv_sorted, X[:, [academic_names.index(n) for n in names_sorted]],
                      feature_names=names_sorted, show=False)
    _fix_colorbar_label()
    plt.xlabel("SHAP值 (对预警概率的影响幅度)", fontsize=12)
    plt.tight_layout()
    _save_figure(output_dir, "shap_summary_classification")
