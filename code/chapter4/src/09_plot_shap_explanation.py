"""
图4-4: 基于真实SHAP值的预警事件归因分析
针对 2017-09-28 MJ1 训练期阶跃段黄色预警事件（in-sample 演示），使用LightGBM分类模型计算SHAP归因。
注：第四章采用 V0 阈值体系下的 4 级预警；原 2020-03-15 橙色事件不适用于新体系。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'utils'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
from typing import List, Tuple
from sklearn.model_selection import TimeSeriesSplit

from read_monitoring_data import load_monitoring_data

plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

BASE_DIR = Path(__file__).parent.parent
FIGURES_DIR = BASE_DIR / 'outputs' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 因子→中文标签映射（用于分组聚合）
FACTOR_CATEGORIES = {
    'RWL':    '库水位',
    'GWT':    '地下水位',
    'Rainfall': '降雨量',
    'aveT':   '日均气温',
    'minT':   '日最低气温',
    'maxT':   '日最高气温',
    'DP':     '露点温度',
    'RH':     '相对湿度',
    'disp':   '历史位移',
}

def build_supervised_samples(
    df: pd.DataFrame,
    window: int,
    point_cols: List[str],
    env_cols: List[str],
    warning_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """构造监督学习样本，返回样本元信息用于定位特定事件"""
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    samples, targets_reg, targets_cls = [], [], []
    meta_dates, meta_points = [], []
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
            meta_dates.append(df_sorted["Date"].iloc[t])
            meta_points.append(point)

    feat_names: List[str] = []
    for k in range(window, 0, -1):
        feat_names.append(f"disp(t-{k})")
    for col in env_cols:
        for k in range(window, 0, -1):
            feat_names.append(f"{col}(t-{k})")
    for p in point_cols:
        feat_names.append(f"point_is_{p}")

    meta = pd.DataFrame({"date": meta_dates, "point": meta_points})
    return np.vstack(samples), np.array(targets_reg), np.array(targets_cls), feat_names, meta


def map_feature_to_category(fname: str) -> str:
    """将原始特征名映射到因子类别（温度因子合并）"""
    if fname.startswith("disp"):
        return "历史位移"
    if fname.startswith("point_is_"):
        return "监测点标识"
    # 温度相关因子统一归为"气温变化"
    if any(fname.startswith(k) for k in ['aveT', 'minT', 'maxT', 'DP']):
        return "气温变化"
    for key, label in FACTOR_CATEGORIES.items():
        if fname.startswith(key) and key not in ['aveT', 'minT', 'maxT', 'DP']:
            return label
    return "其他因素"


def build_chinese_names(feat_names: List[str]) -> List[str]:
    """构建可读的中文特征名"""
    env_cn = {
        'Rainfall/mm': '降雨量', 'GWT/m': '地下水位', 'RWL/m': '库水位',
        'aveT/℃': '日均气温', 'minT/℃': '日最低气温', 'maxT/℃': '日最高气温',
        'DP': '露点温度', 'RH': '相对湿度',
    }
    names = []
    for name in feat_names:
        if name.startswith('disp(t-'):
            lag = name.split('(t-')[1].rstrip(')')
            names.append(f'位移滞后{lag}天')
        elif name.startswith('point_is_'):
            names.append(f'监测点:{name.replace("point_is_", "")}')
        else:
            matched = False
            for key, cn in env_cn.items():
                if name.startswith(f'{key}(t-'):
                    lag = name.split('(t-')[1].rstrip(')')
                    names.append(f'{cn}滞后{lag}天')
                    matched = True
                    break
            if not matched:
                names.append(name)
    return names


def aggregate_shap_by_category(shap_vals: np.ndarray, feat_names: List[str]) -> dict:
    """按因子类别聚合SHAP值（绝对值之和）"""
    cats = {}
    for i, name in enumerate(feat_names):
        cat = map_feature_to_category(name)
        cats[cat] = cats.get(cat, 0.0) + abs(shap_vals[i])
    total = sum(cats.values())
    return {k: v / total * 100 for k, v in cats.items()}  # 百分比


def plot_shap_waterfall(shap_values, base_value, sample_idx: int,
                        feat_names_cn: List[str], event_date: str):
    """绘制SHAP瀑布图"""
    fig, ax = plt.subplots(figsize=(12, 8))

    sv = shap_values[sample_idx]
    # 取top 12特征
    abs_sv = np.abs(sv)
    top_idx = np.argsort(abs_sv)[-12:]
    # 按SHAP值从底到顶排列
    order = np.argsort(sv[top_idx])

    top_sv = sv[top_idx][order]
    top_names = [feat_names_cn[i] for i in top_idx][::-1]
    # reverse order for waterfall (bottom to top)
    top_sv = sv[top_idx][::-1]
    top_names = [feat_names_cn[i] for i in top_idx][::-1]

    # 瀑布图累积
    cumulative = base_value
    y_positions = list(range(len(top_names) + 1))
    values = [base_value] + list(top_sv)
    labels = ['基准值\nE[f(x)]'] + top_names

    colors = []
    for v in values[1:]:
        colors.append('#e74c3c' if v > 0 else '#3498db')

    # 绘制基准值
    ax.barh(0, base_value, color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=1)
    ax.text(base_value / 2, 0, f'{base_value:.3f}',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # 绘制各特征贡献
    for i, (val, color) in enumerate(zip(values[1:], colors)):
        y = i + 1
        if val > 0:
            left = cumulative
            ax.barh(y, val, left=left, color=color, alpha=0.8, edgecolor='black', linewidth=1)
            ax.text(left + val / 2, y, f'+{val:.4f}',
                    ha='center', va='center', fontsize=9, fontweight='bold')
        else:
            left = cumulative + val
            ax.barh(y, abs(val), left=left, color=color, alpha=0.8, edgecolor='black', linewidth=1)
            ax.text(left + abs(val) / 2, y, f'{val:.4f}',
                    ha='center', va='center', fontsize=9, fontweight='bold')
        cumulative += val

    # 最终预测值
    final_y = len(top_names) + 1
    ax.barh(final_y, cumulative, color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
    ax.text(cumulative / 2, final_y, f'{cumulative:.4f}',
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    ax.set_yticks(range(len(labels) + 1))
    ax.set_yticklabels(labels + ['最终预测\nf(x)'], fontsize=10)
    ax.axvline(x=base_value, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('模型输出值 (log-odds)', fontsize=12, fontweight='bold')
    ax.set_title(f'SHAP瀑布图 — 预警事件归因分析\n({event_date} MJ1黄色预警)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    out_png = FIGURES_DIR / 'shap_waterfall.png'
    out_pdf = FIGURES_DIR / 'shap_waterfall.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ SHAP瀑布图已保存: {out_png}")
    plt.close()


def plot_shap_bar_aggregated(category_pct: dict, event_date: str):
    """绘制按因子类别聚合的SHAP贡献柱状图"""
    # 排除监测点标识，合并小于1%的类别到"其他因素"
    plot_data = {}
    other_pct = 0.0
    for k, v in category_pct.items():
        if k == '监测点标识':
            continue
        if v < 1.0:
            other_pct += v
        else:
            plot_data[k] = v
    if other_pct > 0:
        plot_data['其他因素'] = other_pct

    # 按贡献排序（其他因素放最后）
    sorted_items = sorted(plot_data.items(), key=lambda x: (x[0] == '其他因素', -x[1]))
    categories = [item[0] for item in sorted_items]
    pcts = [item[1] for item in sorted_items]
    n = len(categories)

    colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71',
              '#9b59b6', '#1abc9c'][:n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # ── 子图1: 水平条形图 ──
    y_pos = range(n)
    bars = ax1.barh(y_pos, pcts, color=colors, alpha=0.85,
                    edgecolor='black', linewidth=1.5, height=0.55)
    for i, (bar, pct) in enumerate(zip(bars, pcts)):
        ax1.text(pct + 1.0, i, f'{pct:.1f}%', va='center',
                 fontsize=13, fontweight='bold')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(categories, fontsize=12)
    ax1.set_xlabel('对预警概率的贡献度 (%)', fontsize=13, fontweight='bold')
    ax1.set_title('SHAP归因分析 — 各因子贡献度', fontsize=14, fontweight='bold', pad=12)
    ax1.set_xlim(0, max(pcts) * 1.20)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()

    # ── 子图2: 饼图 ──
    explode = [0.06] * n
    max_idx = pcts.index(max(pcts))
    explode[max_idx] = 0.10

    # 小于 2% 的楔块放大到 2%，使其可见；从最大楔块中扣除差额
    min_pct = 1.0
    pcts_display = list(pcts)
    for i in range(n):
        if pcts[i] < min_pct and pcts[i] > 0:
            pcts_display[i] = min_pct
    # 将放大造成的总计超出从最大楔块中扣除
    excess = sum(pcts_display) - 100.0
    if excess > 0:
        max_i = pcts.index(max(pcts))
        pcts_display[max_i] -= excess

    # 小于3%的楔块不显示百分比标签（避免挤在一起），图例中显示真实值
    def autopct_filter(p):
        return f'{p:.1f}%' if p >= 3.0 else ''

    wedges, texts, autotexts = ax2.pie(
        pcts_display, labels=None, autopct=autopct_filter,
        colors=colors, explode=explode,
        startangle=90,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.2},
        pctdistance=0.60,
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')
        at.set_fontsize(12)

    # 图例放在饼图右侧（显示全部类别及百分比）
    ax2.legend(
        wedges, [f'{c} ({p:.1f}%)' for c, p in zip(categories, pcts)],
        title='因子类别',
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        fontsize=11,
        title_fontsize=12,
        framealpha=0.95,
    )
    ax2.set_title('SHAP归因分析 — 因子占比', fontsize=14, fontweight='bold', pad=12)

    fig.suptitle(f'预警事件SHAP归因分析\n({event_date} MJ1黄色预警)',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0, 0.88, 0.92])

    out_png = FIGURES_DIR / 'shap_explanation.png'
    out_pdf = FIGURES_DIR / 'shap_explanation.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ SHAP归因图已保存: {out_png}")
    plt.close()


def main():
    print("=" * 60)
    print("生成图4-4: 基于真实SHAP值的预警归因分析")
    print("=" * 60)

    # 1. 加载数据
    df = load_monitoring_data()
    print(f"[1/6] 数据加载完成: {len(df)} 条记录, {df['Date'].min()} 至 {df['Date'].max()}")

    # 2. 构建监督样本
    window = 5
    point_cols = ["MJ9/mm", "MJ1/mm", "MJ3/mm"]
    env_cols = ["Rainfall/mm", "GWT/m", "RWL/m", "aveT/℃", "minT/℃", "maxT/℃", "DP", "RH"]
    X, y_reg, y_cls, feat_names, meta = build_supervised_samples(df, window, point_cols, env_cols)
    print(f"[2/6] 监督样本构建: {len(X)} 样本, {len(feat_names)} 特征, "
          f"正样本占比 {y_cls.mean():.3f}")

    # 3. 训练LightGBM分类模型（按时序切分，避免数据泄漏）
    print("[3/6] 训练LightGBM分类模型...")
    # 使用时序交叉验证，用最后一折模型
    tscv = TimeSeriesSplit(n_splits=5)
    cls_model = None
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_cls[train_idx], y_cls[test_idx]

        if y_test.sum() == 0:
            continue

        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        cls_model = lgb.train(
            {
                "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
                "num_leaves": 8, "max_depth": 3, "learning_rate": 0.05,
                "min_data_in_leaf": 20, "lambda_l1": 0.3, "lambda_l2": 0.3,
                "feature_fraction": 0.7, "bagging_fraction": 0.7,
                "bagging_freq": 1, "scale_pos_weight": 5, "verbose": -1,
            },
            train_data, num_boost_round=300,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

    if cls_model is None:
        print("错误: 模型训练失败")
        return

    print(f"   模型训练完成, 最佳迭代: {cls_model.best_iteration}")

    # 4. 定位 2017-09-28 MJ1 黄色预警事件样本（V0 体系下训练期 in-sample 演示的典型峰值日）
    event_date = pd.Timestamp('2017-09-28')
    mask = (meta['date'] == event_date) & (meta['point'] == 'MJ1/mm')
    sample_idx = np.where(mask)[0]
    if len(sample_idx) == 0:
        print(f"错误: 找不到 {event_date} MJ1 的样本")
        return
    sample_idx = sample_idx[0]
    print(f"[4/6] 定位样本: 索引={sample_idx}, 日期={event_date.date()}, "
          f"实际增量={y_reg[sample_idx]:.4f}mm, 是否预警={y_cls[sample_idx]}")

    # 打印样本的特征信息
    print(f"   前5天位移: {X[sample_idx, :5]}")
    print(f"   当日降雨量(t-1): {X[sample_idx, 5+0]:.1f}mm")
    print(f"   库水位(t-1): {X[sample_idx, 5+5*2+0]:.2f}m")

    # 5. 计算SHAP值
    print("[5/6] 计算SHAP值...")
    explainer = shap.TreeExplainer(cls_model)
    shap_values = explainer.shap_values(X)
    # LightGBM二分类: shap_values 是 list [负类, 正类]
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values

    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[1]  # 正类的base value

    # 预测概率
    y_prob = cls_model.predict(X, num_iteration=cls_model.best_iteration)
    event_prob = y_prob[sample_idx]
    print(f"   Base value (正类): {base_value:.4f}")
    print(f"   预测概率: {event_prob:.4f}")
    print(f"   实际标签: {y_cls[sample_idx]}")

    # 6. 聚合SHAP值
    feat_names_cn = build_chinese_names(feat_names)
    event_shap = shap_values_pos[sample_idx]
    category_pct = aggregate_shap_by_category(event_shap, feat_names)

    print(f"[6/6] SHAP因子贡献度:")
    for cat, pct in sorted(category_pct.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cat}: {pct:.1f}%")

    # 绘制图表
    event_date_str = "2017年9月28日"
    plot_shap_waterfall(shap_values_pos, base_value, sample_idx, feat_names_cn, event_date_str)
    plot_shap_bar_aggregated(category_pct, event_date_str)

    # 打印top特征详情
    print("\n=== Top 15 特征SHAP贡献 ===")
    abs_sv = np.abs(event_shap)
    top_idx = np.argsort(abs_sv)[-15:][::-1]
    for i in top_idx:
        direction = "↑ 推高预警概率" if event_shap[i] > 0 else "↓ 降低预警概率"
        print(f"  {feat_names_cn[i]:30s}  SHAP={event_shap[i]:+.6f}  {direction}")

    print("\n✓ 所有图表生成完成!")


if __name__ == '__main__':
    main()
