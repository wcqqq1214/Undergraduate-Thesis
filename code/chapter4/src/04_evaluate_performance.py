"""
模块 4: 性能评估

评估口径 (V0 体系, 4 级)：
  - Ground truth：实测月速率 > V0 为"真实越限"（正类）
  - 预测：warning_level >= 1 (黄/橙/红) 视为"触发预警"
  - 两种方法在同一评估窗口内对齐：
        * 概率预警: 测试集 & 跳过 LSTM 前 30 天
        * 传统预警: 全序列 & 跳过前 29 天
    → 公共窗口 = 测试集 & 跳过前 30 天
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


MONTH_WINDOW_DAYS = 30
TRAIN_RATIO = 0.80


def load_v0(intermediate_dir):
    return json.loads((intermediate_dir / 'v0.json').read_text())


def compute_actual_monthly_rate(displacement_series, window=MONTH_WINDOW_DAYS):
    daily_inc = pd.Series(displacement_series).diff().fillna(0)
    return daily_inc.rolling(window=window, min_periods=window).sum().to_numpy()


def evaluate(y_pred, y_true):
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fpr = FP / (FP + TN) if (FP + TN) else 0
    fnr = FN / (FN + TP) if (FN + TP) else 0
    return {
        'confusion_matrix': {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN},
        'accuracy': accuracy, 'recall': recall, 'precision': precision,
        'f1_score': f1, 'false_positive_rate': fpr, 'false_negative_rate': fnr,
        'total_samples': total,
        'positive_samples': int((y_true == 1).sum()),
        'negative_samples': int((y_true == 0).sum()),
    }


def print_metrics(metrics, name):
    cm = metrics['confusion_matrix']
    print(f'\n{name}')
    print(f'  混淆矩阵: TP={cm["TP"]} FP={cm["FP"]} FN={cm["FN"]} TN={cm["TN"]}')
    print(f'  准确率 {metrics["accuracy"]*100:.1f}%  召回率 {metrics["recall"]*100:.1f}%  '
          f'精确率 {metrics["precision"]*100:.1f}%  F1 {metrics["f1_score"]*100:.1f}%')
    print(f'  FPR {metrics["false_positive_rate"]*100:.1f}%  '
          f'FNR {metrics["false_negative_rate"]*100:.1f}%')


def create_comparison_table(prob, trad):
    def fmt(x, plus=False):
        if plus:
            return f'{x:+.3f}'
        return f'{x:.3f}'
    rows = []
    for metric_key, label in [
        ('accuracy', '准确率'), ('recall', '召回率'), ('precision', '精确率'),
        ('f1_score', 'F1分数'), ('false_positive_rate', '误报率'),
        ('false_negative_rate', '漏报率')]:
        rows.append({
            '指标': label,
            '传统速率预警': fmt(trad[metric_key]),
            '本文概率预警': fmt(prob[metric_key]),
            '提升': fmt(prob[metric_key] - trad[metric_key], plus=True),
        })
    return pd.DataFrame(rows)


def main(target='MJ1'):
    base_dir = Path(__file__).parent.parent.parent.parent
    chapter4_out = base_dir / 'code' / 'chapter4' / 'outputs' / 'tables'
    intermediate_dir = chapter4_out / 'intermediate_data' / target
    paper_tables_dir = chapter4_out / 'paper_tables' / target
    stats_dir = chapter4_out / 'statistics' / target
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(f'模块 4: 性能评估 (V0 体系) [{target}]')
    print('=' * 60)

    v0_conf = load_v0(intermediate_dir)
    v0 = v0_conf['v0_mm_per_month']

    actual_df = pd.read_csv(intermediate_dir / f'actual_displacement_{target}.csv')
    actual_disp = actual_df['displacement'].to_numpy()
    actual_monthly = compute_actual_monthly_rate(actual_disp)

    n_total = len(actual_disp)
    train_size = int(n_total * TRAIN_RATIO)
    # 测试集起点
    test_start = train_size
    # 从测试集第 MONTH_WINDOW 天开始才有概率预警（LSTM 预测也需要滚动 30 天）
    eval_start = test_start + MONTH_WINDOW_DAYS
    print(f'评估窗口: 第 {eval_start} 至 {n_total - 1} 天 '
          f'(共 {n_total - eval_start} 天)')

    # Ground truth
    y_true = (actual_monthly[eval_start:] > v0).astype(int)

    # 概率预警: warning_levels 从测试集第 MONTH_WINDOW 天开始
    prob_df = pd.read_csv(intermediate_dir / 'warning_levels.csv')
    prob_levels_full = prob_df['warning_level'].to_numpy()
    # 概率序列长度 = n_total - test_start - MONTH_WINDOW_DAYS + 1（LSTM 测试集差分再滚动）
    # 直接按长度对齐到 y_true
    m = min(len(prob_levels_full), len(y_true))
    prob_levels = prob_levels_full[:m]
    y_true_prob = y_true[:m]
    y_pred_prob = (prob_levels >= 1).astype(int)

    # 传统预警: warning_levels 从全序列第 1 天开始
    trad_df = pd.read_csv(intermediate_dir / 'traditional_warning_levels.csv')
    trad_levels_full = trad_df['warning_level'].to_numpy()
    # 评估窗口对齐：eval_start - 1（传统差分少一天）到末尾
    trad_slice = trad_levels_full[eval_start - 1:eval_start - 1 + m]
    y_pred_trad = (trad_slice >= 1).astype(int)
    y_true_trad = y_true[:len(y_pred_trad)]

    print(f'\n评估样本数: 概率 {len(y_pred_prob)} | 传统 {len(y_pred_trad)}')
    print(f'实际越限天数 (月速率 > V0={v0:.3f} mm/M): {int(y_true_prob.sum())}')

    prob_metrics = evaluate(y_pred_prob, y_true_prob)
    trad_metrics = evaluate(y_pred_trad, y_true_trad)
    print_metrics(prob_metrics, '本文概率预警 (LSTM 月速率 vs V0)')
    print_metrics(trad_metrics, '传统速率预警 (实测月速率 vs V0)')

    comparison = create_comparison_table(prob_metrics, trad_metrics)
    print('\n对比:')
    print(comparison.to_string(index=False))

    (stats_dir / 'probability_warning_metrics.json').write_text(
        json.dumps(prob_metrics, indent=2))
    (stats_dir / 'traditional_warning_metrics.json').write_text(
        json.dumps(trad_metrics, indent=2))
    comparison.to_csv(paper_tables_dir / 'performance_comparison.csv', index=False)

    confusion = pd.DataFrame({
        '方法': ['传统速率预警', '传统速率预警', '本文概率预警', '本文概率预警'],
        '预测结果': ['预警触发', '无预警', '预警触发', '无预警'],
        f'正类(月速率>V0)': [
            trad_metrics['confusion_matrix']['TP'],
            trad_metrics['confusion_matrix']['FN'],
            prob_metrics['confusion_matrix']['TP'],
            prob_metrics['confusion_matrix']['FN'],
        ],
        f'负类(月速率<=V0)': [
            trad_metrics['confusion_matrix']['FP'],
            trad_metrics['confusion_matrix']['TN'],
            prob_metrics['confusion_matrix']['FP'],
            prob_metrics['confusion_matrix']['TN'],
        ],
    })
    confusion.to_csv(paper_tables_dir / 'confusion_matrix.csv', index=False)

    print('\n' + '=' * 60)
    print(f'模块 4 完成！[{target}]')
    print('=' * 60)


if __name__ == '__main__':
    import sys
    tgt = sys.argv[1] if len(sys.argv) > 1 else 'MJ1'
    main(tgt)
