"""
模块4：性能评估
计算混淆矩阵和性能指标，对比概率预警和传统预警方法
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


def evaluate_warning_performance(predicted_warnings, actual_displacement, threshold=0.3):
    """
    评估预警性能

    参数:
        predicted_warnings: (T-1,) - 预测的预警等级
        actual_displacement: (T,) - 实际累计位移
        threshold: 0.3 mm/天 - 日位移增量阈值

    返回:
        metrics: 包含混淆矩阵和性能指标的字典
    """
    # 计算实际日位移增量作为Ground Truth
    actual_daily_increment = np.diff(actual_displacement)  # (T-1,)

    # 确保长度一致
    min_len = min(len(predicted_warnings), len(actual_daily_increment))
    predicted_warnings = predicted_warnings[:min_len]
    actual_daily_increment = actual_daily_increment[:min_len]

    # 构建真实标签：黄色及以上(>=2)为正类，绿色蓝色(<2)为负类
    y_pred = (predicted_warnings >= 2).astype(int)
    y_true = (actual_daily_increment > threshold).astype(int)

    # 计算混淆矩阵
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())

    # 计算性能指标
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 计算误报率和漏报率
    false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
    false_negative_rate = FN / (FN + TP) if (FN + TP) > 0 else 0

    metrics = {
        'confusion_matrix': {
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN
        },
        'accuracy': float(accuracy),
        'recall': float(recall),
        'precision': float(precision),
        'f1_score': float(f1),
        'false_positive_rate': float(false_positive_rate),
        'false_negative_rate': float(false_negative_rate),
        'total_samples': int(total),
        'positive_samples': int((y_true == 1).sum()),
        'negative_samples': int((y_true == 0).sum())
    }

    return metrics


def print_metrics(metrics, method_name):
    """打印性能指标"""
    print(f"\n{method_name} 性能指标:")
    print(f"  混淆矩阵:")
    cm = metrics['confusion_matrix']
    print(f"    TP={cm['TP']}, FP={cm['FP']}")
    print(f"    FN={cm['FN']}, TN={cm['TN']}")
    print(f"  准确率 (Accuracy): {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  召回率 (Recall): {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"  精确率 (Precision): {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"  F1分数: {metrics['f1_score']:.3f} ({metrics['f1_score']*100:.1f}%)")
    print(f"  误报率 (FPR): {metrics['false_positive_rate']:.3f} ({metrics['false_positive_rate']*100:.1f}%)")
    print(f"  漏报率 (FNR): {metrics['false_negative_rate']:.3f} ({metrics['false_negative_rate']*100:.1f}%)")


def create_comparison_table(prob_metrics, trad_metrics):
    """创建对比表格"""
    comparison = pd.DataFrame({
        '指标': ['准确率', '召回率', '精确率', 'F1分数', '误报率', '漏报率'],
        '传统速率预警': [
            f"{trad_metrics['accuracy']:.3f}",
            f"{trad_metrics['recall']:.3f}",
            f"{trad_metrics['precision']:.3f}",
            f"{trad_metrics['f1_score']:.3f}",
            f"{trad_metrics['false_positive_rate']:.3f}",
            f"{trad_metrics['false_negative_rate']:.3f}"
        ],
        '本文概率预警': [
            f"{prob_metrics['accuracy']:.3f}",
            f"{prob_metrics['recall']:.3f}",
            f"{prob_metrics['precision']:.3f}",
            f"{prob_metrics['f1_score']:.3f}",
            f"{prob_metrics['false_positive_rate']:.3f}",
            f"{prob_metrics['false_negative_rate']:.3f}"
        ],
        '提升': [
            f"+{(prob_metrics['accuracy'] - trad_metrics['accuracy']):.3f}",
            f"+{(prob_metrics['recall'] - trad_metrics['recall']):.3f}",
            f"+{(prob_metrics['precision'] - trad_metrics['precision']):.3f}",
            f"+{(prob_metrics['f1_score'] - trad_metrics['f1_score']):.3f}",
            f"{(prob_metrics['false_positive_rate'] - trad_metrics['false_positive_rate']):.3f}",
            f"{(prob_metrics['false_negative_rate'] - trad_metrics['false_negative_rate']):.3f}"
        ]
    })
    return comparison


def main():
    """主函数"""
    # 设置路径
    base_dir = Path(__file__).parent.parent.parent
    chapter4_output = base_dir / "chapter4" / "outputs" / "tables"
    intermediate_dir = chapter4_output / "intermediate_data"
    paper_tables_dir = chapter4_output / "paper_tables"
    statistics_dir = chapter4_output / "statistics"

    # 文件路径
    prob_warning_file = intermediate_dir / "warning_levels.csv"
    trad_warning_file = intermediate_dir / "traditional_warning_levels.csv"
    actual_disp_file = intermediate_dir / "actual_displacement_MJ1.csv"

    print("="*60)
    print("模块4: 性能评估")
    print("="*60)

    # 1. 加载数据
    print("\n加载数据...")
    prob_df = pd.read_csv(prob_warning_file)
    trad_df = pd.read_csv(trad_warning_file)
    actual_df = pd.read_csv(actual_disp_file)

    prob_warnings = prob_df['warning_level'].values
    trad_warnings = trad_df['warning_level'].values
    actual_displacement = actual_df['displacement'].values

    print(f"概率预警数据: {len(prob_warnings)} 天")
    print(f"传统预警数据: {len(trad_warnings)} 天")
    print(f"实际位移数据: {len(actual_displacement)} 天")

    # 2. 评估概率预警方法
    print("\n评估概率预警方法...")
    prob_metrics = evaluate_warning_performance(
        prob_warnings,
        actual_displacement,
        threshold=0.3
    )
    print_metrics(prob_metrics, "概率预警方法")

    # 3. 评估传统预警方法
    print("\n评估传统预警方法...")
    trad_metrics = evaluate_warning_performance(
        trad_warnings,
        actual_displacement,
        threshold=0.3
    )
    print_metrics(trad_metrics, "传统预警方法")

    # 4. 创建对比表格
    print("\n生成对比表格...")
    comparison_table = create_comparison_table(prob_metrics, trad_metrics)
    print("\n性能指标对比:")
    print(comparison_table.to_string(index=False))

    # 5. 保存结果
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    statistics_dir.mkdir(parents=True, exist_ok=True)

    # 保存概率预警指标
    prob_metrics_file = statistics_dir / "probability_warning_metrics.json"
    with open(prob_metrics_file, 'w') as f:
        json.dump(prob_metrics, f, indent=2)
    print(f"\n概率预警指标已保存至: {prob_metrics_file}")

    # 保存传统预警指标
    trad_metrics_file = statistics_dir / "traditional_warning_metrics.json"
    with open(trad_metrics_file, 'w') as f:
        json.dump(trad_metrics, f, indent=2)
    print(f"传统预警指标已保存至: {trad_metrics_file}")

    # 保存对比表格（论文用）
    comparison_file = paper_tables_dir / "performance_comparison.csv"
    comparison_table.to_csv(comparison_file, index=False)
    print(f"对比表格已保存至: {comparison_file}")

    # 保存混淆矩阵（论文用）
    confusion_matrix_df = pd.DataFrame({
        '方法': ['传统速率预警', '传统速率预警', '本文概率预警', '本文概率预警'],
        '预测结果': ['预警触发', '无预警', '预警触发', '无预警'],
        '正类(位移>0.3mm)': [
            trad_metrics['confusion_matrix']['TP'],
            trad_metrics['confusion_matrix']['FN'],
            prob_metrics['confusion_matrix']['TP'],
            prob_metrics['confusion_matrix']['FN']
        ],
        '负类(位移≤0.3mm)': [
            trad_metrics['confusion_matrix']['FP'],
            trad_metrics['confusion_matrix']['TN'],
            prob_metrics['confusion_matrix']['FP'],
            prob_metrics['confusion_matrix']['TN']
        ]
    })
    confusion_file = paper_tables_dir / "confusion_matrix.csv"
    confusion_matrix_df.to_csv(confusion_file, index=False)
    print(f"混淆矩阵已保存至: {confusion_file}")

    print("\n" + "="*60)
    print("模块4 完成！")
    print("="*60)


if __name__ == "__main__":
    main()
