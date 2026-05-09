"""
主运行脚本：对 MJ9/MJ1/MJ3 三个监测点依次执行第四章分析流水线
"""

import json
import sys
from pathlib import Path

# 添加src目录到路径
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

TARGETS = ['MJ9', 'MJ1', 'MJ3']


def run_pipeline(target):
    """对单个目标点运行全流程"""
    modules = [
        ("01b_compute_v0", "计算 V0 阈值"),
        ("01_calculate_exceed_probability", "基于 V0 的概率越限计算"),
        ("02_determine_warning_levels", "确定预警等级"),
        ("03_traditional_velocity_warning", "传统速率预警 (V0 体系)"),
        ("04_evaluate_performance", "性能评估"),
        ("05_calculate_lead_time", "预警提前时间"),
        ("06_plot_warning_timeseries", "绘制预警时间序列图"),
        ("07_plot_detailed_periods", "绘制详细时段分析图"),
        ("08_plot_active_period_fig43", "绘制阶跃变形期预警细节图"),
        ("09_train_period_demo", "训练期 in-sample 阶跃段演示"),
    ]

    print("\n" + "#" * 70)
    print(f"#  目标监测点: {target}")
    print("#" * 70)

    for i, (module_name, description) in enumerate(modules, 1):
        print(f"\n{'=' * 70}")
        print(f"[{target}] 步骤 {i}/{len(modules)}: {description}")
        print(f"模块: {module_name}")
        print(f"{'=' * 70}\n")

        # 动态导入并执行
        module = __import__(module_name)
        module.main(target=target)
        print(f"\n✓ [{target}] 步骤 {i} 完成")


def summarise(tables_root):
    """汇总三点关键指标到一张 CSV 供正文引用"""
    rows = []
    for tgt in TARGETS:
        stats_dir = tables_root / "statistics" / tgt
        prob_metrics_fp = stats_dir / "probability_warning_metrics.json"
        trad_metrics_fp = stats_dir / "traditional_warning_metrics.json"
        prob_lead_fp = stats_dir / "probability_lead_time_statistics.json"
        trad_lead_fp = stats_dir / "traditional_lead_time_statistics.json"

        if not prob_metrics_fp.exists():
            print(f"  缺少 {prob_metrics_fp}，跳过 {tgt}")
            continue

        prob = json.loads(prob_metrics_fp.read_text())
        trad = json.loads(trad_metrics_fp.read_text())
        prob_lt = json.loads(prob_lead_fp.read_text()) if prob_lead_fp.exists() else {}
        trad_lt = json.loads(trad_lead_fp.read_text()) if trad_lead_fp.exists() else {}

        rows.append({
            'target': tgt,
            'prob_accuracy': prob['accuracy'],
            'prob_recall': prob['recall'],
            'prob_precision': prob['precision'],
            'prob_f1': prob['f1_score'],
            'prob_TP': prob['confusion_matrix']['TP'],
            'prob_FP': prob['confusion_matrix']['FP'],
            'prob_FN': prob['confusion_matrix']['FN'],
            'prob_TN': prob['confusion_matrix']['TN'],
            'trad_accuracy': trad['accuracy'],
            'trad_recall': trad['recall'],
            'trad_precision': trad['precision'],
            'trad_f1': trad['f1_score'],
            'trad_TP': trad['confusion_matrix']['TP'],
            'trad_FP': trad['confusion_matrix']['FP'],
            'trad_FN': trad['confusion_matrix']['FN'],
            'trad_TN': trad['confusion_matrix']['TN'],
            'prob_mean_lead_time': prob_lt.get('mean_lead_time', 0),
            'prob_detection_rate': prob_lt.get('detection_rate', 0),
            'trad_mean_lead_time': trad_lt.get('mean_lead_time', 0),
            'trad_detection_rate': trad_lt.get('detection_rate', 0),
        })

    if not rows:
        return

    import pandas as pd
    summary = pd.DataFrame(rows)
    summary_path = tables_root / "paper_tables" / "three_points_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("三点预警性能汇总 (供论文 §4.3.3 / §4.4 引用)")
    print("=" * 70)
    print(summary.to_string(index=False))
    print(f"\n保存至: {summary_path}")


def main():
    print("\n" + "=" * 70)
    print(" " * 20 + "第四章代码执行（三点流水线）")
    print("=" * 70)

    tables_root = Path(__file__).parent / "outputs" / "tables"

    for tgt in TARGETS:
        run_pipeline(tgt)

    summarise(tables_root)

    print("\n" + "=" * 70)
    print(" " * 25 + "全部完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
