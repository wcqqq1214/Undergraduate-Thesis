"""
模块 1: 基于 V0 阈值体系计算概率预警指标

输入:
  - chapter3 50 次 LSTM 累计位移预测 (predictions_{target}.csv)
  - 01b 计算的 V0（黄/橙/红三级阈值，单位 mm/M）

输出:
  对测试集每一天给出：
    - 预测月速率的 50 次分布统计（mean/std/分位数）
    - 4 等级预警概率：绿色/黄色/橙色/红色
    - 越限概率 P(V > V0)（保留旧接口兼容绘图脚本）
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

MONTH_WINDOW_DAYS = 30   # 月速率聚合口径（与 V0 统计保持一致）


def load_lstm_predictions(csv_path):
    df = pd.read_csv(csv_path)
    pivot = df.pivot(index='time_index', columns='run_id', values='prediction')
    pivot = pivot.sort_index()
    # shape: (n_days, n_runs)
    return pivot.values, pivot.index.values


def compute_monthly_rates(predictions_matrix, window=MONTH_WINDOW_DAYS):
    """对每一次 run 做 30 天滚动差分，得到月速率矩阵 (mm/month)

    predictions_matrix: shape (n_days, n_runs) 累计位移预测
    返回: shape (n_days - window, n_runs)
    """
    past = predictions_matrix[:-window]
    future = predictions_matrix[window:]
    return future - past


def probability_levels(monthly_rates, v0, v0_orange, v0_red):
    """
    monthly_rates: shape (n_days, n_runs) 每天的 50 次月速率
    返回:
      stats: DataFrame 每天一行（mean/std/median/min/max/p05/p95）
      level_probs: DataFrame 每天 4 等级概率
    """
    n_runs = monthly_rates.shape[1]

    p_exceed_v0 = (monthly_rates > v0).sum(axis=1) / n_runs
    p_exceed_5v0 = (monthly_rates > v0_orange).sum(axis=1) / n_runs
    p_exceed_10v0 = (monthly_rates > v0_red).sum(axis=1) / n_runs

    p_green = 1.0 - p_exceed_v0
    p_yellow = p_exceed_v0 - p_exceed_5v0
    p_orange = p_exceed_5v0 - p_exceed_10v0
    p_red = p_exceed_10v0

    level_probs = pd.DataFrame({
        'p_green': p_green,
        'p_yellow': p_yellow,
        'p_orange': p_orange,
        'p_red': p_red,
        # 保留旧接口兼容绘图
        'exceed_probability': p_exceed_v0,
    })

    rate_stats = pd.DataFrame({
        'rate_mean': monthly_rates.mean(axis=1),
        'rate_std': monthly_rates.std(axis=1, ddof=1),
        'rate_median': np.median(monthly_rates, axis=1),
        'rate_p05': np.percentile(monthly_rates, 5, axis=1),
        'rate_p95': np.percentile(monthly_rates, 95, axis=1),
        'rate_min': monthly_rates.min(axis=1),
        'rate_max': monthly_rates.max(axis=1),
    })

    return rate_stats, level_probs


def main(target='MJ1'):
    base_dir = Path(__file__).parent.parent.parent.parent
    chapter3_out = base_dir / 'code' / 'chapter3' / 'outputs' / 'tables'
    chapter4_out = base_dir / 'code' / 'chapter4' / 'outputs' / 'tables'

    pred_file = chapter3_out / f'lstm_trend_50runs_predictions_{target}.csv'

    print('=' * 60)
    print(f'模块 1: V0 体系下的概率预警计算 [{target}]')
    print('=' * 60)

    # 1. 载入 V0 阈值
    v0_file = chapter4_out / 'intermediate_data' / target / 'v0.json'
    v0_conf = json.loads(v0_file.read_text())
    v0 = v0_conf['v0_mm_per_month']
    v0_orange = v0_conf['v0_orange_threshold']
    v0_red = v0_conf['v0_red_threshold']
    print(f'\n[{target}] V0 阈值: 黄={v0:.3f}  橙={v0_orange:.3f}  红={v0_red:.3f} mm/M')

    # 2. 载入 LSTM 50 次累计位移预测
    predictions, time_indices = load_lstm_predictions(pred_file)
    print(f'预测矩阵 shape: {predictions.shape}  (n_days, n_runs)')

    # 3. 30 天滚动差分 → 月速率
    monthly_rates = compute_monthly_rates(predictions, MONTH_WINDOW_DAYS)
    valid_time_indices = time_indices[MONTH_WINDOW_DAYS:]
    print(f'月速率矩阵 shape: {monthly_rates.shape}  '
          f'(测试集前 {MONTH_WINDOW_DAYS} 天因滚动窗口不可用)')
    print(f'月速率范围 [{monthly_rates.min():.3f}, {monthly_rates.max():.3f}] mm/M')

    # 4. 计算 4 等级概率
    rate_stats, level_probs = probability_levels(monthly_rates, v0, v0_orange, v0_red)

    # 5. 保存
    out_dir = chapter4_out / 'intermediate_data' / target
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.DataFrame({'time_index': valid_time_indices})
    combined = pd.concat([combined.reset_index(drop=True),
                          rate_stats.reset_index(drop=True),
                          level_probs.reset_index(drop=True)], axis=1)
    combined_file = out_dir / 'exceed_probability.csv'
    combined.to_csv(combined_file, index=False)
    print(f'\n概率预警结果已保存: {combined_file}')

    # 6. 汇总统计
    statistics = {
        'target': target,
        'v0_mm_per_month': v0,
        'v0_orange': v0_orange,
        'v0_red': v0_red,
        'n_days': int(len(valid_time_indices)),
        'n_runs': int(monthly_rates.shape[1]),
        'mean_rate_mm_per_month': float(rate_stats['rate_mean'].mean()),
        'max_rate_mm_per_month': float(rate_stats['rate_mean'].max()),
        'days_exceed_v0': int((level_probs['exceed_probability'] > 0.5).sum()),
        'days_any_yellow_or_above': int(((level_probs[['p_yellow', 'p_orange', 'p_red']].sum(axis=1)) > 0.5).sum()),
        'mean_p_yellow': float(level_probs['p_yellow'].mean()),
        'mean_p_orange': float(level_probs['p_orange'].mean()),
        'mean_p_red': float(level_probs['p_red'].mean()),
    }

    stats_dir = chapter4_out / 'statistics' / target
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / 'exceed_probability_statistics.json').write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False))
    print(f'统计信息已保存: {stats_dir / "exceed_probability_statistics.json"}')

    print('\n' + '=' * 60)
    print(f'模块 1 完成！[{target}]')
    print('=' * 60)


if __name__ == '__main__':
    import sys
    tgt = sys.argv[1] if len(sys.argv) > 1 else 'MJ1'
    main(tgt)
