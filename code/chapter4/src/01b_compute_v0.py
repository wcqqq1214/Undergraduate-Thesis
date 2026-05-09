"""
模块 1b: 计算匀速变形阶段的位移速率 V0

方法（严格遵循你导师研究生 docx 5.2.2 节的 "V0 = MAX(1.5V̄ + 2σ)" 公式）：
  1. 读取某监测点的完整累计位移时间序列（来自 data/monitoring data.xlsx）。
  2. 仅使用训练集（前 80%）数据，剔除日位移增量 > 90% 分位的"加速事件"。
  3. 把剩余日位移增量用 30 天滚动窗口聚合成月速率 V (单位: mm/M)。
  4. 对该月速率序列计算均值 V̄ 与样本标准差 σ。
  5. 输出 V0 = 1.5·V̄ + 2σ（mm/M）以及派生阈值 5·V0、10·V0。

每个监测点 (MJ9/MJ1/MJ3) 各自算一套 V0，结果写入
  outputs/tables/intermediate_data/{target}/v0.json

同时把三点 V0 的算术均值作为 "滑坡整体参考 V0" 落盘到
  outputs/tables/statistics/v0_summary.csv
仅作为汇报时的整体代表值，不用于任何点的判级。

文献支撑：
  - Chen et al. 2024, Bull. Eng. Geol. Environ. 83:437（月口径、稳定段统计）。
  - Xu 等改进切线角方法中 V0 的含义（等速变形阶段均速）。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_FILE = BASE_DIR / 'data' / 'monitoring data.xlsx'
OUTPUT_ROOT = BASE_DIR / 'code' / 'chapter4' / 'outputs' / 'tables'

ACCEL_PERCENTILE = 0.90   # 在月速率层面剔除超过此分位的加速月
MONTH_WINDOW_DAYS = 30    # 月口径：30 天滚动累计增量
TRAIN_RATIO = 0.80


def compute_v0_for_target(target, disp_daily):
    daily_inc = disp_daily.diff().dropna()
    n_train = int(len(disp_daily) * TRAIN_RATIO)
    train_inc = daily_inc.iloc[: max(0, n_train - 1)]

    # 在训练期做 30 天滚动累计增量 → 月速率序列（每一天一个值）
    rolling_month = train_inc.rolling(
        window=MONTH_WINDOW_DAYS, min_periods=MONTH_WINDOW_DAYS).sum().dropna()

    # 月速率层面剔除加速事件
    cutoff = rolling_month.quantile(ACCEL_PERCENTILE)
    steady_month = rolling_month[rolling_month <= cutoff]

    v_bar = float(steady_month.mean())
    sigma = float(steady_month.std(ddof=1))
    v0 = 1.5 * v_bar + 2 * sigma

    return {
        'target': target,
        'method': 'docx V0 = 1.5*V_bar + 2*sigma, 月速率层面剔除加速月 (>90%分位)',
        'n_train_days': int(len(train_inc)),
        'n_monthly_samples': int(len(rolling_month)),
        'n_steady_months': int(len(steady_month)),
        'accel_cutoff_mm_per_month': float(cutoff),
        'v_bar_mm_per_month': v_bar,
        'sigma_mm_per_month': sigma,
        'v0_mm_per_month': float(v0),
        'v0_yellow_threshold': float(v0),
        'v0_orange_threshold': float(5 * v0),
        'v0_red_threshold': float(10 * v0),
    }


def main(target=None):
    """如果 target=None 则一次算三点；否则只算给定点（方便 run_all 按点调用）。

    注：V0 只跟目标点自身位移有关，分多次跑结果一致。为保持接口统一，这里接受 target。
    """
    print('=' * 60)
    print('模块 1b: 计算 V0 (各监测点各自一个)')
    print('=' * 60)

    data = pd.read_excel(DATA_FILE, sheet_name=0)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date').sort_index()

    all_targets = ['MJ9', 'MJ1', 'MJ3']
    targets = [target] if target in all_targets else all_targets
    col_map = {'MJ9': 'MJ9/mm', 'MJ1': 'MJ1/mm', 'MJ3': 'MJ3/mm'}

    summary_rows = []
    for tgt in targets:
        disp = data[col_map[tgt]].astype(float)
        result = compute_v0_for_target(tgt, disp)

        out_dir = OUTPUT_ROOT / 'intermediate_data' / tgt
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / 'v0.json'
        out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))

        print(f"\n[{tgt}] V0 = {result['v0_mm_per_month']:.3f} mm/M "
              f"(V̄={result['v_bar_mm_per_month']:.3f}, σ={result['sigma_mm_per_month']:.3f}, "
              f"剔除加速阈值={result['accel_cutoff_mm_per_month']:.3f} mm/M, "
              f"稳定月样本={result['n_steady_months']}/{result['n_monthly_samples']})")
        print(f"    黄色阈值={result['v0_yellow_threshold']:.3f}, "
              f"橙色阈值={result['v0_orange_threshold']:.3f}, "
              f"红色阈值={result['v0_red_threshold']:.3f} mm/M")
        print(f"    -> {out_file}")

        summary_rows.append({
            'target': tgt,
            'v_bar_mm_per_month': result['v_bar_mm_per_month'],
            'sigma_mm_per_month': result['sigma_mm_per_month'],
            'v0_mm_per_month': result['v0_mm_per_month'],
        })

    # 只有全量调用才写汇总表
    if target is None:
        stats_dir = OUTPUT_ROOT / 'statistics'
        stats_dir.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame(summary_rows)
        mean_v0 = summary['v0_mm_per_month'].mean()
        summary.loc[len(summary)] = {
            'target': '三点均值（参考）',
            'v_bar_mm_per_month': summary['v_bar_mm_per_month'].mean(),
            'sigma_mm_per_month': summary['sigma_mm_per_month'].mean(),
            'v0_mm_per_month': mean_v0,
        }
        summary_file = stats_dir / 'v0_summary.csv'
        summary.to_csv(summary_file, index=False)

        print('\n' + '=' * 60)
        print(f'三点 V0 均值（滑坡整体参考）= {mean_v0:.3f} mm/M')
        print(f'汇总表 -> {summary_file}')
        print('=' * 60)


if __name__ == '__main__':
    main()
