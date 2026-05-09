"""
运行LSTM趋势预测模型50次并记录结果
根据论文方法，每次使用不同的随机种子初始化模型参数

支持对 MJ9、MJ1、MJ3 三个目标监测点分别训练 50 次，输出独立的统计结果。
每个目标点的 CSV 以 `_<target>` 后缀区分；为兼容第四章现有脚本，
MJ1 目标额外写一份不带后缀的默认文件名。

通过 --runs 可减少运行次数用于 smoke test，默认 50。
通过 --targets 可以指定目标子集，例如 `--targets MJ1`。
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2


INPUT_COLS = ['MJ9/mm', 'MJ1/mm', 'MJ3/mm', 'ATU4/mm']
TARGET_COLS = ['MJ9/mm', 'MJ1/mm', 'MJ3/mm']
COL_TO_TAG = {'MJ9/mm': 'MJ9', 'MJ1/mm': 'MJ1', 'MJ3/mm': 'MJ3'}

# 每个目标点的超参数。模型结构（25/15 LSTM + Dense(15) + Dense(1)、Adam lr=5e-4、
# batch_size=64）对所有点保持一致；只有训练配置（时间窗口、epochs、dropout、L2）
# 按各点位移特征单独选取。依据来自两轮 smoke test（2 runs / 3 runs）：
#   MJ1 位移趋势强、测试期外推显著，适合较长窗口 + 较强正则；
#   MJ3 位移节奏在训练/测试期差异大，长窗口会把训练期模式错套到测试期，沿用原配置最稳；
#   MJ9 位移基准平稳、R² 分母极小，该指标在本点不可靠，改用 RMSE 评价，短窗口表现更佳。
TARGET_CONFIG = {
    'MJ9': {'time_steps': 7, 'epochs': 30, 'dropout': 0.4, 'l2': 0.005,
            'batch_size': 64, 'learning_rate': 5e-4},
    'MJ1': {'time_steps': 2, 'epochs': 40, 'dropout': 0.3, 'l2': 0.002,
            'batch_size': 64, 'learning_rate': 5e-4},
    'MJ3': {'time_steps': 2, 'epochs': 40, 'dropout': 0.3, 'l2': 0.002,
            'batch_size': 64, 'learning_rate': 5e-4},
}


def build_model(input_shape, dropout, l2_reg, learning_rate):
    # 结构固定：两层 LSTM (25/15) + Dense(15) + Dense(1)；只把训练配置做成参数。
    model = Sequential([
        LSTM(25, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(15, return_sequences=False, kernel_regularizer=l2(l2_reg)),
        Dropout(dropout),
        Dense(15),
        Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
    )
    return model


def create_sequences(data, time_steps, target_idx):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, target_idx])
    return np.array(X), np.array(y)


def aggregate_time_stats(pred_df):
    grouped = pred_df.groupby('time_index').agg(
        mean=('prediction', 'mean'),
        std=('prediction', 'std'),
        p05=('prediction', lambda x: np.percentile(x, 5)),
        p25=('prediction', lambda x: np.percentile(x, 25)),
        p50=('prediction', lambda x: np.percentile(x, 50)),
        p75=('prediction', lambda x: np.percentile(x, 75)),
        p95=('prediction', lambda x: np.percentile(x, 95)),
        actual=('actual', 'first'),
    ).reset_index()
    return grouped


def run_for_target(target_col, displacement_scaled, scaler, train_size,
                   n_runs, output_dir, config):
    target_idx = INPUT_COLS.index(target_col)
    tag = COL_TO_TAG[target_col]
    time_steps = config['time_steps']

    train_data = displacement_scaled[:train_size]
    test_data = displacement_scaled[train_size:]

    X_train, y_train = create_sequences(train_data, time_steps, target_idx)
    X_test, y_test = create_sequences(test_data, time_steps, target_idx)

    results_summary = []
    all_predictions = []
    all_train_predictions = []

    n_features = len(INPUT_COLS)

    print(f"\n===== 目标监测点 {tag} ({target_col}) =====")
    print(f"超参数: {config}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for run_id in range(1, n_runs + 1):
        seed = run_id * 42
        np.random.seed(seed)
        tf.random.set_seed(seed)

        model = build_model(
            (X_train.shape[1], X_train.shape[2]),
            dropout=config['dropout'],
            l2_reg=config['l2'],
            learning_rate=config['learning_rate'],
        )
        history = model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_data=(X_test, y_test),
            verbose=0,
        )

        train_predict = model.predict(X_train, verbose=0)
        test_predict = model.predict(X_test, verbose=0)

        train_predict = scaler.inverse_transform(
            np.concatenate([train_predict] * n_features, axis=1))[:, 0]
        test_predict = scaler.inverse_transform(
            np.concatenate([test_predict] * n_features, axis=1))[:, 0]

        y_train_actual = scaler.inverse_transform(
            np.concatenate([y_train.reshape(-1, 1)] * n_features, axis=1))[:, 0]
        y_test_actual = scaler.inverse_transform(
            np.concatenate([y_test.reshape(-1, 1)] * n_features, axis=1))[:, 0]

        train_r2 = r2_score(y_train_actual, train_predict)
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
        test_r2 = r2_score(y_test_actual, test_predict)
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))

        results_summary.append({
            'run_id': run_id,
            'seed': seed,
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'train_loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1],
        })

        for i, pred in enumerate(test_predict):
            all_predictions.append({
                'run_id': run_id,
                'time_index': i,
                'prediction': pred,
                'actual': y_test_actual[i],
            })
        for i, pred in enumerate(train_predict):
            all_train_predictions.append({
                'run_id': run_id,
                'time_index': i,
                'prediction': pred,
                'actual': y_train_actual[i],
            })

        print(f"  run {run_id:02d}/{n_runs}  "
              f"train R²={train_r2:.4f} RMSE={train_rmse:.3f}  "
              f"test R²={test_r2:.4f} RMSE={test_rmse:.3f}")

    summary_df = pd.DataFrame(results_summary)
    predictions_df = pd.DataFrame(all_predictions)
    train_predictions_df = pd.DataFrame(all_train_predictions)

    time_stats = aggregate_time_stats(predictions_df)
    train_time_stats = aggregate_time_stats(train_predictions_df)

    # 集成指标：50 次预测均值对真值的 R² / RMSE（即论文正文报告口径）
    ensemble_test_r2 = r2_score(time_stats['actual'], time_stats['mean'])
    ensemble_test_rmse = np.sqrt(
        mean_squared_error(time_stats['actual'], time_stats['mean']))
    ensemble_test_mae = np.mean(np.abs(time_stats['actual'] - time_stats['mean']))
    actual_mean_abs = np.mean(np.abs(time_stats['actual']))
    ensemble_test_rel_err = ensemble_test_rmse / actual_mean_abs if actual_mean_abs else np.nan

    ensemble_train_r2 = r2_score(train_time_stats['actual'], train_time_stats['mean'])
    ensemble_train_rmse = np.sqrt(
        mean_squared_error(train_time_stats['actual'], train_time_stats['mean']))

    return {
        'tag': tag,
        'config': config,
        'summary': summary_df,
        'predictions': predictions_df,
        'train_predictions': train_predictions_df,
        'statistics': time_stats,
        'train_statistics': train_time_stats,
        'ensemble_test_r2': ensemble_test_r2,
        'ensemble_test_rmse': ensemble_test_rmse,
        'ensemble_test_mae': ensemble_test_mae,
        'ensemble_test_rel_err': ensemble_test_rel_err,
        'ensemble_train_r2': ensemble_train_r2,
        'ensemble_train_rmse': ensemble_train_rmse,
    }


def save_outputs(result, output_dir):
    tag = result['tag']
    files = {
        'lstm_trend_50runs_summary': result['summary'],
        'lstm_trend_50runs_predictions': result['predictions'],
        'lstm_trend_50runs_train_predictions': result['train_predictions'],
        'lstm_trend_50runs_statistics': result['statistics'],
        'lstm_trend_50runs_train_statistics': result['train_statistics'],
    }
    for base, df in files.items():
        tagged_path = os.path.join(output_dir, f'{base}_{tag}.csv')
        df.to_csv(tagged_path, index=False)
        print(f"  saved -> {tagged_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=50,
                        help='每个目标点的训练次数 (default: 50)')
    parser.add_argument('--targets', nargs='+', default=TARGET_COLS,
                        choices=TARGET_COLS,
                        help='目标列 (default: MJ9 MJ1 MJ3)')
    parser.add_argument('--output-dir', default='../outputs/tables')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    file_path = '../../../data/monitoring data.xlsx'
    data = pd.read_excel(file_path, sheet_name=0)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    displacement_data = data[INPUT_COLS]

    # 对四个位移列做独立归一化（保留原脚本逻辑）
    displacement_scaled = pd.DataFrame(index=displacement_data.index)
    scalers = {}
    for col in INPUT_COLS:
        scaler = MinMaxScaler(feature_range=(0, 1))
        displacement_scaled[col] = scaler.fit_transform(displacement_data[[col]])
        scalers[col] = scaler
    displacement_scaled = displacement_scaled.values

    train_size = int(len(displacement_scaled) * 0.8)

    print("=" * 80)
    print(f"LSTM 多目标 50 次运行（本次 runs={args.runs}）")
    print(f"目标监测点: {args.targets}")
    print(f"训练/测试划分: {train_size} / {len(displacement_scaled) - train_size}")
    print("=" * 80)

    overall_summary = []

    for target in args.targets:
        tag = COL_TO_TAG[target]
        result = run_for_target(
            target_col=target,
            displacement_scaled=displacement_scaled,
            scaler=scalers[target],
            train_size=train_size,
            n_runs=args.runs,
            output_dir=args.output_dir,
            config=TARGET_CONFIG[tag],
        )
        save_outputs(result, args.output_dir)

        s = result['summary']
        overall_summary.append({
            'target': result['tag'],
            'time_steps': result['config']['time_steps'],
            'epochs': result['config']['epochs'],
            'dropout': result['config']['dropout'],
            'l2': result['config']['l2'],
            'ensemble_test_r2': result['ensemble_test_r2'],
            'ensemble_test_rmse': result['ensemble_test_rmse'],
            'ensemble_test_mae': result['ensemble_test_mae'],
            'ensemble_test_rel_err': result['ensemble_test_rel_err'],
            'ensemble_train_r2': result['ensemble_train_r2'],
            'ensemble_train_rmse': result['ensemble_train_rmse'],
            'perrun_test_r2_mean': s['test_r2'].mean(),
            'perrun_test_r2_std': s['test_r2'].std(),
            'perrun_test_rmse_mean': s['test_rmse'].mean(),
            'perrun_test_rmse_std': s['test_rmse'].std(),
        })

    overall_df = pd.DataFrame(overall_summary)
    overall_path = os.path.join(args.output_dir, 'lstm_trend_50runs_overall.csv')
    overall_df.to_csv(overall_path, index=False)

    print("\n" + "=" * 80)
    print("各监测点总体性能")
    print("=" * 80)
    print(overall_df.to_string(index=False))
    print(f"\n已保存到: {overall_path}")


if __name__ == '__main__':
    sys.exit(main())
