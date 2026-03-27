"""
运行LSTM趋势预测模型50次并记录结果
根据论文方法，每次使用不同的随机种子初始化模型参数
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import os
from datetime import datetime

# 设置输出路径
output_dir = '../outputs/tables'
os.makedirs(output_dir, exist_ok=True)

# 读取数据
file_path = '../../../data/monitoring data.xlsx'
data = pd.read_excel(file_path, sheet_name=0)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 仅选择位移数据
displacement_cols = data.columns[0:4]
displacement_data = data[displacement_cols]

# 归一化
scalers = {}
displacement_scaled = pd.DataFrame(index=displacement_data.index)

for col in displacement_cols:
    scaler = MinMaxScaler(feature_range=(0, 1))
    displacement_scaled[col] = scaler.fit_transform(displacement_data[[col]])
    scalers[col] = scaler

displacement_scaled = displacement_scaled.values

# 划分训练集和测试集
train_size = int(len(displacement_scaled) * 0.8)
train_data = displacement_scaled[:train_size]
test_data = displacement_scaled[train_size:]

# 定义时间步长
time_steps = 2

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, 1])  # MJ1在第2列，索引为1
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)

# 存储50次运行的结果
results_summary = []
all_predictions = []

print("开始运行LSTM模型50次...")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 80)

for run_id in range(1, 51):
    print(f"\n运行 {run_id}/50...")

    # 设置随机种子
    seed = run_id * 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 构建LSTM模型
    model = Sequential([
        LSTM(25, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(15, return_sequences=False, kernel_regularizer=l2(0.002)),
        Dropout(0.3),
        Dense(15),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='mean_squared_error')

    # 训练模型（关闭输出）
    history = model.fit(X_train, y_train,
                       epochs=40,
                       batch_size=64,
                       validation_data=(X_test, y_test),
                       verbose=0)

    # 进行预测
    train_predict = model.predict(X_train, verbose=0)
    test_predict = model.predict(X_test, verbose=0)

    # 将预测结果逆归一化
    train_predict = scalers[displacement_cols[1]].inverse_transform(
        np.concatenate([train_predict] * len(displacement_cols), axis=1))[:, 0]
    test_predict = scalers[displacement_cols[1]].inverse_transform(
        np.concatenate([test_predict] * len(displacement_cols), axis=1))[:, 0]

    # 将真实值逆归一化
    y_train_actual = scalers[displacement_cols[1]].inverse_transform(
        np.concatenate([y_train.reshape(-1, 1)] * len(displacement_cols), axis=1))[:, 0]
    y_test_actual = scalers[displacement_cols[1]].inverse_transform(
        np.concatenate([y_test.reshape(-1, 1)] * len(displacement_cols), axis=1))[:, 0]

    # 计算评价指标
    train_r2 = r2_score(y_train_actual, train_predict)
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
    test_r2 = r2_score(y_test_actual, test_predict)
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))

    # 记录本次运行的汇总结果
    results_summary.append({
        'run_id': run_id,
        'seed': seed,
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'train_loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1]
    })

    # 记录本次运行的所有预测值
    for i, pred in enumerate(test_predict):
        all_predictions.append({
            'run_id': run_id,
            'time_index': i,
            'prediction': pred,
            'actual': y_test_actual[i]
        })

    print(f"  训练集 R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
    print(f"  测试集 R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")

print("\n" + "=" * 80)
print("所有运行完成！")
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 保存汇总结果
summary_df = pd.DataFrame(results_summary)
summary_output_path = os.path.join(output_dir, 'lstm_trend_50runs_summary.csv')
summary_df.to_csv(summary_output_path, index=False)
print(f"\n汇总结果已保存到: {summary_output_path}")

# 保存所有预测值
predictions_df = pd.DataFrame(all_predictions)
predictions_output_path = os.path.join(output_dir, 'lstm_trend_50runs_predictions.csv')
predictions_df.to_csv(predictions_output_path, index=False)
print(f"所有预测值已保存到: {predictions_output_path}")

# 计算统计量
print("\n" + "=" * 80)
print("统计结果:")
print("-" * 80)
print(f"测试集 R² - 均值: {summary_df['test_r2'].mean():.4f}, 标准差: {summary_df['test_r2'].std():.4f}")
print(f"测试集 RMSE - 均值: {summary_df['test_rmse'].mean():.4f}, 标准差: {summary_df['test_rmse'].std():.4f}")
print(f"训练集 R² - 均值: {summary_df['train_r2'].mean():.4f}, 标准差: {summary_df['train_r2'].std():.4f}")
print(f"训练集 RMSE - 均值: {summary_df['train_rmse'].mean():.4f}, 标准差: {summary_df['train_rmse'].std():.4f}")

# 计算每个时间点的统计量
time_stats = predictions_df.groupby('time_index').agg({
    'prediction': ['mean', 'std', lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 25),
                   lambda x: np.percentile(x, 50), lambda x: np.percentile(x, 75), lambda x: np.percentile(x, 95)],
    'actual': 'first'
}).reset_index()

time_stats.columns = ['time_index', 'mean', 'std', 'p05', 'p25', 'p50', 'p75', 'p95', 'actual']
time_stats_output_path = os.path.join(output_dir, 'lstm_trend_50runs_statistics.csv')
time_stats.to_csv(time_stats_output_path, index=False)
print(f"\n时间序列统计量已保存到: {time_stats_output_path}")

print("\n完成！")
