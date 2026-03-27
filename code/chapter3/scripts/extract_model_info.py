"""
提取LSTM和GRU模型的参数量和训练时间信息
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import time

# 读取数据
file_path = '../../../data/monitoring data.xlsx'
data = pd.read_excel(file_path, sheet_name=2)
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
        y.append(data[i + time_steps, 3])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)

print("="*60)
print("数据集信息")
print("="*60)
print(f"训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")
print(f"输入形状: {X_train.shape}")
print()

# ============ LSTM 模型 ============
print("="*60)
print("LSTM 模型信息")
print("="*60)

lstm_model = Sequential([
    LSTM(25, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(15, return_sequences=False, kernel_regularizer=l2(0.002)),
    Dropout(0.3),
    Dense(15),
    Dense(1)
])

lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                   loss='mean_squared_error')

# 打印模型结构
lstm_model.summary()

# 计算参数量
lstm_params = lstm_model.count_params()
print(f"\nLSTM总参数量: {lstm_params:,}")
print()

# 测量训练时间
print("开始训练LSTM模型（40 epochs）...")
start_time = time.time()
lstm_history = lstm_model.fit(X_train, y_train,
                               epochs=40,
                               batch_size=64,
                               validation_data=(X_test, y_test),
                               verbose=0)
lstm_train_time = time.time() - start_time

print(f"LSTM训练总时间: {lstm_train_time:.2f} 秒 ({lstm_train_time/60:.2f} 分钟)")
print(f"LSTM平均每epoch时间: {lstm_train_time/40:.2f} 秒")
print()

# ============ GRU 模型 ============
print("="*60)
print("GRU 模型信息")
print("="*60)

gru_model = Sequential([
    GRU(15, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),
        kernel_regularizer=l2(0.002)),
    Dropout(0.3),
    GRU(15, return_sequences=False, kernel_regularizer=l2(0.002)),
    Dropout(0.3),
    Dense(15),
    Dense(1)
])

gru_model.compile(optimizer='adam', loss='mean_squared_error')

# 打印模型结构
gru_model.summary()

# 计算参数量
gru_params = gru_model.count_params()
print(f"\nGRU总参数量: {gru_params:,}")
print()

# 测量训练时间
print("开始训练GRU模型（40 epochs）...")
start_time = time.time()
gru_history = gru_model.fit(X_train, y_train,
                             epochs=40,
                             batch_size=64,
                             validation_data=(X_test, y_test),
                             verbose=0)
gru_train_time = time.time() - start_time

print(f"GRU训练总时间: {gru_train_time:.2f} 秒 ({gru_train_time/60:.2f} 分钟)")
print(f"GRU平均每epoch时间: {gru_train_time/40:.2f} 秒")
print()

# ============ 对比分析 ============
print("="*60)
print("LSTM vs GRU 对比")
print("="*60)
print(f"参数量对比:")
print(f"  LSTM: {lstm_params:,} 参数")
print(f"  GRU:  {gru_params:,} 参数")
print(f"  GRU参数量为LSTM的: {gru_params/lstm_params*100:.1f}%")
print()

print(f"训练时间对比:")
print(f"  LSTM: {lstm_train_time:.2f} 秒 ({lstm_train_time/60:.2f} 分钟)")
print(f"  GRU:  {gru_train_time:.2f} 秒 ({gru_train_time/60:.2f} 分钟)")
print(f"  GRU比LSTM快: {(1-gru_train_time/lstm_train_time)*100:.1f}%")
print()

# 保存结果到CSV
results = pd.DataFrame({
    '模型': ['LSTM', 'GRU'],
    '总参数量': [lstm_params, gru_params],
    '训练时间(秒)': [lstm_train_time, gru_train_time],
    '训练时间(分钟)': [lstm_train_time/60, gru_train_time/60],
    '平均每epoch时间(秒)': [lstm_train_time/40, gru_train_time/40]
})

results.to_csv('../outputs/tables/model_comparison.csv', index=False, encoding='utf-8-sig')
print("结果已保存到: ../outputs/tables/model_comparison.csv")
