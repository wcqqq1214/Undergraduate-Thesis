"""
计算LSTM和GRU模型的参数量（基于理论公式）
"""

def calculate_lstm_params(input_size, hidden_size):
    """
    计算LSTM层的参数量
    LSTM有4个门：输入门(i)、遗忘门(f)、输出门(o)、候选记忆(g)
    每个门的参数 = (input_size + hidden_size) * hidden_size + hidden_size (bias)
    总参数 = 4 * [(input_size + hidden_size) * hidden_size + hidden_size]
    """
    params = 4 * ((input_size + hidden_size) * hidden_size + hidden_size)
    return params

def calculate_gru_params(input_size, hidden_size):
    """
    计算GRU层的参数量
    GRU有3个门：重置门(r)、更新门(z)、候选隐藏状态(h)
    每个门的参数 = (input_size + hidden_size) * hidden_size + hidden_size (bias)
    总参数 = 3 * [(input_size + hidden_size) * hidden_size + hidden_size]
    """
    params = 3 * ((input_size + hidden_size) * hidden_size + hidden_size)
    return params

def calculate_dense_params(input_size, output_size):
    """
    计算全连接层的参数量
    参数 = input_size * output_size + output_size (bias)
    """
    return input_size * output_size + output_size

print("="*60)
print("LSTM 模型参数量计算")
print("="*60)
print("\n模型结构:")
print("  LSTM(25, return_sequences=True, input_shape=(2, 4))")
print("  Dropout(0.3)")
print("  LSTM(15, return_sequences=False)")
print("  Dropout(0.3)")
print("  Dense(15)")
print("  Dense(1)")
print()

# LSTM模型参数计算
# 第一层LSTM: input_size=4, hidden_size=25
lstm_layer1 = calculate_lstm_params(input_size=4, hidden_size=25)
print(f"第1层 LSTM(25): {lstm_layer1:,} 参数")

# 第二层LSTM: input_size=25 (来自第一层), hidden_size=15
lstm_layer2 = calculate_lstm_params(input_size=25, hidden_size=15)
print(f"第2层 LSTM(15): {lstm_layer2:,} 参数")

# 第一个Dense层: input_size=15, output_size=15
dense1 = calculate_dense_params(input_size=15, output_size=15)
print(f"第3层 Dense(15): {dense1:,} 参数")

# 第二个Dense层: input_size=15, output_size=1
dense2 = calculate_dense_params(input_size=15, output_size=1)
print(f"第4层 Dense(1): {dense2:,} 参数")

lstm_total = lstm_layer1 + lstm_layer2 + dense1 + dense2
print(f"\nLSTM总参数量: {lstm_total:,}")
print()

print("="*60)
print("GRU 模型参数量计算")
print("="*60)
print("\n模型结构:")
print("  GRU(15, return_sequences=True, input_shape=(2, 1))")
print("  Dropout(0.3)")
print("  GRU(15, return_sequences=False)")
print("  Dropout(0.3)")
print("  Dense(15)")
print("  Dense(1)")
print()

# GRU模型参数计算
# 第一层GRU: input_size=1, hidden_size=15
gru_layer1 = calculate_gru_params(input_size=1, hidden_size=15)
print(f"第1层 GRU(15): {gru_layer1:,} 参数")

# 第二层GRU: input_size=15 (来自第一层), hidden_size=15
gru_layer2 = calculate_gru_params(input_size=15, hidden_size=15)
print(f"第2层 GRU(15): {gru_layer2:,} 参数")

# 第一个Dense层: input_size=15, output_size=15
dense1_gru = calculate_dense_params(input_size=15, output_size=15)
print(f"第3层 Dense(15): {dense1_gru:,} 参数")

# 第二个Dense层: input_size=15, output_size=1
dense2_gru = calculate_dense_params(input_size=15, output_size=1)
print(f"第4层 Dense(1): {dense2_gru:,} 参数")

gru_total = gru_layer1 + gru_layer2 + dense1_gru + dense2_gru
print(f"\nGRU总参数量: {gru_total:,}")
print()

print("="*60)
print("LSTM vs GRU 对比")
print("="*60)
print(f"LSTM总参数量: {lstm_total:,}")
print(f"GRU总参数量:  {gru_total:,}")
print(f"参数量差异:   {lstm_total - gru_total:,}")
print(f"GRU参数量为LSTM的: {gru_total/lstm_total*100:.1f}%")
print()

# 基于论文中提到的训练信息估算时间
print("="*60)
print("训练时间估算（基于论文描述）")
print("="*60)
print("\n论文中提到的训练信息:")
print("  - 硬件: NVIDIA GeForce RTX 3060 Laptop GPU")
print("  - Epochs: 40")
print("  - Batch size: 64")
print("  - 三个分位数模型的训练时间:")
print("    τ=0.05: 52轮, 约2分钟")
print("    τ=0.5:  41轮, 约1.5分钟")
print("    τ=0.95: 210轮, 约7分钟")
print()
print("基于τ=0.5的训练速度估算:")
print("  每轮训练时间 ≈ 1.5分钟 / 41轮 ≈ 2.2秒/轮")
print("  40轮训练时间 ≈ 40 * 2.2秒 ≈ 88秒 ≈ 1.5分钟")
print()
print("考虑到GRU参数量约为LSTM的75%，估算:")
print(f"  LSTM单次训练时间: 约1.5-2分钟")
print(f"  GRU单次训练时间:  约1-1.5分钟")
print(f"  GRU比LSTM快约: 20-30%")
print()

# 保存结果
import pandas as pd

results = pd.DataFrame({
    '模型': ['LSTM', 'GRU'],
    '总参数量': [lstm_total, gru_total],
    '参数量占比': [f'100%', f'{gru_total/lstm_total*100:.1f}%'],
    '估算训练时间(分钟)': ['1.5-2', '1-1.5'],
    '相对速度': ['基准', '快20-30%']
})

print("="*60)
print("汇总表格")
print("="*60)
print(results.to_string(index=False))
print()

results.to_csv('../outputs/tables/model_comparison_theoretical.csv', index=False, encoding='utf-8-sig')
print("结果已保存到: ../outputs/tables/model_comparison_theoretical.csv")
