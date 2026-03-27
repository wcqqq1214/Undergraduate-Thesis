"""
正确计算LSTM模型的参数量
基于实际的notebook配置
"""

def calculate_lstm_params(input_size, hidden_size):
    """LSTM参数 = 4 * [(input_size + hidden_size) * hidden_size + hidden_size]"""
    params = 4 * ((input_size + hidden_size) * hidden_size + hidden_size)
    return params

def calculate_dense_params(input_size, output_size):
    """Dense参数 = input_size * output_size + output_size"""
    return input_size * output_size + output_size

print("="*70)
print("LSTM 模型参数量计算（趋势项预测）")
print("="*70)
print("\n实际模型结构（来自LSTM-trend-submit.ipynb）:")
print("  输入: (time_steps=2, features=4)  # 4个位移监测点")
print("  LSTM(25, return_sequences=True)")
print("  Dropout(0.3)")
print("  LSTM(15, return_sequences=False, kernel_regularizer=l2(0.002))")
print("  Dropout(0.3)")
print("  Dense(15)")
print("  Dense(1)")
print()

# LSTM模型参数计算
lstm_layer1 = calculate_lstm_params(input_size=4, hidden_size=25)
print(f"第1层 LSTM(25, input=4):  {lstm_layer1:,} 参数")
print(f"  计算: 4 * [(4+25)*25 + 25] = 4 * [725 + 25] = {lstm_layer1}")

lstm_layer2 = calculate_lstm_params(input_size=25, hidden_size=15)
print(f"第2层 LSTM(15, input=25): {lstm_layer2:,} 参数")
print(f"  计算: 4 * [(25+15)*15 + 15] = 4 * [600 + 15] = {lstm_layer2}")

dense1 = calculate_dense_params(input_size=15, output_size=15)
print(f"第3层 Dense(15, input=15): {dense1:,} 参数")
print(f"  计算: 15*15 + 15 = {dense1}")

dense2 = calculate_dense_params(input_size=15, output_size=1)
print(f"第4层 Dense(1, input=15):  {dense2:,} 参数")
print(f"  计算: 15*1 + 1 = {dense2}")

lstm_total = lstm_layer1 + lstm_layer2 + dense1 + dense2
print(f"\n{'='*50}")
print(f"LSTM总参数量: {lstm_total:,}")
print(f"{'='*50}")
print()

print("="*70)
print("训练时间估算")
print("="*70)
print("\n基于论文第636行的训练信息:")
print("  硬件: NVIDIA GeForce RTX 3060 Laptop GPU")
print("  三个分位数模型的实际训练时间:")
print("    τ=0.05: 52轮, 约2分钟   → 每轮2.3秒")
print("    τ=0.5:  41轮, 约1.5分钟 → 每轮2.2秒")
print("    τ=0.95: 210轮, 约7分钟  → 每轮2.0秒")
print()
print("  平均训练速度: 约2.2秒/轮")
print()
print("基于此估算40轮训练时间:")
print(f"  LSTM (40轮): 40 × 2.2秒 ≈ 88秒 ≈ 1.5分钟")
print()

# 保存结果
import pandas as pd

results = pd.DataFrame({
    '模型': ['LSTM'],
    '第1层参数': [f'LSTM(25): {lstm_layer1:,}'],
    '第2层参数': [f'LSTM(15): {lstm_layer2:,}'],
    'Dense层参数': [f'{dense1+dense2:,}'],
    '总参数量': [f'{lstm_total:,}'],
    '估算训练时间': ['约1.5分钟'],
})

print("="*70)
print("汇总表格")
print("="*70)
print(results.to_string(index=False))
print()

results.to_csv('../outputs/tables/model_params_correct.csv', index=False, encoding='utf-8-sig')
print("✓ 结果已保存到: ../outputs/tables/model_params_correct.csv")

# 生成论文用的简洁表格
paper_table = pd.DataFrame({
    '模型': ['LSTM'],
    '总参数量': [lstm_total],
    '单次训练时间(分钟)': [1.5],
})

print("\n" + "="*70)
print("论文用表格")
print("="*70)
print(paper_table.to_string(index=False))
print()

paper_table.to_csv('../outputs/tables/model_comparison_for_paper.csv', index=False, encoding='utf-8-sig')
print("✓ 论文用表格已保存到: ../outputs/tables/model_comparison_for_paper.csv")
