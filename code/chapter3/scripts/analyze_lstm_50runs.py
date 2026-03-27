#!/usr/bin/env python3
"""
分析LSTM-50runs数据的脚本
读取result.xlsx中的LSTM-50runs sheet，计算真实的统计指标
"""

import pandas as pd
import numpy as np
import sys

def main():
    # 读取Excel文件
    excel_path = 'data/result.xlsx'

    print("=" * 80)
    print("LSTM-50runs 数据分析报告")
    print("=" * 80)

    # 1. 读取所有sheet名称
    excel_file = pd.ExcelFile(excel_path)
    print(f"\n文件中的所有sheet:")
    for i, sheet in enumerate(excel_file.sheet_names, 1):
        print(f"  {i}. {sheet}")

    # 2. 读取LSTM-50runs数据
    print(f"\n正在读取 'LSTM-50runs' sheet...")
    df = pd.read_excel(excel_path, sheet_name='LSTM-50runs', skiprows=1)

    print(f"数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"日期范围: {df['Date'].min()} 到 {df['Date'].max()}")

    # 3. 提取预测列
    prediction_cols = [col for col in df.columns if 'Predict' in col]
    print(f"\n预测列数量: {len(prediction_cols)}")

    if len(prediction_cols) > 50:
        print(f"警告: 发现 {len(prediction_cols)} 列预测数据，可能包含重复列")
        print(f"前10列: {prediction_cols[:10]}")
        print(f"后10列: {prediction_cols[-10:]}")

        # 只取前50列
        prediction_cols = prediction_cols[:50]
        print(f"\n使用前50列进行分析")

    # 4. 提取预测数据
    predictions = df[prediction_cols].values

    # 5. 计算统计量
    print("\n" + "=" * 80)
    print("统计分析结果")
    print("=" * 80)

    # 每天的平均预测值（50次运行的平均）
    daily_mean = np.mean(predictions, axis=1)

    # 每天的标准差（50次运行之间的标准差 - 这是模型不确定性）
    daily_std = np.std(predictions, axis=1, ddof=1)  # 使用样本标准差

    # 整体统计
    overall_mean = np.mean(daily_mean)
    mean_std = np.mean(daily_std)
    max_std = np.max(daily_std)
    min_std = np.min(daily_std)
    cv = (mean_std / overall_mean) * 100  # 变异系数

    print(f"\n【点预测统计】")
    print(f"  整体平均预测值: {overall_mean:.2f} mm")
    print(f"  预测值范围: {np.min(daily_mean):.2f} - {np.max(daily_mean):.2f} mm")
    print(f"  预测值中位数: {np.median(daily_mean):.2f} mm")

    print(f"\n【不确定性统计（50次运行之间的标准差）】")
    print(f"  平均标准差: {mean_std:.2f} mm")
    print(f"  最大标准差: {max_std:.2f} mm")
    print(f"  最小标准差: {min_std:.2f} mm")
    print(f"  变异系数: {cv:.2f}%")

    # 6. 对比监测点数据
    print("\n" + "=" * 80)
    print("监测点对比分析")
    print("=" * 80)

    monitoring_data = pd.read_excel('data/monitoring data.xlsx')

    monitoring_points = {}
    for col in monitoring_data.columns:
        if 'MJ' in str(col):
            monitoring_points[col] = monitoring_data[col].mean()

    print(f"\n各监测点的平均位移:")
    for point, mean_val in monitoring_points.items():
        diff = abs(overall_mean - mean_val)
        print(f"  {point}: {mean_val:.2f} mm (差异: {diff:.2f} mm)")

    # 判断预测的是哪个监测点
    closest_point = min(monitoring_points.items(), key=lambda x: abs(x[1] - overall_mean))
    print(f"\n结论: 预测平均值 {overall_mean:.2f} mm 最接近 {closest_point[0]} ({closest_point[1]:.2f} mm)")

    # 7. 详细分析：检查这是否真的是"50次运行"
    print("\n" + "=" * 80)
    print("数据性质分析")
    print("=" * 80)

    # 计算前5列之间的相关系数
    if len(prediction_cols) >= 5:
        corr_matrix = np.corrcoef(predictions[:, :5].T)
        avg_corr = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        print(f"\n前5列预测之间的平均相关系数: {avg_corr:.6f}")

        if avg_corr > 0.999:
            print("  → 相关性极高 (>0.999)，说明这是同一个模型的多次运行")
            print("  → 标准差应该很小（几个mm），反映模型的稳定性")
        else:
            print("  → 相关性较低，可能不是真正的'50次运行'")

    # 检查标准差的合理性
    print(f"\n标准差合理性检查:")
    if mean_std < 10:
        print(f"  ✓ 平均标准差 {mean_std:.2f} mm < 10 mm")
        print(f"    说明模型非常稳定，50次运行结果高度一致")
    elif mean_std < 50:
        print(f"  ⚠ 平均标准差 {mean_std:.2f} mm 在 10-50 mm 之间")
        print(f"    说明模型有一定波动，但仍在合理范围")
    else:
        print(f"  ✗ 平均标准差 {mean_std:.2f} mm > 50 mm")
        print(f"    这不太可能是'50次运行的标准差'")
        print(f"    可能是：")
        print(f"      1. 计算错误（混淆了时间维度和ensemble维度）")
        print(f"      2. 这150列不是真正的'50次运行'")
        print(f"      3. 数据包含了其他信息")

    # 8. 生成论文用的统计表
    print("\n" + "=" * 80)
    print("论文用统计表（表3-3）")
    print("=" * 80)

    print(f"\n统计量 & 数值 \\\\")
    print(f"\\midrule")
    print(f"平均预测值 (mm) & {overall_mean:.2f} \\\\")
    print(f"平均标准差 (mm) & {mean_std:.2f} \\\\")
    print(f"最大标准差 (mm) & {max_std:.2f} \\\\")
    print(f"最小标准差 (mm) & {min_std:.2f} \\\\")
    print(f"变异系数 (\\%) & {cv:.2f} \\\\")

    # 9. 保存详细结果到CSV
    output_df = pd.DataFrame({
        'Date': df['Date'],
        'Mean_Prediction': daily_mean,
        'Std_Prediction': daily_std,
        'Min_Prediction': np.min(predictions, axis=1),
        'Max_Prediction': np.max(predictions, axis=1),
        'Q25_Prediction': np.percentile(predictions, 25, axis=1),
        'Q75_Prediction': np.percentile(predictions, 75, axis=1),
    })

    output_path = 'code/chapter3/outputs/tables/lstm_50runs_statistics.csv'
    output_df.to_csv(output_path, index=False)
    print(f"\n详细统计结果已保存到: {output_path}")

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
