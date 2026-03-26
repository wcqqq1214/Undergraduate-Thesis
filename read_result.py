#!/usr/bin/env python3
"""
读取 result.xlsx 文件的内容
"""
import pandas as pd
import sys

def read_excel_file(file_path):
    """读取Excel文件并显示所有sheet的内容"""
    try:
        # 读取Excel文件
        xls = pd.ExcelFile(file_path)

        print(f"文件路径: {file_path}")
        print(f"Sheet数量: {len(xls.sheet_names)}")
        print(f"Sheet名称: {xls.sheet_names}\n")
        print("=" * 80)

        # 遍历每个sheet
        for sheet_name in xls.sheet_names:
            print(f"\n【Sheet: {sheet_name}】")
            print("-" * 80)

            # 读取sheet数据
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # 显示基本信息
            print(f"行数: {len(df)}, 列数: {len(df.columns)}")
            print(f"列名: {list(df.columns)}")

            # 显示前几行数据
            print("\n前5行数据:")
            print(df.head())

            # 显示数据统计信息
            print("\n数据统计:")
            print(df.describe())

            print("\n" + "=" * 80)

        return xls

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    file_path = "/home/wcqqq21/Undergraduate-Thesis/data/result.xlsx"
    read_excel_file(file_path)
