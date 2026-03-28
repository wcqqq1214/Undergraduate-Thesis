"""
主运行脚本：按顺序执行所有模块
"""

import sys
from pathlib import Path

# 添加src目录到路径
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

def main():
    print("\n" + "="*70)
    print(" "*20 + "第四章代码执行")
    print("="*70)

    modules = [
        ("01_calculate_exceed_probability", "计算越限概率"),
        ("02_determine_warning_levels", "确定预警等级"),
        ("03_traditional_velocity_warning", "传统速率预警"),
        ("04_evaluate_performance", "性能评估"),
        ("05_calculate_lead_time", "计算预警提前时间")
    ]

    for i, (module_name, description) in enumerate(modules, 1):
        print(f"\n{'='*70}")
        print(f"步骤 {i}/{len(modules)}: {description}")
        print(f"模块: {module_name}")
        print(f"{'='*70}\n")

        try:
            # 动态导入并执行模块
            module = __import__(module_name)
            module.main()
            print(f"\n✓ 步骤 {i} 完成")
        except Exception as e:
            print(f"\n✗ 步骤 {i} 失败: {str(e)}")
            import traceback
            traceback.print_exc()

            # 询问是否继续
            response = input("\n是否继续执行下一步？(y/n): ")
            if response.lower() != 'y':
                print("执行中止")
                return

    print("\n" + "="*70)
    print(" "*25 + "全部完成！")
    print("="*70)
    print("\n生成的文件位置:")
    print("  - 论文表格: code/chapter4/outputs/tables/paper_tables/")
    print("  - 中间数据: code/chapter4/outputs/tables/intermediate_data/")
    print("  - 统计信息: code/chapter4/outputs/tables/statistics/")
    print("  - 图表: code/chapter4/outputs/figures/ (需要运行绘图脚本)")
    print("\n下一步:")
    print("  1. 检查论文表格: paper_tables/")
    print("  2. 运行绘图脚本生成图4-1至图4-4")
    print("  3. 将图表插入LaTeX论文")


if __name__ == "__main__":
    main()
