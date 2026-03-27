#!/usr/bin/env python3
"""
将PNG图片转换为PDF格式
"""
from PIL import Image
from pathlib import Path

# 设置路径
FIGURES_DIR = Path(__file__).parent.parent / "outputs" / "figures"

# 要转换的PNG文件
png_files = [
    "lstm_prediction.png",
    "lstm_uncertainty.png"
]

print("=" * 60)
print("PNG转PDF转换工具")
print("=" * 60)
print()

for png_file in png_files:
    png_path = FIGURES_DIR / png_file
    pdf_path = FIGURES_DIR / png_file.replace('.png', '.pdf')

    if not png_path.exists():
        print(f"⚠ 文件不存在: {png_file}")
        continue

    try:
        # 打开PNG图片
        image = Image.open(png_path)

        # 转换为RGB模式（PDF需要）
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 保存为PDF
        image.save(pdf_path, 'PDF', resolution=300.0)

        print(f"✓ {png_file} -> {pdf_path.name}")
        print(f"  大小: {pdf_path.stat().st_size / 1024:.1f} KB")

    except Exception as e:
        print(f"✗ 转换失败 {png_file}: {e}")

print()
print("=" * 60)
print("转换完成！")
print(f"输出目录: {FIGURES_DIR}")
print("=" * 60)
