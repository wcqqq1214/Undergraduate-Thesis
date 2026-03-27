# 工具函数模块 (Utils)

本目录包含项目中可复用的通用工具函数。

## 文件说明

### read_monitoring_data.py
读取监测数据的工具函数。

**功能：**
- 从 `data/monitoring data.xlsx` 读取监测数据
- 返回完整的 pandas DataFrame
- 自动处理项目路径

**使用示例：**
```python
from code.utils.read_monitoring_data import load_monitoring_data

# 读取监测数据
df = load_monitoring_data()
print(f"数据形状: {df.shape}")
```

## 使用说明

这些工具函数可以在项目的任何地方导入使用，例如：
- 在 chapter2、chapter3 的脚本中
- 在 Jupyter notebooks 中
- 在其他分析脚本中

## 添加新工具

如果需要添加新的通用工具函数：
1. 在此目录创建新的 `.py` 文件
2. 编写清晰的文档字符串
3. 确保函数具有通用性，可在多个章节中复用
