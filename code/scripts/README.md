# 项目级脚本 (Scripts)

本目录包含项目级别的辅助脚本，用于数据检查、分析和维护。

## 文件说明

### read_result.py
查看 `result.xlsx` 文件内容的脚本。

**功能：**
- 显示所有 sheet 的名称和数量
- 显示每个 sheet 的行列数和列名
- 显示前5行数据预览
- 显示数据统计信息

**使用方法：**
```bash
cd code/scripts
python3 read_result.py
```

**输出示例：**
```
文件路径: /home/wcqqq21/Undergraduate-Thesis/data/result.xlsx
Sheet数量: 6
Sheet名称: ['ST-GNN', 'ST-GNN置信区间', 'ST-GNN无注意力', 'ST-GNN-50runs', 'LSTM-50runs', 'GRU-50runs']

【Sheet: LSTM-50runs】
行数: 1458, 列数: 53
列名: ['Date', 'Predict1/mm', 'Predict2/mm', ...]
...
```

## 脚本分类

此目录用于存放：
- **数据检查脚本**：查看、验证数据文件
- **批处理脚本**：批量处理多个文件或任务
- **维护脚本**：项目维护和管理工具

## 与 chapter 脚本的区别

- **code/scripts/**：项目级脚本，通用性强，不特定于某个章节
- **code/chapterX/scripts/**：章节专用脚本，仅用于该章节的分析

## 添加新脚本

添加新的项目级脚本时：
1. 确保脚本具有通用性，不局限于某个章节
2. 添加清晰的文档字符串和使用说明
3. 使用相对路径或 `Path(__file__)` 来定位文件
4. 在本 README 中添加说明
