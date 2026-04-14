# 基于机器学习方法的水库滑坡位移预测及预警研究——以藕塘滑坡为例

本科毕业论文项目，研究三峡库区藕塘滑坡的位移预测与预警方法。

**作者**：韦承谦  
**学号**：2203110625  
**学院**：土木建筑工程学院  
**专业**：土木工程  
**班级**：岩土与地下工程222班  
**指导老师**：陈铭熙  
**完成时间**：2026年6月

## 研究内容

本研究以三峡库区典型阶跃型滑坡——藕塘滑坡为研究对象，运用机器学习方法开展滑坡位移预测及预警研究，主要包括：

1. **多源特征工程与可解释性分析**：基于 LightGBM 和 SHAP 方法，定量分析库水位、降雨、地下水位等诱发因子对滑坡位移的贡献度与非线性响应规律，揭示水文气象因素与滑坡变形的内在映射关系

2. **机器学习位移预测模型**：基于 LightGBM 构建回归模型进行位移增量预测，同时构建分类模型进行预警状态判定，通过时序交叉验证评估模型性能

3. **概率区间预测与分级预警体系**：通过多次独立运行策略构建概率预测区间，量化模型不确定性，建立基于越限概率的五级预警体系（绿-蓝-黄-橙-红），为防灾减灾提供科学决策依据

## 项目结构

```
├── docs/latex/              # 论文 LaTeX 源文件
│   ├── main.tex            # 主文档
│   ├── gxufrontmatter.tex   # 封面和前置页面模板
│   ├── gxuthesis.cls        # 论文类文件
│   ├── gxulogo.png          # 学校logo
│   └── fonts/               # 字体文件
│       ├── simsun.ttc       # 宋体
│       ├── simhei.ttf       # 黑体
│       ├── simli.ttf        # 隶书
│       ├── fangsong.ttf     # 仿宋
│       ├── times.ttf        # Times New Roman
│       ├── timesbd.ttf      # Times New Roman Bold
│       ├── timesi.ttf       # Times New Roman Italic
│       └── timesbi.ttf      # Times New Roman Bold Italic
├── code/                    # 实验代码
│   ├── chapter2/            # 第二章：特征提取与贡献度分析
│   ├── chapter3/            # 第三章：LSTM 位移预测
│   └── chapter4/            # 第四章：概率预测与预警
└── README.md
```

## 技术栈

- **编程语言**：Python 3.x
- **深度学习框架**：PyTorch
- **机器学习库**：scikit-learn, LightGBM
- **可解释性分析**：SHAP
- **时序预测**：LSTM (长短期记忆网络)
- **论文排版**：LaTeX (XeTeX)

## 论文编译

使用 XeTeX 编译论文：

```bash
cd docs/latex
xelatex -interaction=nonstopmode main.tex
```

## 注意事项

- 字体文件已包含在 `fonts/` 目录中，编译时会自动加载
- 编译生成的中间文件（`.xdv`, `.aux`, `.log` 等）已添加到 `.gitignore`
- 仅保留最终的 PDF 文件在版本控制中