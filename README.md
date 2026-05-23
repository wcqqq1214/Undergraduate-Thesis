# 基于机器学习方法的水库滑坡位移预测及预警研究——以藕塘滑坡为例

本科毕业论文项目，研究三峡库区藕塘滑坡的位移预测与预警方法。

## 研究内容

本研究以三峡库区典型阶跃型滑坡——藕塘滑坡为研究对象，运用机器学习方法开展滑坡位移预测及预警研究，主要包括：

1. **多源特征工程与可解释性分析**：基于 LightGBM 和 SHAP 方法，定量分析库水位、降雨、地下水位等诱发因子对滑坡位移的贡献度与非线性响应规律，揭示水文气象因素与滑坡变形的内在映射关系

2. **机器学习位移预测模型**：基于 LightGBM 构建回归模型进行位移增量预测，同时构建分类模型进行预警状态判定，通过时序交叉验证评估模型性能

3. **LSTM 时序位移预测与概率区间估计**：基于多监测点空间运动学输入构建 LSTM 时序预测模型，通过 50 次独立训练获得预测分布，构建 50%/90% 置信区间，量化模型认知不确定性；MJ1 测试集 R²=0.7916，RMSE=5.55 mm，相对误差 0.56%

4. **基于 $V_0$ 阈值体系的四级动态概率预警**：以各监测点匀速变形段速率统计量 $V_0 = 1.5\bar{V} + 2\sigma$ 为基准，按 $V_0/5V_0/10V_0$ 递进设定黄/橙/红阈值，将 LSTM 50 次预测月速率分布映射到四级区间（绿/黄/橙/红）直接统计概率并取最高者为当日预警等级；测试集三点均保持 100% 特异性、零误报，2017 年阶跃段演示中 MJ1 与 MJ3 分别触发 70 天与 129 天黄色预警

## 项目结构

```
├── docs/latex/                  # 论文 LaTeX 源文件
│   ├── main.tex                # 论文正文
│   ├── slides.tex              # 答辩幻灯片（beamer）
│   ├── references.bib          # 参考文献
│   ├── thesis_cover_page.pdf   # 封面（外部 PDF 导入）
│   ├── thesis_end_page.pdf     # 封底（外部 PDF 导入）
│   ├── fonts/                  # 字体文件
│   └── figures/                # 插图（按章节分目录）
├── code/                        # 实验代码
│   ├── chapter2/                # 第二章：LightGBM+SHAP 特征分析
│   ├── chapter3/                # 第三章：LSTM 位移预测与概率区间
│   └── chapter4/                # 第四章：V0 四级动态概率预警
├── data/                        # 原始监测数据
├── scripts/                     # 项目级脚本
├── .gitignore
├── LICENSE
└── README.md
```

## 技术栈

- **编程语言**：Python 3.x
- **深度学习框架**：Pytorch
- **机器学习库**：scikit-learn, LightGBM
- **可解释性分析**：SHAP
- **时序预测**：LSTM（长短期记忆网络）
- **论文排版**：LaTeX (XeTeX)

## 论文编译

使用 latexmk + XeLaTeX 编译：

```bash
cd docs/latex
latexmk -xelatex main.tex    # 论文正文
latexmk -xelatex slides.tex  # 答辩幻灯片 (beamer)
```

## 📄 License

- **Code** (`/code`, `/scripts`): [MIT License](LICENSE)
- **Thesis Text & Figures** (`/docs`): [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)

## 注意事项

- 封面为外部 PDF（`thesis_cover_page.pdf`），通过 `\includepdf` 导入
- 字体文件已包含在 `fonts/` 目录中，编译时会自动加载
- 编译生成的中间文件（`.xdv`, `.aux`, `.log` 等）已添加到 `.gitignore`
- 仅保留最终的 PDF 文件在版本控制中
