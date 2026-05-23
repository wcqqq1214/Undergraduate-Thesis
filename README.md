# 基于机器学习方法的水库滑坡位移预测及预警研究——以藕塘滑坡为例

本科毕业论文项目，研究三峡库区藕塘滑坡的位移预测与预警方法。

[English](README_EN.md) | **Download Thesis PDF**: [Wei_2026_ML_Reservoir_Landslide_Prediction.pdf](https://github.com/wcqqq1214/Undergraduate-Thesis/releases/latest/download/Wei_2026_ML_Reservoir_Landslide_Prediction.pdf)

## 研究目标

以三峡库区典型阶跃型滑坡——藕塘滑坡为研究对象，运用机器学习方法开展滑坡位移预测及预警研究，具体包括：

- 定量分析库水位、降雨、地下水位等诱发因子对滑坡位移的贡献度与非线性响应规律
- 构建机器学习与深度学习位移预测模型
- 建立基于预测位移速率的四级动态概率预警框架

## 数据集

数据集来源于藕塘滑坡现场长期监测记录，主要包括：

| 数据类型 | 说明 |
|---------|------|
| **地表位移** | 多个 GPS 监测点（MJ1、MJ3、MJ9 等）的累积位移（mm），涵盖缓慢蠕变与阶跃变形阶段 |
| **库水位** | 三峡水库逐日水位（m），每年在 145–175 m 之间周期性调度 |
| **降雨量** | 邻近气象站记录的逐日降雨量（mm） |
| **地下水位** | 滑坡体钻孔地下水位高程（m） |

监测时段约为 2007–2019 年，覆盖多个水文周期和数次典型阶跃加速事件。

## 研究方法

1. **多源特征工程与可解释性分析（LightGBM + SHAP）**：基于 LightGBM 构建梯度提升模型，以水文气象因子预测位移增量，同时构建分类模型进行预警状态判定；通过 SHAP 值量化各诱发因子的边际贡献，揭示库水位下降与累计降雨对位移加速的阈值型非线性驱动规律

2. **LSTM 时序位移预测与概率区间估计**：以多监测点空间运动学特征为输入，构建 LSTM 网络同时预测三个监测点的逐日位移；通过 50 次独立训练获得预测分布，构建 50%/90% 置信区间，量化模型认知不确定性

3. **基于 $V_0$ 阈值体系的四级动态概率预警**：以各监测点匀速变形段速率统计量 $V_0 = 1.5\bar{V} + 2\sigma$ 为基准，按 $V_0 / 5V_0 / 10V_0$ 递进设定黄/橙/红阈值；将 LSTM 50 次预测月速率分布映射至四级区间（绿/黄/橙/红），以最高概率等级作为当日预警等级

## 关键结果

| 指标 | 数值 |
|------|------|
| **LSTM 测试集 R²**（MJ1） | 0.7916 |
| **LSTM 测试集 RMSE**（MJ1） | 5.55 mm |
| **相对误差**（MJ1） | 0.56% |
| **特异性**（三个监测点） | 100% |
| **误报率**（三个监测点） | 0% |

- SHAP 分析表明库水位消落与累计降雨是阶跃变形的首要触发因子
- LSTM 集成模型在稳定期给出较窄置信区间，在阶跃期自适应展宽，预测分布合理
- 2017 年阶跃变形事件中，预警系统对 MJ1 触发 70 天黄色预警、MJ3 触发 129 天黄色预警，实现无漏报、零误报

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

## License

- **Code** (`/code`, `/scripts`): [MIT License](LICENSE)
- **Thesis Text & Figures** (`/docs`): [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)

## 注意事项

- 封面为外部 PDF（`thesis_cover_page.pdf`），通过 `\includepdf` 导入
- 字体文件已包含在 `fonts/` 目录中，编译时会自动加载
- 编译生成的中间文件（`.xdv`, `.aux`, `.log` 等）已添加到 `.gitignore`
- 仅保留最终的 PDF 文件在版本控制中
