"""
第三章配置文件
包含所有可调参数、路径配置和环境设置
"""
from pathlib import Path
import torch
import numpy as np
import random

# ==================== 路径配置 ====================
# 获取当前文件所在目录的绝对路径
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # my_code/
# 修复：数据文件实际在chapter2目录下
DATA_PATH = PROJECT_ROOT / 'chapter2' / 'monitoring data.xlsx'

# 输出目录
OUTPUT_DIR = BASE_DIR / 'outputs'
FIGURE_DIR = OUTPUT_DIR / 'figures'
TABLE_DIR = OUTPUT_DIR / 'tables'
MODEL_DIR = OUTPUT_DIR / 'models'

# 自动创建输出目录
for dir_path in [OUTPUT_DIR, FIGURE_DIR, TABLE_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== 随机种子（可复现性） ====================
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """设置全局随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保CUDA的确定性行为（可能略微降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== 硬件设备配置 ====================
# 自动检测可用设备（CUDA > MPS > CPU）
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"使用GPU加速: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("使用Apple Silicon MPS加速")
else:
    DEVICE = torch.device('cpu')
    print("使用CPU训练（建议使用GPU加速）")


# ==================== 数据列名映射 ====================
# 根据 monitoring data.xlsx 的实际列名定义
COL_DATE = 'Date'
COL_RAINFALL = 'Rainfall/mm'
COL_GWT = 'GWT/m'  # 地下水位
COL_RWL = 'RWL/m'  # 库水位
COL_AVE_TEMP = 'aveT/℃'
COL_MIN_TEMP = 'minT/℃'
COL_MAX_TEMP = 'maxT/℃'
COL_DEW_POINT = 'DP'
COL_HUMIDITY = 'RH'

# 监测点位移列名（动态生成）
def get_displacement_col(point_name):
    """获取指定监测点的位移列名"""
    return f'{point_name}/mm'

# ==================== 监测点配置 ====================
TARGET_POINT = 'MJ9'  # 主要分析的监测点（生成详细图表）
ALL_POINTS = ['MJ9', 'MJ1', 'MJ3']  # 所有监测点（计算评价指标）

# ==================== 数据处理配置 ====================
TRAIN_RATIO = 0.70  # 训练集比例（修正：70%）
VAL_RATIO = 0.10    # 验证集比例（新增：10%）
TEST_RATIO = 0.20   # 测试集比例（修正：20%）
LOOKBACK_DAYS = 10  # 时间窗口：使用过去10天的数据预测（增加以捕捉滞后效应）

# 特征列表（根据第二章SHAP筛选结果）
FEATURE_COLS = [
    COL_RAINFALL,
    COL_RWL,
    COL_GWT,
    COL_AVE_TEMP,
    COL_MIN_TEMP,
    COL_MAX_TEMP,
    COL_DEW_POINT,
    COL_HUMIDITY
]


# ==================== 趋势提取模型配置 ====================
# 使用多项式去趋势（替代MVIF）
DETREND_METHOD = 'polynomial'  # 'polynomial' 或 'mvif'（已弃用）
POLYNOMIAL_DEGREE = 2  # 多项式阶数（推荐2，即二次多项式）

# MVIF模型配置（已弃用，保留用于对比实验）
MVIF_INIT_PARAMS = {
    'A': 1000.0,   # 位移饱和值初始猜测（mm）
    'B': 1.0,      # 曲线位置参数
    'C': 0.001,    # 蠕变速率参数（1/day）
    'tf': 5000.0   # 破坏时间初始猜测（day）
}

# MVIF拟合配置
MVIF_FIT_CONFIG = {
    'method': 'trf',  # Trust Region Reflective（支持边界约束）
    'max_nfev': 10000,
    'ftol': 1e-8,
    'xtol': 1e-8
}

# 周期项平滑配置
PERIODIC_SMOOTH_CONFIG = {
    'enable': True,  # 是否启用平滑
    'method': 'savgol',  # 'savgol' 或 'moving_average'
    'window_length': 7,  # Savitzky-Golay窗口长度（必须为奇数）
    'polyorder': 2       # Savitzky-Golay多项式阶数
}

# 监测点特定配置（Gemini建议）
POINT_SPECIFIC_CONFIG = {
    'MJ9': {
        'sg_window': 7,
        'sg_polyorder': 2,
        'mvif_init_scale': 1.2
    },
    'MJ1': {
        'sg_window': 11,
        'sg_polyorder': 3,
        'mvif_init_scale': 1.5
    },
    'MJ3': {
        'sg_window': 9,
        'sg_polyorder': 2,
        'mvif_init_scale': 1.3
    }
}

# ==================== LSTM模型配置 ====================
LSTM_CONFIG = {
    'input_size': None,  # 运行时根据特征数自动确定
    'hidden_size': 64,   # 隐藏层单元数（保持简单以避免过拟合）
    'num_layers': 1,     # LSTM层数（单层足够）
    'dropout': 0.3,      # Dropout比率（增加以提升泛化能力）
    'use_layer_norm': True,  # 是否使用LayerNorm
    'batch_size': 16,    # 批次大小（减小以增加训练噪声，提升泛化）
    'epochs': 500,       # 训练轮数（增加以给困难分位数更多时间）
    'learning_rate': 0.0001,  # 学习率（小火慢炖，稳定训练）
    'early_stopping_patience': 100,  # 早停耐心值（大幅增加，防止下界过早放弃）
    'weight_decay': 1e-4  # L2正则化系数
}

# ==================== 分位数回归配置 ====================
QUANTILES = [0.05, 0.5, 0.95]  # 5%, 50%, 95%分位数
CONFIDENCE_LEVEL = 0.95  # 置信水平（用于PICP计算）


# ==================== 可视化配置 ====================
FIGURE_CONFIG = {
    'format': 'pdf',  # 图片格式：'pdf', 'png', 'svg'
    'dpi': 300,       # 分辨率
    'figsize': (10, 6),  # 图片尺寸（英寸）
    'font_size': 12,     # 字体大小
    'style': 'seaborn-v0_8-darkgrid'  # matplotlib样式
}

# 中文字体配置（避免中文显示为方块）
# Linux环境使用DejaVu Sans + 中文fallback
FONT_CONFIG = {
    'family': 'DejaVu Sans',  # Linux通用字体
    'size': 12,
    'use_chinese': True  # 启用中文支持
}

# ==================== 对比模型配置 ====================
# 用于性能对比的基线模型
BASELINE_MODELS = ['LSTM', 'GRU', 'MLP']

# ==================== 日志配置 ====================
LOG_LEVEL = 'INFO'  # 日志级别：DEBUG, INFO, WARNING, ERROR
LOG_FILE = OUTPUT_DIR / 'training.log'  # 日志文件路径

# ==================== 导出配置 ====================
# 是否保存训练好的模型
SAVE_MODEL = True

# 是否生成LaTeX表格代码
GENERATE_LATEX_TABLES = True

# 是否保存中间结果（如MVIF拟合参数）
SAVE_INTERMEDIATE_RESULTS = True

