"""
数据加载和预处理模块 - （支持周期项预测）
修改：LSTM预测周期项而非总位移
Author: wcqqq21
Date: 2026-03-25
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
import sys

# 添加父目录到路径以导入config
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from models.polynomial_detrend import PolynomialDetrendModel

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MonitoringDataLoader:
    """监测数据加载器 支持周期项预测"""

    def __init__(self, data_path=DATA_PATH, target_point=TARGET_POINT):
        """
        初始化数据加载器

        Args:
            data_path: 数据文件路径
            target_point: 目标监测点名称
        """
        self.data_path = data_path
        self.target_point = target_point
        self.target_col = get_displacement_col(target_point)

        # 数据容器
        self.raw_data = None
        self.processed_data = None

        # 标准化器
        self.feature_scaler = StandardScaler()
        self.periodic_scaler = StandardScaler()  # 用于周期项的scaler

        # 多项式去趋势
        self.poly_model = None
        self.y_trend = None
        self.y_periodic = None

        # 数据集划分索引
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

        logger.info(f"初始化数据加载器: 目标点={target_point}, 数据路径={data_path}")

    def load_data(self):
        """加载原始数据"""
        try:
            logger.info(f"正在加载数据: {self.data_path}")
            self.raw_data = pd.read_excel(self.data_path)
            logger.info(f"数据加载成功: {self.raw_data.shape[0]} 行, {self.raw_data.shape[1]} 列")

            # 检查必要列是否存在
            required_cols = [COL_DATE, self.target_col] + FEATURE_COLS
            missing_cols = [col for col in required_cols if col not in self.raw_data.columns]
            if missing_cols:
                raise ValueError(f"缺少必要列: {missing_cols}")

            return self.raw_data

        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

    def handle_missing_values(self):
        """处理缺失值"""
        logger.info("开始处理缺失值...")

        # 统计缺失值
        missing_before = self.raw_data.isnull().sum()
        if missing_before.sum() > 0:
            logger.warning(f"发现缺失值:\n{missing_before[missing_before > 0]}")

        # 前向填充 + 后向填充
        self.processed_data = self.raw_data.copy()
        self.processed_data = self.processed_data.ffill().bfill()

        # 检查是否还有缺失值
        missing_after = self.processed_data.isnull().sum()
        if missing_after.sum() > 0:
            logger.error(f"填充后仍有缺失值:\n{missing_after[missing_after > 0]}")
            raise ValueError("缺失值处理失败")

        logger.info("缺失值处理完成")
        return self.processed_data

    def split_data(self):
        """划分训练集、验证集、测试集"""
        n = len(self.processed_data)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        self.train_idx = (0, train_end)
        self.val_idx = (train_end, val_end)
        self.test_idx = (val_end, n)

        logger.info(f"数据集划分完成:")
        logger.info(f"  训练集: {self.train_idx[0]}-{self.train_idx[1]} ({train_end} 样本)")
        logger.info(f"  验证集: {self.val_idx[0]}-{self.val_idx[1]} ({val_end - train_end} 样本)")
        logger.info(f"  测试集: {self.test_idx[0]}-{self.test_idx[1]} ({n - val_end} 样本)")

        return self.train_idx, self.val_idx, self.test_idx

    def extract_trend_periodic(self):
        """
        使用多项式提取趋势项和周期项
        ⭐ 这是的核心改动
        """
        logger.info("开始提取趋势项和周期项...")
        
        # 获取原始位移
        y_original = self.processed_data[self.target_col].values
        t = np.arange(len(y_original))
        
        # 多项式去趋势
        self.poly_model = PolynomialDetrendModel(
            point_name=self.target_point,
            degree=POLYNOMIAL_DEGREE
        )
        self.y_trend, self.y_periodic, _ = self.poly_model.decompose(
            t, y_original, smooth=True
        )
        
        logger.info(f"趋势项范围: [{self.y_trend.min():.2f}, {self.y_trend.max():.2f}] mm")
        logger.info(f"周期项范围: [{self.y_periodic.min():.2f}, {self.y_periodic.max():.2f}] mm")
        
        return self.y_trend, self.y_periodic

    def normalize_data(self):
        """
        标准化特征和周期项
        ⭐ 关键修改：标准化周期项而非总位移
        """
        logger.info("开始数据标准化...")
        
        # 提取特征
        X = self.processed_data[FEATURE_COLS].values
        
        # ⭐ 使用周期项作为目标（而非原始位移）
        y = self.y_periodic  # 周期项已经在extract_trend_periodic中计算
        
        # 只在训练集上fit
        train_start, train_end = self.train_idx
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end].reshape(-1, 1)
        
        self.feature_scaler.fit(X_train)
        self.periodic_scaler.fit(y_train)  # 使用periodic_scaler
        
        # transform所有数据
        X_normalized = self.feature_scaler.transform(X)
        y_normalized = self.periodic_scaler.transform(y.reshape(-1, 1)).flatten()
        
        logger.info(f"特征标准化完成: mean={self.feature_scaler.mean_[:3]}, std={self.feature_scaler.scale_[:3]}")
        logger.info(f"周期项标准化完成: mean={self.periodic_scaler.mean_[0]:.2f}, std={self.periodic_scaler.scale_[0]:.2f}")
        
        return X_normalized, y_normalized

    def create_sequences(self, X, y, lookback=LOOKBACK_DAYS):
        """创建时间序列窗口"""
        X_seq, y_seq = [], []

        for i in range(lookback, len(X)):
            X_seq.append(X[i - lookback:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def inverse_transform_periodic(self, y_periodic_normalized):
        """反标准化周期项"""
        return self.periodic_scaler.inverse_transform(y_periodic_normalized.reshape(-1, 1)).flatten()

    def get_trend_at_indices(self, indices):
        """获取指定索引处的趋势值"""
        return self.y_trend[indices]

    def prepare_data(self):
        """
        完整的数据准备流程
        ⭐ ：增加了趋势提取步骤
        """
        # 1. 加载数据
        self.load_data()

        # 2. 处理缺失值
        self.handle_missing_values()

        # 3. 划分数据集
        self.split_data()

        # 4. ⭐ 提取趋势和周期（新增步骤）
        self.extract_trend_periodic()

        # 5. 标准化（现在标准化的是周期项）
        X_normalized, y_normalized = self.normalize_data()

        # 6. 创建时间序列窗口
        X_seq, y_seq = self.create_sequences(X_normalized, y_normalized)

        # 7. 根据索引划分
        lookback = LOOKBACK_DAYS
        train_start = max(0, self.train_idx[0] - lookback)
        train_end = self.train_idx[1] - lookback
        val_start = self.val_idx[0] - lookback
        val_end = self.val_idx[1] - lookback
        test_start = self.test_idx[0] - lookback
        test_end = self.test_idx[1] - lookback

        X_train, y_train = X_seq[train_start:train_end], y_seq[train_start:train_end]
        X_val, y_val = X_seq[val_start:val_end], y_seq[val_start:val_end]
        X_test, y_test = X_seq[test_start:test_end], y_seq[test_start:test_end]

        logger.info(f"数据准备完成:")
        logger.info(f"  训练集: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"  验证集: X={X_val.shape}, y={y_val.shape}")
        logger.info(f"  测试集: X={X_test.shape}, y={y_test.shape}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class TimeSeriesDataset(Dataset):
    """PyTorch时间序列数据集"""

    def __init__(self, X, y):
        """
        Args:
            X: numpy array, shape (n_samples, lookback, n_features)
            y: numpy array, shape (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(batch_size=LSTM_CONFIG['batch_size']):
    """
    获取训练、验证、测试的DataLoader ()
    
    Returns:
        train_loader, val_loader, test_loader, data_loader_obj
    """
    # 创建数据加载器
    data_loader = MonitoringDataLoader()

    # 准备数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.prepare_data()

    # 创建PyTorch数据集
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"DataLoader创建完成: batch_size={batch_size}")

    return train_loader, val_loader, test_loader, data_loader
