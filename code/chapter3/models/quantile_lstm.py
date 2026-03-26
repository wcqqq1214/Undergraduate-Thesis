"""
基于Pinball Loss的分位数LSTM预测模型
Author: wcqqq21
Date: 2026-03-25
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from pathlib import Path
import sys

# 添加父目录到路径以导入config
sys.path.append(str(Path(__file__).parent.parent))
from config import *

logger = logging.getLogger(__name__)


class PinballLoss(nn.Module):
    """
    Pinball Loss (分位数损失函数)

    L_τ(y, ŷ) = {
        τ * (y - ŷ),     if y >= ŷ
        (τ - 1) * (y - ŷ), if y < ŷ
    }
    """

    def __init__(self, quantile):
        """
        Args:
            quantile: 分位数 τ ∈ (0, 1)
        """
        super(PinballLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: 预测值, shape (batch_size,)
            y_true: 真实值, shape (batch_size,)

        Returns:
            loss: 标量
        """
        errors = y_true - y_pred
        loss = torch.max(self.quantile * errors, (self.quantile - 1) * errors)
        return torch.mean(loss)


class QuantileLSTM(nn.Module):
    """分位数LSTM网络"""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, use_layer_norm=True):
        """
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比率
            use_layer_norm: 是否使用LayerNorm
        """
        super(QuantileLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # LayerNorm（Gemini建议）
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)

        # 全连接层
        self.fc = nn.Linear(hidden_size, 1)

        logger.info(f"初始化QuantileLSTM: input_size={input_size}, hidden_size={hidden_size}, "
                   f"num_layers={num_layers}, dropout={dropout}, use_layer_norm={use_layer_norm}")

    def forward(self, x):
        """
        Args:
            x: 输入序列, shape (batch_size, seq_len, input_size)

        Returns:
            output: 预测值, shape (batch_size, 1)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)

        # 取最后一个时间步
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # LayerNorm
        if self.use_layer_norm:
            last_hidden = self.layer_norm(last_hidden)

        # 全连接层
        output = self.fc(last_hidden)  # (batch_size, 1)

        return output.squeeze(-1)  # (batch_size,)


class QuantileLSTMTrainer:
    """分位数LSTM训练器"""

    def __init__(self, input_size, quantiles=QUANTILES, device=DEVICE):
        """
        Args:
            input_size: 输入特征维度
            quantiles: 分位数列表
            device: 训练设备
        """
        self.input_size = input_size
        self.quantiles = quantiles
        self.device = device

        # 为每个分位数创建独立的模型
        self.models = {}
        self.optimizers = {}
        self.loss_fns = {}

        # 区间扩展偏置（最终冲刺：针对不对称分布，激进扩展）
        self.interval_bias = {
            0.05: -0.7,   # 下界激进向下偏移（宁可保守，不可漏报）
            0.5: 0.0,     # 中位数不偏移
            0.95: 0.5     # 上界加大偏移（平衡区间宽度）
        }

        for q in quantiles:
            model = QuantileLSTM(
                input_size=input_size,
                hidden_size=LSTM_CONFIG['hidden_size'],
                num_layers=LSTM_CONFIG['num_layers'],
                dropout=LSTM_CONFIG['dropout'],
                use_layer_norm=LSTM_CONFIG['use_layer_norm']
            ).to(device)

            optimizer = optim.Adam(
                model.parameters(),
                lr=LSTM_CONFIG['learning_rate'],
                weight_decay=LSTM_CONFIG['weight_decay']
            )

            loss_fn = PinballLoss(quantile=q)

            self.models[q] = model
            self.optimizers[q] = optimizer
            self.loss_fns[q] = loss_fn

        # 训练历史
        self.train_history = {q: [] for q in quantiles}
        self.val_history = {q: [] for q in quantiles}

        logger.info(f"初始化QuantileLSTMTrainer: {len(quantiles)} 个分位数模型")
        logger.info(f"区间扩展偏置: {self.interval_bias}")

    def train_epoch(self, train_loader, quantile):
        """训练一个epoch"""
        model = self.models[quantile]
        optimizer = self.optimizers[quantile]
        loss_fn = self.loss_fns[quantile]

        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # 前向传播
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(X_batch)

        return epoch_loss / len(train_loader.dataset)

    def validate(self, val_loader, quantile):
        """验证"""
        model = self.models[quantile]
        loss_fn = self.loss_fns[quantile]

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)

                val_loss += loss.item() * len(X_batch)

        return val_loss / len(val_loader.dataset)

    def train(self, train_loader, val_loader, epochs=LSTM_CONFIG['epochs'],
              patience=LSTM_CONFIG['early_stopping_patience']):
        """
        训练所有分位数模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 最大训练轮数
            patience: 早停耐心值

        Returns:
            训练历史
        """
        logger.info("=" * 50)
        logger.info("开始训练分位数LSTM模型")
        logger.info("=" * 50)

        for q in self.quantiles:
            logger.info(f"\n训练分位数 τ={q} 的模型...")

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(epochs):
                # 训练
                train_loss = self.train_epoch(train_loader, q)
                self.train_history[q].append(train_loss)

                # 验证
                val_loss = self.validate(val_loader, q)
                self.val_history[q].append(val_loss)

                # 日志
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"Epoch [{epoch+1}/{epochs}] - "
                              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    if SAVE_MODEL:
                        self.save_model(q, epoch, val_loss)
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"早停触发: 验证损失在 {patience} 轮内未改善")
                    break

            logger.info(f"分位数 τ={q} 训练完成: 最佳验证损失={best_val_loss:.4f}")

        logger.info("\n所有分位数模型训练完成！")

        return self.train_history, self.val_history

    def predict(self, data_loader, quantile):
        """
        预测指定分位数（应用区间扩展偏置）

        Args:
            data_loader: 数据加载器
            quantile: 分位数

        Returns:
            predictions: numpy array
        """
        model = self.models[quantile]
        model.eval()

        predictions = []
        bias = self.interval_bias.get(quantile, 0.0)

        with torch.no_grad():
            for X_batch, _ in data_loader:
                X_batch = X_batch.to(self.device)
                y_pred = model(X_batch)
                # 应用区间扩展偏置
                y_pred = y_pred + bias
                predictions.append(y_pred.cpu().numpy())

        return np.concatenate(predictions)

    def predict_all_quantiles(self, data_loader):
        """
        预测所有分位数

        Args:
            data_loader: 数据加载器

        Returns:
            predictions: {quantile: predictions}
        """
        predictions = {}

        for q in self.quantiles:
            predictions[q] = self.predict(data_loader, q)

        return predictions

    def save_model(self, quantile, epoch, val_loss):
        """保存模型"""
        model_path = MODEL_DIR / f"quantile_lstm_q{quantile:.2f}_epoch{epoch+1}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.models[quantile].state_dict(),
            'optimizer_state_dict': self.optimizers[quantile].state_dict(),
            'val_loss': val_loss,
            'quantile': quantile,
            'config': LSTM_CONFIG
        }, model_path)

        logger.debug(f"模型已保存: {model_path}")

    def load_model(self, quantile, model_path):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        self.models[quantile].load_state_dict(checkpoint['model_state_dict'])
        self.optimizers[quantile].load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"模型已加载: {model_path} (epoch={checkpoint['epoch']}, "
                   f"val_loss={checkpoint['val_loss']:.4f})")

        return checkpoint


if __name__ == "__main__":
    # 测试代码
    logger.info("=" * 50)
    logger.info("测试分位数LSTM模块")
    logger.info("=" * 50)

    # 生成模拟数据
    batch_size = 32
    seq_len = 5
    input_size = 8
    n_samples = 100

    X = torch.randn(n_samples, seq_len, input_size)
    y = torch.randn(n_samples)

    # 创建数据集
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 创建训练器
    trainer = QuantileLSTMTrainer(input_size=input_size, quantiles=[0.05, 0.5, 0.95])

    # 训练（仅2个epoch用于测试）
    train_history, val_history = trainer.train(train_loader, val_loader, epochs=2, patience=10)

    # 预测
    predictions = trainer.predict_all_quantiles(val_loader)

    logger.info(f"预测结果:")
    for q, pred in predictions.items():
        logger.info(f"  τ={q}: shape={pred.shape}, 范围=[{pred.min():.2f}, {pred.max():.2f}]")

