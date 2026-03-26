"""
第三章主程序：多项式去趋势 + 分位数LSTM ()
完整流程：数据加载 -> 多项式分解 -> LSTM训练(预测周期项) -> 评估 -> 可视化
Author: wcqqq21
Date: 2026-03-25
"""

import sys
from pathlib import Path
import logging
import json
import numpy as np
import torch

# 添加路径
sys.path.append(str(Path(__file__).parent))

# 导入配置
from config import *

# 导入自定义模块
from data.data_loader import get_dataloaders, MonitoringDataLoader
from models.polynomial_detrend import PolynomialDetrendModel
from models.quantile_lstm import QuantileLSTMTrainer
from utils.metrics import calculate_quantile_metrics, print_metrics
from utils.visualization import plot_all_results, save_latex_table

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


def step1_load_data():
    """步骤1: 加载和预处理数据 ()"""
    logger.info("=" * 70)
    logger.info("步骤1: 数据加载和预处理")
    logger.info("=" * 70)

    # 设置随机种子
    set_seed(RANDOM_SEED)

    # 获取DataLoader ()
    train_loader, val_loader, test_loader, data_loader_obj = get_dataloaders()

    logger.info("✓ 数据加载完成")

    return train_loader, val_loader, test_loader, data_loader_obj


def step2_train_lstm(train_loader, val_loader):
    """步骤2: 训练分位数LSTM (预测周期项)"""
    logger.info("\n" + "=" * 70)
    logger.info("步骤2: 训练分位数LSTM (预测周期项)")
    logger.info("=" * 70)

    # 获取输入维度
    X_sample, _ = next(iter(train_loader))
    input_size = X_sample.shape[2]

    # 创建训练器
    trainer = QuantileLSTMTrainer(
        input_size=input_size,
        quantiles=QUANTILES,
        device=DEVICE
    )

    # 训练
    train_history, val_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=LSTM_CONFIG['epochs'],
        patience=LSTM_CONFIG['early_stopping_patience']
    )

    logger.info("✓ LSTM训练完成")

    return trainer, train_history, val_history


def step3_evaluate(trainer, test_loader, data_loader_obj):
    """步骤3: 模型评估 -  (组合趋势+周期)"""
    logger.info("\n" + "=" * 70)
    logger.info("步骤3: 模型评估")
    logger.info("=" * 70)

    # 预测所有分位数（预测的是周期项）
    predictions_periodic = trainer.predict_all_quantiles(test_loader)

    # 获取真实的周期项
    y_periodic_normalized = []
    for _, y_batch in test_loader:
        y_periodic_normalized.append(y_batch.numpy())
    y_periodic_normalized = np.concatenate(y_periodic_normalized)

    # 反标准化周期项
    y_periodic_true = data_loader_obj.inverse_transform_periodic(y_periodic_normalized)
    predictions_periodic_denorm = {
        q: data_loader_obj.inverse_transform_periodic(pred)
        for q, pred in predictions_periodic.items()
    }

    # ⭐ 关键步骤：加上趋势项得到最终位移预测
    test_indices = range(data_loader_obj.test_idx[0], data_loader_obj.test_idx[1])
    y_trend_test = data_loader_obj.get_trend_at_indices(test_indices)
    
    logger.info(f"测试集趋势项范围: [{y_trend_test.min():.2f}, {y_trend_test.max():.2f}] mm")
    logger.info(f"测试集周期项预测范围: [{predictions_periodic_denorm[0.5].min():.2f}, {predictions_periodic_denorm[0.5].max():.2f}] mm")
    
    # 最终预测 = 趋势 + 周期
    y_true = y_periodic_true + y_trend_test
    predictions_denorm = {
        q: pred + y_trend_test
        for q, pred in predictions_periodic_denorm.items()
    }
    
    logger.info(f"最终预测范围: [{predictions_denorm[0.5].min():.2f}, {predictions_denorm[0.5].max():.2f}] mm")
    logger.info(f"真实值范围: [{y_true.min():.2f}, {y_true.max():.2f}] mm")

    # 计算指标（总位移）
    metrics = calculate_quantile_metrics(y_true, predictions_denorm)

    # 额外计算周期项的 R²（用于论文中说明残差预测难度）
    from utils.metrics import r2_score
    periodic_r2 = r2_score(y_periodic_true, predictions_periodic_denorm[0.5])
    metrics['R2_periodic'] = periodic_r2

    print_metrics(metrics, title="测试集评估指标（总位移）")
    logger.info(f"周期项 R²: {periodic_r2:.4f} (残差预测难度指标)")

    logger.info("✓ 模型评估完成")

    return {
        'y_true': y_true,
        'y_pred': predictions_denorm[0.5],
        'y_lower': predictions_denorm[0.05],
        'y_upper': predictions_denorm[0.95],
        'metrics': metrics
    }


def step4_visualize(detrend_results, training_results, test_results):
    """步骤4: 结果可视化"""
    logger.info("\n" + "=" * 70)
    logger.info("步骤4: 结果可视化")
    logger.info("=" * 70)

    # 整合所有结果
    results = {
        'detrend': detrend_results,
        'training': {
            'train_history': training_results['train_history'],
            'val_history': training_results['val_history'],
            'quantiles': QUANTILES
        },
        'test': test_results
    }

    # 生成所有图表
    plot_all_results(results, save_dir=FIGURE_DIR)

    # 生成LaTeX表格
    if GENERATE_LATEX_TABLES:
        metrics_dict = {
            'Polynomial-QLSTM': test_results['metrics']
        }
        save_latex_table(
            metrics_dict,
            save_path=TABLE_DIR / 'model_performance.tex',
            caption='多项式去趋势+分位数LSTM模型性能评估 ()',
            label='tab:poly_qlstm_performance'
        )

    logger.info("✓ 可视化完成")


def main():
    """主函数 - """
    logger.info("\n" + "=" * 70)
    logger.info("第三章：多项式去趋势 + 分位数LSTM ()")
    logger.info("=" * 70)
    logger.info(f"设备: {DEVICE}")
    logger.info(f"随机种子: {RANDOM_SEED}")
    logger.info(f"目标监测点: {TARGET_POINT}")
    logger.info(f"去趋势方法: {DETREND_METHOD} (阶数={POLYNOMIAL_DEGREE})")
    logger.info(f"⭐ 改进: LSTM预测周期项，最终预测=趋势+周期")
    logger.info("=" * 70)

    try:
        # 步骤1: 加载数据（包含趋势提取）
        train_loader, val_loader, test_loader, data_loader_obj = step1_load_data()

        # 步骤2: 训练LSTM（预测周期项）
        trainer, train_history, val_history = step2_train_lstm(train_loader, val_loader)

        # 步骤3: 评估（组合趋势+周期）
        test_results = step3_evaluate(trainer, test_loader, data_loader_obj)

        # 步骤4: 可视化
        detrend_results = {
            't': np.arange(len(data_loader_obj.processed_data)),
            'y_original': data_loader_obj.processed_data[data_loader_obj.target_col].values,
            'y_trend': data_loader_obj.y_trend,
            'y_periodic': data_loader_obj.y_periodic,
            'params': data_loader_obj.poly_model.get_params_dict(),
            'model': data_loader_obj.poly_model
        }
        
        training_results = {
            'train_history': train_history,
            'val_history': val_history
        }
        step4_visualize(detrend_results, training_results, test_results)

        # 最终总结
        logger.info("\n" + "=" * 70)
        logger.info("✓ 所有步骤完成！")
        logger.info("=" * 70)
        logger.info(f"图表保存位置: {FIGURE_DIR}")
        logger.info(f"表格保存位置: {TABLE_DIR}")
        logger.info(f"模型保存位置: {MODEL_DIR}")
        logger.info("=" * 70)

        # 打印最终指标
        logger.info("\n最终性能指标:")
        for metric, value in test_results['metrics'].items():
            if metric in ['PICP', 'PINAW', 'CWC', 'R2']:
                logger.info(f"  {metric}: {value:.4f}")
            elif metric == 'MAPE':
                logger.info(f"  {metric}: {value:.2f}%")
            else:
                logger.info(f"  {metric}: {value:.4f}")

    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise


if __name__ == "__main__":
    main()
