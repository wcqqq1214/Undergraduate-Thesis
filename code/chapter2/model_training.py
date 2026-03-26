"""
模型训练模块

包含 LightGBM 回归与分类模型的训练逻辑，以及时序交叉验证流程。
"""
import numpy as np
import lightgbm as lgb
from typing import List, Tuple

from sklearn.metrics import (
    mean_squared_error, accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE


def print_fold_metrics(
    fold_idx: int,
    n_splits: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mse_train: float,
    mse_test: float,
    acc_train: float,
    auc_train: float,
    acc_test: float,
    auc_test: float,
    precision_test: float,
    recall_test: float,
    f1_test: float,
    cm: np.ndarray,
) -> None:
    """打印单折的详细评估指标"""
    print(f"\n=== Fold {fold_idx}/{n_splits} ===")
    print(f"[Regression] MSE train: {mse_train:.4f}, MSE test: {mse_test:.4f}")
    print(f"[Classification] ACC train: {acc_train:.4f}, AUC train: {auc_train:.4f}")
    print(f"[Classification] ACC test: {acc_test:.4f}, AUC test: {auc_test:.4f}")
    print(f"[Info] 训练集: {len(X_train)}样本, 正样本: {y_train.sum()}/{len(y_train)} = {y_train.mean():.4f}")
    print(f"[Info] 测试集: {len(X_test)}样本, 正样本: {y_test.sum()}/{len(y_test)} = {y_test.mean():.4f}")
    print(f"[Classification] Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1: {f1_test:.4f}")
    print(f"[Info] 混淆矩阵 (测试集):")
    print(f"       预测负  预测正")
    print(f"实际负  {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"实际正  {cm[1,0]:4d}   {cm[1,1]:4d}")


# 回归模型超参数
PARAMS_REG = {
    "objective": "regression",
    "metric": "l2",
    "boosting_type": "gbdt",
    "num_leaves": 4,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_data_in_leaf": 5,
    "min_split_gain": 1e-4,
    "lambda_l1": 0.2,
    "lambda_l2": 0.2,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}

# 分类模型超参数
PARAMS_CLS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 8,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_data_in_leaf": 20,
    "lambda_l1": 0.3,
    "lambda_l2": 0.3,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "scale_pos_weight": 5,
    "verbose": -1,
}


def cross_validate_models(
    X: np.ndarray,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    feat_names: List[str],
    n_splits: int = 5,
    use_smote: bool = True,
) -> Tuple[lgb.Booster, lgb.Booster, dict]:
    """
    严格时序交叉验证（避免数据泄漏）

    Returns:
        reg_model: 最后一折的回归模型
        cls_model: 最后一折的分类模型
        cv_results: 各折评估指标字典
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_results = {
        'reg_mse_train': [], 'reg_mse_test': [],
        'cls_acc_train': [], 'cls_acc_test': [],
        'cls_auc_train': [], 'cls_auc_test': [],
        'cls_precision_test': [], 'cls_recall_test': [], 'cls_f1_test': [],
    }

    print(f"\n=== 开始{n_splits}折时序交叉验证（严格避免数据泄漏）===")
    if use_smote:
        print(f"[Info] 使用SMOTE对训练集进行重采样以平衡类别")

    reg_model = None
    cls_model = None

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_reg_train, y_reg_test = y_reg[train_idx], y_reg[test_idx]
        y_cls_train, y_cls_test = y_cls[train_idx], y_cls[test_idx]

        if y_cls_test.sum() == 0:
            print(f"\n=== Fold {fold_idx}/{n_splits} ===")
            print(f"[Warning] 测试集无正样本，跳过此fold")
            continue

        # SMOTE 仅作用于训练集
        X_train_cls = X_train.copy()
        y_cls_train_resampled = y_cls_train.copy()
        if use_smote and 0 < y_cls_train.sum() < len(y_cls_train):
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, y_cls_train.sum() - 1))
                X_train_cls, y_cls_train_resampled = smote.fit_resample(X_train, y_cls_train)
            except Exception as e:
                print(f"[Warning] SMOTE失败: {e}，使用原始训练集")

        # ── 回归模型 ──────────────────────────────────────────────────────
        train_data_reg = lgb.Dataset(X_train, label=y_reg_train)
        test_data_reg = lgb.Dataset(X_test, label=y_reg_test, reference=train_data_reg)
        reg_model = lgb.train(
            PARAMS_REG, train_data_reg, num_boost_round=500,
            valid_sets=[test_data_reg],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        mse_train = mean_squared_error(y_reg_train, reg_model.predict(X_train, num_iteration=reg_model.best_iteration))
        mse_test = mean_squared_error(y_reg_test, reg_model.predict(X_test, num_iteration=reg_model.best_iteration))
        cv_results['reg_mse_train'].append(mse_train)
        cv_results['reg_mse_test'].append(mse_test)

        # ── 分类模型 ──────────────────────────────────────────────────────
        train_data_cls = lgb.Dataset(X_train_cls, label=y_cls_train_resampled)
        test_data_cls = lgb.Dataset(X_test, label=y_cls_test, reference=train_data_cls)
        cls_model = lgb.train(
            PARAMS_CLS, train_data_cls, num_boost_round=300,
            valid_sets=[test_data_cls],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        threshold = 0.3
        y_prob_train = cls_model.predict(X_train, num_iteration=cls_model.best_iteration)
        y_prob_test = cls_model.predict(X_test, num_iteration=cls_model.best_iteration)
        y_pred_train = (y_prob_train >= threshold).astype(int)
        y_pred_test = (y_prob_test >= threshold).astype(int)

        acc_train = accuracy_score(y_cls_train, y_pred_train)
        acc_test = accuracy_score(y_cls_test, y_pred_test)
        auc_train = roc_auc_score(y_cls_train, y_prob_train) if len(np.unique(y_cls_train)) > 1 else np.nan
        auc_test = roc_auc_score(y_cls_test, y_prob_test) if len(np.unique(y_cls_test)) > 1 else np.nan

        cv_results['cls_acc_train'].append(acc_train)
        cv_results['cls_acc_test'].append(acc_test)
        if not np.isnan(auc_train):
            cv_results['cls_auc_train'].append(auc_train)
        if not np.isnan(auc_test):
            cv_results['cls_auc_test'].append(auc_test)

        if y_cls_test.sum() > 0:
            precision_test = precision_score(y_cls_test, y_pred_test, zero_division=0)
            recall_test = recall_score(y_cls_test, y_pred_test, zero_division=0)
            f1_test = f1_score(y_cls_test, y_pred_test, zero_division=0)
            cm = confusion_matrix(y_cls_test, y_pred_test)
        else:
            precision_test = recall_test = f1_test = 0.0
            cm = np.array([[len(y_cls_test), 0], [0, 0]])

        cv_results['cls_precision_test'].append(precision_test)
        cv_results['cls_recall_test'].append(recall_test)
        cv_results['cls_f1_test'].append(f1_test)

        print_fold_metrics(
            fold_idx, n_splits,
            X_train, y_cls_train, X_test, y_cls_test,
            mse_train, mse_test,
            acc_train, auc_train, acc_test, auc_test,
            precision_test, recall_test, f1_test, cm,
        )

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print(f"\n=== 交叉验证总结 ===")
    print(f"[Regression]")
    print(f"  MSE (train): {np.mean(cv_results['reg_mse_train']):.4f} ± {np.std(cv_results['reg_mse_train']):.4f}")
    print(f"  MSE (test):  {np.mean(cv_results['reg_mse_test']):.4f} ± {np.std(cv_results['reg_mse_test']):.4f}")
    print(f"\n[Classification]")
    if cv_results['cls_auc_train']:
        print(f"  AUC (train):  {np.mean(cv_results['cls_auc_train']):.4f} ± {np.std(cv_results['cls_auc_train']):.4f}")
    if cv_results['cls_auc_test']:
        print(f"  AUC (test):   {np.mean(cv_results['cls_auc_test']):.4f} ± {np.std(cv_results['cls_auc_test']):.4f}")
    print(f"  F1 (test):    {np.mean(cv_results['cls_f1_test']):.4f} ± {np.std(cv_results['cls_f1_test']):.4f}")
    print(f"  Precision (test): {np.mean(cv_results['cls_precision_test']):.4f} ± {np.std(cv_results['cls_precision_test']):.4f}")
    print(f"  Recall (test):    {np.mean(cv_results['cls_recall_test']):.4f} ± {np.std(cv_results['cls_recall_test']):.4f}")

    return reg_model, cls_model, cv_results
