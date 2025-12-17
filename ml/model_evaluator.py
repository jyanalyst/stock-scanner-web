"""
ML Model Evaluator - Phase 3.2
Evaluates model performance with comprehensive metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation for classification and regression
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.results = {}
        logger.info("Initialized ModelEvaluator")
    
    def evaluate_classification(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_pred_proba: Optional[np.ndarray] = None,
                               model_name: str = "model") -> Dict:
        """
        Evaluate classification model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional, for ROC-AUC)
            model_name: Name of the model
        
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating classification model: {model_name}")
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC-AUC (if probabilities provided)
        if y_pred_proba is not None:
            try:
                if y_pred_proba.ndim == 2:
                    # Use probability of positive class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                metrics['roc_auc'] = np.nan
        else:
            metrics['roc_auc'] = np.nan
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # True/False Positives/Negatives
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
        
        # Sample counts
        metrics['n_samples'] = len(y_true)
        metrics['n_positive'] = int(y_true.sum())
        metrics['n_negative'] = int(len(y_true) - y_true.sum())
        
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"F1: {metrics['f1']:.4f}, ROC-AUC: {metrics.get('roc_auc', np.nan):.4f}")
        
        self.results[f"{model_name}_classification"] = metrics
        return metrics
    
    def evaluate_regression(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           model_name: str = "model") -> Dict:
        """
        Evaluate regression model
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
        
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating regression model: {model_name}")
        
        metrics = {}
        
        # Basic regression metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Information Coefficient (Spearman correlation)
        ic, ic_pvalue = spearmanr(y_pred, y_true)
        metrics['ic'] = ic
        metrics['ic_pvalue'] = ic_pvalue
        
        # Directional accuracy (% correct sign)
        y_true_sign = np.sign(y_true)
        y_pred_sign = np.sign(y_pred)
        metrics['directional_accuracy'] = (y_true_sign == y_pred_sign).mean()
        
        # Mean and std of predictions
        metrics['pred_mean'] = y_pred.mean()
        metrics['pred_std'] = y_pred.std()
        metrics['true_mean'] = y_true.mean()
        metrics['true_std'] = y_true.std()
        
        # Sample counts
        metrics['n_samples'] = len(y_true)
        
        logger.info(f"{model_name} - MAE: {metrics['mae']:.4f}, "
                   f"RMSE: {metrics['rmse']:.4f}, "
                   f"R²: {metrics['r2']:.4f}, "
                   f"IC: {metrics['ic']:.4f}")
        
        self.results[f"{model_name}_regression"] = metrics
        return metrics
    
    def calculate_profit_metrics(self,
                                 y_true_returns: np.ndarray,
                                 y_pred_labels: np.ndarray,
                                 model_name: str = "model") -> Dict:
        """
        Calculate profit metrics by simulating trading
        
        Args:
            y_true_returns: Actual returns (e.g., return_3d)
            y_pred_labels: Predicted labels (1 = trade, 0 = no trade)
            model_name: Name of the model
        
        Returns:
            Dictionary of profit metrics
        """
        logger.info(f"Calculating profit metrics for: {model_name}")
        
        metrics = {}
        
        # Filter to only trades where model predicted 1
        trade_mask = y_pred_labels == 1
        
        if trade_mask.sum() == 0:
            logger.warning(f"{model_name}: No trades predicted")
            return {
                'n_trades': 0,
                'total_return': 0.0,
                'mean_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
        trade_returns = y_true_returns[trade_mask]
        
        # Basic stats
        metrics['n_trades'] = int(trade_mask.sum())
        metrics['total_return'] = float(trade_returns.sum())
        metrics['mean_return'] = float(trade_returns.mean())
        metrics['std_return'] = float(trade_returns.std())
        
        # Win/loss stats
        wins = trade_returns > 0
        losses = trade_returns < 0
        
        metrics['n_wins'] = int(wins.sum())
        metrics['n_losses'] = int(losses.sum())
        metrics['win_rate'] = float(wins.mean())
        
        if wins.sum() > 0:
            metrics['mean_win'] = float(trade_returns[wins].mean())
        else:
            metrics['mean_win'] = 0.0
        
        if losses.sum() > 0:
            metrics['mean_loss'] = float(trade_returns[losses].mean())
        else:
            metrics['mean_loss'] = 0.0
        
        # Profit factor (total wins / abs(total losses))
        total_wins = trade_returns[wins].sum() if wins.sum() > 0 else 0
        total_losses = abs(trade_returns[losses].sum()) if losses.sum() > 0 else 0
        
        if total_losses > 0:
            metrics['profit_factor'] = float(total_wins / total_losses)
        else:
            metrics['profit_factor'] = float('inf') if total_wins > 0 else 0.0
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        if metrics['std_return'] > 0:
            metrics['sharpe_ratio'] = float((metrics['mean_return'] / metrics['std_return']) * np.sqrt(252))
        else:
            metrics['sharpe_ratio'] = 0.0
        
        logger.info(f"{model_name} - Trades: {metrics['n_trades']}, "
                   f"Total Return: {metrics['total_return']:.2%}, "
                   f"Win Rate: {metrics['win_rate']:.2%}, "
                   f"Profit Factor: {metrics['profit_factor']:.2f}")
        
        self.results[f"{model_name}_profit"] = metrics
        return metrics
    
    def compare_models(self, 
                      models_metrics: Dict[str, Dict],
                      metric_type: str = 'classification') -> pd.DataFrame:
        """
        Compare multiple models side-by-side
        
        Args:
            models_metrics: Dictionary of {model_name: metrics_dict}
            metric_type: 'classification' or 'regression'
        
        Returns:
            DataFrame with comparison
        """
        logger.info(f"Comparing {len(models_metrics)} models ({metric_type})")
        
        if metric_type == 'classification':
            columns = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        else:  # regression
            columns = ['mae', 'rmse', 'r2', 'ic', 'directional_accuracy']
        
        comparison = {}
        for model_name, metrics in models_metrics.items():
            comparison[model_name] = {col: metrics.get(col, np.nan) for col in columns}
        
        df = pd.DataFrame(comparison).T
        df = df.sort_values(by=columns[0], ascending=(metric_type == 'regression'))
        
        return df
    
    def get_feature_importance_comparison(self,
                                         models_importance: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare feature importance across models
        
        Args:
            models_importance: Dictionary of {model_name: {feature: importance}}
        
        Returns:
            DataFrame with feature importance comparison
        """
        logger.info(f"Comparing feature importance across {len(models_importance)} models")
        
        df = pd.DataFrame(models_importance)
        
        # Sort by mean importance
        df['mean_importance'] = df.mean(axis=1)
        df = df.sort_values('mean_importance', ascending=False)
        
        return df
    
    def generate_summary_report(self) -> str:
        """
        Generate text summary of all evaluations
        
        Returns:
            Formatted summary string
        """
        report = []
        report.append("=" * 80)
        report.append("MODEL EVALUATION SUMMARY")
        report.append("=" * 80)
        
        # Classification results
        classification_results = {k: v for k, v in self.results.items() if 'classification' in k}
        if classification_results:
            report.append("\n📊 CLASSIFICATION RESULTS:")
            report.append("-" * 80)
            
            for model_name, metrics in classification_results.items():
                model_name = model_name.replace('_classification', '')
                report.append(f"\n{model_name}:")
                report.append(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
                report.append(f"  Precision: {metrics.get('precision', 0):.4f}")
                report.append(f"  Recall:    {metrics.get('recall', 0):.4f}")
                report.append(f"  F1-Score:  {metrics.get('f1', 0):.4f}")
                report.append(f"  ROC-AUC:   {metrics.get('roc_auc', 0):.4f}")
                
                if 'confusion_matrix' in metrics:
                    cm = metrics['confusion_matrix']
                    if cm.shape == (2, 2):
                        report.append(f"  Confusion Matrix:")
                        report.append(f"    TN: {metrics.get('true_negatives', 0):5d}  FP: {metrics.get('false_positives', 0):5d}")
                        report.append(f"    FN: {metrics.get('false_negatives', 0):5d}  TP: {metrics.get('true_positives', 0):5d}")
        
        # Regression results
        regression_results = {k: v for k, v in self.results.items() if 'regression' in k}
        if regression_results:
            report.append("\n\n📈 REGRESSION RESULTS:")
            report.append("-" * 80)
            
            for model_name, metrics in regression_results.items():
                model_name = model_name.replace('_regression', '')
                report.append(f"\n{model_name}:")
                report.append(f"  MAE:                  {metrics.get('mae', 0):.6f}")
                report.append(f"  RMSE:                 {metrics.get('rmse', 0):.6f}")
                report.append(f"  R²:                   {metrics.get('r2', 0):.4f}")
                report.append(f"  IC:                   {metrics.get('ic', 0):.4f}")
                report.append(f"  Directional Accuracy: {metrics.get('directional_accuracy', 0):.4f}")
        
        # Profit results
        profit_results = {k: v for k, v in self.results.items() if 'profit' in k}
        if profit_results:
            report.append("\n\n💰 PROFIT SIMULATION:")
            report.append("-" * 80)
            
            for model_name, metrics in profit_results.items():
                model_name = model_name.replace('_profit', '')
                report.append(f"\n{model_name}:")
                report.append(f"  Total Trades:   {metrics.get('n_trades', 0):,}")
                report.append(f"  Win Rate:       {metrics.get('win_rate', 0):.2%}")
                report.append(f"  Total Return:   {metrics.get('total_return', 0):.2%}")
                report.append(f"  Mean Return:    {metrics.get('mean_return', 0):.4%}")
                report.append(f"  Mean Win:       {metrics.get('mean_win', 0):.4%}")
                report.append(f"  Mean Loss:      {metrics.get('mean_loss', 0):.4%}")
                report.append(f"  Profit Factor:  {metrics.get('profit_factor', 0):.2f}")
                report.append(f"  Sharpe Ratio:   {metrics.get('sharpe_ratio', 0):.2f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def get_best_model(self, 
                      models_metrics: Dict[str, Dict],
                      metric: str = 'f1',
                      higher_is_better: bool = True) -> Tuple[str, Dict]:
        """
        Get best model based on a specific metric
        
        Args:
            models_metrics: Dictionary of {model_name: metrics_dict}
            metric: Metric to use for comparison
            higher_is_better: Whether higher values are better
        
        Returns:
            Tuple of (best_model_name, best_metrics)
        """
        best_model = None
        best_value = -float('inf') if higher_is_better else float('inf')
        best_metrics = None
        
        for model_name, metrics in models_metrics.items():
            value = metrics.get(metric, np.nan)
            
            if np.isnan(value):
                continue
            
            if higher_is_better:
                if value > best_value:
                    best_value = value
                    best_model = model_name
                    best_metrics = metrics
            else:
                if value < best_value:
                    best_value = value
                    best_model = model_name
                    best_metrics = metrics
        
        logger.info(f"Best model: {best_model} ({metric}={best_value:.4f})")
        
        return best_model, best_metrics
    
    def diagnose_overfitting(self, model, X_train, y_train, X_test, y_test) -> Dict:
        """
        Diagnose overfitting by comparing train vs test performance
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with diagnosis results
        """
        from sklearn.metrics import accuracy_score, f1_score
        
        # Training performance
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
        
        # Test performance
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        
        # Calculate gaps
        acc_gap = train_acc - test_acc
        f1_gap = train_f1 - test_f1
        
        print("=" * 60)
        print("OVERFITTING DIAGNOSIS")
        print("=" * 60)
        print(f"{'Metric':<15} {'Train':<12} {'Test':<12} {'Gap':<12} {'Status'}")
        print("-" * 60)
        
        # Accuracy
        acc_status = "✅ OK" if acc_gap < 0.05 else "⚠️ WARNING" if acc_gap < 0.10 else "🔴 OVERFIT"
        print(f"{'Accuracy':<15} {train_acc:<12.2%} {test_acc:<12.2%} {acc_gap:<12.2%} {acc_status}")
        
        # F1
        f1_status = "✅ OK" if f1_gap < 0.05 else "⚠️ WARNING" if f1_gap < 0.10 else "🔴 OVERFIT"
        print(f"{'F1-Score':<15} {train_f1:<12.2%} {test_f1:<12.2%} {f1_gap:<12.2%} {f1_status}")
        
        print("-" * 60)
        
        if acc_gap > 0.10 or f1_gap > 0.10:
            print("🔴 MODEL IS OVERFITTING - Increase regularization:")
            print("   - Reduce max_depth")
            print("   - Increase min_samples_split and min_samples_leaf")
            print("   - Reduce max_samples")
        elif acc_gap > 0.05 or f1_gap > 0.05:
            print("⚠️ MILD OVERFITTING - Consider light regularization")
        else:
            print("✅ MODEL GENERALIZES WELL")
        
        return {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'acc_gap': acc_gap,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'f1_gap': f1_gap,
            'is_overfitting': acc_gap > 0.10 or f1_gap > 0.10
        }
