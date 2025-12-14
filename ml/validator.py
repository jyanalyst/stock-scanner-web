"""
ML Model Validator - Phase 4
Walk-forward validation and threshold optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import logging
from ml.model_trainer import MLModelTrainer
from ml.model_evaluator import ModelEvaluator
from ml.feature_preprocessor import MLFeaturePreprocessor

logger = logging.getLogger(__name__)


class MLValidator:
    """
    Validate ML models using walk-forward and threshold optimization
    """
    
    def __init__(self, target: str = 'win_3d'):
        """
        Initialize validator
        
        Args:
            target: Target variable to predict
        """
        self.target = target
        self.preprocessor = MLFeaturePreprocessor()
        self.trainer = MLModelTrainer(random_state=42)
        self.evaluator = ModelEvaluator()
        
        logger.info(f"Initialized MLValidator (target={target})")
    
    def walk_forward_validation(self,
                                test_periods: List[Tuple[str, str]] = None,
                                embargo_days: int = 20) -> Dict:
        """
        Perform EXPANDING WINDOW walk-forward validation

        CRITICAL FIX: Retrains model for each test period using all available data before it

        Args:
            test_periods: List of (start, end) tuples for test periods
            embargo_days: Days to exclude before test period (prevents data leakage)

        Returns:
            Dictionary with validation results
        """
        logger.info("Starting EXPANDING WINDOW walk-forward validation...")

        # Default test periods (2024 quarters)
        if test_periods is None:
            test_periods = [
                ('2024-01-01', '2024-03-31'),  # Q1
                ('2024-04-01', '2024-06-30'),  # Q2
                ('2024-07-01', '2024-09-30'),  # Q3
                ('2024-10-01', '2024-12-31'),  # Q4
            ]

        # Load full dataset
        logger.info("Loading training data...")
        df_full = self.preprocessor.load_training_data()

        # Ensure entry_date is datetime
        df_full['entry_date'] = pd.to_datetime(df_full['entry_date'])

        # Load Phase 2 features
        if self.preprocessor.selected_features is None:
            self.preprocessor.load_phase2_results()

        # Select features
        features = self.preprocessor.select_features(
            include_technical=True,
            include_fundamental=False
        )

        results = {
            'validation_type': 'expanding_window_walk_forward',
            'embargo_days': embargo_days,
            'test_periods': [],
            'overall_metrics': {},
            'period_metrics': []
        }

        # Test on each period with expanding training window
        all_y_true = []
        all_y_pred = []
        all_y_proba = []

        for i, (period_start, period_end) in enumerate(test_periods):
            logger.info(f"Testing period {i+1}/{len(test_periods)}: {period_start} to {period_end}")

            # CRITICAL FIX: Expanding window - train on ALL data before this test period
            embargo_date = pd.to_datetime(period_start) - pd.Timedelta(days=embargo_days)
            train_mask = df_full['entry_date'] < embargo_date

            df_train = df_full[train_mask].copy()

            if len(df_train) == 0:
                logger.warning(f"No training data available before embargo date {embargo_date}")
                continue

            logger.info(f"Training on expanding window: {df_train['entry_date'].min()} to {df_train['entry_date'].max()} ({len(df_train)} samples)")

            # Prepare training features
            X_train = df_train[features].values
            y_train = df_train[self.target].values

            # Fresh scaler for each period (prevents data leakage)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # CRITICAL FIX: Retrain model for each test period
            logger.info(f"Retraining model for period {period_start} to {period_end}...")
            model = self.trainer.train_random_forest_classifier(
                X_train_scaled,
                y_train,
                tune_hyperparameters=False,  # Use default params for speed
                use_timeseries_cv=False      # Don't use TS CV during WF (already temporal)
            )

            # Filter test data for this period
            test_mask = (df_full['entry_date'] >= period_start) & (df_full['entry_date'] <= period_end)
            df_period = df_full[test_mask].copy()

            if len(df_period) == 0:
                logger.warning(f"No test data for period {period_start} to {period_end}")
                continue

            logger.info(f"Testing on {len(df_period)} samples")

            # Prepare test features
            X_period = df_period[features].values
            y_period = df_period[self.target].values

            # Scale test features with training scaler
            X_period_scaled = scaler.transform(X_period)

            # Predict
            y_pred = model.predict(X_period_scaled)
            y_proba = model.predict_proba(X_period_scaled)

            # Evaluate
            metrics = self.evaluator.evaluate_classification(
                y_period, y_pred, y_proba
            )

            # Calculate profit metrics
            returns = df_period['return_3d'].values if 'return_3d' in df_period.columns else None
            if returns is not None:
                profit_metrics = self.evaluator.calculate_profit_metrics(
                    y_period, y_pred, returns
                )
                metrics.update(profit_metrics)

            period_result = {
                'period': f"{period_start} to {period_end}",
                'train_samples': len(df_train),
                'test_samples': len(df_period),
                'train_end_date': str(df_train['entry_date'].max()),
                'test_start_date': period_start,
                'metrics': metrics
            }

            results['period_metrics'].append(period_result)
            results['test_periods'].append((period_start, period_end))

            # Accumulate for overall metrics
            all_y_true.extend(y_period)
            all_y_pred.extend(y_pred)
            all_y_proba.extend(y_proba)

        # Calculate overall metrics across all test periods
        if all_y_true:
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            all_y_proba = np.array(all_y_proba)

            overall_metrics = self.evaluator.evaluate_classification(
                all_y_true, all_y_pred, all_y_proba
            )

            results['overall_metrics'] = overall_metrics
            results['n_total_samples'] = len(all_y_true)

        logger.info("Expanding window walk-forward validation complete")
        return results
    
    def optimize_threshold(self, 
                          confidence_thresholds: List[float] = None) -> Dict:
        """
        Find optimal confidence threshold
        
        Args:
            confidence_thresholds: List of thresholds to test
        
        Returns:
            Dictionary with threshold optimization results
        """
        logger.info("Starting threshold optimization...")
        
        if confidence_thresholds is None:
            confidence_thresholds = [0.55, 0.60, 0.65, 0.70]
        
        # Load test data
        data = self.preprocessor.get_preprocessed_data(
            target=self.target,
            include_technical=True,
            include_fundamental=False,
            normalization='standard',
            split_method='timeseries'
        )
        
        # Train model
        logger.info("Training model for threshold optimization...")
        model = self.trainer.train_random_forest_classifier(
            data['X_train'], 
            data['y_train'],
            tune_hyperparameters=False
        )
        
        # Get predictions on test set
        y_pred = model.predict(data['X_test'])
        y_proba = model.predict_proba(data['X_test'])
        
        # Get returns from test dataframe
        df_test = data['X_test_df']
        returns = None
        if 'return_3d' in self.preprocessor.df.columns:
            # Get returns for test samples
            test_indices = df_test.index
            returns = self.preprocessor.df.loc[test_indices, 'return_3d'].values
        
        results = {
            'thresholds_tested': confidence_thresholds,
            'threshold_results': []
        }
        
        # Test each threshold
        for threshold in confidence_thresholds:
            logger.info(f"Testing threshold={threshold}...")
            
            # Apply threshold
            high_confidence = y_proba[:, 1] >= threshold
            trade_signals = (y_pred == 1) & high_confidence
            
            # Calculate metrics for trades only
            if trade_signals.sum() > 0:
                y_true_trades = data['y_test'][trade_signals]
                y_pred_trades = y_pred[trade_signals]
                
                # Win rate
                win_rate = y_true_trades.sum() / len(y_true_trades)
                
                # Profit metrics if returns available
                profit_metrics = {}
                if returns is not None and len(returns) == len(trade_signals):
                    returns_trades = returns[trade_signals]
                    total_return = returns_trades.sum()
                    mean_return = returns_trades.mean()
                    
                    wins = returns_trades[y_true_trades == 1]
                    losses = returns_trades[y_true_trades == 0]
                    
                    profit_metrics = {
                        'total_return': total_return,
                        'mean_return': mean_return,
                        'mean_win': wins.mean() if len(wins) > 0 else 0,
                        'mean_loss': losses.mean() if len(losses) > 0 else 0,
                        'profit_factor': abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0
                    }
                
                threshold_result = {
                    'threshold': threshold,
                    'n_trades': trade_signals.sum(),
                    'trade_rate': trade_signals.mean(),
                    'win_rate': win_rate,
                    **profit_metrics
                }
            else:
                threshold_result = {
                    'threshold': threshold,
                    'n_trades': 0,
                    'trade_rate': 0,
                    'win_rate': 0
                }
            
            results['threshold_results'].append(threshold_result)
        
        # Find optimal threshold (best profit factor or win rate)
        if results['threshold_results']:
            best_threshold = max(
                results['threshold_results'],
                key=lambda x: x.get('profit_factor', x.get('win_rate', 0))
            )
            results['optimal_threshold'] = best_threshold
        
        logger.info("Threshold optimization complete")
        return results
    
    def generate_validation_summary(self,
                                   walk_forward_results: Dict,
                                   threshold_results: Dict) -> Dict:
        """
        Generate comprehensive validation summary

        Args:
            walk_forward_results: Results from walk-forward validation
            threshold_results: Results from threshold optimization

        Returns:
            Summary dictionary
        """
        # Get training period from first period result
        first_period = walk_forward_results['period_metrics'][0] if walk_forward_results['period_metrics'] else {}
        train_period = f"{first_period.get('train_end_date', 'Unknown')} (expanding window)"

        summary = {
            'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'target': self.target,
            'walk_forward': {
                'validation_type': walk_forward_results.get('validation_type', 'unknown'),
                'embargo_days': walk_forward_results.get('embargo_days', 0),
                'train_period': train_period,
                'n_test_periods': len(walk_forward_results['test_periods']),
                'overall_accuracy': walk_forward_results['overall_metrics'].get('accuracy', 0),
                'overall_f1': walk_forward_results['overall_metrics'].get('f1', 0),
                'period_accuracies': [
                    p['metrics'].get('accuracy', 0)
                    for p in walk_forward_results['period_metrics']
                ],
                'accuracy_std': np.std([
                    p['metrics'].get('accuracy', 0)
                    for p in walk_forward_results['period_metrics']
                ])
            },
            'threshold_optimization': {
                'optimal_threshold': threshold_results['optimal_threshold']['threshold'],
                'optimal_win_rate': threshold_results['optimal_threshold'].get('win_rate', 0),
                'optimal_profit_factor': threshold_results['optimal_threshold'].get('profit_factor', 0),
                'optimal_n_trades': threshold_results['optimal_threshold'].get('n_trades', 0)
            },
            'recommendation': self._generate_recommendation(walk_forward_results, threshold_results)
        }

        return summary
    
    def _generate_recommendation(self, 
                                walk_forward_results: Dict,
                                threshold_results: Dict) -> str:
        """Generate deployment recommendation"""
        
        overall_acc = walk_forward_results['overall_metrics'].get('accuracy', 0)
        acc_std = np.std([p['metrics'].get('accuracy', 0) for p in walk_forward_results['period_metrics']])
        
        optimal = threshold_results['optimal_threshold']
        win_rate = optimal.get('win_rate', 0)
        profit_factor = optimal.get('profit_factor', 0)
        
        # Decision criteria
        if overall_acc >= 0.52 and acc_std < 0.05 and win_rate >= 0.50 and profit_factor >= 1.5:
            return "DEPLOY - Model shows consistent performance and positive edge"
        elif overall_acc >= 0.50 and win_rate >= 0.48:
            return "DEPLOY WITH CAUTION - Model shows edge but monitor closely"
        else:
            return "DO NOT DEPLOY - Model performance below acceptable threshold. Retrain or collect more data."
