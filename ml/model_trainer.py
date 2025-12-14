"""
ML Model Trainer - Phase 3.2
Trains baseline, Random Forest, and XGBoost models for classification and regression
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from typing import Dict, List, Tuple, Optional
import logging
import joblib
import os
from datetime import datetime

# XGBoost import with fallback
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

logger = logging.getLogger(__name__)


class BaselineClassifier(BaseEstimator, ClassifierMixin):
    """
    Baseline classifier using weighted composite score
    Uses Phase 2 optimal weights to create a score, then applies threshold

    CRITICAL FIX: Standardizes features before applying weights to prevent
    scale mismatch issues (e.g., Daily_Flow=500 vs IBS_Accel=0.05)
    """

    def __init__(self, weights: Dict[str, float], scaler: Optional[object] = None, threshold: float = 0.0):
        """
        Args:
            weights: Dictionary of feature weights from Phase 2
            scaler: StandardScaler fitted on training data (required for proper scaling)
            threshold: Decision threshold (default: 0.0)
        """
        self.weights = weights
        self.scaler = scaler
        self.threshold = threshold
        self.feature_names_ = list(weights.keys())
        self.classes_ = np.array([0, 1])

        if scaler is None:
            logger.warning("⚠️ BaselineClassifier: No scaler provided - predictions may be dominated by high-magnitude features")

    def fit(self, X, y):
        """Fit is a no-op for baseline model"""
        return self

    def predict_proba(self, X):
        """Calculate weighted score and convert to probabilities"""
        if isinstance(X, pd.DataFrame):
            # Convert to numpy array for scaling
            X_array = X[self.feature_names_].values
        else:
            X_array = X

        # CRITICAL FIX: Standardize features before applying weights
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_array)
        else:
            # Fallback to no scaling (not recommended)
            X_scaled = X_array

        # Apply weights to standardized features
        scores = np.zeros(len(X_scaled))
        for i, weight in enumerate(self.weights.values()):
            scores += X_scaled[:, i] * weight

        # Convert scores to probabilities using sigmoid
        proba_positive = 1 / (1 + np.exp(-scores))
        proba_negative = 1 - proba_positive

        return np.column_stack([proba_negative, proba_positive])

    def predict(self, X):
        """Predict class based on threshold"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


class BaselineRegressor(BaseEstimator, RegressorMixin):
    """
    Baseline regressor using IC-weighted linear combination
    Uses Phase 2 IC values and weights for prediction

    CRITICAL FIX: Standardizes features before applying weights to prevent
    scale mismatch issues (e.g., Daily_Flow=500 vs IBS_Accel=0.05)
    """

    def __init__(self, weights: Dict[str, float], scaler: Optional[object] = None, ic_values: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: Dictionary of feature weights from Phase 2
            scaler: StandardScaler fitted on training data (required for proper scaling)
            ic_values: Dictionary of IC values (optional, uses weights if not provided)
        """
        self.weights = weights
        self.scaler = scaler
        self.ic_values = ic_values if ic_values else weights
        self.feature_names_ = list(weights.keys())

        if scaler is None:
            logger.warning("⚠️ BaselineRegressor: No scaler provided - predictions may be dominated by high-magnitude features")

    def fit(self, X, y):
        """Fit is a no-op for baseline model"""
        return self

    def predict(self, X):
        """Predict using weighted linear combination"""
        if isinstance(X, pd.DataFrame):
            # Convert to numpy array for scaling
            X_array = X[self.feature_names_].values
        else:
            X_array = X

        # CRITICAL FIX: Standardize features before applying weights
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_array)
        else:
            # Fallback to no scaling (not recommended)
            X_scaled = X_array

        # Apply weights to standardized features
        predictions = np.zeros(len(X_scaled))
        for i, (weight, ic) in enumerate(zip(self.weights.values(), self.ic_values.values())):
            predictions += X_scaled[:, i] * weight * ic

        return predictions


class MLModelTrainer:
    """
    Main model trainer for Random Forest and XGBoost
    Handles hyperparameter tuning, cross-validation, and model persistence
    """
    
    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        
        logger.info(f"Initialized MLModelTrainer (random_state={random_state})")
    
    def create_baseline_classifier(self, weights: Dict[str, float], scaler: Optional[object] = None) -> BaselineClassifier:
        """Create baseline classifier"""
        logger.info("Creating baseline classifier")
        return BaselineClassifier(weights=weights, scaler=scaler)

    def create_baseline_regressor(self, weights: Dict[str, float], scaler: Optional[object] = None,
                                  ic_values: Optional[Dict[str, float]] = None) -> BaselineRegressor:
        """Create baseline regressor"""
        logger.info("Creating baseline regressor")
        return BaselineRegressor(weights=weights, scaler=scaler, ic_values=ic_values)
    
    def train_random_forest_classifier(self,
                                       X_train: np.ndarray,
                                       y_train: np.ndarray,
                                       tune_hyperparameters: bool = True,
                                       cv_folds: int = 5,
                                       use_timeseries_cv: bool = True) -> RandomForestClassifier:
        """
        Train Random Forest Classifier

        CRITICAL FIX: Uses TimeSeriesSplit to prevent look-ahead bias in CV

        Args:
            X_train: Training features
            y_train: Training labels
            tune_hyperparameters: Whether to perform GridSearch
            cv_folds: Number of cross-validation folds
            use_timeseries_cv: Use TimeSeriesSplit (prevents look-ahead bias)

        Returns:
            Trained RandomForestClassifier
        """
        logger.info("Training Random Forest Classifier...")

        if tune_hyperparameters:
            # Quick hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }

            rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)

            # CRITICAL FIX: Use TimeSeriesSplit to prevent look-ahead bias
            if use_timeseries_cv:
                cv_splitter = TimeSeriesSplit(n_splits=cv_folds, gap=5)
                logger.info(f"Using TimeSeriesSplit with {cv_folds} folds (gap=5) to prevent look-ahead bias")
            else:
                cv_splitter = cv_folds
                logger.info(f"Using standard {cv_folds}-fold CV (WARNING: may have look-ahead bias)")

            grid_search = GridSearchCV(
                rf, param_grid, cv=cv_splitter,
                scoring='f1', n_jobs=-1, verbose=1
            )

            grid_search.fit(X_train, y_train)

            self.best_params['rf_classifier'] = grid_search.best_params_
            self.cv_scores['rf_classifier'] = grid_search.best_score_

            logger.info(f"Best params: {grid_search.best_params_}")
            logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")

            model = grid_search.best_estimator_
        else:
            # Use default parameters
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            logger.info("Trained with default parameters")

        self.models['rf_classifier'] = model
        return model
    
    def train_random_forest_regressor(self,
                                      X_train: np.ndarray,
                                      y_train: np.ndarray,
                                      tune_hyperparameters: bool = True,
                                      cv_folds: int = 5,
                                      use_timeseries_cv: bool = True) -> RandomForestRegressor:
        """
        Train Random Forest Regressor

        CRITICAL FIX: Uses TimeSeriesSplit to prevent look-ahead bias in CV

        Args:
            X_train: Training features
            y_train: Training targets
            tune_hyperparameters: Whether to perform GridSearch
            cv_folds: Number of cross-validation folds
            use_timeseries_cv: Use TimeSeriesSplit (prevents look-ahead bias)

        Returns:
            Trained RandomForestRegressor
        """
        logger.info("Training Random Forest Regressor...")

        if tune_hyperparameters:
            # Quick hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }

            rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)

            # CRITICAL FIX: Use TimeSeriesSplit to prevent look-ahead bias
            if use_timeseries_cv:
                cv_splitter = TimeSeriesSplit(n_splits=cv_folds, gap=5)
                logger.info(f"Using TimeSeriesSplit with {cv_folds} folds (gap=5) to prevent look-ahead bias")
            else:
                cv_splitter = cv_folds
                logger.info(f"Using standard {cv_folds}-fold CV (WARNING: may have look-ahead bias)")

            grid_search = GridSearchCV(
                rf, param_grid, cv=cv_splitter,
                scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1
            )

            grid_search.fit(X_train, y_train)

            self.best_params['rf_regressor'] = grid_search.best_params_
            self.cv_scores['rf_regressor'] = -grid_search.best_score_  # Convert back to positive MAE

            logger.info(f"Best params: {grid_search.best_params_}")
            logger.info(f"Best CV MAE: {-grid_search.best_score_:.4f}")

            model = grid_search.best_estimator_
        else:
            # Use default parameters
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            logger.info("Trained with default parameters")

        self.models['rf_regressor'] = model
        return model
    
    def train_xgboost_classifier(self,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 tune_hyperparameters: bool = True,
                                 cv_folds: int = 5,
                                 use_timeseries_cv: bool = True) -> Optional[object]:
        """
        Train XGBoost Classifier

        CRITICAL FIX: Uses TimeSeriesSplit to prevent look-ahead bias in CV

        Args:
            X_train: Training features
            y_train: Training labels
            tune_hyperparameters: Whether to perform GridSearch
            cv_folds: Number of cross-validation folds
            use_timeseries_cv: Use TimeSeriesSplit (prevents look-ahead bias)

        Returns:
            Trained XGBClassifier or None if XGBoost not available
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping")
            return None

        logger.info("Training XGBoost Classifier...")

        if tune_hyperparameters:
            # Quick hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }

            xgb = XGBClassifier(random_state=self.random_state, n_jobs=-1, eval_metric='logloss')

            # CRITICAL FIX: Use TimeSeriesSplit to prevent look-ahead bias
            if use_timeseries_cv:
                cv_splitter = TimeSeriesSplit(n_splits=cv_folds, gap=5)
                logger.info(f"Using TimeSeriesSplit with {cv_folds} folds (gap=5) to prevent look-ahead bias")
            else:
                cv_splitter = cv_folds
                logger.info(f"Using standard {cv_folds}-fold CV (WARNING: may have look-ahead bias)")

            grid_search = GridSearchCV(
                xgb, param_grid, cv=cv_splitter,
                scoring='f1', n_jobs=-1, verbose=1
            )

            grid_search.fit(X_train, y_train)

            self.best_params['xgb_classifier'] = grid_search.best_params_
            self.cv_scores['xgb_classifier'] = grid_search.best_score_

            logger.info(f"Best params: {grid_search.best_params_}")
            logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")

            model = grid_search.best_estimator_
        else:
            # Use default parameters
            model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)

            logger.info("Trained with default parameters")

        self.models['xgb_classifier'] = model
        return model
    
    def train_xgboost_regressor(self,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                tune_hyperparameters: bool = True,
                                cv_folds: int = 5,
                                use_timeseries_cv: bool = True) -> Optional[object]:
        """
        Train XGBoost Regressor

        CRITICAL FIX: Uses TimeSeriesSplit to prevent look-ahead bias in CV

        Args:
            X_train: Training features
            y_train: Training targets
            tune_hyperparameters: Whether to perform GridSearch
            cv_folds: Number of cross-validation folds
            use_timeseries_cv: Use TimeSeriesSplit (prevents look-ahead bias)

        Returns:
            Trained XGBRegressor or None if XGBoost not available
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping")
            return None

        logger.info("Training XGBoost Regressor...")

        if tune_hyperparameters:
            # Quick hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }

            xgb = XGBRegressor(random_state=self.random_state, n_jobs=-1)

            # CRITICAL FIX: Use TimeSeriesSplit to prevent look-ahead bias
            if use_timeseries_cv:
                cv_splitter = TimeSeriesSplit(n_splits=cv_folds, gap=5)
                logger.info(f"Using TimeSeriesSplit with {cv_folds} folds (gap=5) to prevent look-ahead bias")
            else:
                cv_splitter = cv_folds
                logger.info(f"Using standard {cv_folds}-fold CV (WARNING: may have look-ahead bias)")

            grid_search = GridSearchCV(
                xgb, param_grid, cv=cv_splitter,
                scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1
            )

            grid_search.fit(X_train, y_train)

            self.best_params['xgb_regressor'] = grid_search.best_params_
            self.cv_scores['xgb_regressor'] = -grid_search.best_score_

            logger.info(f"Best params: {grid_search.best_params_}")
            logger.info(f"Best CV MAE: {-grid_search.best_score_:.4f}")

            model = grid_search.best_estimator_
        else:
            # Use default parameters
            model = XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            logger.info("Trained with default parameters")

        self.models['xgb_regressor'] = model
        return model
    
    def train_logistic_regression(self,
                                  X_train: np.ndarray,
                                  y_train: np.ndarray) -> LogisticRegression:
        """
        Train Logistic Regression (for comparison)
        
        Args:
            X_train: Training features (should be normalized)
            y_train: Training labels
        
        Returns:
            Trained LogisticRegression
        """
        logger.info("Training Logistic Regression...")
        
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        logger.info("Logistic Regression trained")
        
        self.models['logistic_regression'] = model
        return model
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """
        Extract feature importance from trained model
        
        Args:
            model: Trained model
            feature_names: List of feature names
        
        Returns:
            Dictionary of feature: importance
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning(f"Model {type(model).__name__} has no feature importance")
            return {}
        
        return dict(zip(feature_names, importances))
    
    def save_model(self, model, model_name: str, 
                   output_dir: str = "models/experiments",
                   metadata: Optional[Dict] = None):
        """
        Save trained model to disk
        
        Args:
            model: Trained model
            model_name: Name for the model file
            output_dir: Directory to save model
            metadata: Optional metadata dictionary
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(output_dir, f"{model_name}_{timestamp}.pkl")
        
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save metadata if provided
        if metadata:
            import json
            metadata_path = os.path.join(output_dir, f"{model_name}_{timestamp}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")
        
        return model_path
    
    def load_model(self, model_path: str):
        """Load trained model from disk"""
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
