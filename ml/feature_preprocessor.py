"""
ML Feature Preprocessor - Phase 3.1
Handles feature selection, normalization, and train-test splitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import logging
import json
import os

logger = logging.getLogger(__name__)


class MLFeaturePreprocessor:
    """
    Prepares features for model training
    
    Key Methods:
    - load_phase2_results(): Load selected features and weights from Phase 2
    - select_features(): Filter features by category
    - normalize_features(): Apply StandardScaler or MinMaxScaler
    - split_data(): Time-series or random split
    - get_preprocessed_data(): Complete preprocessing pipeline
    """
    
    def __init__(self, 
                 data_path: str = "data/ml_training/raw/training_data_complete.parquet",
                 phase2_path: str = "data/ml_training/analysis/"):
        """
        Initialize preprocessor
        
        Args:
            data_path: Path to training data
            phase2_path: Path to Phase 2 results
        """
        self.data_path = data_path
        self.phase2_path = phase2_path
        self.df = None
        self.selected_features = None
        self.optimal_weights = None
        self.scaler = None
        
        logger.info("Initialized MLFeaturePreprocessor")
    
    def load_phase2_results(self) -> Tuple[List[str], Dict[str, float]]:
        """
        Load Phase 2 selected features and optimal weights
        
        Returns:
            Tuple of (selected_features, optimal_weights)
        """
        logger.info("Loading Phase 2 results...")
        
        # Load selected features
        features_path = os.path.join(self.phase2_path, 'selected_features.json')
        with open(features_path, 'r') as f:
            self.selected_features = json.load(f)
        
        # Load optimal weights
        weights_path = os.path.join(self.phase2_path, 'optimal_weights.json')
        with open(weights_path, 'r') as f:
            self.optimal_weights = json.load(f)
        
        logger.info(f"Loaded {len(self.selected_features)} features from Phase 2")
        
        return self.selected_features, self.optimal_weights
    
    def load_training_data(self) -> pd.DataFrame:
        """Load training data"""
        logger.info(f"Loading training data from {self.data_path}")
        
        self.df = pd.read_parquet(self.data_path)
        
        logger.info(f"Loaded {len(self.df):,} samples with {len(self.df.columns)} columns")
        
        return self.df
    
    def categorize_features(self, features: List[str]) -> Dict[str, List[str]]:
        """
        Categorize features by type
        
        Args:
            features: List of feature names
        
        Returns:
            Dictionary of category: [features]
        """
        # Define fundamental features
        fundamental_features = [
            'income_available_for_distribution', 'total_debt', 'dpu',
            'revenue', 'net_income', 'total_assets', 'total_liabilities',
            'portfolio_occupancy', 'gross_margin', 'operating_margin', 
            'net_margin', 'debt_to_equity', 'revenue_yoy_change',
            'eps_yoy_change', 'dpu_yoy_change'
        ]
        
        # Define signal features
        signal_features = [
            'Signal_Bias_Numeric', 'Signal_State_Numeric', 
            'Conviction_Level_Numeric', 'Sentiment_Label_Numeric'
        ]
        
        # Categorize
        categories = {
            'technical': [],
            'fundamental': [],
            'signal': [],
            'other': []
        }
        
        for feature in features:
            if feature in fundamental_features:
                categories['fundamental'].append(feature)
            elif feature in signal_features:
                categories['signal'].append(feature)
            elif any(keyword in feature.lower() for keyword in ['price', 'volume', 'flow', 'mpi', 'ibs', 'vpi', 'rvol', 'vwap', 'percentile', 'accel', 'velocity']):
                categories['technical'].append(feature)
            else:
                categories['other'].append(feature)
        
        logger.info(f"Categorized features: Technical={len(categories['technical'])}, "
                   f"Fundamental={len(categories['fundamental'])}, "
                   f"Signal={len(categories['signal'])}, "
                   f"Other={len(categories['other'])}")
        
        return categories
    
    def select_features(self, 
                       include_technical: bool = True,
                       include_fundamental: bool = False,
                       include_signal: bool = True,
                       custom_features: Optional[List[str]] = None) -> List[str]:
        """
        Select features based on categories or custom list
        
        Args:
            include_technical: Include technical features
            include_fundamental: Include fundamental features
            include_signal: Include signal features
            custom_features: Custom feature list (overrides category selection)
        
        Returns:
            List of selected feature names
        """
        if custom_features is not None:
            logger.info(f"Using custom feature selection: {len(custom_features)} features")
            return custom_features
        
        # Load Phase 2 results if not already loaded
        if self.selected_features is None:
            self.load_phase2_results()
        
        # Categorize features
        categories = self.categorize_features(self.selected_features)
        
        # Select based on flags
        selected = []
        
        if include_technical:
            selected.extend(categories['technical'])
        
        if include_fundamental:
            selected.extend(categories['fundamental'])
        
        if include_signal:
            selected.extend(categories['signal'])
        
        # Always include 'other' category
        selected.extend(categories['other'])
        
        logger.info(f"Selected {len(selected)} features based on category filters")
        
        return selected
    
    def renormalize_weights(self, features: List[str]) -> Dict[str, float]:
        """
        Renormalize weights for selected features
        
        Args:
            features: List of selected features
        
        Returns:
            Dictionary of renormalized weights
        """
        if self.optimal_weights is None:
            self.load_phase2_results()
        
        # Get weights for selected features
        selected_weights = {f: self.optimal_weights[f] for f in features if f in self.optimal_weights}
        
        # Renormalize to sum to 1
        total_weight = sum(selected_weights.values())
        renormalized = {f: w / total_weight for f, w in selected_weights.items()}
        
        logger.info(f"Renormalized weights for {len(renormalized)} features")
        
        return renormalized
    
    def normalize_features(self, 
                          X_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          method: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize features using StandardScaler or MinMaxScaler
        
        Args:
            X_train: Training features
            X_test: Test features
            method: 'standard' or 'minmax' or 'none'
        
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        if method == 'none':
            logger.info("No normalization applied")
            return X_train.values, X_test.values
        
        if method == 'standard':
            self.scaler = StandardScaler()
            logger.info("Applying StandardScaler normalization")
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
            logger.info("Applying MinMaxScaler normalization")
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Fit on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Normalized features: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")
        
        return X_train_scaled, X_test_scaled
    
    def split_data_timeseries(self, 
                              df: pd.DataFrame,
                              features: List[str],
                              target: str,
                              split_date: str = '2024-01-01') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data by date (time-series split)
        
        Args:
            df: Full dataset
            features: Feature columns
            target: Target column
            split_date: Date to split on (train before, test after)
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Performing time-series split at {split_date}")
        
        # Ensure entry_date is datetime
        if 'entry_date' not in df.columns:
            raise ValueError("entry_date column not found in data")
        
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        split_date = pd.to_datetime(split_date)
        
        # Split by date
        train_mask = df['entry_date'] < split_date
        test_mask = df['entry_date'] >= split_date
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        # Extract features and target
        X_train = train_df[features]
        X_test = test_df[features]
        y_train = train_df[target]
        y_test = test_df[target]
        
        logger.info(f"Train set: {len(X_train):,} samples ({train_df['entry_date'].min()} to {train_df['entry_date'].max()})")
        logger.info(f"Test set: {len(X_test):,} samples ({test_df['entry_date'].min()} to {test_df['entry_date'].max()})")
        logger.info(f"Train/Test split: {len(X_train)/(len(X_train)+len(X_test))*100:.1f}% / {len(X_test)/(len(X_train)+len(X_test))*100:.1f}%")
        
        return X_train, X_test, y_train, y_test
    
    def split_data_random(self,
                         df: pd.DataFrame,
                         features: List[str],
                         target: str,
                         test_size: float = 0.3,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data randomly
        
        Args:
            df: Full dataset
            features: Feature columns
            target: Target column
            test_size: Fraction for test set
            random_state: Random seed
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Performing random split (test_size={test_size})")
        
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Train set: {len(X_train):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")
        logger.info(f"Train/Test split: {len(X_train)/(len(X_train)+len(X_test))*100:.1f}% / {len(X_test)/(len(X_train)+len(X_test))*100:.1f}%")
        
        return X_train, X_test, y_train, y_test
    
    def get_preprocessed_data(self,
                             target: str = 'win_3d',
                             include_technical: bool = True,
                             include_fundamental: bool = False,
                             include_signal: bool = True,
                             normalization: str = 'standard',
                             split_method: str = 'timeseries',
                             split_date: str = '2024-01-01',
                             test_size: float = 0.3) -> Dict:
        """
        Complete preprocessing pipeline
        
        Args:
            target: Target variable ('win_3d', 'return_3d', etc.)
            include_technical: Include technical features
            include_fundamental: Include fundamental features
            include_signal: Include signal features
            normalization: 'standard', 'minmax', or 'none'
            split_method: 'timeseries' or 'random'
            split_date: Date for time-series split
            test_size: Test size for random split
        
        Returns:
            Dictionary with preprocessed data and metadata
        """
        logger.info("=" * 80)
        logger.info("STARTING FEATURE PREPROCESSING PIPELINE")
        logger.info("=" * 80)
        
        # Step 1: Load data
        if self.df is None:
            self.load_training_data()
        
        # Step 2: Load Phase 2 results
        if self.selected_features is None:
            self.load_phase2_results()
        
        # Step 3: Select features
        features = self.select_features(
            include_technical=include_technical,
            include_fundamental=include_fundamental,
            include_signal=include_signal
        )
        
        # Step 4: Renormalize weights
        weights = self.renormalize_weights(features)
        
        # Step 5: Split data
        if split_method == 'timeseries':
            X_train, X_test, y_train, y_test = self.split_data_timeseries(
                self.df, features, target, split_date
            )
        else:
            X_train, X_test, y_train, y_test = self.split_data_random(
                self.df, features, target, test_size
            )
        
        # Step 6: Normalize features
        X_train_scaled, X_test_scaled = self.normalize_features(
            X_train, X_test, method=normalization
        )
        
        # Step 7: Compile results
        results = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train.values,
            'y_test': y_test.values,
            'X_train_df': X_train,  # Original unscaled for reference
            'X_test_df': X_test,
            'features': features,
            'weights': weights,
            'scaler': self.scaler,
            'target': target,
            'normalization': normalization,
            'split_method': split_method,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': len(features)
        }
        
        logger.info("=" * 80)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Features: {len(features)}")
        logger.info(f"Train samples: {len(X_train):,}")
        logger.info(f"Test samples: {len(X_test):,}")
        logger.info(f"Normalization: {normalization}")
        logger.info(f"Split method: {split_method}")
        
        return results
