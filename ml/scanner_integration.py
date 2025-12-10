"""
ML Scanner Integration - Phase 5
Bridges ML predictions with production scanner
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Optional, Tuple
import json

logger = logging.getLogger(__name__)


class ScannerMLIntegration:
    """
    Integrates ML predictions into scanner results
    
    Key Features:
    - Non-invasive: Augments existing scanner, doesn't replace
    - Graceful fallback: Scanner works even if ML fails
    - Transparent: Shows confidence scores and probabilities
    """
    
    def __init__(self, 
                 model_path: str = "models/production",
                 threshold: float = 0.60):
        """
        Initialize ML integration
        
        Args:
            model_path: Path to production model directory
            threshold: Confidence threshold for BUY signals (from validation)
        """
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        self.scaler = None
        self.features = None
        self.model_metadata = None
        
        # Try to load model components
        self._load_model_components()
        
        logger.info(f"Initialized ScannerMLIntegration (threshold={threshold})")
    
    def _load_model_components(self) -> bool:
        """Load model, scaler, and feature list"""
        try:
            from ml.model_loader import MLModelLoader
            from pages.common.error_handler import structured_logger
            
            loader = MLModelLoader()
            
            # Load production model (uses load_production_model method)
            model_components = loader.load_production_model()
            
            self.model = model_components['model']
            self.scaler = model_components['scaler']
            self.model_metadata = model_components['metadata']
            
            # Get features from metadata
            if self.model_metadata and 'features' in self.model_metadata:
                self.features = self.model_metadata['features']
            else:
                # Fallback: Load from Phase 2 results
                features_path = "data/ml_training/analysis/selected_features.json"
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        self.features = json.load(f)
                else:
                    logger.warning("Feature list not found")
                    return False
            
            structured_logger.log('INFO', 'ScannerMLIntegration', 
                                f"Loaded model with {len(self.features)} features")
            return True
            
        except Exception as e:
            from pages.common.error_handler import structured_logger
            structured_logger.log('ERROR', 'ScannerMLIntegration', 
                                f"Failed to load model components: {e}")
            return False
    
    def is_ml_available(self) -> bool:
        """Check if ML model is available and ready"""
        return (self.model is not None and 
                self.scaler is not None and 
                self.features is not None)
    
    def get_ml_status(self) -> Dict:
        """
        Get ML model status and metadata
        
        Returns:
            Dictionary with model info
        """
        if not self.is_ml_available():
            return {
                'available': False,
                'message': 'ML model not loaded'
            }
        
        # Get validation metrics from metadata or use defaults
        if self.model_metadata:
            accuracy = self.model_metadata.get('test_accuracy', 0.5248)
            win_rate = 0.53  # From validation
            profit_factor = 3.22  # From validation
        else:
            accuracy = 0.5248
            win_rate = 0.53
            profit_factor = 3.22
        
        return {
            'available': True,
            'threshold': self.threshold,
            'accuracy': accuracy,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'n_features': len(self.features),
            'model_type': self.model_metadata.get('model_type', 'RandomForest') if self.model_metadata else 'RandomForest',
            'target': 'win_3d'
        }
    
    def add_ml_predictions(self, scan_results: pd.DataFrame) -> pd.DataFrame:
        """
        Add ML predictions to scanner results
        
        Args:
            scan_results: DataFrame from scanner with technical indicators
        
        Returns:
            DataFrame with added ML columns:
            - ML_Prediction: 0 (loss) or 1 (win)
            - ML_Confidence: Probability score (0-1)
            - ML_Signal: "🤖 BUY" if confidence >= threshold
            - ML_Win_Probability: Formatted percentage
            - ML_Trade_Eligible: Boolean for filtering
        """
        if not self.is_ml_available():
            logger.warning("ML not available - skipping predictions")
            return scan_results
        
        if scan_results.empty:
            logger.warning("Empty scan results - skipping ML predictions")
            return scan_results
        
        try:
            # Create copy to avoid modifying original
            results = scan_results.copy()
            
            # Check for required features
            missing_features = [f for f in self.features if f not in results.columns]
            if missing_features:
                logger.error(f"Missing features: {missing_features[:5]}...")
                # Add empty ML columns
                return self._add_empty_ml_columns(results)
            
            # Extract features in correct order
            X = results[self.features].values
            
            # Handle NaN values
            if np.isnan(X).any():
                logger.warning("NaN values detected in features - filling with 0")
                X = np.nan_to_num(X, nan=0.0)
            
            # Scale features
            if self.scaler:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Generate predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            # Extract win probability (class 1)
            win_probs = probabilities[:, 1]
            
            # Add ML columns
            results['ML_Prediction'] = predictions
            results['ML_Confidence'] = win_probs
            results['ML_Win_Probability'] = (win_probs * 100).round(1)
            
            # Generate BUY signals based on threshold
            results['ML_Signal'] = results['ML_Confidence'].apply(
                lambda x: '🤖 BUY' if x >= self.threshold else '—'
            )
            
            # Trade eligibility flag
            results['ML_Trade_Eligible'] = results['ML_Confidence'] >= self.threshold
            
            # Add confidence tier for display
            results['ML_Confidence_Tier'] = results['ML_Confidence'].apply(
                self._get_confidence_tier
            )
            
            logger.info(f"Added ML predictions: {results['ML_Trade_Eligible'].sum()} BUY signals out of {len(results)} stocks")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to add ML predictions: {e}")
            # Return original results with empty ML columns
            return self._add_empty_ml_columns(scan_results)
    
    def _add_empty_ml_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add empty ML columns when predictions fail"""
        df = df.copy()
        df['ML_Prediction'] = np.nan
        df['ML_Confidence'] = np.nan
        df['ML_Win_Probability'] = np.nan
        df['ML_Signal'] = '—'
        df['ML_Trade_Eligible'] = False
        df['ML_Confidence_Tier'] = '—'
        return df
    
    def _get_confidence_tier(self, confidence: float) -> str:
        """
        Categorize confidence into tiers
        
        Args:
            confidence: Confidence score (0-1)
        
        Returns:
            Tier label
        """
        if pd.isna(confidence):
            return '—'
        elif confidence >= 0.70:
            return '⭐⭐⭐ HIGH'
        elif confidence >= 0.60:
            return '⭐⭐ GOOD'
        elif confidence >= 0.55:
            return '⭐ MARGINAL'
        else:
            return '⚠️ LOW'
    
    def get_ml_summary(self, results_with_ml: pd.DataFrame) -> Dict:
        """
        Generate ML prediction summary statistics
        
        Args:
            results_with_ml: Scanner results with ML predictions
        
        Returns:
            Dictionary with summary stats
        """
        if 'ML_Signal' not in results_with_ml.columns:
            return {'available': False}
        
        ml_buy_signals = results_with_ml[results_with_ml['ML_Signal'] == '🤖 BUY']
        
        if len(ml_buy_signals) == 0:
            return {
                'available': True,
                'total_stocks': len(results_with_ml),
                'ml_buy_count': 0,
                'avg_confidence': 0,
                'avg_win_prob': 0,
                'tech_ml_agree': 0,
                'agreement_rate': 0,
                'high_confidence_count': 0,
                'expected_win_rate': 0.53,
                'expected_profit_factor': 3.22
            }
        
        # Calculate agreement with technical signals
        tech_ml_agree = 0
        if 'Signal_Bias' in results_with_ml.columns:
            tech_ml_agree = (
                (results_with_ml['Signal_Bias'] == '🟢 BULLISH') &
                (results_with_ml['ML_Signal'] == '🤖 BUY')
            ).sum()
        
        return {
            'available': True,
            'total_stocks': len(results_with_ml),
            'ml_buy_count': len(ml_buy_signals),
            'avg_confidence': ml_buy_signals['ML_Confidence'].mean(),
            'avg_win_prob': ml_buy_signals['ML_Win_Probability'].mean(),
            'tech_ml_agree': tech_ml_agree,
            'agreement_rate': (tech_ml_agree / len(ml_buy_signals) * 100) if len(ml_buy_signals) > 0 else 0,
            'high_confidence_count': (ml_buy_signals['ML_Confidence'] >= 0.70).sum(),
            'expected_win_rate': 0.53,  # From validation
            'expected_profit_factor': 3.22  # From validation
        }
    
    def filter_by_ml_signal(self, 
                           results: pd.DataFrame,
                           filter_type: str = 'all') -> pd.DataFrame:
        """
        Filter results by ML signal type
        
        Args:
            results: Scanner results with ML predictions
            filter_type: 'all', 'ml_buy', 'tech_ml_agree', 'divergence'
        
        Returns:
            Filtered DataFrame
        """
        if 'ML_Signal' not in results.columns:
            return results
        
        if filter_type == 'all':
            return results
        
        elif filter_type == 'ml_buy':
            return results[results['ML_Signal'] == '🤖 BUY']
        
        elif filter_type == 'tech_ml_agree':
            # Both technical and ML bullish
            if 'Signal_Bias' in results.columns:
                return results[
                    (results['Signal_Bias'] == '🟢 BULLISH') &
                    (results['ML_Signal'] == '🤖 BUY')
                ]
            else:
                return results[results['ML_Signal'] == '🤖 BUY']
        
        elif filter_type == 'divergence':
            # Technical and ML disagree
            if 'Signal_Bias' in results.columns:
                return results[
                    ((results['Signal_Bias'] == '🟢 BULLISH') & (results['ML_Signal'] != '🤖 BUY')) |
                    ((results['Signal_Bias'] == '🔴 BEARISH') & (results['ML_Signal'] == '🤖 BUY'))
                ]
            else:
                return results
        
        else:
            return results
