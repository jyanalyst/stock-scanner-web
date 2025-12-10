"""
ML Prediction Engine - Phase 3.3
Apply trained models to new data
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from ml.model_loader import MLModelLoader

logger = logging.getLogger(__name__)


class MLPredictionEngine:
    """
    Apply trained ML models to generate predictions
    """
    
    def __init__(self, model_loader: Optional[MLModelLoader] = None):
        """
        Initialize prediction engine
        
        Args:
            model_loader: MLModelLoader instance (creates new one if None)
        """
        self.model_loader = model_loader if model_loader else MLModelLoader()
        logger.info("Initialized MLPredictionEngine")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for prediction
        
        Args:
            df: DataFrame with raw features
        
        Returns:
            Tuple of (feature_array, feature_names)
        """
        if not self.model_loader.is_model_loaded():
            raise ValueError("No model loaded. Call model_loader.load_production_model() first.")
        
        metadata = self.model_loader.loaded_metadata
        if metadata is None:
            raise ValueError("No metadata available")
        
        required_features = metadata.get('features', [])
        if not required_features:
            raise ValueError("No features specified in metadata")
        
        # Validate features
        is_valid, missing_features = self.model_loader.validate_model_features(df)
        if not is_valid:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Extract features in correct order
        X = df[required_features].values
        
        logger.info(f"Prepared {len(X)} samples with {len(required_features)} features")
        return X, required_features
    
    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize features using loaded scaler
        
        Args:
            X: Raw feature array
        
        Returns:
            Normalized feature array
        """
        if self.model_loader.loaded_scaler is None:
            logger.warning("No scaler loaded - returning unnormalized features")
            return X
        
        X_scaled = self.model_loader.loaded_scaler.transform(X)
        logger.info(f"Normalized features: mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")
        
        return X_scaled
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions (class labels)
        
        Args:
            df: DataFrame with features
        
        Returns:
            Array of predictions (0 or 1)
        """
        if not self.model_loader.is_model_loaded():
            raise ValueError("No model loaded")
        
        # Prepare and normalize features
        X, _ = self.prepare_features(df)
        X_scaled = self.normalize_features(X)
        
        # Predict
        predictions = self.model_loader.loaded_model.predict(X_scaled)
        
        logger.info(f"Generated {len(predictions)} predictions")
        logger.info(f"Positive predictions: {predictions.sum()} ({predictions.mean()*100:.1f}%)")
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate prediction probabilities
        
        Args:
            df: DataFrame with features
        
        Returns:
            Array of probabilities (n_samples, 2) for [negative, positive] class
        """
        if not self.model_loader.is_model_loaded():
            raise ValueError("No model loaded")
        
        # Check if model supports predict_proba
        if not hasattr(self.model_loader.loaded_model, 'predict_proba'):
            logger.warning("Model does not support predict_proba")
            # Return dummy probabilities based on predictions
            predictions = self.predict(df)
            proba = np.zeros((len(predictions), 2))
            proba[predictions == 0, 0] = 1.0
            proba[predictions == 1, 1] = 1.0
            return proba
        
        # Prepare and normalize features
        X, _ = self.prepare_features(df)
        X_scaled = self.normalize_features(X)
        
        # Predict probabilities
        proba = self.model_loader.loaded_model.predict_proba(X_scaled)
        
        logger.info(f"Generated probabilities for {len(proba)} samples")
        logger.info(f"Mean positive probability: {proba[:, 1].mean():.3f}")
        
        return proba
    
    def predict_with_confidence(self, df: pd.DataFrame, 
                               confidence_threshold: float = 0.5) -> pd.DataFrame:
        """
        Generate predictions with confidence scores
        
        Args:
            df: DataFrame with features
            confidence_threshold: Minimum confidence to predict positive (default: 0.5)
        
        Returns:
            DataFrame with predictions and confidence scores
        """
        # Get predictions and probabilities
        predictions = self.predict(df)
        proba = self.predict_proba(df)
        
        # Create results dataframe
        results = df.copy()
        results['ml_prediction'] = predictions
        results['ml_confidence'] = proba[:, 1]  # Probability of positive class
        results['ml_prediction_label'] = results['ml_prediction'].astype(int).map({0: 'LOSS', 1: 'WIN'})
        
        # Apply confidence threshold
        results['ml_high_confidence'] = results['ml_confidence'] >= confidence_threshold
        results['ml_trade_signal'] = (
            (results['ml_prediction'] == 1) & 
            (results['ml_high_confidence'])
        ).map({True: 'BUY', False: 'PASS'})
        
        logger.info(f"Generated predictions with confidence threshold={confidence_threshold}")
        logger.info(f"BUY signals: {(results['ml_trade_signal'] == 'BUY').sum()}")
        
        return results
    
    def batch_predict(self, df: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
        """
        Generate predictions in batches (for large datasets)
        
        Args:
            df: DataFrame with features
            batch_size: Number of samples per batch
        
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Batch prediction: {len(df)} samples, batch_size={batch_size}")
        
        all_predictions = []
        all_proba = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            predictions = self.predict(batch)
            proba = self.predict_proba(batch)
            
            all_predictions.append(predictions)
            all_proba.append(proba)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
        
        # Combine results
        predictions = np.concatenate(all_predictions)
        proba = np.concatenate(all_proba)
        
        # Create results dataframe
        results = df.copy()
        results['ml_prediction'] = predictions
        results['ml_confidence'] = proba[:, 1]
        results['ml_prediction_label'] = results['ml_prediction'].map({0: 'LOSS', 1: 'WIN'})
        
        return results
    
    def get_feature_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get feature values used for prediction
        
        Args:
            df: DataFrame with features
        
        Returns:
            DataFrame with only model features
        """
        if not self.model_loader.is_model_loaded():
            raise ValueError("No model loaded")
        
        metadata = self.model_loader.loaded_metadata
        if metadata is None:
            raise ValueError("No metadata available")
        
        required_features = metadata.get('features', [])
        
        return df[required_features].copy()
    
    def explain_prediction(self, df: pd.DataFrame, index: int = 0) -> Dict:
        """
        Explain a single prediction (feature contributions)
        
        Args:
            df: DataFrame with features
            index: Index of sample to explain
        
        Returns:
            Dictionary with explanation
        """
        if not self.model_loader.is_model_loaded():
            raise ValueError("No model loaded")
        
        # Get single sample
        sample = df.iloc[[index]]
        
        # Get prediction
        prediction = self.predict(sample)[0]
        proba = self.predict_proba(sample)[0]
        
        # Get feature values
        X, feature_names = self.prepare_features(sample)
        feature_values = X[0]
        
        # Get feature importance (if available)
        feature_importance = {}
        if hasattr(self.model_loader.loaded_model, 'feature_importances_'):
            importances = self.model_loader.loaded_model.feature_importances_
            feature_importance = dict(zip(feature_names, importances))
        
        explanation = {
            'prediction': int(prediction),
            'prediction_label': 'WIN' if prediction == 1 else 'LOSS',
            'confidence': float(proba[1]),
            'feature_values': dict(zip(feature_names, feature_values)),
            'feature_importance': feature_importance
        }
        
        return explanation
