"""
ML Model Loader - Phase 3.3
Load and manage production ML models
"""

import os
import json
import joblib
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class MLModelLoader:
    """
    Load and manage production ML models
    """
    
    def __init__(self, production_dir: str = "models/production"):
        """
        Initialize model loader
        
        Args:
            production_dir: Directory containing production models
        """
        self.production_dir = production_dir
        self.loaded_model = None
        self.loaded_scaler = None
        self.loaded_metadata = None
        
        logger.info(f"Initialized MLModelLoader (dir={production_dir})")
    
    def list_available_models(self) -> List[Dict]:
        """
        List all available models in production directory
        
        Returns:
            List of model info dictionaries
        """
        models = []
        
        if not os.path.exists(self.production_dir):
            logger.warning(f"Production directory not found: {self.production_dir}")
            return models
        
        # Find all .pkl files
        for filename in os.listdir(self.production_dir):
            if filename.endswith('.pkl') and not filename.startswith('scaler'):
                model_path = os.path.join(self.production_dir, filename)
                metadata_path = model_path.replace('.pkl', '_metadata.json')
                
                model_info = {
                    'filename': filename,
                    'model_path': model_path,
                    'metadata_path': metadata_path if os.path.exists(metadata_path) else None,
                    'size_mb': os.path.getsize(model_path) / (1024 * 1024),
                    'modified': datetime.fromtimestamp(os.path.getmtime(model_path))
                }
                
                # Load metadata if available
                if model_info['metadata_path']:
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        model_info['metadata'] = metadata
                        model_info['model_type'] = metadata.get('model_type', 'Unknown')
                        model_info['accuracy'] = metadata.get('metrics', {}).get('accuracy', None)
                        model_info['f1'] = metadata.get('metrics', {}).get('f1', None)
                    except Exception as e:
                        logger.warning(f"Could not load metadata for {filename}: {e}")
                        model_info['metadata'] = None
                
                models.append(model_info)
        
        # Sort by modification date (newest first)
        models.sort(key=lambda x: x['modified'], reverse=True)
        
        logger.info(f"Found {len(models)} models in production")
        return models
    
    def load_production_model(self, model_filename: Optional[str] = None) -> Dict:
        """
        Load production model, scaler, and metadata
        
        Args:
            model_filename: Specific model file to load (if None, loads latest)
        
        Returns:
            Dictionary with model, scaler, and metadata
        """
        logger.info("Loading production model...")
        
        # Find model to load
        if model_filename is None:
            # Load latest model
            models = self.list_available_models()
            if not models:
                raise FileNotFoundError("No models found in production directory")
            model_info = models[0]  # Latest model
            model_path = model_info['model_path']
            metadata_path = model_info['metadata_path']
        else:
            model_path = os.path.join(self.production_dir, model_filename)
            metadata_path = model_path.replace('.pkl', '_metadata.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        self.loaded_model = model
        
        # Load scaler
        scaler_path = os.path.join(self.production_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            logger.info(f"Loading scaler from {scaler_path}")
            scaler = joblib.load(scaler_path)
            self.loaded_scaler = scaler
        else:
            logger.warning("Scaler not found - predictions may fail")
            scaler = None
        
        # Load metadata
        metadata = None
        if os.path.exists(metadata_path):
            logger.info(f"Loading metadata from {metadata_path}")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.loaded_metadata = metadata
        else:
            logger.warning("Metadata not found")
        
        result = {
            'model': model,
            'scaler': scaler,
            'metadata': metadata,
            'model_path': model_path,
            'scaler_path': scaler_path if os.path.exists(scaler_path) else None,
            'metadata_path': metadata_path if os.path.exists(metadata_path) else None
        }
        
        logger.info("Model loaded successfully")
        return result
    
    def get_model_metadata(self, model_filename: Optional[str] = None) -> Optional[Dict]:
        """
        Get metadata for a specific model
        
        Args:
            model_filename: Model file to get metadata for (if None, uses loaded model)
        
        Returns:
            Metadata dictionary or None
        """
        if model_filename is None:
            # Return loaded metadata
            return self.loaded_metadata
        
        # Load metadata from file
        metadata_path = os.path.join(
            self.production_dir,
            model_filename.replace('.pkl', '_metadata.json')
        )
        
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata not found: {metadata_path}")
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return None
    
    def validate_model_features(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that dataframe has all required features
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Tuple of (is_valid, missing_features)
        """
        if self.loaded_metadata is None:
            logger.warning("No metadata loaded - cannot validate features")
            return False, []
        
        required_features = self.loaded_metadata.get('features', [])
        
        if not required_features:
            logger.warning("No features specified in metadata")
            return True, []
        
        missing_features = [f for f in required_features if f not in df.columns]
        
        is_valid = len(missing_features) == 0
        
        if not is_valid:
            logger.warning(f"Missing {len(missing_features)} features: {missing_features}")
        else:
            logger.info("All required features present")
        
        return is_valid, missing_features
    
    def get_model_summary(self) -> Dict:
        """
        Get summary of loaded model
        
        Returns:
            Summary dictionary
        """
        if self.loaded_model is None:
            return {'status': 'No model loaded'}
        
        summary = {
            'status': 'Model loaded',
            'model_type': type(self.loaded_model).__name__,
            'has_scaler': self.loaded_scaler is not None,
            'has_metadata': self.loaded_metadata is not None
        }
        
        if self.loaded_metadata:
            summary.update({
                'task': self.loaded_metadata.get('task', 'Unknown'),
                'target': self.loaded_metadata.get('target', 'Unknown'),
                'n_features': self.loaded_metadata.get('n_features', 0),
                'features': self.loaded_metadata.get('features', []),
                'train_samples': self.loaded_metadata.get('train_samples', 0),
                'test_samples': self.loaded_metadata.get('test_samples', 0),
                'metrics': self.loaded_metadata.get('metrics', {}),
                'normalization': self.loaded_metadata.get('normalization', 'Unknown'),
                'split_method': self.loaded_metadata.get('split_method', 'Unknown')
            })
        
        return summary
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded"""
        return self.loaded_model is not None
    
    def unload_model(self):
        """Unload current model from memory"""
        self.loaded_model = None
        self.loaded_scaler = None
        self.loaded_metadata = None
        logger.info("Model unloaded")
