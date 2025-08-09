"""
Unified Model Loader for Anomaly Detection
==========================================
Handles loading models trained by the new unified training script

Usage:
    from src.model_loader import ModelLoader
    
    loader = ModelLoader()
    models, scalers = loader.load_models('nsl_kdd')  # Load specific dataset models
    models, scalers = loader.load_best_models()      # Load best performing models
"""

import os
import joblib
import yaml
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class ModelLoader:
    """Unified model loader for different training approaches"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize model loader"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_dir = self.config['models']['save_path']
        
    def list_available_models(self) -> Dict[str, list]:
        """List all available trained models"""
        available = {
            'datasets': [],
            'model_files': []
        }
        
        if not os.path.exists(self.models_dir):
            return available
        
        files = os.listdir(self.models_dir)
        
        # Check for dataset-specific models (new format)
        datasets = ['nsl_kdd', 'cicids2017', 'unsw_nb15', 'ton_iot']
        for dataset in datasets:
            if (f'{dataset}_isolation_forest.pkl' in files or 
                f'{dataset}_random_forest.pkl' in files or
                f'{dataset}_xgboost.pkl' in files or
                f'{dataset}_kmeans.pkl' in files or
                f'{dataset}_autoencoder.pkl' in files):
                available['datasets'].append(dataset)
        
        # Check for generic models (old format)
        if 'isolation_forest.pkl' in files or 'random_forest.pkl' in files:
            available['model_files'] = ['isolation_forest.pkl', 'random_forest.pkl']
        
        return available
    
    def load_models(self, dataset: str = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load models for a specific dataset or best available models
        
        Args:
            dataset (str): Dataset name (nsl_kdd, cicids2017, unsw_nb15, ton_iot)
                          If None, loads best available models
        
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: (models, scalers)
        """
        models = {}
        scalers = {}
        
        if dataset:
            # Load dataset-specific models
            models, scalers = self._load_dataset_models(dataset)
        else:
            # Load best available models
            models, scalers = self._load_best_models()
        
        if not models:
            raise FileNotFoundError("No trained models found. Please run training first.")
        
        logger.info(f"Loaded {len(models)} models and {len(scalers)} scalers")
        return models, scalers
    
    def _load_dataset_models(self, dataset: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load models for a specific dataset"""
        models = {}
        scalers = {}
        
        # Load Isolation Forest
        iso_path = os.path.join(self.models_dir, f'{dataset}_isolation_forest.pkl')
        if os.path.exists(iso_path):
            models['isolation_forest'] = joblib.load(iso_path)
            logger.info(f"Loaded {dataset} Isolation Forest model")
        
        # Load Random Forest  
        rf_path = os.path.join(self.models_dir, f'{dataset}_random_forest.pkl')
        if os.path.exists(rf_path):
            models['random_forest'] = joblib.load(rf_path)
            logger.info(f"Loaded {dataset} Random Forest model")
        
        # Load XGBoost
        xgb_path = os.path.join(self.models_dir, f'{dataset}_xgboost.pkl')
        if os.path.exists(xgb_path):
            models['xgboost'] = joblib.load(xgb_path)
            logger.info(f"Loaded {dataset} XGBoost model")
        
        # Load K-Means
        kmeans_path = os.path.join(self.models_dir, f'{dataset}_kmeans.pkl')
        if os.path.exists(kmeans_path):
            models['kmeans'] = joblib.load(kmeans_path)
            logger.info(f"Loaded {dataset} K-Means model")
        
        # Load Autoencoder
        autoencoder_path = os.path.join(self.models_dir, f'{dataset}_autoencoder.pkl')
        if os.path.exists(autoencoder_path):
            models['autoencoder'] = joblib.load(autoencoder_path)
            logger.info(f"Loaded {dataset} Autoencoder model")
            
            # Load autoencoder threshold as metadata, not a separate model
            threshold_path = os.path.join(self.models_dir, f'{dataset}_autoencoder_threshold.pkl')
            if os.path.exists(threshold_path):
                threshold = joblib.load(threshold_path)
                # Store threshold as metadata with the autoencoder
                models['autoencoder_threshold'] = threshold
                logger.info(f"Loaded {dataset} Autoencoder threshold: {threshold:.4f}")
        
        # Load scaler
        scaler_path = os.path.join(self.models_dir, f'{dataset}_scaler.pkl')
        if os.path.exists(scaler_path):
            scalers[dataset] = joblib.load(scaler_path)
            logger.info(f"Loaded {dataset} scaler")
        
        return models, scalers
    
    def _load_best_models(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load best available models (fallback to generic models)"""
        models = {}
        scalers = {}
        
        available = self.list_available_models()
        
        # Try to load from best performing dataset (NSL-KDD first, then others)
        priority_datasets = ['nsl_kdd', 'unsw_nb15', 'cicids2017', 'ton_iot']
        
        for dataset in priority_datasets:
            if dataset in available['datasets']:
                return self._load_dataset_models(dataset)
        
        # Fallback to generic models (old format)
        iso_path = os.path.join(self.models_dir, 'isolation_forest.pkl')
        if os.path.exists(iso_path):
            models['isolation_forest'] = joblib.load(iso_path)
            logger.info("Loaded generic Isolation Forest model")
        
        rf_path = os.path.join(self.models_dir, 'random_forest.pkl')
        if os.path.exists(rf_path):
            models['random_forest'] = joblib.load(rf_path)
            logger.info("Loaded generic Random Forest model")
        
        return models, scalers
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        available = self.list_available_models()
        
        info = {
            'available_datasets': available['datasets'],
            'generic_models': available['model_files'],
            'models_directory': self.models_dir,
            'total_model_files': len(os.listdir(self.models_dir)) if os.path.exists(self.models_dir) else 0
        }
        
        return info 