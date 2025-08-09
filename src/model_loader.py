"""
Model Loader for Network Anomaly Detection System
================================================

Handles loading trained machine learning models and their associated scalers
for anomaly detection inference. Supports dataset-specific model loading
and automatic fallback to best available models.

Features:
- Load models by dataset (NSL-KDD, CICIDS2017, UNSW-NB15, TON-IoT)
- Automatic model discovery and validation
- Scaler loading for feature preprocessing
- Model metadata and threshold loading

"""

import os
import joblib
import yaml
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ModelLoader:
    """Loads trained models and scalers for anomaly detection inference"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize model loader with configuration file"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_dir = self.config['models']['save_path']
        
        logger.info("Model loader initialized")
    
    def list_available_models(self) -> Dict[str, list]:
        """Discover and list all available trained models in the models directory"""
        available = {
            'datasets': [],
            'model_files': []
        }
        
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory not found: {self.models_dir}")
            return available
        
        files = os.listdir(self.models_dir)
        
        # Check for dataset-specific models (primary format)
        datasets = ['nsl_kdd', 'cicids2017', 'unsw_nb15', 'ton_iot']
        for dataset in datasets:
            # Check if any model type exists for this dataset
            dataset_models = [
                f'{dataset}_isolation_forest.pkl',
                f'{dataset}_random_forest.pkl',
                f'{dataset}_xgboost.pkl',
                f'{dataset}_kmeans.pkl',
                f'{dataset}_autoencoder.pkl'
            ]
            
            if any(model_file in files for model_file in dataset_models):
                available['datasets'].append(dataset)
        
        # Check for generic models (fallback format)
        generic_models = ['isolation_forest.pkl', 'random_forest.pkl', 'xgboost.pkl']
        available['model_files'] = [model for model in generic_models if model in files]
        
        return available
    
    def load_models(self, dataset: str = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load models and scalers for inference
        
        Args:
            dataset: Specific dataset name (nsl_kdd, cicids2017, unsw_nb15, ton_iot)
                    If None, loads best available models automatically
        
        Returns:
            Tuple containing (models_dict, scalers_dict)
        
        Raises:
            FileNotFoundError: If no trained models are found
        """
        models = {}
        scalers = {}
        
        if dataset:
            # Load models for specific dataset
            models, scalers = self._load_dataset_models(dataset)
        else:
            # Load best available models with automatic fallback
            models, scalers = self._load_best_models()
        
        if not models:
            raise FileNotFoundError(
                "No trained models found. Please run training first: python src/train.py --dataset all"
            )
        
        logger.info(f"Successfully loaded {len(models)} models and {len(scalers)} scalers")
        return models, scalers
    
    def _load_dataset_models(self, dataset: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load all available models for a specific dataset"""
        models = {}
        scalers = {}
        
        # Model loading configuration
        model_configs = [
            ('isolation_forest', f'{dataset}_isolation_forest.pkl'),
            ('random_forest', f'{dataset}_random_forest.pkl'),
            ('xgboost', f'{dataset}_xgboost.pkl'),
            ('kmeans', f'{dataset}_kmeans.pkl'),
            ('autoencoder', f'{dataset}_autoencoder.pkl')
        ]
        
        # Load each model type if available
        for model_name, filename in model_configs:
            model_path = os.path.join(self.models_dir, filename)
            if os.path.exists(model_path):
                try:
                    models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {dataset} {model_name} model")
                except Exception as e:
                    logger.error(f"Failed to load {model_name} model: {e}")
        
        # Load autoencoder threshold if autoencoder was loaded
        if 'autoencoder' in models:
            threshold_path = os.path.join(self.models_dir, f'{dataset}_autoencoder_threshold.pkl')
            if os.path.exists(threshold_path):
                try:
                    threshold = joblib.load(threshold_path)
                    models['autoencoder_threshold'] = threshold
                    logger.info(f"Loaded {dataset} autoencoder threshold: {threshold:.4f}")
                except Exception as e:
                    logger.error(f"Failed to load autoencoder threshold: {e}")
        
        # Load feature scaler
        scaler_path = os.path.join(self.models_dir, f'{dataset}_scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                scalers[dataset] = joblib.load(scaler_path)
                logger.info(f"Loaded {dataset} feature scaler")
            except Exception as e:
                logger.error(f"Failed to load scaler: {e}")
        
        return models, scalers
    
    def _load_best_models(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load best available models with priority-based selection"""
        models = {}
        scalers = {}
        
        available = self.list_available_models()
        
        # Priority order for dataset selection (based on typical performance)
        priority_datasets = ['nsl_kdd', 'unsw_nb15', 'cicids2017', 'ton_iot']
        
        # Try to load from highest priority dataset with available models
        for dataset in priority_datasets:
            if dataset in available['datasets']:
                logger.info(f"Loading best available models from {dataset} dataset")
                return self._load_dataset_models(dataset)
        
        # Fallback to generic models if no dataset-specific models found
        if available['model_files']:
            logger.info("Loading generic fallback models")
            
            generic_models = {
                'isolation_forest': 'isolation_forest.pkl',
                'random_forest': 'random_forest.pkl',
                'xgboost': 'xgboost.pkl'
            }
            
            for model_name, filename in generic_models.items():
                model_path = os.path.join(self.models_dir, filename)
                if os.path.exists(model_path):
                    try:
                        models[model_name] = joblib.load(model_path)
                        logger.info(f"Loaded generic {model_name} model")
                    except Exception as e:
                        logger.error(f"Failed to load generic {model_name}: {e}")
        
        return models, scalers
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about available models and system status"""
        available = self.list_available_models()
        
        # Count total model files
        total_files = 0
        if os.path.exists(self.models_dir):
            try:
                total_files = len([f for f in os.listdir(self.models_dir) 
                                 if f.endswith('.pkl')])
            except OSError:
                total_files = 0
        
        info = {
            'available_datasets': available['datasets'],
            'generic_models': available['model_files'],
            'models_directory': self.models_dir,
            'total_model_files': total_files,
            'directory_exists': os.path.exists(self.models_dir)
        }
        
        return info 