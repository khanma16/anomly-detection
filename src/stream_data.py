"""
Updated Real-time Anomaly Detection Streaming Script
====================================================
Compatible with models trained by the new unified training system

Usage:
    python stream_data.py --dataset nsl_kdd
    python stream_data.py --dataset cicids2017
    python stream_data.py  # Uses best available models
"""

import argparse
import pandas as pd
import numpy as np
import yaml
import logging
import os
import time
import threading
from datetime import datetime
from typing import Dict, Any, List, Tuple
import signal
import sys

# Add current directory and parent to path for imports
sys.path.append('.')
sys.path.append('..')

try:
    from model_loader import ModelLoader
    from alert_system import EmailAlerter
except ImportError:
    from src.model_loader import ModelLoader
    from src.alert_system import EmailAlerter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedAnomalyDetector:
    """Updated real-time anomaly detector for unified training system"""
    
    def __init__(self, config_path: str = 'config.yaml', dataset: str = None):
        """Initialize the detector"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset = dataset
        self.models = {}
        self.scalers = {}
        self.is_running = False
        self.current_row_index = 0
        self.anomaly_count = 0
        self.total_processed = 0
        
        # Setup logging
        self._setup_logging()
        
        # Load models
        self._load_models()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """Setup file logging"""
        log_config = self.config['logging']
        os.makedirs(os.path.dirname(log_config['log_file']), exist_ok=True)
        
        file_handler = logging.FileHandler(log_config['log_file'])
        file_handler.setLevel(getattr(logging, log_config['log_level']))
        
        formatter = logging.Formatter(log_config['format'])
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.info("Real-time anomaly detection system initialized")
    
    def _load_models(self):
        """Load models using unified model loader"""
        try:
            model_loader = ModelLoader()
            self.models, self.scalers = model_loader.load_models(self.dataset)
            
            logger.info(f"Loaded {len(self.models)} models for dataset: {self.dataset or 'best available'}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.is_running = False
        print(f"\nDETECTION STOPPED. Processed {self.total_processed} rows, detected {self.anomaly_count} anomalies")
        sys.exit(0)
    
    def load_streaming_data(self) -> pd.DataFrame:
        """Load data for streaming simulation"""
        logger.info("Loading streaming data...")
        
        # Default to NSL-KDD test data for streaming
        if self.dataset == 'nsl_kdd' or self.dataset is None:
            df = pd.read_csv('data/NSL_KDD/KDDTest+.csv', header=None)
            
            # Add column names (same as training script)
            with open('data/NSL_KDD/Field Names.csv', 'r') as f:
                columns = [line.split(',')[0] for line in f.readlines()]
            df.columns = columns + ['attack_type', 'difficulty']
            
        elif self.dataset == 'cicids2017':
            df = pd.read_csv('data/CICIDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
            
        elif self.dataset == 'unsw_nb15':
            df = pd.read_csv('data/UNSW_NB15/UNSW_NB15_testing-set.csv')
            
        elif self.dataset == 'ton_iot':
            df = pd.read_csv('data/TON_IOT/train_test_network.csv')
            # Use last 1000 rows for streaming
            df = df.tail(1000)
            
        else:
            # Default fallback
            df = pd.read_csv('data/NSL_KDD/KDDTest+.csv', header=None)
            with open('data/NSL_KDD/Field Names.csv', 'r') as f:
                columns = [line.split(',')[0] for line in f.readlines()]
            df.columns = columns + ['difficulty']
        
        logger.info(f"Loaded {len(df)} rows for streaming simulation")
        return df
    
    def preprocess_row(self, row: pd.Series) -> pd.DataFrame:
        """Preprocess a single row for prediction"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([row])
            
            # Handle missing values
            df = df.fillna(df.median(numeric_only=True))
            df = df.fillna(0)  # Fallback for any remaining NaNs
            
            # Encode categorical features (simple approach)
            for col in df.select_dtypes(include=['object']).columns:
                if col not in ['label', 'class', 'Label']:
                    # Simple hash-based encoding for consistency
                    df[col] = df[col].astype(str).apply(lambda x: hash(x) % 1000)
            
            # Remove target columns if present
            target_cols = ['label', 'class', 'Label', 'attack_cat', 'type', 'attack_type', 'difficulty']
            for col in target_cols:
                if col in df.columns:
                    df = df.drop([col], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing row: {str(e)}")
            return None
    
    def predict_anomaly(self, processed_row: pd.DataFrame) -> Dict[str, Any]:
        """Make anomaly predictions"""
        predictions = {}
        
        try:
            # Scale features if scaler available
            if self.dataset in self.scalers:
                X_scaled = self.scalers[self.dataset].transform(processed_row)
            else:
                # Use raw features if no scaler
                X_scaled = processed_row.values
            
            # Make predictions with each model
            for model_name, model in self.models.items():
                if model_name == 'isolation_forest':
                    score = model.decision_function(X_scaled)[0]
                    is_anomaly = model.predict(X_scaled)[0] == -1
                    confidence = abs(score)
                    
                    predictions[model_name] = {
                        'is_anomaly': is_anomaly,
                        'anomaly_score': score,
                        'confidence': confidence
                    }
                    
                elif model_name == 'random_forest':
                    pred = model.predict(X_scaled)[0]
                    prob = model.predict_proba(X_scaled)[0]
                    
                    predictions[model_name] = {
                        'is_anomaly': bool(pred == 1),
                        'anomaly_probability': float(prob[1]),
                        'confidence': float(max(prob))
                    }
                
                elif model_name == 'xgboost':
                    pred = model.predict(X_scaled)[0]
                    prob = model.predict_proba(X_scaled)[0]
                    
                    predictions[model_name] = {
                        'is_anomaly': bool(pred == 1),
                        'anomaly_probability': float(prob[1]),
                        'confidence': float(max(prob))
                    }
                
                elif model_name == 'kmeans':
                    cluster_label = model.predict(X_scaled)[0]
                    
                    # Determine which cluster is the anomaly cluster
                    # (This should be stored during training, but we'll use a simple heuristic)
                    distances = model.transform(X_scaled)[0]
                    
                    # Assume cluster with fewer points is anomaly cluster
                    # For simplicity, use cluster 0 as anomaly cluster
                    anomaly_cluster = 0
                    is_anomaly = cluster_label == anomaly_cluster
                    
                    # Use distance to cluster center as confidence
                    confidence = float(distances[cluster_label])
                    
                    predictions[model_name] = {
                        'is_anomaly': is_anomaly,
                        'cluster_label': int(cluster_label),
                        'distance_to_center': confidence,
                        'confidence': confidence
                    }
                
                elif model_name == 'autoencoder':
                    # Autoencoder anomaly detection
                    X_reconstructed = model.predict(X_scaled)
                    reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
                    
                    # Get threshold from models dict
                    threshold_key = 'autoencoder_threshold'
                    threshold = self.models.get(threshold_key, reconstruction_error * 1.5)  # Fallback threshold
                    
                    is_anomaly = reconstruction_error > threshold
                    confidence = float(reconstruction_error)
                    
                    predictions[model_name] = {
                        'is_anomaly': is_anomaly,
                        'reconstruction_error': confidence,
                        'threshold': float(threshold),
                        'confidence': confidence
                    }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {}
    
    def should_alert(self, predictions: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Determine if an alert should be triggered"""
        threshold = self.config['streaming']['anomaly_threshold']
        
        # Check each model's prediction
        for model_name, pred in predictions.items():
            if model_name == 'isolation_forest':
                if pred['is_anomaly'] and pred['confidence'] > threshold:
                    return True, model_name, pred['confidence']
            
            elif model_name == 'random_forest':
                if pred['is_anomaly'] and pred['confidence'] > threshold:
                    return True, model_name, pred['confidence']
            
            elif model_name == 'xgboost':
                if pred['is_anomaly'] and pred['confidence'] > threshold:
                    return True, model_name, pred['confidence']
            
            elif model_name == 'kmeans':
                if pred['is_anomaly'] and pred['confidence'] > threshold:
                    return True, model_name, pred['confidence']
            
            elif model_name == 'autoencoder':
                if pred['is_anomaly'] and pred['confidence'] > threshold:
                    return True, model_name, pred['confidence']
        
        return False, '', 0.0
    
    def log_anomaly(self, row_index: int, predictions: Dict[str, Any]):
        """Log detected anomaly"""
        timestamp = datetime.now().isoformat()
        
        logger.warning(f"ANOMALY DETECTED - Row {row_index}: {predictions}")
        
        # Log to separate anomaly file
        anomaly_log_path = os.path.join(
            os.path.dirname(self.config['logging']['log_file']), 
            'anomalies.log'
        )
        
        with open(anomaly_log_path, 'a') as f:
            f.write(f"{timestamp} - Row {row_index} - {predictions}\n")
    
    def process_single_row(self, row: pd.Series, row_index: int) -> bool:
        """Process a single row and check for anomalies"""
        try:
            # Preprocess
            processed_row = self.preprocess_row(row)
            if processed_row is None or processed_row.empty:
                logger.warning(f"Failed to preprocess row {row_index}")
                return False
            
            # Make predictions
            predictions = self.predict_anomaly(processed_row)
            if not predictions:
                logger.warning(f"No predictions available for row {row_index}")
                return False
            
            # Check for alerts
            should_alert, best_model, confidence = self.should_alert(predictions)
            
            if should_alert:
                self.log_anomaly(row_index, predictions)
                self.anomaly_count += 1
                
                # Print alert to console
                print(f"ANOMALY DETECTED - Row {row_index}")
                print(f"   Model: {best_model}")
                print(f"   Confidence: {confidence:.4f}")
                print(f"   Predictions: {predictions}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing row {row_index}: {str(e)}")
            return False
    
    def start_detection(self):
        """Start real-time anomaly detection"""
        logger.info("Starting real-time anomaly detection...")
        
        # Load streaming data
        streaming_data = self.load_streaming_data()
        
        self.is_running = True
        delay = self.config['streaming']['delay_seconds']
        loop_dataset = self.config['streaming']['loop_dataset']
        
        print(f"\n{'='*60}")
        print("REAL-TIME ANOMALY DETECTION SYSTEM STARTED")
        print(f"{'='*60}")
        print(f"Dataset: {self.dataset or 'best available'}")
        print(f"Processing {len(streaming_data)} rows...")
        print(f"Delay between rows: {delay} seconds")
        print(f"Loop dataset: {loop_dataset}")
        print(f"Anomaly threshold: {self.config['streaming']['anomaly_threshold']}")
        print("Press Ctrl+C to stop")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        try:
            while self.is_running:
                for index, row in streaming_data.iterrows():
                    if not self.is_running:
                        break
                    
                    # Process row
                    is_anomaly = self.process_single_row(row, self.current_row_index)
                    self.total_processed += 1
                    self.current_row_index += 1
                    
                    # Progress indicator
                    if self.total_processed % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = self.total_processed / elapsed
                        print(f"PROGRESS: Processed: {self.total_processed} | Anomalies: {self.anomaly_count} | Rate: {rate:.1f} rows/sec")
                    
                    # Delay
                    if delay > 0:
                        time.sleep(delay)
                
                # Loop dataset if configured
                if not loop_dataset:
                    break
                
                print("LOOPING DATASET...")
        
        except KeyboardInterrupt:
            print("\nDETECTION STOPPED BY USER")
        
        finally:
            self.is_running = False
            elapsed = time.time() - start_time
            
            print(f"\n{'='*60}")
            print("DETECTION SUMMARY")
            print(f"{'='*60}")
            print(f"Total processed: {self.total_processed}")
            print(f"Anomalies detected: {self.anomaly_count}")
            print(f"Detection rate: {self.anomaly_count/self.total_processed*100:.2f}%" if self.total_processed > 0 else "0%")
            print(f"Processing time: {elapsed:.2f} seconds")
            print(f"Processing rate: {self.total_processed/elapsed:.2f} rows/sec" if elapsed > 0 else "N/A")
            print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Real-time Anomaly Detection')
    parser.add_argument('--dataset', choices=['nsl_kdd', 'cicids2017', 'unsw_nb15', 'ton_iot'],
                       help='Dataset to use for detection (default: best available)')
    
    args = parser.parse_args()
    
    print("Unified Real-time Anomaly Detection")
    print("=" * 60)
    print(f"Dataset: {args.dataset or 'best available'}")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 60)
    
    try:
        detector = UnifiedAnomalyDetector(dataset=args.dataset)
        detector.start_detection()
        
    except Exception as e:
        print(f"ERROR: Detection failed: {str(e)}")
        logger.error(f"Detection error: {str(e)}")

if __name__ == "__main__":
    main() 