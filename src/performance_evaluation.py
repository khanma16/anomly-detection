#!/usr/bin/env python3
"""
Comprehensive Performance Evaluation for Anomaly Detection System
Measures F1-score, ROC-AUC, latency, and compares against traditional IDS benchmarks
"""

import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
sys.path.append('.')

from src.model_loader import ModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceEvaluator:
    def __init__(self, models_dir='models'):
        """Initialize performance evaluator"""
        self.models_dir = Path(models_dir)
        self.model_loader = ModelLoader()
        self.results = {}
        self.latency_results = {}
        
    def load_test_data(self, dataset_name):
        """Load test data for evaluation"""
        try:
            if dataset_name == 'nsl_kdd':
                data_path = Path('data/NSL_KDD/KDDTest+.csv')
                columns = [
                    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
                ]
            else:
                logger.error(f"Dataset {dataset_name} not supported")
                return None, None
            
            if not data_path.exists():
                logger.error(f"Test data not found: {data_path}")
                return None, None
            
            # Load data
            df = pd.read_csv(data_path, header=None, names=columns)
            logger.info(f"Loaded {len(df)} test records from {dataset_name}")
            
            # Prepare features and labels
            X = df.drop(['attack_type', 'difficulty'], axis=1)
            
            # Create binary labels (normal=0, anomaly=1)
            y = (df['attack_type'] != 'normal').astype(int)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return None, None
    
    def preprocess_data(self, X, dataset_name):
        """Preprocess data for model evaluation"""
        try:
            # Load preprocessors
            models, scalers = self.model_loader.load_models(dataset_name)
            
            # Handle categorical columns
            categorical_cols = ['protocol_type', 'service', 'flag']
            X_processed = X.copy()
            
            # Simple categorical encoding (for evaluation)
            label_encoders = {}
            for col in categorical_cols:
                if col in X_processed.columns:
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                    label_encoders[col] = le
            
            # Handle missing values
            X_processed = X_processed.fillna(0)
            
            # Handle infinite values
            X_processed = X_processed.replace([np.inf, -np.inf], 0)
            
            # Scale features if scaler available
            if scalers:
                scaler = list(scalers.values())[0]  # Get first scaler
                try:
                    X_scaled = scaler.transform(X_processed)
                    X_processed = pd.DataFrame(X_scaled, columns=X_processed.columns)
                except Exception as e:
                    logger.warning(f"Scaling failed, using raw features: {e}")
            
            return X_processed
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return X
    
    def evaluate_model_performance(self, dataset_name='nsl_kdd'):
        """Evaluate all models on test data"""
        logger.info(f"Starting performance evaluation for {dataset_name}")
        
        # Load test data
        X_test, y_test = self.load_test_data(dataset_name)
        if X_test is None:
            return
        
        # Use subset for faster evaluation
        n_samples = min(1000, len(X_test))
        X_test = X_test.head(n_samples)
        y_test = y_test.head(n_samples)
        
        logger.info(f"Using {n_samples} samples for evaluation")
        logger.info(f"Anomaly ratio: {y_test.mean():.1%}")
        
        # Preprocess data
        X_processed = self.preprocess_data(X_test, dataset_name)
        
        # Load models
        try:
            models, scalers = self.model_loader.load_models(dataset_name)
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return
        
        self.results[dataset_name] = {}
        
        # Evaluate each model
        for model_name, model in models.items():
            if model_name == 'autoencoder_threshold':
                continue  # Skip threshold metadata
            
            logger.info(f"Evaluating {model_name}...")
            
            try:
                # Measure latency
                start_time = time.time()
                
                if model_name == 'isolation_forest':
                    predictions = model.predict(X_processed)
                    y_pred = (predictions == -1).astype(int)  # Convert to binary
                    if hasattr(model, 'decision_function'):
                        y_scores = -model.decision_function(X_processed)  # Higher score = more anomalous
                    else:
                        y_scores = y_pred
                
                elif model_name == 'kmeans':
                    clusters = model.predict(X_processed)
                    # Assume cluster 1 is anomaly cluster (from training)
                    y_pred = (clusters == 1).astype(int)
                    # Use distance to cluster centers as scores
                    distances = np.linalg.norm(X_processed.values[:, np.newaxis] - model.cluster_centers_, axis=2)
                    y_scores = np.min(distances, axis=1)
                
                elif model_name == 'autoencoder':
                    # Reconstruction error
                    reconstructed = model.predict(X_processed)
                    errors = np.mean(np.square(X_processed.values - reconstructed), axis=1)
                    
                    # Load threshold
                    threshold_path = self.models_dir / f'{dataset_name}_autoencoder_threshold.pkl'
                    if threshold_path.exists():
                        threshold = joblib.load(threshold_path)
                    else:
                        threshold = np.percentile(errors, 95)  # Use 95th percentile as threshold
                    
                    y_pred = (errors > threshold).astype(int)
                    y_scores = errors
                
                else:  # Supervised models (random_forest, xgboost)
                    y_pred = model.predict(X_processed)
                    if hasattr(model, 'predict_proba'):
                        y_scores = model.predict_proba(X_processed)[:, 1]
                    else:
                        y_scores = y_pred
                
                # Measure inference time
                inference_time = time.time() - start_time
                avg_latency = (inference_time / len(X_processed)) * 1000  # ms per sample
                
                # Calculate metrics
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                
                # ROC-AUC (handle edge cases)
                try:
                    if len(np.unique(y_test)) > 1:
                        roc_auc = roc_auc_score(y_test, y_scores)
                    else:
                        roc_auc = 0.5  # Random performance
                except Exception as e:
                    logger.warning(f"ROC-AUC calculation failed for {model_name}: {e}")
                    roc_auc = 0.5
                
                # Store results
                self.results[dataset_name][model_name] = {
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'roc_auc': roc_auc,
                    'avg_latency_ms': avg_latency,
                    'total_time_s': inference_time,
                    'samples': len(X_processed),
                    'predictions': y_pred,
                    'scores': y_scores
                }
                
                logger.info(f"  F1-Score: {f1:.3f}")
                logger.info(f"  Precision: {precision:.3f}")
                logger.info(f"  Recall: {recall:.3f}")
                logger.info(f"  ROC-AUC: {roc_auc:.3f}")
                logger.info(f"  Avg Latency: {avg_latency:.2f} ms/sample")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        # Store ground truth for later analysis
        self.results[dataset_name]['ground_truth'] = y_test.values
    
    def generate_performance_report(self, dataset_name='nsl_kdd'):
        """Generate comprehensive performance report"""
        if dataset_name not in self.results:
            logger.error(f"No results found for {dataset_name}")
            return
        
        results = self.results[dataset_name]
        
        print("\n" + "="*80)
        print(f"COMPREHENSIVE PERFORMANCE EVALUATION - {dataset_name.upper()}")
        print("="*80)
        
        # Summary table
        print(f"\nMODEL PERFORMANCE SUMMARY")
        print("-" * 80)
        print(f"{'Model':<20} {'F1-Score':<10} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10} {'Latency (ms)':<12}")
        print("-" * 80)
        
        model_scores = []
        for model_name, metrics in results.items():
            if model_name == 'ground_truth':
                continue
            
            f1 = metrics['f1_score']
            precision = metrics['precision']
            recall = metrics['recall']
            roc_auc = metrics['roc_auc']
            latency = metrics['avg_latency_ms']
            
            print(f"{model_name:<20} {f1:<10.3f} {precision:<10.3f} {recall:<10.3f} {roc_auc:<10.3f} {latency:<12.2f}")
            
            model_scores.append({
                'model': model_name,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'latency_ms': latency
            })
        
        # Best performing models
        print(f"\nBEST PERFORMING MODELS")
        print("-" * 40)
        
        best_f1 = max(model_scores, key=lambda x: x['f1_score'])
        best_roc = max(model_scores, key=lambda x: x['roc_auc'])
        fastest = min(model_scores, key=lambda x: x['latency_ms'])
        
        print(f"Best F1-Score: {best_f1['model']} ({best_f1['f1_score']:.3f})")
        print(f"Best ROC-AUC: {best_roc['model']} ({best_roc['roc_auc']:.3f})")
        print(f"Fastest: {fastest['model']} ({fastest['latency_ms']:.2f} ms/sample)")
        
        # Traditional IDS comparison
        self.compare_with_traditional_ids()
        
        # Throughput analysis
        print(f"\nTHROUGHPUT ANALYSIS")
        print("-" * 40)
        
        for model_name, metrics in results.items():
            if model_name == 'ground_truth':
                continue
            
            throughput = 1000 / metrics['avg_latency_ms']  # samples per second
            print(f"{model_name:<20}: {throughput:>8.1f} samples/second")
        
        # System requirements analysis
        print(f"\nREAL-TIME SYSTEM REQUIREMENTS")
        print("-" * 40)
        
        target_throughput = 1000  # samples/second
        suitable_models = []
        
        for model_name, metrics in results.items():
            if model_name == 'ground_truth':
                continue
            
            model_throughput = 1000 / metrics['avg_latency_ms']
            if model_throughput >= target_throughput:
                suitable_models.append((model_name, model_throughput, metrics['f1_score']))
        
        if suitable_models:
            print(f"Models suitable for ≥{target_throughput} samples/second:")
            for model, throughput, f1 in sorted(suitable_models, key=lambda x: x[2], reverse=True):
                print(f"  {model:<20}: {throughput:>8.1f} samples/s (F1: {f1:.3f})")
        else:
            print(f"No models meet {target_throughput} samples/second requirement")
            print("Consider distributed processing or model optimization")
    
    def compare_with_traditional_ids(self):
        """Compare with traditional IDS benchmarks"""
        print(f"\nTRADITIONAL IDS COMPARISON")
        print("-" * 40)
        
        # Typical traditional IDS performance benchmarks
        traditional_benchmarks = {
            'Signature-based IDS': {
                'f1_score': 0.85,
                'precision': 0.95,
                'recall': 0.78,
                'false_positive_rate': 0.05,
                'detection_rate': 0.78,
                'description': 'Rule-based detection with known attack signatures'
            },
            'Statistical Anomaly Detection': {
                'f1_score': 0.72,
                'precision': 0.68,
                'recall': 0.76,
                'false_positive_rate': 0.32,
                'detection_rate': 0.76,
                'description': 'Statistical deviation from normal behavior'
            },
            'Hybrid IDS': {
                'f1_score': 0.88,
                'precision': 0.91,
                'recall': 0.85,
                'false_positive_rate': 0.09,
                'detection_rate': 0.85,
                'description': 'Combination of signature and anomaly detection'
            }
        }
        
        print("Traditional IDS Benchmarks:")
        for ids_name, metrics in traditional_benchmarks.items():
            print(f"\n{ids_name}:")
            print(f"  F1-Score: {metrics['f1_score']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  Description: {metrics['description']}")
        
        # Compare our models
        print(f"\nOur ML Models vs Traditional IDS:")
        
        dataset_name = list(self.results.keys())[0]
        results = self.results[dataset_name]
        
        for model_name, metrics in results.items():
            if model_name == 'ground_truth':
                continue
            
            f1_score = metrics['f1_score']
            
            # Find best comparable traditional method
            best_traditional = max(traditional_benchmarks.items(), key=lambda x: x[1]['f1_score'])
            traditional_f1 = best_traditional[1]['f1_score']
            
            improvement = ((f1_score - traditional_f1) / traditional_f1) * 100
            
            if improvement > 0:
                print(f"  {model_name}: {improvement:+.1f}% better than {best_traditional[0]}")
            else:
                print(f"  {model_name}: {improvement:+.1f}% vs {best_traditional[0]}")
    
    def create_visualizations(self, dataset_name='nsl_kdd'):
        """Create performance visualization plots"""
        if dataset_name not in self.results:
            return
        
        results = self.results[dataset_name]
        
        # Create output directory
        output_dir = Path('evaluation_results')
        output_dir.mkdir(exist_ok=True)
        
        # Model comparison plot
        models = []
        f1_scores = []
        roc_scores = []
        latencies = []
        
        for model_name, metrics in results.items():
            if model_name == 'ground_truth':
                continue
            
            models.append(model_name)
            f1_scores.append(metrics['f1_score'])
            roc_scores.append(metrics['roc_auc'])
            latencies.append(metrics['avg_latency_ms'])
        
        # Performance comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # F1-Score comparison
        bars1 = ax1.bar(models, f1_scores, color='skyblue')
        ax1.set_title('F1-Score Comparison')
        ax1.set_ylabel('F1-Score')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars1, f1_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # ROC-AUC comparison
        bars2 = ax2.bar(models, roc_scores, color='lightcoral')
        ax2.set_title('ROC-AUC Comparison')
        ax2.set_ylabel('ROC-AUC')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars2, roc_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Latency comparison
        bars3 = ax3.bar(models, latencies, color='lightgreen')
        ax3.set_title('Average Latency Comparison')
        ax3.set_ylabel('Latency (ms/sample)')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, latency in zip(bars3, latencies):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{latency:.2f}', ha='center', va='bottom')
        
        # Performance vs Latency scatter plot
        ax4.scatter(latencies, f1_scores, s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax4.annotate(model, (latencies[i], f1_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.set_xlabel('Latency (ms/sample)')
        ax4.set_ylabel('F1-Score')
        ax4.set_title('Performance vs Latency Trade-off')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset_name}_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_dir / f'{dataset_name}_performance_comparison.png'}")
    
    def run_comprehensive_evaluation(self, dataset_name='nsl_kdd'):
        """Run complete performance evaluation"""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE PERFORMANCE EVALUATION")
        print("="*80)
        
        # Evaluate model performance
        self.evaluate_model_performance(dataset_name)
        
        # Generate report
        self.generate_performance_report(dataset_name)
        
        # Create visualizations
        self.create_visualizations(dataset_name)
        
        print(f"\nCOMPREHENSIVE EVALUATION COMPLETED!")
        print(f"\nResults saved to: evaluation_results/")
        print(f"Next steps:")
        print(f"  1. Review model performance metrics")
        print(f"  2. Select optimal model for deployment")
        print(f"  3. Consider ensemble approaches")
        print(f"  4. Optimize for production requirements")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Performance Evaluation')
    parser.add_argument('--dataset', default='nsl_kdd', help='Dataset to evaluate')
    parser.add_argument('--models-dir', default='models', help='Models directory')
    
    args = parser.parse_args()
    
    # Create evaluator and run evaluation
    evaluator = PerformanceEvaluator(args.models_dir)
    evaluator.run_comprehensive_evaluation(args.dataset)

if __name__ == "__main__":
    main() 