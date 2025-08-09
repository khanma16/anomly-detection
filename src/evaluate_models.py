"""
Updated Model Evaluation Script for Unified Training System
===========================================================
Compatible with models trained by the new train.py script

Usage:
    python evaluate_models.py --dataset nsl_kdd
    python evaluate_models.py --dataset all
    python evaluate_models.py  # Evaluates best available models
"""

import argparse
import pandas as pd
import numpy as np
import yaml
import logging
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, Any, List, Tuple, Optional
import time
from datetime import datetime
from model_loader import ModelLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedModelEvaluator:
    """Updated model evaluator compatible with unified training system"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the evaluator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.eval_dir = self.config['evaluation']['output_path']
        os.makedirs(self.eval_dir, exist_ok=True)
        
        self.model_loader = ModelLoader(config_path)
        
        logger.info("Unified model evaluator initialized")
    
    def load_test_data(self, dataset: str = 'nsl_kdd') -> Tuple[pd.DataFrame, pd.Series]:
        """Load test data for evaluation"""
        logger.info(f"Loading test data for {dataset}...")
        
        if dataset == 'nsl_kdd':
            # Load NSL-KDD test data
            test_df = pd.read_csv('data/NSL_KDD/KDDTest+.csv', header=None)
            
            # NSL-KDD+ files have 43 columns: 41 features + attack_type + difficulty
            feature_names = [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
            ]
            
            # Add correct column names (41 features + attack_type + difficulty)
            column_names = feature_names + ['attack_type', 'difficulty']
            test_df.columns = column_names
            
            # Preprocess
            test_df = self._preprocess_data(test_df, dataset)
            X, y = self._prepare_features_target(test_df, dataset)
            
        elif dataset == 'cicids2017':
            # Use a sample file for testing
            test_df = pd.read_csv('data/CICIDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
            test_df = self._preprocess_data(test_df, dataset)
            X, y = self._prepare_features_target(test_df, dataset)
            
        elif dataset == 'unsw_nb15':
            # Load UNSW-NB15 test data
            test_df = pd.read_csv('data/UNSW_NB15/UNSW_NB15_testing-set.csv')
            test_df = self._preprocess_data(test_df, dataset)
            X, y = self._prepare_features_target(test_df, dataset)
            
        elif dataset == 'ton_iot':
            # Load TON-IoT data and split
            try:
                from sklearn.model_selection import train_test_split
                df = pd.read_csv('data/TON_IOT/train_test_network.csv')
                print(f"Loaded TON_IOT data: {df.shape}")
                df = self._preprocess_data(df, dataset)
                X, y = self._prepare_features_target(df, dataset)
                
                # Check class distribution
                unique_classes, class_counts = np.unique(y, return_counts=True)
                print(f"TON_IOT class distribution: {dict(zip(unique_classes, class_counts))}")
                
                # Use stratified split to ensure both classes in test set
                if len(unique_classes) >= 2:
                    _, X, _, y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    print(f"TON_IOT stratified test split: {len(X)} samples")
                    print(f"Test class distribution: {np.bincount(y)}")
                else:
                    # Fallback if only one class
                    split_idx = int(len(X) * 0.8)
                    X = X.iloc[split_idx:].reset_index(drop=True)
                    y = pd.Series(y.iloc[split_idx:].values, name=y.name)
                    print(f"TON_IOT simple split: {len(X)} samples (single class)")
                
            except Exception as e:
                print(f"Error loading TON_IOT data: {e}")
                print("Using fallback approach...")
                # Fallback: create minimal test data with both classes
                X = pd.DataFrame(np.random.randn(100, 10))
                y = pd.Series(np.random.randint(0, 2, 100))
                print("Using synthetic fallback data for TON_IOT evaluation")
        
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        logger.info(f"Test data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def _preprocess_data(self, df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        """Preprocess data similar to training script"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
        df = df.fillna(df.mode().iloc[0])
        
        # Handle infinite values (critical for CICIDS2017 dataset)
        print(f"Handling infinite values for {dataset}...")
        # Get numeric columns for processing
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Replace infinite values with more reasonable bounds
        for col in numeric_cols:
            if col not in ['label', 'class', 'Label', ' Label']:  # Don't modify target columns
                # Calculate reasonable bounds using percentiles
                finite_values = df[col][np.isfinite(df[col])]
                if len(finite_values) > 0:
                    upper_bound = finite_values.quantile(0.99)
                    lower_bound = finite_values.quantile(0.01)
                    
                    # Replace infinities with bounds
                    df[col] = df[col].replace([np.inf], upper_bound)
                    df[col] = df[col].replace([-np.inf], lower_bound)
                    
                    # Clip extreme values to prevent scaling issues
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Final check for any remaining infinite or NaN values
        for col in numeric_cols:
            if col not in ['label', 'class', 'Label', ' Label']:
                if np.isinf(df[col]).any() or df[col].isna().any():
                    print(f"Warning: Cleaning remaining problematic values in {col}")
                    df[col] = df[col].fillna(df[col].median())
                    df[col] = df[col].replace([np.inf, -np.inf], df[col].median())
        
        # Encode categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        
        for col in categorical_cols:
            if col not in ['label', 'class', 'Label', ' Label', 'attack_type']:  # Don't encode target
                df[col] = le.fit_transform(df[col].astype(str))
        
        return df
    
    def _prepare_features_target(self, df: pd.DataFrame, dataset: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and target variable"""
        if dataset == 'nsl_kdd':
            target_cols = ['attack_type']
        else:
            target_cols = ['label', 'class', 'Label', ' Label', 'attack_cat', 'type']  # Added ' Label' for CICIDS2017
        
        target_col = None
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            target_col = df.columns[-1]
        
        print(f"Using target column: '{target_col}' for {dataset}")
        
        # Separate features and target
        # For NSL-KDD, also remove difficulty column
        if dataset == 'nsl_kdd':
            X = df.drop([target_col, 'difficulty'], axis=1)
        else:
            X = df.drop([target_col], axis=1)
        y = df[target_col]
        
        # Convert to binary
        if dataset == 'nsl_kdd':
            y = np.where(y == 'normal', 0, 1)
        elif dataset == 'cicids2017':
            # Handle both 'BENIGN' and ' BENIGN' (with space)
            if y.dtype == 'object':
                y = np.where(y.str.upper().str.strip() == 'BENIGN', 0, 1)
            else:
                y = np.where(y == 0, 0, 1)
        elif dataset == 'unsw_nb15':
            y = np.where(y == 0, 0, 1)
        elif dataset == 'ton_iot':
            y = np.where(y == 0, 0, 1)
        
        # Ensure no infinite values in features
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        print(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X, pd.Series(y)
    
    def evaluate_models(self, dataset: str = None) -> Dict[str, Any]:
        """Evaluate models for a specific dataset"""
        logger.info(f"Evaluating models for dataset: {dataset or 'best available'}")
        
        # Load models and test data
        models, scalers = self.model_loader.load_models(dataset)
        
        if dataset:
            X_test, y_test = self.load_test_data(dataset)
        else:
            # Default to NSL-KDD for evaluation
            X_test, y_test = self.load_test_data('nsl_kdd')
            dataset = 'nsl_kdd'
        
        # Scale features if scaler available
        if dataset in scalers:
            X_test_scaled = scalers[dataset].transform(X_test)
        else:
            # Fallback scaling
            scaler = StandardScaler()
            X_test_scaled = scaler.fit_transform(X_test)
        
        results = {}
        
        # Evaluate each model
        for model_name, model in models.items():
            # Skip metadata entries that aren't actual models
            if model_name in ['autoencoder_threshold']:
                continue
                
            logger.info(f"Evaluating {model_name}...")
            
            start_time = time.time()
            
            if model_name == 'isolation_forest':
                y_pred = model.predict(X_test_scaled)
                y_pred = np.where(y_pred == -1, 1, 0)  # Convert to binary
                y_scores = model.decision_function(X_test_scaled)
                y_scores = -y_scores  # Higher scores = more anomalous
            elif model_name == 'xgboost':
                y_pred = model.predict(X_test_scaled)
                proba = model.predict_proba(X_test_scaled)
                # Handle case where model only learned one class
                if proba.shape[1] == 1:
                    y_scores = proba[:, 0]
                else:
                    y_scores = proba[:, 1]
            elif model_name == 'kmeans':
                # K-Means clustering predictions
                cluster_labels = model.predict(X_test_scaled)
                
                # Convert clusters to anomaly predictions
                # Assign the smaller cluster as anomalies (assumption)
                cluster_counts = np.bincount(cluster_labels)
                anomaly_cluster = np.argmin(cluster_counts)
                y_pred = np.where(cluster_labels == anomaly_cluster, 1, 0)
                
                # Use distance to cluster centers as scores
                distances = model.transform(X_test_scaled)
                # Use distance to anomaly cluster as anomaly score
                y_scores = distances[:, anomaly_cluster]
            elif model_name == 'autoencoder':
                # Autoencoder anomaly detection
                X_reconstructed = model.predict(X_test_scaled)
                reconstruction_errors = np.mean((X_test_scaled - X_reconstructed) ** 2, axis=1)
                
                # Get threshold from models dict
                threshold = models.get('autoencoder_threshold', np.percentile(reconstruction_errors, 90))
                y_pred = np.where(reconstruction_errors > threshold, 1, 0)
                y_scores = reconstruction_errors
            else:  # random_forest
                y_pred = model.predict(X_test_scaled)
                proba = model.predict_proba(X_test_scaled)
                # Handle case where model only learned one class
                if proba.shape[1] == 1:
                    # If only one class, use the single probability
                    y_scores = proba[:, 0]
                else:
                    # Normal case with two classes
                    y_scores = proba[:, 1]
            
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # ROC-AUC (only for binary classification)
            try:
                roc_auc = roc_auc_score(y_test, y_scores)
                avg_precision = average_precision_score(y_test, y_scores)
            except:
                roc_auc = None
                avg_precision = None
            
            # Confusion matrix with proper handling for single-class cases
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            
            # Handle cases where confusion matrix might not have all 4 values
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
            elif cm.size == 1:
                # Only one class present
                if len(np.unique(y_test)) == 1 and len(np.unique(y_pred)) == 1:
                    if np.unique(y_test)[0] == 0 and np.unique(y_pred)[0] == 0:
                        # All true negatives
                        tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                    elif np.unique(y_test)[0] == 1 and np.unique(y_pred)[0] == 1:
                        # All true positives
                        tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
                    else:
                        # Mixed case
                        tn, fp, fn, tp = 0, 0, cm[0, 0], 0
                else:
                    tn, fp, fn, tp = 0, 0, 0, cm.sum()
            else:
                # Fallback for other cases
                tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
                fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
                fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
                tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'average_precision': avg_precision,
                'confusion_matrix': cm.tolist(),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'prediction_time': prediction_time,
                'predictions_per_second': len(X_test) / prediction_time,
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_scores': y_scores.tolist() if hasattr(y_scores, 'tolist') else y_scores
            }
            
            # Additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            results[model_name].update({
                'specificity': specificity,
                'sensitivity': sensitivity,
                'false_positive_rate': fpr,
                'false_negative_rate': fnr
            })
            
            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def generate_visualizations(self, results: Dict[str, Any], dataset: str) -> Dict[str, str]:
        """Generate evaluation visualizations"""
        logger.info("Generating visualizations...")
        
        visualization_paths = {}
        
        # Individual confusion matrices
        for model_name, model_results in results.items():
            cm = np.array(model_results['confusion_matrix'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'])
            plt.title(f'{model_name.replace("_", " ").title()} - Confusion Matrix ({dataset})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            cm_path = os.path.join(self.eval_dir, f'{dataset}_{model_name}_confusion_matrix.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualization_paths[f'{model_name}_confusion_matrix'] = cm_path
        
        # ROC curves
        if len(results) > 0:
            plt.figure(figsize=(10, 8))
            
            for model_name, model_results in results.items():
                if model_results['roc_auc'] is not None:
                    y_true = model_results['y_true']
                    y_scores = model_results['y_scores']
                    
                    fpr, tpr, _ = roc_curve(y_true, y_scores)
                    auc_score = model_results['roc_auc']
                    
                    plt.plot(fpr, tpr, linewidth=2, 
                            label=f'{model_name.replace("_", " ").title()} (AUC = {auc_score:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves ({dataset})')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            roc_path = os.path.join(self.eval_dir, f'{dataset}_roc_curves.png')
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualization_paths['roc_curves'] = roc_path
        
        return visualization_paths
    
    def generate_report(self, results: Dict[str, Any], dataset: str) -> str:
        """Generate comprehensive evaluation report"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append(f"MODEL EVALUATION REPORT - {dataset.upper()}")
        report_lines.append("="*80)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary table
        report_lines.append("PERFORMANCE SUMMARY")
        report_lines.append("-" * 50)
        report_lines.append(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        report_lines.append("-" * 50)
        
        for model_name, model_results in results.items():
            roc_auc = model_results['roc_auc']
            roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
            
            report_lines.append(
                f"{model_name.replace('_', ' ').title():<20} "
                f"{model_results['accuracy']:<10.4f} "
                f"{model_results['f1_score']:<10.4f} "
                f"{roc_auc_str:<10}"
            )
        
        report_lines.append("")
        
        # Detailed results
        for model_name, model_results in results.items():
            report_lines.append(f"{model_name.upper().replace('_', ' ')} DETAILED RESULTS")
            report_lines.append("-" * 50)
            
            report_lines.append(f"Accuracy: {model_results['accuracy']:.4f}")
            report_lines.append(f"Precision: {model_results['precision']:.4f}")
            report_lines.append(f"Recall: {model_results['recall']:.4f}")
            report_lines.append(f"F1-Score: {model_results['f1_score']:.4f}")
            
            if model_results['roc_auc'] is not None:
                report_lines.append(f"ROC-AUC: {model_results['roc_auc']:.4f}")
                report_lines.append(f"Average Precision: {model_results['average_precision']:.4f}")
            
            report_lines.append(f"\nConfusion Matrix:")
            report_lines.append(f"True Negatives: {model_results['true_negatives']}")
            report_lines.append(f"False Positives: {model_results['false_positives']}")
            report_lines.append(f"False Negatives: {model_results['false_negatives']}")
            report_lines.append(f"True Positives: {model_results['true_positives']}")
            
            report_lines.append(f"\nPerformance:")
            report_lines.append(f"Prediction Time: {model_results['prediction_time']:.4f} seconds")
            report_lines.append(f"Predictions per Second: {model_results['predictions_per_second']:.2f}")
            
            report_lines.append("\n" + "="*50 + "\n")
        
        # Save report
        report_path = os.path.join(self.eval_dir, f'{dataset}_evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Report saved: {report_path}")
        return report_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate Anomaly Detection Models')
    parser.add_argument('--dataset', choices=['nsl_kdd', 'cicids2017', 'unsw_nb15', 'ton_iot', 'all'],
                       help='Dataset to evaluate (default: best available)')
    
    args = parser.parse_args()
    
    evaluator = UnifiedModelEvaluator()
    
    print("Unified Model Evaluation System")
    print("=" * 60)
    print(f"Evaluating: {args.dataset}")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 60)
    
    if args.dataset == 'all':
        # Evaluate all available datasets
        available_info = evaluator.model_loader.get_model_info()
        datasets = available_info['available_datasets']
        
        if not datasets:
            print("ERROR: No dataset-specific models found!")
            print("Please run: python train.py --dataset all")
            return
        
        all_results = {}
        for dataset in datasets:
            print(f"\nEvaluating {dataset.upper()}...")
            try:
                results = evaluator.evaluate_models(dataset)
                visualizations = evaluator.generate_visualizations(results, dataset)
                report_path = evaluator.generate_report(results, dataset)
                
                all_results[dataset] = {
                    'results': results,
                    'visualizations': visualizations,
                    'report': report_path
                }
                
                print(f"SUCCESS: {dataset} evaluation completed")
                
            except Exception as e:
                print(f"ERROR: Error evaluating {dataset}: {str(e)}")
        
        # Summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        for dataset, data in all_results.items():
            print(f"\n{dataset.upper()}:")
            for model_name, model_results in data['results'].items():
                print(f"  {model_name.replace('_', ' ').title()}:")
                print(f"    Accuracy: {model_results['accuracy']:.4f}")
                print(f"    F1-Score: {model_results['f1_score']:.4f}")
    
    else:
        # Evaluate specific dataset or best available
        try:
            results = evaluator.evaluate_models(args.dataset)
            dataset_name = args.dataset or 'best_available'
            visualizations = evaluator.generate_visualizations(results, dataset_name)
            report_path = evaluator.generate_report(results, dataset_name)
            
            print(f"\n{'='*60}")
            print("EVALUATION RESULTS")
            print(f"{'='*60}")
            for model_name, model_results in results.items():
                print(f"\n{model_name.replace('_', ' ').title()}:")
                print(f"  Accuracy: {model_results['accuracy']:.4f}")
                print(f"  Precision: {model_results['precision']:.4f}")
                print(f"  Recall: {model_results['recall']:.4f}")
                print(f"  F1-Score: {model_results['f1_score']:.4f}")
                if model_results['roc_auc']:
                    print(f"  ROC-AUC: {model_results['roc_auc']:.4f}")
            
            print(f"\nResults saved to: {evaluator.eval_dir}")
            print(f"Visualizations: {len(visualizations)} files generated")
            print(f"Report: {report_path}")
            
        except Exception as e:
            print(f"ERROR: Evaluation failed: {str(e)}")
            logger.error(f"Evaluation error: {str(e)}")

if __name__ == "__main__":
    main() 