#!/usr/bin/env python3
"""
Network Anomaly Detection Training Module
=========================================

Trains machine learning models for network anomaly detection on multiple datasets
using various algorithms including supervised and unsupervised methods.

Models trained:
- Isolation Forest (unsupervised anomaly detection)
- Random Forest (supervised classification)
- XGBoost (supervised classification)
- K-Means Clustering (unsupervised clustering)
- Autoencoder (unsupervised neural network)

Supported datasets:
- NSL-KDD: Network intrusion detection dataset
- CICIDS2017: Canadian Institute for Cybersecurity dataset
- UNSW-NB15: University of New South Wales dataset
- TON-IoT: IoT network traffic dataset

Usage:
    # Linux/Mac
    source venv/bin/activate
    python src/train.py --dataset nsl_kdd
    python src/train.py --dataset all
    python src/train.py --dataset nsl_kdd --feature_engineering
    
    # Windows
    venv\Scripts\activate
    python src/train.py --dataset nsl_kdd
    python src/train.py --dataset all
    python src/train.py --dataset nsl_kdd --feature_engineering

Author: Network Anomaly Detection System
Date: 2024
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, silhouette_score, mean_squared_error
import xgboost as xgb
import joblib
import os
import yaml
from datetime import datetime

# Import feature engineering module
try:
    from src.feature_engineering import AdvancedFeatureEngineer
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False
    print("Warning: Feature engineering module not available. Install umap-learn to enable advanced features.")

class AnomalyTrainer:
    """Main training class for anomaly detection models"""
    
    def __init__(self, enable_feature_engineering=False, pca_components=None, 
                 feature_selection=None, dimensionality_analysis=False):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Feature engineering configuration
        self.enable_feature_engineering = enable_feature_engineering and FEATURE_ENGINEERING_AVAILABLE
        self.pca_components = pca_components
        self.feature_selection = feature_selection
        self.dimensionality_analysis = dimensionality_analysis
        
        # Initialize feature engineer if available
        if self.enable_feature_engineering:
            self.feature_engineer = AdvancedFeatureEngineer()
            print("Advanced Feature Engineering: ENABLED")
        else:
            self.feature_engineer = None
            if enable_feature_engineering and not FEATURE_ENGINEERING_AVAILABLE:
                print("Advanced Feature Engineering: DISABLED (dependencies missing)")
            else:
                print("Advanced Feature Engineering: DISABLED")
    
    def load_config(self):
        """Load system configuration from YAML file"""
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    
    def preprocess_data(self, df, dataset_name):
        """Clean and preprocess data for training"""
        print(f"Preprocessing {dataset_name} data...")
        
        # Remove duplicate records
        df = df.drop_duplicates()
        
        # Handle missing values using median for numeric and mode for categorical
        df = df.fillna(df.median(numeric_only=True))
        df = df.fillna(df.mode().iloc[0])
        
        # Handle infinite values that can cause training issues
        print("Handling infinite values...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['label', 'class', 'Label']:  # Preserve target columns
                # Use percentile-based bounds to replace infinite values
                finite_values = df[col][np.isfinite(df[col])]
                if len(finite_values) > 0:
                    upper_bound = finite_values.quantile(0.99)
                    lower_bound = finite_values.quantile(0.01)
                    
                    # Replace infinities with reasonable bounds
                    df[col] = df[col].replace([np.inf], upper_bound)
                    df[col] = df[col].replace([-np.inf], lower_bound)
                    
                    # Clip extreme values to prevent scaling issues
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Final cleanup for any remaining problematic values
        for col in numeric_cols:
            if col not in ['label', 'class', 'Label']:
                if np.isinf(df[col]).any() or df[col].isna().any():
                    print(f"Warning: Cleaning remaining problematic values in {col}")
                    df[col] = df[col].fillna(df[col].median())
                    df[col] = df[col].replace([np.inf, -np.inf], df[col].median())
        
        # Encode categorical features as numerical values
        categorical_cols = df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        
        for col in categorical_cols:
            if col not in ['label', 'class', 'Label', 'attack_type']:  # Don't encode target
                df[col] = le.fit_transform(df[col].astype(str))
        
        return df
    
    def prepare_features_target(self, df, dataset_name):
        """Separate features and target variable from dataset"""
        # Dataset-specific target column identification
        if dataset_name == 'nsl_kdd':
            target_cols = ['attack_type']
        else:
            target_cols = ['label', 'class', 'Label', ' Label', 'attack_cat', 'type']
        
        target_col = None
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            # Use last column as fallback
            target_col = df.columns[-1]
        
        print(f"Using target column: '{target_col}'")
        
        # Separate features and target
        if dataset_name == 'nsl_kdd':
            X = df.drop([target_col, 'difficulty'], axis=1)
        else:
            X = df.drop([target_col], axis=1)
        y = df[target_col]
        
        # Convert target to binary classification (0=normal, 1=anomaly)
        if dataset_name == 'nsl_kdd':
            y = np.where(y == 'normal', 0, 1)
        elif dataset_name == 'cicids2017':
            if y.dtype == 'object':
                y = np.where(y.str.upper().str.strip() == 'BENIGN', 0, 1)
            else:
                y = np.where(y == 0, 0, 1)
        elif dataset_name == 'unsw_nb15':
            y = np.where(y == 0, 0, 1)
        elif dataset_name == 'ton_iot':
            y = np.where(y == 0, 0, 1)
        
        # Report class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print(f"Class distribution before balancing:")
        for cls, count in zip(unique_classes, class_counts):
            label_name = "Normal" if cls == 0 else "Anomaly"
            print(f"  {label_name} (class {cls}): {count} ({count/len(y)*100:.1f}%)")
        
        # Handle severe class imbalance using balanced sampling
        if len(unique_classes) == 2:
            minority_class = unique_classes[np.argmin(class_counts)]
            majority_class = unique_classes[np.argmax(class_counts)]
            minority_count = np.min(class_counts)
            majority_count = np.max(class_counts)
            
            imbalance_ratio = majority_count / minority_count if minority_count > 0 else float('inf')
            
            if imbalance_ratio > 100:
                print(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
                print("Applying balanced sampling...")
                
                # Get indices for each class
                minority_indices = np.where(y == minority_class)[0]
                majority_indices = np.where(y == majority_class)[0]
                
                # Sample to create better balance (maximum 10:1 ratio)
                target_minority_count = min(minority_count, 10000)  # Cap at 10k samples
                target_majority_count = min(target_minority_count * 10, majority_count)
                
                # Sample indices with reproducible randomness
                np.random.seed(42)
                sampled_minority = np.random.choice(minority_indices, 
                                                  size=min(target_minority_count, len(minority_indices)), 
                                                  replace=False)
                sampled_majority = np.random.choice(majority_indices, 
                                                  size=min(target_majority_count, len(majority_indices)), 
                                                  replace=False)
                
                # Combine and shuffle samples
                balanced_indices = np.concatenate([sampled_minority, sampled_majority])
                np.random.shuffle(balanced_indices)
                
                # Apply sampling
                X = X.iloc[balanced_indices].reset_index(drop=True)
                y = y[balanced_indices]
                
                print(f"Balanced dataset: {len(X)} samples")
                unique_classes, class_counts = np.unique(y, return_counts=True)
                for cls, count in zip(unique_classes, class_counts):
                    label_name = "Normal" if cls == 0 else "Anomaly"
                    print(f"  {label_name} (class {cls}): {count} ({count/len(y)*100:.1f}%)")
        
        return X, y
    
    def apply_feature_engineering(self, X_train, X_test, y_train, dataset_name):
        """Apply advanced feature engineering techniques if enabled"""
        if not self.enable_feature_engineering:
            return X_train, X_test
        
        print(f"Applying Advanced Feature Engineering on {dataset_name}...")
        
        # Perform dimensionality analysis if requested
        if self.dimensionality_analysis:
            print("Performing dimensionality analysis...")
            self.feature_engineer.analyze_feature_distribution(X_train, y_train, dataset_name)
            self.feature_engineer.create_dimensionality_comparison(X_train, y_train, dataset_name)
        
        # Apply Principal Component Analysis
        if self.pca_components:
            print(f"Applying PCA with {self.pca_components} components...")
            X_train_pca, pca_info = self.feature_engineer.apply_pca(
                X_train, n_components=self.pca_components, dataset_name=dataset_name
            )
            
            # Transform test data using same PCA transformation
            pca_transformer = self.feature_engineer.transformers[f'pca_{dataset_name}']
            X_test_scaled = pca_transformer['scaler'].transform(X_test)
            X_test_pca = pca_transformer['pca'].transform(X_test_scaled)
            X_test_pca = pd.DataFrame(X_test_pca, columns=X_train_pca.columns, index=X_test.index)
            
            print(f"PCA applied: {X_train.shape[1]} -> {X_train_pca.shape[1]} features")
            print(f"Total variance explained: {pca_info['total_variance_explained']:.3f}")
            
            X_train, X_test = X_train_pca, X_test_pca
        
        # Apply feature selection methods
        if self.feature_selection:
            print(f"Applying feature selection: {self.feature_selection}")
            
            if self.feature_selection == 'variance_threshold':
                X_train = self.feature_engineer.apply_variance_threshold(X_train)
                selector = self.feature_engineer.transformers['variance_threshold']
                X_test_selected = selector.transform(X_test)
                X_test = pd.DataFrame(X_test_selected, columns=X_train.columns, index=X_test.index)
            
            elif self.feature_selection.startswith('select_k_best'):
                k = int(self.feature_selection.split('_')[-1]) if '_' in self.feature_selection else 20
                X_train, selected_features = self.feature_engineer.select_best_features(
                    X_train, y_train, k=k, method='f_classif'
                )
                X_test = X_test[selected_features]
            
            elif self.feature_selection.startswith('rfe'):
                n_features = int(self.feature_selection.split('_')[-1]) if '_' in self.feature_selection else 20
                X_train, selected_features = self.feature_engineer.recursive_feature_elimination(
                    X_train, y_train, n_features=n_features
                )
                X_test = X_test[selected_features]
            
            print(f"Feature selection applied: {X_test.shape[1]} features selected")
        
        # Save transformers and generate reports
        if self.feature_engineer:
            self.feature_engineer.save_transformers(dataset_name)
            self.feature_engineer.generate_feature_report(dataset_name)
        
        return X_train, X_test
    
    def train_models(self, X_train, X_test, y_train, y_test, dataset_name):
        """Train all machine learning models on the dataset"""
        print(f"Training models on {dataset_name}...")
        
        # Check class distribution for supervised learning
        unique_classes = np.unique(y_train)
        class_counts = np.bincount(y_train)
        print(f"Training set class distribution: {class_counts}")
        
        # Determine if we can train supervised models
        has_both_classes = len(unique_classes) >= 2 and np.min(class_counts) > 0
        
        if not has_both_classes:
            print("WARNING: Training set has only one class. Supervised models will be skipped.")
        
        # Scale features for all models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler for inference
        self.scalers[dataset_name] = scaler
        
        results = {}
        
        # Train Isolation Forest (unsupervised anomaly detection)
        print("Training Isolation Forest...")
        iso_forest = IsolationForest(
            contamination=0.1,
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(X_train_scaled)
        
        # Evaluate Isolation Forest
        y_pred_iso = iso_forest.predict(X_test_scaled)
        y_pred_iso = np.where(y_pred_iso == -1, 1, 0)  # Convert to binary
        iso_accuracy = accuracy_score(y_test, y_pred_iso)
        iso_f1 = f1_score(y_test, y_pred_iso, zero_division=0)
        
        results['isolation_forest'] = {
            'accuracy': iso_accuracy,
            'f1_score': iso_f1
        }
        
        # Train supervised models only if both classes are present
        if has_both_classes:
            # Train Random Forest classifier
            print("Training Random Forest...")
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            rf.fit(X_train_scaled, y_train)
            
            # Evaluate Random Forest
            y_pred_rf = rf.predict(X_test_scaled)
            rf_accuracy = accuracy_score(y_test, y_pred_rf)
            rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)
            
            # Calculate ROC-AUC if possible
            try:
                y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
                rf_auc = roc_auc_score(y_test, y_proba_rf)
            except:
                rf_auc = 0.0
                
            results['random_forest'] = {
                'accuracy': rf_accuracy,
                'f1_score': rf_f1,
                'roc_auc': rf_auc
            }
            
            # Train XGBoost classifier
            print("Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train_scaled, y_train)
            
            # Evaluate XGBoost
            y_pred_xgb = xgb_model.predict(X_test_scaled)
            xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
            xgb_f1 = f1_score(y_test, y_pred_xgb, zero_division=0)
            
            # Calculate ROC-AUC if possible
            try:
                y_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
                xgb_auc = roc_auc_score(y_test, y_proba_xgb)
            except:
                xgb_auc = 0.0
                
            results['xgboost'] = {
                'accuracy': xgb_accuracy,
                'f1_score': xgb_f1,
                'roc_auc': xgb_auc
            }
        else:
            # Skip supervised models and set default values
            print("SKIPPING Random Forest and XGBoost due to class imbalance")
            results['random_forest'] = {'accuracy': 0.0, 'f1_score': 0.0, 'roc_auc': 0.0}
            results['xgboost'] = {'accuracy': 0.0, 'f1_score': 0.0, 'roc_auc': 0.0}
        
        # Train K-Means clustering (unsupervised)
        print("Training K-Means Clustering...")
        
        n_clusters = 2  # Binary classification setup
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        kmeans.fit(X_train_scaled)
        
        # Predict clusters for test data
        cluster_labels = kmeans.predict(X_test_scaled)
        
        # Convert clusters to anomaly predictions (smaller cluster = anomalies)
        cluster_counts = np.bincount(cluster_labels)
        anomaly_cluster = np.argmin(cluster_counts)
        y_pred_kmeans = np.where(cluster_labels == anomaly_cluster, 1, 0)
        
        # Calculate clustering metrics
        kmeans_accuracy = accuracy_score(y_test, y_pred_kmeans)
        kmeans_f1 = f1_score(y_test, y_pred_kmeans, zero_division=0)
        
        # Calculate silhouette score for clustering quality
        try:
            silhouette_avg = silhouette_score(X_test_scaled, cluster_labels)
        except:
            silhouette_avg = 0.0
        
        results['kmeans'] = {
            'accuracy': kmeans_accuracy,
            'f1_score': kmeans_f1,
            'silhouette_score': silhouette_avg,
            'n_clusters': n_clusters,
            'anomaly_cluster': int(anomaly_cluster)
        }
        
        # Train Autoencoder (unsupervised neural network)
        print("Training Autoencoder...")
        
        # Design autoencoder architecture
        input_dim = X_train_scaled.shape[1]
        hidden_dim = max(10, input_dim // 2)  # Compression layer
        
        autoencoder = MLPRegressor(
            hidden_layer_sizes=(hidden_dim, input_dim),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # Train on normal data only if available
        if has_both_classes:
            normal_indices = np.where(y_train == 0)[0]
            if len(normal_indices) > 0:
                X_normal = X_train_scaled[normal_indices]
            else:
                X_normal = X_train_scaled
        else:
            X_normal = X_train_scaled
        
        autoencoder.fit(X_normal, X_normal)  # Learn to reconstruct input
        
        # Evaluate autoencoder using reconstruction error
        X_test_reconstructed = autoencoder.predict(X_test_scaled)
        reconstruction_errors = np.mean((X_test_scaled - X_test_reconstructed) ** 2, axis=1)
        
        # Set threshold for anomaly detection (top 10% reconstruction errors)
        threshold = np.percentile(reconstruction_errors, 90)
        y_pred_autoencoder = np.where(reconstruction_errors > threshold, 1, 0)
        
        # Calculate autoencoder metrics
        autoencoder_accuracy = accuracy_score(y_test, y_pred_autoencoder)
        autoencoder_f1 = f1_score(y_test, y_pred_autoencoder, zero_division=0)
        reconstruction_loss = mean_squared_error(X_test_scaled.flatten(), X_test_reconstructed.flatten())
        
        results['autoencoder'] = {
            'accuracy': autoencoder_accuracy,
            'f1_score': autoencoder_f1,
            'reconstruction_loss': reconstruction_loss,
            'threshold': threshold,
            'hidden_dim': hidden_dim
        }
        
        # Store trained models
        self.models[f'{dataset_name}_isolation_forest'] = iso_forest
        self.models[f'{dataset_name}_kmeans'] = kmeans
        self.models[f'{dataset_name}_autoencoder'] = autoencoder
        self.models[f'{dataset_name}_autoencoder_threshold'] = threshold
        
        # Store supervised models if they were trained
        if has_both_classes:
            self.models[f'{dataset_name}_random_forest'] = rf
            self.models[f'{dataset_name}_xgboost'] = xgb_model
        
        # Print training results
        print(f"Results for {dataset_name}:")
        print(f"   Isolation Forest - Accuracy: {iso_accuracy:.4f}, F1: {iso_f1:.4f}")
        
        if has_both_classes:
            print(f"   Random Forest    - Accuracy: {rf_accuracy:.4f}, F1: {rf_f1:.4f}, AUC: {rf_auc:.4f}")
            print(f"   XGBoost         - Accuracy: {xgb_accuracy:.4f}, F1: {xgb_f1:.4f}, AUC: {xgb_auc:.4f}")
        else:
            print(f"   Random Forest    - SKIPPED (class imbalance)")
            print(f"   XGBoost         - SKIPPED (class imbalance)")
            
        print(f"   K-Means         - Accuracy: {kmeans_accuracy:.4f}, F1: {kmeans_f1:.4f}, Silhouette: {silhouette_avg:.4f}")
        print(f"   Autoencoder     - Accuracy: {autoencoder_accuracy:.4f}, F1: {autoencoder_f1:.4f}, Loss: {reconstruction_loss:.4f}")
        
        print(f"Autoencoder Classification Report:")
        print(classification_report(y_test, y_pred_autoencoder))
        
        return results
    
    def save_models(self, dataset_name):
        """Save trained models and scalers to disk"""
        os.makedirs('models', exist_ok=True)
        
        # Save models that were successfully trained
        model_types = ['isolation_forest', 'random_forest', 'xgboost', 'kmeans', 'autoencoder']
        saved_models = []
        
        for model_type in model_types:
            model_key = f'{dataset_name}_{model_type}'
            if model_key in self.models:
                joblib.dump(self.models[model_key], f'models/{dataset_name}_{model_type}.pkl')
                print(f"Saved: {dataset_name}_{model_type}.pkl")
                saved_models.append(model_type)
        
        # Save autoencoder threshold
        threshold_key = f'{dataset_name}_autoencoder_threshold'
        if threshold_key in self.models:
            joblib.dump(self.models[threshold_key], f'models/{dataset_name}_autoencoder_threshold.pkl')
            print(f"Saved: {dataset_name}_autoencoder_threshold.pkl")
        
        # Save feature scaler
        if dataset_name in self.scalers:
            joblib.dump(self.scalers[dataset_name], f'models/{dataset_name}_scaler.pkl')
            print(f"Saved: {dataset_name}_scaler.pkl")
        
        print(f"Models saved for {dataset_name}")
        print(f"Successfully saved models: {', '.join(saved_models)}")
        
        if len(saved_models) < 5:
            skipped = set(model_types) - set(saved_models)
            print(f"Skipped models (due to data issues): {', '.join(skipped)}")
    
    def train_nsl_kdd(self):
        """Train models on NSL-KDD dataset"""
        print("Training on NSL-KDD Dataset")
        print("=" * 50)
        
        # Load pre-split training and test data
        train_df = pd.read_csv('data/NSL_KDD/KDDTrain+.csv', header=None)
        test_df = pd.read_csv('data/NSL_KDD/KDDTest+.csv', header=None)
        
        # Define NSL-KDD feature names (41 features + attack_type + difficulty)
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
        
        # Add column names
        column_names = feature_names + ['attack_type', 'difficulty']
        train_df.columns = column_names
        test_df.columns = column_names
        
        # Preprocess data
        train_df = self.preprocess_data(train_df, 'nsl_kdd')
        test_df = self.preprocess_data(test_df, 'nsl_kdd')
        
        # Prepare features and target
        X_train, y_train = self.prepare_features_target(train_df, 'nsl_kdd')
        X_test, y_test = self.prepare_features_target(test_df, 'nsl_kdd')
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Class distribution: {np.bincount(y_train)}")
        
        # Apply feature engineering if enabled
        X_train, X_test = self.apply_feature_engineering(X_train, X_test, y_train, 'nsl_kdd')
        
        # Train models
        results = self.train_models(X_train, X_test, y_train, y_test, 'nsl_kdd')
        
        # Save models
        self.save_models('nsl_kdd')
        
        return results
    
    def train_cicids2017(self):
        """Train models on CICIDS2017 dataset"""
        print("Training on CICIDS2017 Dataset")
        print("=" * 50)
        
        # Load multiple CICIDS2017 files for better representation
        files = [
            'data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv',
            'data/CICIDS2017/Tuesday-WorkingHours.pcap_ISCX.csv',
            'data/CICIDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
        ]
        
        dfs = []
        for file in files:
            if os.path.exists(file):
                print(f"Loading {file}...")
                try:
                    df = pd.read_csv(file)
                    print(f"  Loaded {len(df)} records from {file}")
                    dfs.append(df)
                except Exception as e:
                    print(f"  ERROR loading {file}: {e}")
            else:
                print(f"  File not found: {file}")
        
        if not dfs:
            print("ERROR: No CICIDS2017 data files found!")
            return {}
        
        # Combine all datasets
        df = pd.concat(dfs, ignore_index=True)
        print(f"Combined dataset: {len(df)} total records")
        
        # Report initial class distribution
        if 'Label' in df.columns:
            initial_counts = df['Label'].value_counts()
            print(f"Initial class distribution:")
            for label, count in initial_counts.items():
                print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
        
        # Preprocess data
        df = self.preprocess_data(df, 'cicids2017')
        
        # Prepare features and target
        X, y = self.prepare_features_target(df, 'cicids2017')
        
        # Check class distribution after preprocessing
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print(f"Class distribution after preprocessing:")
        for cls, count in zip(unique_classes, class_counts):
            label_name = "Normal" if cls == 0 else "Anomaly"
            print(f"  {label_name} (class {cls}): {count} ({count/len(y)*100:.1f}%)")
        
        # Check if we have both classes
        if len(unique_classes) < 2:
            print("WARNING: Only one class found after preprocessing!")
            print("Continuing with unsupervised models only...")
        
        # Split data with stratification if possible
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError as e:
            print(f"Stratified split failed: {e}")
            print("Using random split instead...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Training class distribution: {np.bincount(y_train)}")
        print(f"Test class distribution: {np.bincount(y_test)}")
        
        # Apply feature engineering if enabled
        X_train, X_test = self.apply_feature_engineering(X_train, X_test, y_train, 'cicids2017')
        
        # Train models
        results = self.train_models(X_train, X_test, y_train, y_test, 'cicids2017')
        
        # Save models
        self.save_models('cicids2017')
        
        return results
    
    def train_unsw_nb15(self):
        """Train models on UNSW-NB15 dataset"""
        print("Training on UNSW-NB15 Dataset")
        print("=" * 50)
        
        # Load pre-split training and test data
        train_df = pd.read_csv('data/UNSW_NB15/UNSW_NB15_training-set.csv')
        test_df = pd.read_csv('data/UNSW_NB15/UNSW_NB15_testing-set.csv')
        
        # Preprocess data
        train_df = self.preprocess_data(train_df, 'unsw_nb15')
        test_df = self.preprocess_data(test_df, 'unsw_nb15')
        
        # Prepare features and target
        X_train, y_train = self.prepare_features_target(train_df, 'unsw_nb15')
        X_test, y_test = self.prepare_features_target(test_df, 'unsw_nb15')
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Class distribution: {np.bincount(y_train)}")
        
        # Apply feature engineering if enabled
        X_train, X_test = self.apply_feature_engineering(X_train, X_test, y_train, 'unsw_nb15')
        
        # Train models
        results = self.train_models(X_train, X_test, y_train, y_test, 'unsw_nb15')
        
        # Save models
        self.save_models('unsw_nb15')
        
        return results
    
    def train_ton_iot(self):
        """Train models on TON-IoT dataset"""
        print("Training on TON-IoT Dataset")
        print("=" * 50)
        
        # Load dataset
        df = pd.read_csv('data/TON_IOT/train_test_network.csv')
        
        # Preprocess data
        df = self.preprocess_data(df, 'ton_iot')
        
        # Prepare features and target
        X, y = self.prepare_features_target(df, 'ton_iot')
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Class distribution: {np.bincount(y_train)}")
        
        # Apply feature engineering if enabled
        X_train, X_test = self.apply_feature_engineering(X_train, X_test, y_train, 'ton_iot')
        
        # Train models
        results = self.train_models(X_train, X_test, y_train, y_test, 'ton_iot')
        
        # Save models
        self.save_models('ton_iot')
        
        return results

def main():
    """Main function for command-line training execution"""
    parser = argparse.ArgumentParser(description='Network Anomaly Detection Model Training')
    parser.add_argument('--dataset', choices=['nsl_kdd', 'cicids2017', 'unsw_nb15', 'ton_iot', 'all'],
                       required=True, help='Dataset to train on')
    
    # Feature engineering options
    parser.add_argument('--feature_engineering', action='store_true',
                       help='Enable advanced feature engineering')
    parser.add_argument('--pca_components', type=int,
                       help='Number of PCA components (e.g., 20)')
    parser.add_argument('--feature_selection', 
                       choices=['variance_threshold', 'select_k_best_10', 'select_k_best_20', 
                               'rfe_10', 'rfe_20', 'rfe_30'],
                       help='Feature selection method')
    parser.add_argument('--dimensionality_analysis', action='store_true',
                       help='Perform comprehensive dimensionality analysis')
    
    args = parser.parse_args()
    
    # Create trainer with specified options
    trainer = AnomalyTrainer(
        enable_feature_engineering=args.feature_engineering,
        pca_components=args.pca_components,
        feature_selection=args.feature_selection,
        dimensionality_analysis=args.dimensionality_analysis
    )
    
    print("Network Anomaly Detection Training System")
    print("=" * 60)
    print(f"Training on: {args.dataset}")
    if args.feature_engineering:
        print("Feature Engineering: ENABLED")
        if args.pca_components:
            print(f"  - PCA Components: {args.pca_components}")
        if args.feature_selection:
            print(f"  - Feature Selection: {args.feature_selection}")
        if args.dimensionality_analysis:
            print(f"  - Dimensionality Analysis: ENABLED")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 60)
    
    results = {}
    
    # Train on specified datasets
    if args.dataset == 'nsl_kdd' or args.dataset == 'all':
        results['nsl_kdd'] = trainer.train_nsl_kdd()
    
    if args.dataset == 'cicids2017' or args.dataset == 'all':
        results['cicids2017'] = trainer.train_cicids2017()
    
    if args.dataset == 'unsw_nb15' or args.dataset == 'all':
        results['unsw_nb15'] = trainer.train_unsw_nb15()
    
    if args.dataset == 'ton_iot' or args.dataset == 'all':
        results['ton_iot'] = trainer.train_ton_iot()
    
    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for dataset, scores in results.items():
        print(f"{dataset.upper()}:")
        print(f"  Isolation Forest: {scores['isolation_forest']['accuracy']:.4f}")
        print(f"  Random Forest:    {scores['random_forest']['accuracy']:.4f}")
        print(f"  XGBoost:          {scores['xgboost']['accuracy']:.4f}")
        print(f"  K-Means:          {scores['kmeans']['accuracy']:.4f}")
        print(f"  Autoencoder:      {scores['autoencoder']['accuracy']:.4f}")
    print("=" * 60)
    print("TRAINING COMPLETED! Models saved to 'models/' directory")

if __name__ == "__main__":
    main() 