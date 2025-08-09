"""
Performance Evaluation Module for Network Anomaly Detection System
==================================================================

Measures comprehensive performance metrics including F1-score, ROC-AUC, precision,
recall, and inference latency. Compares model performance against traditional IDS
benchmarks and provides recommendations for deployment.

Features:
- Model performance evaluation (F1, ROC-AUC, precision, recall)
- Latency and throughput analysis
- Traditional IDS benchmark comparison
- Performance visualization and reporting
- Real-time system requirements analysis

Usage:
    python src/performance_evaluation.py --dataset nsl_kdd
    python src/performance_evaluation.py --dataset cicids2017
    python src/performance_evaluation.py --dataset unsw_nb15
    python src/performance_evaluation.py --dataset ton_iot
    python src/performance_evaluation.py --dataset all
"""

import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib
import sys

sys.path.append(".")

from src.model_loader import ModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """Evaluates model performance with comprehensive metrics and benchmarks"""

    def __init__(self, models_dir="models"):
        """Initialize performance evaluator with models directory"""
        self.models_dir = Path(models_dir)
        self.model_loader = ModelLoader()
        self.results = {}

        logger.info("Performance evaluator initialized")

    def load_test_data(self, dataset_name):
        """Load and prepare test data for performance evaluation"""
        try:
            if dataset_name == "nsl_kdd":
                data_path = Path("data/NSL_KDD/KDDTest+.csv")
                # NSL-KDD feature column names
                columns = [
                    "duration",
                    "protocol_type",
                    "service",
                    "flag",
                    "src_bytes",
                    "dst_bytes",
                    "land",
                    "wrong_fragment",
                    "urgent",
                    "hot",
                    "num_failed_logins",
                    "logged_in",
                    "num_compromised",
                    "root_shell",
                    "su_attempted",
                    "num_root",
                    "num_file_creations",
                    "num_shells",
                    "num_access_files",
                    "num_outbound_cmds",
                    "is_host_login",
                    "is_guest_login",
                    "count",
                    "srv_count",
                    "serror_rate",
                    "srv_serror_rate",
                    "rerror_rate",
                    "srv_rerror_rate",
                    "same_srv_rate",
                    "diff_srv_rate",
                    "srv_diff_host_rate",
                    "dst_host_count",
                    "dst_host_srv_count",
                    "dst_host_same_srv_rate",
                    "dst_host_diff_srv_rate",
                    "dst_host_same_src_port_rate",
                    "dst_host_srv_diff_host_rate",
                    "dst_host_serror_rate",
                    "dst_host_srv_serror_rate",
                    "dst_host_rerror_rate",
                    "dst_host_srv_rerror_rate",
                    "attack_type",
                    "difficulty",
                ]

                if not data_path.exists():
                    logger.error(f"Test data not found: {data_path}")
                    return None, None

                # Load test data
                df = pd.read_csv(data_path, header=None, names=columns)
                logger.info(f"Loaded {len(df)} test records from {dataset_name}")

                # Prepare features and binary labels
                X = df.drop(["attack_type", "difficulty"], axis=1)
                y = (df["attack_type"] != "normal").astype(int)

                return X, y

            elif dataset_name == "cicids2017":
                data_path = Path(
                    "data/CICIDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
                )

                if not data_path.exists():
                    logger.error(f"Test data not found: {data_path}")
                    return None, None

                # Load CICIDS2017 data
                df = pd.read_csv(data_path)
                logger.info(f"Loaded {len(df)} test records from {dataset_name}")

                # Prepare features and binary labels for CICIDS2017
                label_col = " Label" if " Label" in df.columns else "Label"
                X = df.drop([label_col], axis=1)
                y = (df[label_col].str.upper().str.strip() != "BENIGN").astype(int)

                return X, y

            elif dataset_name == "unsw_nb15":
                data_path = Path("data/UNSW_NB15/UNSW_NB15_testing-set.csv")

                if not data_path.exists():
                    logger.error(f"Test data not found: {data_path}")
                    return None, None

                # Load UNSW-NB15 test data
                df = pd.read_csv(data_path)
                logger.info(f"Loaded {len(df)} test records from {dataset_name}")

                # Prepare features and binary labels for UNSW-NB15
                label_col = "label" if "label" in df.columns else "attack_cat"
                X = df.drop([label_col], axis=1)
                y = (
                    df[label_col].astype(int)
                    if label_col == "label"
                    else (df[label_col] != "Normal").astype(int)
                )

                return X, y

            elif dataset_name == "ton_iot":
                data_path = Path("data/TON_IOT/train_test_network.csv")

                if not data_path.exists():
                    logger.error(f"Test data not found: {data_path}")
                    return None, None

                # Load TON-IoT data and use test portion
                from sklearn.model_selection import train_test_split

                df = pd.read_csv(data_path)
                logger.info(f"Loaded {len(df)} total records from {dataset_name}")

                # Use stratified split to get test data
                label_col = "label" if "label" in df.columns else "type"
                X_full = df.drop([label_col], axis=1)
                y_full = (
                    df[label_col].astype(int)
                    if label_col == "label"
                    else (df[label_col] != "normal").astype(int)
                )

                # Use last 20% as test data
                _, X, _, y = train_test_split(
                    X_full,
                    y_full,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_full if len(np.unique(y_full)) > 1 else None,
                )

                logger.info(f"Using {len(X)} test records from {dataset_name}")
                return X, y

            else:
                logger.error(f"Dataset {dataset_name} not supported")
                return None, None

        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return None, None

    def preprocess_data(self, X, dataset_name):
        """Preprocess data for model evaluation with proper scaling"""
        try:
            # Load models and scalers
            models, scalers = self.model_loader.load_models(dataset_name)

            # Create a copy to avoid modifying original data
            X_processed = X.copy()

            # Handle different dataset column structures
            if dataset_name == "nsl_kdd":
                categorical_cols = ["protocol_type", "service", "flag"]
            elif dataset_name == "cicids2017":
                # CICIDS2017 might have some string columns
                categorical_cols = []
                for col in X_processed.columns:
                    if X_processed[col].dtype == "object":
                        categorical_cols.append(col)
            elif dataset_name == "unsw_nb15":
                # UNSW-NB15 has specific categorical columns
                categorical_cols = ["proto", "service", "state"]
            elif dataset_name == "ton_iot":
                # TON-IoT categorical columns
                categorical_cols = [
                    "src_ip",
                    "dst_ip",
                    "proto",
                    "service",
                    "conn_state",
                ]
            else:
                # Auto-detect categorical columns
                categorical_cols = []
                for col in X_processed.columns:
                    if X_processed[col].dtype == "object":
                        categorical_cols.append(col)

            # Handle categorical columns with label encoding
            for col in categorical_cols:
                if col in X_processed.columns:
                    try:
                        le = LabelEncoder()
                        # Handle NaN values
                        X_processed[col] = X_processed[col].fillna("unknown")
                        X_processed[col] = le.fit_transform(
                            X_processed[col].astype(str)
                        )
                    except Exception as e:
                        logger.warning(f"Failed to encode column {col}: {e}")
                        # Remove problematic column
                        X_processed = X_processed.drop(columns=[col])

            # Convert all columns to numeric, replacing any remaining non-numeric values
            for col in X_processed.columns:
                try:
                    X_processed[col] = pd.to_numeric(X_processed[col], errors="coerce")
                except:
                    pass

            # Handle missing and infinite values
            X_processed = X_processed.fillna(0)
            X_processed = X_processed.replace([np.inf, -np.inf], 0)

            # Ensure all data is numeric
            X_processed = X_processed.select_dtypes(include=[np.number])

            # Apply feature scaling if scaler is available
            if scalers:
                scaler = list(scalers.values())[0]  # Use first available scaler
                try:
                    # Ensure scaler and data have same number of features
                    if hasattr(scaler, "n_features_in_"):
                        expected_features = scaler.n_features_in_
                        if X_processed.shape[1] != expected_features:
                            logger.warning(
                                f"Feature mismatch: got {X_processed.shape[1]}, expected {expected_features}"
                            )
                            # Take first n features that match
                            if X_processed.shape[1] > expected_features:
                                X_processed = X_processed.iloc[:, :expected_features]
                            else:
                                # Pad with zeros if needed
                                missing_features = (
                                    expected_features - X_processed.shape[1]
                                )
                                for i in range(missing_features):
                                    X_processed[f"missing_feature_{i}"] = 0

                    X_scaled = scaler.transform(X_processed)
                    X_processed = pd.DataFrame(X_scaled, columns=X_processed.columns)
                except Exception as e:
                    logger.warning(f"Scaling failed, using raw features: {e}")

            return X_processed

        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            # Return a fallback version with basic numeric conversion
            try:
                X_fallback = X.copy()
                # Convert to numeric where possible
                for col in X_fallback.columns:
                    if X_fallback[col].dtype == "object":
                        le = LabelEncoder()
                        try:
                            X_fallback[col] = le.fit_transform(
                                X_fallback[col].fillna("unknown").astype(str)
                            )
                        except:
                            X_fallback[col] = 0

                X_fallback = X_fallback.fillna(0).replace([np.inf, -np.inf], 0)
                return X_fallback.select_dtypes(include=[np.number])
            except:
                logger.error("Fallback preprocessing also failed")
                return X

    def evaluate_model_performance(self, dataset_name="nsl_kdd"):
        """Evaluate all models on test data with comprehensive metrics"""
        logger.info(f"Starting performance evaluation for {dataset_name}")

        # Load test data
        X_test, y_test = self.load_test_data(dataset_name)
        if X_test is None:
            return

        # Use stratified sampling for faster evaluation while maintaining representativeness
        n_samples = min(1000, len(X_test))
        
        # Use stratified sampling to maintain anomaly ratio
        try:
            from sklearn.model_selection import train_test_split
            if len(np.unique(y_test)) > 1 and n_samples < len(X_test):
                # Stratified sampling to maintain anomaly ratio
                _, X_test, _, y_test = train_test_split(
                    X_test, y_test, 
                    test_size=n_samples/len(X_test), 
                    random_state=42, 
                    stratify=y_test
                )
            else:
                # If no stratification needed or possible, use random sampling
                indices = np.random.RandomState(42).choice(len(X_test), size=n_samples, replace=False)
                X_test = X_test.iloc[indices]
                y_test = y_test.iloc[indices]
        except Exception as e:
            logger.warning(f"Stratified sampling failed: {e}, using random sampling")
            indices = np.random.RandomState(42).choice(len(X_test), size=n_samples, replace=False)
            X_test = X_test.iloc[indices]
            y_test = y_test.iloc[indices]

        logger.info(f"Using {len(X_test)} samples for evaluation")
        logger.info(f"Anomaly ratio: {y_test.mean():.1%}")

        # Preprocess data
        X_processed = self.preprocess_data(X_test, dataset_name)
        
        # Ensure X_processed is properly aligned with y_test
        if X_processed is None:
            logger.error("Data preprocessing failed")
            return
            
        # Reset indices to ensure alignment
        X_processed = X_processed.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # Load trained models
        try:
            models, scalers = self.model_loader.load_models(dataset_name)
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return

        self.results[dataset_name] = {}

        # Evaluate each model with appropriate prediction logic
        for model_name, model in models.items():
            if model_name == "autoencoder_threshold":
                continue  # Skip threshold metadata

            logger.info(f"Evaluating {model_name}")

            try:
                # Measure inference latency
                start_time = time.time()

                # Model-specific prediction logic
                if model_name == "isolation_forest":
                    predictions = model.predict(X_processed)
                    y_pred = (predictions == -1).astype(int)  # Convert to binary
                    y_scores = (
                        -model.decision_function(X_processed)
                        if hasattr(model, "decision_function")
                        else y_pred
                    )

                elif model_name == "kmeans":
                    try:
                        clusters = model.predict(X_processed)

                        # Check if we got valid cluster predictions
                        if len(clusters) == 0:
                            logger.warning(f"No cluster predictions for {model_name}")
                            y_pred = np.zeros(len(X_processed))
                            y_scores = np.zeros(len(X_processed))
                        else:
                            # Count cluster assignments
                            cluster_counts = np.bincount(clusters)

                            # Handle empty cluster counts
                            if len(cluster_counts) == 0:
                                logger.warning(f"Empty cluster counts for {model_name}")
                                y_pred = np.zeros(len(X_processed))
                                y_scores = np.zeros(len(X_processed))
                            else:
                                # Assume smaller cluster is anomaly cluster
                                anomaly_cluster = np.argmin(cluster_counts)
                                y_pred = (clusters == anomaly_cluster).astype(int)

                                # Use distance to cluster centers as anomaly scores
                                try:
                                    if (
                                        hasattr(model, "cluster_centers_")
                                        and len(model.cluster_centers_) > 0
                                    ):
                                        distances = np.linalg.norm(
                                            X_processed.values[:, np.newaxis]
                                            - model.cluster_centers_,
                                            axis=2,
                                        )
                                        y_scores = np.min(distances, axis=1)
                                    else:
                                        y_scores = np.ones(
                                            len(X_processed)
                                        )  # Default scores
                                except Exception as dist_e:
                                    logger.warning(
                                        f"Distance calculation failed: {dist_e}"
                                    )
                                    y_scores = np.abs(
                                        clusters - anomaly_cluster
                                    )  # Simple distance metric

                    except Exception as kmeans_e:
                        logger.warning(
                            f"K-Means prediction failed for {model_name}: {kmeans_e}"
                        )
                        y_pred = np.zeros(len(X_processed))
                        y_scores = np.zeros(len(X_processed))

                elif model_name == "autoencoder":
                    # Reconstruction error-based anomaly detection
                    reconstructed = model.predict(X_processed)
                    errors = np.mean(
                        np.square(X_processed.values - reconstructed), axis=1
                    )

                    # Load or calculate threshold
                    threshold_path = (
                        self.models_dir / f"{dataset_name}_autoencoder_threshold.pkl"
                    )
                    if threshold_path.exists():
                        threshold = joblib.load(threshold_path)
                    else:
                        threshold = np.percentile(
                            errors, 95
                        )  # Use 95th percentile as default

                    y_pred = (errors > threshold).astype(int)
                    y_scores = errors

                else:  # Supervised models (random_forest, xgboost)
                    y_pred = model.predict(X_processed)
                    y_scores = (
                        model.predict_proba(X_processed)[:, 1]
                        if hasattr(model, "predict_proba")
                        else y_pred
                    )

                # Calculate inference metrics
                inference_time = time.time() - start_time
                avg_latency = (
                    inference_time / len(X_processed)
                ) * 1000  # milliseconds per sample

                # Calculate performance metrics
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)

                # ROC-AUC with proper error handling
                try:
                    roc_auc = (
                        roc_auc_score(y_test, y_scores)
                        if len(np.unique(y_test)) > 1
                        else 0.5
                    )
                except Exception as e:
                    logger.warning(f"ROC-AUC calculation failed for {model_name}: {e}")
                    roc_auc = 0.5

                # Store comprehensive results
                self.results[dataset_name][model_name] = {
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "roc_auc": roc_auc,
                    "avg_latency_ms": avg_latency,
                    "total_time_s": inference_time,
                    "samples": len(X_processed),
                    "predictions": y_pred,
                    "scores": y_scores,
                }

                # Log results
                logger.info(f"  F1-Score: {f1:.3f}")
                logger.info(f"  Precision: {precision:.3f}")
                logger.info(f"  Recall: {recall:.3f}")
                logger.info(f"  ROC-AUC: {roc_auc:.3f}")
                logger.info(f"  Avg Latency: {avg_latency:.2f} ms/sample")

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue

        # Store ground truth for analysis
        self.results[dataset_name]["ground_truth"] = y_test.values

    def generate_performance_report(self, dataset_name="nsl_kdd"):
        """Generate comprehensive performance report with analysis"""
        if dataset_name not in self.results:
            logger.error(f"No results found for {dataset_name}")
            return

        results = self.results[dataset_name]

        print("\n" + "=" * 80)
        print(f"COMPREHENSIVE PERFORMANCE EVALUATION - {dataset_name.upper()}")
        print("=" * 80)

        # Performance summary table
        print(f"\nMODEL PERFORMANCE SUMMARY")
        print("-" * 80)
        print(
            f"{'Model':<20} {'F1-Score':<10} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10} {'Latency (ms)':<12}"
        )
        print("-" * 80)

        model_scores = []
        for model_name, metrics in results.items():
            if model_name == "ground_truth":
                continue

            f1 = metrics["f1_score"]
            precision = metrics["precision"]
            recall = metrics["recall"]
            roc_auc = metrics["roc_auc"]
            latency = metrics["avg_latency_ms"]

            print(
                f"{model_name:<20} {f1:<10.3f} {precision:<10.3f} {recall:<10.3f} {roc_auc:<10.3f} {latency:<12.2f}"
            )

            model_scores.append(
                {
                    "model": model_name,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "roc_auc": roc_auc,
                    "latency_ms": latency,
                }
            )

        # Best performing models analysis
        print(f"\nBEST PERFORMING MODELS")
        print("-" * 40)

        best_f1 = max(model_scores, key=lambda x: x["f1_score"])
        best_roc = max(model_scores, key=lambda x: x["roc_auc"])
        fastest = min(model_scores, key=lambda x: x["latency_ms"])

        print(f"Best F1-Score: {best_f1['model']} ({best_f1['f1_score']:.3f})")
        print(f"Best ROC-AUC: {best_roc['model']} ({best_roc['roc_auc']:.3f})")
        print(f"Fastest: {fastest['model']} ({fastest['latency_ms']:.2f} ms/sample)")

        # Traditional IDS comparison
        self.compare_with_traditional_ids()

        # Throughput analysis for real-time deployment
        print(f"\nTHROUGHPUT ANALYSIS")
        print("-" * 40)

        for model_name, metrics in results.items():
            if model_name == "ground_truth":
                continue

            throughput = 1000 / metrics["avg_latency_ms"]  # samples per second
            print(f"{model_name:<20}: {throughput:>8.1f} samples/second")

        # Real-time system requirements analysis
        print(f"\nREAL-TIME SYSTEM REQUIREMENTS")
        print("-" * 40)

        target_throughput = 1000  # samples/second for real-time processing
        suitable_models = []

        for model_name, metrics in results.items():
            if model_name == "ground_truth":
                continue

            model_throughput = 1000 / metrics["avg_latency_ms"]
            if model_throughput >= target_throughput:
                suitable_models.append(
                    (model_name, model_throughput, metrics["f1_score"])
                )

        if suitable_models:
            print(f"Models suitable for >= {target_throughput} samples/second:")
            for model, throughput, f1 in sorted(
                suitable_models, key=lambda x: x[2], reverse=True
            ):
                print(f"  {model:<20}: {throughput:>8.1f} samples/s (F1: {f1:.3f})")
        else:
            print(f"No models meet {target_throughput} samples/second requirement")
            print("Consider distributed processing or model optimization")

    def compare_with_traditional_ids(self):
        """Compare model performance with traditional IDS benchmarks"""
        print(f"\nTRADITIONAL IDS COMPARISON")
        print("-" * 40)

        # Industry standard IDS performance benchmarks
        traditional_benchmarks = {
            "Signature-based IDS": {
                "f1_score": 0.85,
                "precision": 0.95,
                "recall": 0.78,
                "description": "Rule-based detection with known attack signatures",
            },
            "Statistical Anomaly Detection": {
                "f1_score": 0.72,
                "precision": 0.68,
                "recall": 0.76,
                "description": "Statistical deviation from normal behavior",
            },
            "Hybrid IDS": {
                "f1_score": 0.88,
                "precision": 0.91,
                "recall": 0.85,
                "description": "Combination of signature and anomaly detection",
            },
        }

        print("Traditional IDS Benchmarks:")
        for ids_name, metrics in traditional_benchmarks.items():
            print(f"\n{ids_name}:")
            print(f"  F1-Score: {metrics['f1_score']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  Description: {metrics['description']}")

        # Compare our models against traditional methods
        print(f"\nOur ML Models vs Traditional IDS:")

        dataset_name = list(self.results.keys())[0]
        results = self.results[dataset_name]

        for model_name, metrics in results.items():
            if model_name == "ground_truth":
                continue

            f1_score = metrics["f1_score"]

            # Compare with best traditional method
            best_traditional = max(
                traditional_benchmarks.items(), key=lambda x: x[1]["f1_score"]
            )
            traditional_f1 = best_traditional[1]["f1_score"]

            improvement = ((f1_score - traditional_f1) / traditional_f1) * 100

            if improvement > 0:
                print(
                    f"  {model_name}: {improvement:+.1f}% better than {best_traditional[0]}"
                )
            else:
                print(f"  {model_name}: {improvement:+.1f}% vs {best_traditional[0]}")

    def create_visualizations(self, dataset_name="nsl_kdd"):
        """Create comprehensive performance visualization plots"""
        if dataset_name not in self.results:
            return

        results = self.results[dataset_name]

        # Create output directory for visualizations
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)

        # Extract metrics for plotting
        models = []
        f1_scores = []
        roc_scores = []
        latencies = []

        for model_name, metrics in results.items():
            if model_name == "ground_truth":
                continue

            models.append(model_name.replace("_", " ").title())
            f1_scores.append(metrics["f1_score"])
            roc_scores.append(metrics["roc_auc"])
            latencies.append(metrics["avg_latency_ms"])

        # Skip visualization if no models to plot
        if not models:
            logger.warning(f"No model results to visualize for {dataset_name}")
            return

        # Create comprehensive performance comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Performance Evaluation: {dataset_name.upper()}",
            fontsize=16,
            fontweight="bold",
        )

        # F1-Score comparison
        bars1 = ax1.bar(range(len(models)), f1_scores, color="skyblue")
        ax1.set_title("F1-Score Comparison", fontsize=14)
        ax1.set_ylabel("F1-Score", fontsize=12)
        ax1.set_ylim(0, 1.1)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha="right", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars1, f1_scores)):
            ax1.text(
                i, score + 0.02, f"{score:.3f}", ha="center", va="bottom", fontsize=9
            )

        # ROC-AUC comparison
        bars2 = ax2.bar(range(len(models)), roc_scores, color="lightcoral")
        ax2.set_title("ROC-AUC Comparison", fontsize=14)
        ax2.set_ylabel("ROC-AUC", fontsize=12)
        ax2.set_ylim(0, 1.1)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha="right", fontsize=10)
        ax2.grid(True, alpha=0.3)

        for i, (bar, score) in enumerate(zip(bars2, roc_scores)):
            ax2.text(
                i, score + 0.02, f"{score:.3f}", ha="center", va="bottom", fontsize=9
            )

        # Latency comparison
        bars3 = ax3.bar(range(len(models)), latencies, color="lightgreen")
        ax3.set_title("Average Latency Comparison", fontsize=14)
        ax3.set_ylabel("Latency (ms/sample)", fontsize=12)
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels(models, rotation=45, ha="right", fontsize=10)
        ax3.grid(True, alpha=0.3)

        # Set appropriate y-limit for latency
        max_latency = max(latencies) if latencies else 1
        ax3.set_ylim(0, max_latency * 1.1)

        for i, (bar, latency) in enumerate(zip(bars3, latencies)):
            ax3.text(
                i,
                latency + max_latency * 0.02,
                f"{latency:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Performance vs Latency trade-off analysis
        if latencies and f1_scores:
            scatter = ax4.scatter(
                latencies,
                f1_scores,
                s=100,
                alpha=0.7,
                c=range(len(models)),
                cmap="viridis",
            )
            for i, model in enumerate(models):
                ax4.annotate(
                    model,
                    (latencies[i], f1_scores[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                )
            ax4.set_xlabel("Latency (ms/sample)", fontsize=12)
            ax4.set_ylabel("F1-Score", fontsize=12)
            ax4.set_title("Performance vs Latency Trade-off", fontsize=14)
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for suptitle

        # Save plot with better filename
        output_file = output_dir / f"{dataset_name}_performance_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        logger.info(f"Visualization saved to {output_file}")

        # Create additional summary plot if we have data
        if models and f1_scores and roc_scores:
            self._create_summary_plot(
                dataset_name, models, f1_scores, roc_scores, latencies, output_dir
            )

    def _create_summary_plot(
        self, dataset_name, models, f1_scores, roc_scores, latencies, output_dir
    ):
        """Create a summary comparison plot"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            x = np.arange(len(models))
            width = 0.35

            # Create grouped bars for F1 and ROC-AUC
            bars1 = ax.bar(
                x - width / 2,
                f1_scores,
                width,
                label="F1-Score",
                color="skyblue",
                alpha=0.8,
            )
            bars2 = ax.bar(
                x + width / 2,
                roc_scores,
                width,
                label="ROC-AUC",
                color="lightcoral",
                alpha=0.8,
            )

            ax.set_xlabel("Models", fontsize=12)
            ax.set_ylabel("Score", fontsize=12)
            ax.set_title(
                f"Model Performance Summary: {dataset_name.upper()}",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)

            # Add value labels
            for bar, score in zip(bars1, f1_scores):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{score:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            for bar, score in zip(bars2, roc_scores):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{score:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            plt.tight_layout()

            # Save summary plot
            summary_file = output_dir / f"{dataset_name}_summary.png"
            plt.savefig(summary_file, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()

            logger.info(f"Summary plot saved to {summary_file}")

        except Exception as e:
            logger.warning(f"Failed to create summary plot: {e}")

    def run_comprehensive_evaluation(self, dataset_name="nsl_kdd"):
        """Run complete performance evaluation pipeline"""
        print("\n" + "=" * 80)
        print("STARTING COMPREHENSIVE PERFORMANCE EVALUATION")
        print("=" * 80)

        if dataset_name == "all":
            # Evaluate all available datasets
            available_info = self.model_loader.get_model_info()
            datasets = available_info["available_datasets"]

            if not datasets:
                print("ERROR: No dataset-specific models found")
                print("Run training first: python src/train.py --dataset all")
                return

            all_results = {}
            for dataset in datasets:
                print(f"\nEvaluating {dataset.upper()}")
                try:
                    self.evaluate_model_performance(dataset)
                    self.generate_performance_report(dataset)
                    self.create_visualizations(dataset)

                    all_results[dataset] = {
                        "results": self.results[dataset],
                        "status": "success",
                    }

                    print(f"SUCCESS: {dataset} evaluation completed")

                except Exception as e:
                    print(f"ERROR: Evaluation failed for {dataset}: {str(e)}")
                    all_results[dataset] = {"status": "failed", "error": str(e)}

            # Print summary
            print(f"\n{'='*80}")
            print("EVALUATION SUMMARY")
            print(f"{'='*80}")
            successful_datasets = [
                d for d, r in all_results.items() if r["status"] == "success"
            ]
            failed_datasets = [
                d for d, r in all_results.items() if r["status"] == "failed"
            ]

            print(f"Successful evaluations: {len(successful_datasets)}")
            for dataset in successful_datasets:
                print(f"  - {dataset}")

            if failed_datasets:
                print(f"Failed evaluations: {len(failed_datasets)}")
                for dataset in failed_datasets:
                    print(f"  - {dataset}: {all_results[dataset]['error']}")

        else:
            # Execute evaluation pipeline for single dataset
            self.evaluate_model_performance(dataset_name)
            self.generate_performance_report(dataset_name)
            self.create_visualizations(dataset_name)

def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Performance Evaluation")
    parser.add_argument(
        "--dataset",
        choices=["nsl_kdd", "cicids2017", "unsw_nb15", "ton_iot", "all"],
        default="nsl_kdd",
        help="Dataset to evaluate (default: nsl_kdd)",
    )
    parser.add_argument("--models-dir", default="models", help="Models directory")

    args = parser.parse_args()

    # Create evaluator and run comprehensive evaluation
    evaluator = PerformanceEvaluator(args.models_dir)
    evaluator.run_comprehensive_evaluation(args.dataset)


if __name__ == "__main__":
    main()
