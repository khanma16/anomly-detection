"""
Advanced Feature Engineering Module - Phase 2
==============================================
Implements dimensionality reduction and feature selection techniques

Features:
- PCA (Principal Component Analysis)
- t-SNE (t-distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)
- Feature Selection (SelectKBest, RFE, Variance Threshold)
- Feature Importance Analysis
- Visualization and Analysis Tools

Usage:
    from src.feature_engineering import AdvancedFeatureEngineer
    
    engineer = AdvancedFeatureEngineer()
    X_reduced = engineer.apply_pca(X_train, n_components=20)
    X_embedded = engineer.apply_umap(X_train, n_components=2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFE, VarianceThreshold, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
import umap.umap_ as umap

import joblib
import os
import yaml
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """Advanced feature engineering with dimensionality reduction and selection"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the feature engineer"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except:
            # Default configuration if config file not found
            self.config = {
                'feature_engineering': {
                    'output_path': 'feature_analysis/',
                    'save_transformers': True
                }
            }
        
        self.output_dir = self.config.get('feature_engineering', {}).get('output_path', 'feature_analysis/')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Store fitted transformers
        self.transformers = {}
        self.feature_info = {}
        
        logger.info("Advanced Feature Engineer initialized")
    
    def analyze_feature_distribution(self, X: pd.DataFrame, y: pd.Series = None, 
                                   dataset_name: str = "dataset") -> Dict[str, Any]:
        """Analyze feature distributions and correlations"""
        logger.info(f"Analyzing feature distribution for {dataset_name}")
        
        analysis = {
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'feature_types': {},
            'missing_values': {},
            'variance_analysis': {},
            'correlation_analysis': {}
        }
        
        # Feature types and missing values
        for col in X.columns:
            analysis['feature_types'][col] = str(X[col].dtype)
            analysis['missing_values'][col] = int(X[col].isnull().sum())  # Convert to int
        
        # Variance analysis
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            variances = X[numeric_cols].var()
            analysis['variance_analysis'] = {
                'low_variance_features': variances[variances < 0.01].index.tolist(),
                'high_variance_features': variances[variances > variances.quantile(0.95)].index.tolist(),
                'mean_variance': float(variances.mean()),
                'variance_distribution': {k: float(v) for k, v in variances.describe().to_dict().items()}
            }
        
        # Correlation analysis (sample if too many features)
        if len(numeric_cols) > 0:
            sample_cols = numeric_cols[:50] if len(numeric_cols) > 50 else numeric_cols
            corr_matrix = X[sample_cols].corr()
            
            # Find highly correlated features
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': float(corr_val)
                        })
            
            analysis['correlation_analysis'] = {
                'high_correlation_pairs': high_corr_pairs,
                'mean_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean())
            }
        
        # Save analysis
        analysis_path = os.path.join(self.output_dir, f'{dataset_name}_feature_analysis.json')
        import json
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Feature analysis saved to {analysis_path}")
        return analysis
    
    def apply_variance_threshold(self, X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Remove features with low variance"""
        logger.info(f"Applying variance threshold: {threshold}")
        
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        X_result = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        self.transformers['variance_threshold'] = selector
        
        logger.info(f"Variance threshold: {X.shape[1]} -> {X_result.shape[1]} features")
        return X_result
    
    def apply_pca(self, X: pd.DataFrame, n_components: Union[int, float] = 0.95, 
                  dataset_name: str = "dataset") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply Principal Component Analysis"""
        logger.info(f"Applying PCA with {n_components} components")
        
        # Standardize features first
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create DataFrame with component names
        if isinstance(n_components, int):
            n_comp = n_components
        else:
            n_comp = pca.n_components_
        
        component_names = [f'PC{i+1}' for i in range(n_comp)]
        X_pca_df = pd.DataFrame(X_pca, columns=component_names, index=X.index)
        
        # Store transformer
        self.transformers[f'pca_{dataset_name}'] = {
            'pca': pca,
            'scaler': scaler
        }
        
        # Analysis results
        pca_info = {
            'n_components': n_comp,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'total_variance_explained': float(np.sum(pca.explained_variance_ratio_)),
            'components': pca.components_.tolist(),
            'feature_importance': {}
        }
        
        # Feature importance in each component
        for i, component in enumerate(pca.components_):
            top_features_idx = np.argsort(np.abs(component))[-10:][::-1]
            pca_info['feature_importance'][f'PC{i+1}'] = [
                {
                    'feature': X.columns[idx],
                    'weight': float(component[idx])
                } for idx in top_features_idx
            ]
        
        self.feature_info[f'pca_{dataset_name}'] = pca_info
        
        # Generate visualization
        self._visualize_pca(pca_info, dataset_name)
        
        logger.info(f"PCA: {X.shape[1]} -> {n_comp} components, "
                   f"{pca_info['total_variance_explained']:.3f} variance explained")
        
        return X_pca_df, pca_info
    
    def apply_tsne(self, X: pd.DataFrame, n_components: int = 2, 
                   dataset_name: str = "dataset", sample_size: int = 5000) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply t-SNE for visualization"""
        logger.info(f"Applying t-SNE with {n_components} components")
        
        # Sample data if too large (t-SNE is computationally expensive)
        if len(X) > sample_size:
            logger.info(f"Sampling {sample_size} points for t-SNE")
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_idx]
        else:
            X_sample = X.copy()
            sample_idx = X.index
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)
        
        # Apply t-SNE
        tsne = TSNE(
            n_components=n_components,
            random_state=42,
            perplexity=min(30, len(X_sample)-1),
            n_iter=1000,
            learning_rate='auto'
        )
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Create DataFrame
        component_names = [f'tSNE{i+1}' for i in range(n_components)]
        X_tsne_df = pd.DataFrame(X_tsne, columns=component_names, index=sample_idx)
        
        # Store transformer info
        tsne_info = {
            'n_components': n_components,
            'sample_size': len(X_sample),
            'kl_divergence': float(tsne.kl_divergence_),
            'n_iter': int(tsne.n_iter_)
        }
        
        self.feature_info[f'tsne_{dataset_name}'] = tsne_info
        
        logger.info(f"t-SNE: {X.shape[1]} -> {n_components} components, "
                   f"KL divergence: {tsne_info['kl_divergence']:.3f}")
        
        return X_tsne_df, tsne_info
    
    def apply_umap(self, X: pd.DataFrame, n_components: int = 2, 
                   dataset_name: str = "dataset") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply UMAP for dimensionality reduction"""
        logger.info(f"Applying UMAP with {n_components} components")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply UMAP
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean'
        )
        X_umap = reducer.fit_transform(X_scaled)
        
        # Create DataFrame
        component_names = [f'UMAP{i+1}' for i in range(n_components)]
        X_umap_df = pd.DataFrame(X_umap, columns=component_names, index=X.index)
        
        # Store transformer
        self.transformers[f'umap_{dataset_name}'] = {
            'umap': reducer,
            'scaler': scaler
        }
        
        # UMAP info
        umap_info = {
            'n_components': n_components,
            'n_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'euclidean'
        }
        
        self.feature_info[f'umap_{dataset_name}'] = umap_info
        
        logger.info(f"UMAP: {X.shape[1]} -> {n_components} components")
        
        return X_umap_df, umap_info
    
    def select_best_features(self, X: pd.DataFrame, y: pd.Series, k: int = 20, 
                           method: str = 'f_classif') -> Tuple[pd.DataFrame, List[str]]:
        """Select k best features using statistical tests"""
        logger.info(f"Selecting {k} best features using {method}")
        
        # Choose scoring function
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'chi2':
            # Ensure non-negative values for chi2
            X = X - X.min() + 1e-10
            score_func = chi2
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply feature selection
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        X_result = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        # Store transformer
        self.transformers[f'select_k_best_{method}'] = selector
        
        # Get feature scores
        scores = selector.scores_
        feature_scores = list(zip(X.columns, scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Feature selection: {X.shape[1]} -> {k} features")
        logger.info(f"Top 5 features: {[f[0] for f in feature_scores[:5]]}")
        
        return X_result, selected_features
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, 
                                    n_features: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """Apply Recursive Feature Elimination"""
        logger.info(f"Applying RFE to select {n_features} features")
        
        # Use Random Forest as base estimator
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        # Apply RFE
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = rfe.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[rfe.get_support()].tolist()
        X_result = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        # Store transformer
        self.transformers['rfe'] = rfe
        
        # Feature ranking
        feature_ranking = list(zip(X.columns, rfe.ranking_))
        feature_ranking.sort(key=lambda x: x[1])
        
        logger.info(f"RFE: {X.shape[1]} -> {n_features} features")
        logger.info(f"Top 5 features: {[f[0] for f in feature_ranking[:5]]}")
        
        return X_result, selected_features
    
    def _visualize_pca(self, pca_info: Dict[str, Any], dataset_name: str):
        """Create PCA visualization"""
        try:
            # Explained variance plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Individual explained variance
            components = list(range(1, len(pca_info['explained_variance_ratio']) + 1))
            ax1.bar(components, pca_info['explained_variance_ratio'])
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Explained Variance Ratio')
            ax1.set_title(f'PCA Explained Variance - {dataset_name}')
            ax1.grid(True, alpha=0.3)
            
            # Cumulative explained variance
            ax2.plot(components, pca_info['cumulative_variance_ratio'], 'bo-')
            ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
            ax2.set_xlabel('Principal Component')
            ax2.set_ylabel('Cumulative Explained Variance Ratio')
            ax2.set_title(f'PCA Cumulative Variance - {dataset_name}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, f'{dataset_name}_pca_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"PCA visualization saved to {plot_path}")
            
        except Exception as e:
            logger.warning(f"Could not create PCA visualization: {str(e)}")
    
    def create_dimensionality_comparison(self, X: pd.DataFrame, y: pd.Series = None, 
                                       dataset_name: str = "dataset") -> Dict[str, Any]:
        """Compare different dimensionality reduction techniques"""
        logger.info(f"Creating dimensionality reduction comparison for {dataset_name}")
        
        comparison_results = {}
        
        try:
            # Apply different techniques
            logger.info("Applying PCA...")
            X_pca, pca_info = self.apply_pca(X, n_components=2, dataset_name=dataset_name)
            comparison_results['pca'] = {'data': X_pca, 'info': pca_info}
            
            logger.info("Applying t-SNE...")
            X_tsne, tsne_info = self.apply_tsne(X, n_components=2, dataset_name=dataset_name)
            comparison_results['tsne'] = {'data': X_tsne, 'info': tsne_info}
            
            logger.info("Applying UMAP...")
            X_umap, umap_info = self.apply_umap(X, n_components=2, dataset_name=dataset_name)
            comparison_results['umap'] = {'data': X_umap, 'info': umap_info}
            
            # Create comparison visualization if labels available
            if y is not None:
                self._create_comparison_plot(comparison_results, y, dataset_name)
            
            logger.info("Dimensionality reduction comparison completed")
            
        except Exception as e:
            logger.error(f"Error in dimensionality comparison: {str(e)}")
        
        return comparison_results
    
    def _create_comparison_plot(self, results: Dict[str, Any], y: pd.Series, dataset_name: str):
        """Create comparison plot for different dimensionality reduction techniques"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            techniques = ['pca', 'tsne', 'umap']
            titles = ['PCA', 't-SNE', 'UMAP']
            
            for i, (technique, title) in enumerate(zip(techniques, titles)):
                if technique in results:
                    data = results[technique]['data']
                    
                    # Create scatter plot
                    scatter = axes[i].scatter(
                        data.iloc[:, 0], data.iloc[:, 1], 
                        c=y, cmap='viridis', alpha=0.6, s=1
                    )
                    axes[i].set_xlabel(data.columns[0])
                    axes[i].set_ylabel(data.columns[1])
                    axes[i].set_title(f'{title} - {dataset_name}')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add colorbar
                    plt.colorbar(scatter, ax=axes[i])
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, f'{dataset_name}_dimensionality_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Comparison plot saved to {plot_path}")
            
        except Exception as e:
            logger.warning(f"Could not create comparison plot: {str(e)}")
    
    def save_transformers(self, dataset_name: str):
        """Save fitted transformers for later use"""
        if not self.config.get('feature_engineering', {}).get('save_transformers', True):
            return
        
        transformers_path = os.path.join(self.output_dir, f'{dataset_name}_transformers.pkl')
        info_path = os.path.join(self.output_dir, f'{dataset_name}_feature_info.pkl')
        
        try:
            joblib.dump(self.transformers, transformers_path)
            joblib.dump(self.feature_info, info_path)
            
            logger.info(f"Transformers saved to {transformers_path}")
            logger.info(f"Feature info saved to {info_path}")
            
        except Exception as e:
            logger.error(f"Error saving transformers: {str(e)}")
    
    def load_transformers(self, dataset_name: str):
        """Load previously fitted transformers"""
        transformers_path = os.path.join(self.output_dir, f'{dataset_name}_transformers.pkl')
        info_path = os.path.join(self.output_dir, f'{dataset_name}_feature_info.pkl')
        
        try:
            if os.path.exists(transformers_path):
                self.transformers = joblib.load(transformers_path)
                logger.info(f"Transformers loaded from {transformers_path}")
            
            if os.path.exists(info_path):
                self.feature_info = joblib.load(info_path)
                logger.info(f"Feature info loaded from {info_path}")
                
        except Exception as e:
            logger.error(f"Error loading transformers: {str(e)}")
    
    def generate_feature_report(self, dataset_name: str) -> str:
        """Generate comprehensive feature engineering report"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append(f"FEATURE ENGINEERING REPORT - {dataset_name.upper()}")
        report_lines.append("="*80)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # PCA Analysis
        if f'pca_{dataset_name}' in self.feature_info:
            pca_info = self.feature_info[f'pca_{dataset_name}']
            report_lines.append("PRINCIPAL COMPONENT ANALYSIS (PCA)")
            report_lines.append("-" * 50)
            report_lines.append(f"Number of components: {pca_info['n_components']}")
            report_lines.append(f"Total variance explained: {pca_info['total_variance_explained']:.4f}")
            report_lines.append(f"Top 3 components variance: {pca_info['explained_variance_ratio'][:3]}")
            report_lines.append("")
        
        # t-SNE Analysis
        if f'tsne_{dataset_name}' in self.feature_info:
            tsne_info = self.feature_info[f'tsne_{dataset_name}']
            report_lines.append("T-DISTRIBUTED STOCHASTIC NEIGHBOR EMBEDDING (t-SNE)")
            report_lines.append("-" * 50)
            report_lines.append(f"Number of components: {tsne_info['n_components']}")
            report_lines.append(f"Sample size used: {tsne_info['sample_size']}")
            report_lines.append(f"KL divergence: {tsne_info['kl_divergence']:.4f}")
            report_lines.append("")
        
        # UMAP Analysis
        if f'umap_{dataset_name}' in self.feature_info:
            umap_info = self.feature_info[f'umap_{dataset_name}']
            report_lines.append("UNIFORM MANIFOLD APPROXIMATION AND PROJECTION (UMAP)")
            report_lines.append("-" * 50)
            report_lines.append(f"Number of components: {umap_info['n_components']}")
            report_lines.append(f"Number of neighbors: {umap_info['n_neighbors']}")
            report_lines.append(f"Minimum distance: {umap_info['min_dist']}")
            report_lines.append("")
        
        # Save report
        report_path = os.path.join(self.output_dir, f'{dataset_name}_feature_engineering_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Feature engineering report saved: {report_path}")
        return report_path 