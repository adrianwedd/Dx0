#!/usr/bin/env python3
"""
Feature Importance Analysis and ML Model Explainability Tools
Tools for understanding and interpreting ML model predictions and feature contributions
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns

import structlog

logger = structlog.get_logger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class FeatureImportanceResult:
    """Results from feature importance analysis."""
    
    feature_name: str
    importance_score: float
    importance_rank: int
    importance_type: str  # 'builtin', 'permutation', 'shap', 'mutual_info'
    confidence_interval: Optional[Tuple[float, float]] = None
    stability_score: Optional[float] = None  # 0-1, higher means more stable
    interpretation: Optional[str] = None


@dataclass
class ModelExplanation:
    """Explanation of model predictions."""
    
    model_name: str
    feature_importances: List[FeatureImportanceResult]
    model_performance: Dict[str, float]
    feature_interactions: Optional[Dict[str, float]] = None
    global_explanation: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class PredictionExplanation:
    """Explanation for individual predictions."""
    
    sample_id: str
    prediction: float
    actual_value: Optional[float]
    feature_contributions: Dict[str, float]
    top_contributing_features: List[Tuple[str, float]]
    confidence_score: Optional[float] = None
    explanation_text: Optional[str] = None


class FeatureImportanceAnalyzer:
    """Analyze feature importance using multiple methods."""
    
    def __init__(self, model: Optional[BaseEstimator] = None):
        self.model = model or RandomForestRegressor(n_estimators=100, random_state=42)
        self.feature_names_ = []
        self.importance_results_ = []
        self.fitted_ = False
        
    def analyze_importance(self, X: np.ndarray, y: np.ndarray,
                         feature_names: Optional[List[str]] = None,
                         methods: List[str] = ['builtin', 'permutation']) -> List[FeatureImportanceResult]:
        """Analyze feature importance using multiple methods."""
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        self.feature_names_ = feature_names
        
        # Fit model if not already fitted
        if not hasattr(self.model, 'feature_importances_') or not self.fitted_:
            self.model.fit(X, y)
            self.fitted_ = True
            
        importance_results = []
        
        # Built-in feature importance (for tree-based models)
        if 'builtin' in methods and hasattr(self.model, 'feature_importances_'):
            builtin_importance = self._analyze_builtin_importance(X, y)
            importance_results.extend(builtin_importance)
            
        # Permutation importance
        if 'permutation' in methods:
            perm_importance = self._analyze_permutation_importance(X, y)
            importance_results.extend(perm_importance)
            
        # Mutual information importance
        if 'mutual_info' in methods:
            mi_importance = self._analyze_mutual_info_importance(X, y)
            importance_results.extend(mi_importance)
            
        # Statistical correlation importance
        if 'correlation' in methods:
            corr_importance = self._analyze_correlation_importance(X, y)
            importance_results.extend(corr_importance)
            
        self.importance_results_ = importance_results
        
        logger.info(f"Feature importance analysis complete: {len(importance_results)} results")
        
        return importance_results
        
    def _analyze_builtin_importance(self, X: np.ndarray, y: np.ndarray) -> List[FeatureImportanceResult]:
        """Analyze built-in feature importance."""
        
        importances = self.model.feature_importances_
        
        results = []
        for i, (name, importance) in enumerate(zip(self.feature_names_, importances)):
            results.append(FeatureImportanceResult(
                feature_name=name,
                importance_score=float(importance),
                importance_rank=0,  # Will be set later
                importance_type='builtin',
                interpretation=self._interpret_builtin_importance(name, importance)
            ))
            
        # Sort by importance and assign ranks
        results.sort(key=lambda x: x.importance_score, reverse=True)
        for i, result in enumerate(results):
            result.importance_rank = i + 1
            
        return results
        
    def _analyze_permutation_importance(self, X: np.ndarray, y: np.ndarray) -> List[FeatureImportanceResult]:
        """Analyze permutation importance."""
        
        # Use train/test split for more robust permutation importance
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Fit model on training data
        temp_model = type(self.model)(**self.model.get_params())
        temp_model.fit(X_train, y_train)
        
        # Calculate permutation importance on test data
        perm_importance = permutation_importance(
            temp_model, X_test, y_test, 
            n_repeats=10, random_state=42, scoring='neg_mean_absolute_error'
        )
        
        results = []
        for i, name in enumerate(self.feature_names_):
            importance = perm_importance.importances_mean[i]
            std = perm_importance.importances_std[i]
            
            # Calculate confidence interval (95%)
            confidence_interval = (
                importance - 1.96 * std,
                importance + 1.96 * std
            )
            
            # Calculate stability score (inverse of coefficient of variation)
            stability = 1 / (1 + std / (abs(importance) + 1e-8))
            
            results.append(FeatureImportanceResult(
                feature_name=name,
                importance_score=float(importance),
                importance_rank=0,  # Will be set later
                importance_type='permutation',
                confidence_interval=confidence_interval,
                stability_score=float(stability),
                interpretation=self._interpret_permutation_importance(name, importance, std)
            ))
            
        # Sort by importance and assign ranks
        results.sort(key=lambda x: x.importance_score, reverse=True)
        for i, result in enumerate(results):
            result.importance_rank = i + 1
            
        return results
        
    def _analyze_mutual_info_importance(self, X: np.ndarray, y: np.ndarray) -> List[FeatureImportanceResult]:
        """Analyze mutual information importance."""
        
        from sklearn.feature_selection import mutual_info_regression
        
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        results = []
        for i, (name, score) in enumerate(zip(self.feature_names_, mi_scores)):
            results.append(FeatureImportanceResult(
                feature_name=name,
                importance_score=float(score),
                importance_rank=0,  # Will be set later
                importance_type='mutual_info',
                interpretation=self._interpret_mutual_info_importance(name, score)
            ))
            
        # Sort by importance and assign ranks
        results.sort(key=lambda x: x.importance_score, reverse=True)
        for i, result in enumerate(results):
            result.importance_rank = i + 1
            
        return results
        
    def _analyze_correlation_importance(self, X: np.ndarray, y: np.ndarray) -> List[FeatureImportanceResult]:
        """Analyze correlation-based importance."""
        
        correlations = []
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            if len(np.unique(feature_data)) > 1:  # Skip constant features
                corr = np.corrcoef(feature_data, y)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
                else:
                    correlations.append(0)
            else:
                correlations.append(0)
                
        results = []
        for i, (name, corr) in enumerate(zip(self.feature_names_, correlations)):
            results.append(FeatureImportanceResult(
                feature_name=name,
                importance_score=float(corr),
                importance_rank=0,  # Will be set later
                importance_type='correlation',
                interpretation=self._interpret_correlation_importance(name, corr)
            ))
            
        # Sort by importance and assign ranks
        results.sort(key=lambda x: x.importance_score, reverse=True)
        for i, result in enumerate(results):
            result.importance_rank = i + 1
            
        return results
        
    def _interpret_builtin_importance(self, feature_name: str, importance: float) -> str:
        """Interpret built-in importance score."""
        
        if importance > 0.1:
            level = "very high"
        elif importance > 0.05:
            level = "high"
        elif importance > 0.02:
            level = "moderate"
        elif importance > 0.01:
            level = "low"
        else:
            level = "very low"
            
        return f"Feature '{feature_name}' has {level} importance ({importance:.4f}) for model predictions"
        
    def _interpret_permutation_importance(self, feature_name: str, importance: float, std: float) -> str:
        """Interpret permutation importance score."""
        
        if importance > 0.1:
            level = "critical"
        elif importance > 0.05:
            level = "very important"
        elif importance > 0.02:
            level = "important"
        elif importance > 0.01:
            level = "moderately important"
        else:
            level = "minimally important"
            
        stability = "stable" if std < importance * 0.5 else "unstable"
        
        return f"Feature '{feature_name}' is {level} ({importance:.4f} ± {std:.4f}, {stability})"
        
    def _interpret_mutual_info_importance(self, feature_name: str, score: float) -> str:
        """Interpret mutual information score."""
        
        if score > 0.5:
            level = "very high"
        elif score > 0.3:
            level = "high"
        elif score > 0.1:
            level = "moderate"
        elif score > 0.05:
            level = "low"
        else:
            level = "very low"
            
        return f"Feature '{feature_name}' has {level} mutual information with target ({score:.4f})"
        
    def _interpret_correlation_importance(self, feature_name: str, correlation: float) -> str:
        """Interpret correlation importance."""
        
        if correlation > 0.7:
            level = "very strong"
        elif correlation > 0.5:
            level = "strong"
        elif correlation > 0.3:
            level = "moderate"
        elif correlation > 0.1:
            level = "weak"
        else:
            level = "very weak"
            
        return f"Feature '{feature_name}' has {level} correlation with target ({correlation:.4f})"
        
    def get_top_features(self, n: int = 10, importance_type: str = 'builtin') -> List[FeatureImportanceResult]:
        """Get top N most important features."""
        
        filtered_results = [r for r in self.importance_results_ if r.importance_type == importance_type]
        filtered_results.sort(key=lambda x: x.importance_score, reverse=True)
        
        return filtered_results[:n]
        
    def compare_importance_methods(self) -> pd.DataFrame:
        """Compare feature importance across different methods."""
        
        if not self.importance_results_:
            return pd.DataFrame()
            
        # Group results by feature name
        feature_dict = {}
        for result in self.importance_results_:
            if result.feature_name not in feature_dict:
                feature_dict[result.feature_name] = {}
            feature_dict[result.feature_name][result.importance_type] = result.importance_score
            
        # Convert to DataFrame
        comparison_df = pd.DataFrame.from_dict(feature_dict, orient='index')
        comparison_df = comparison_df.fillna(0)
        
        # Add rank columns
        for col in comparison_df.columns:
            rank_col = f"{col}_rank"
            comparison_df[rank_col] = comparison_df[col].rank(ascending=False, method='min')
            
        return comparison_df


class ModelExplainer:
    """Comprehensive model explanation and interpretation."""
    
    def __init__(self, model: BaseEstimator, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.importance_analyzer = FeatureImportanceAnalyzer(model)
        
    def explain_model(self, X: np.ndarray, y: np.ndarray,
                     importance_methods: List[str] = ['builtin', 'permutation']) -> ModelExplanation:
        """Generate comprehensive model explanation."""
        
        logger.info("Generating comprehensive model explanation")
        
        # Analyze feature importance
        importance_results = self.importance_analyzer.analyze_importance(
            X, y, self.feature_names, importance_methods
        )
        
        # Calculate model performance
        predictions = self.model.predict(X)
        performance = {
            'mae': mean_absolute_error(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions),
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions)
        }
        
        # Generate global explanation
        global_explanation = self._generate_global_explanation(importance_results, performance)
        
        # Analyze feature interactions (simplified)
        feature_interactions = self._analyze_feature_interactions(X, y)
        
        explanation = ModelExplanation(
            model_name=type(self.model).__name__,
            feature_importances=importance_results,
            model_performance=performance,
            feature_interactions=feature_interactions,
            global_explanation=global_explanation
        )
        
        logger.info("Model explanation complete")
        
        return explanation
        
    def explain_prediction(self, X_sample: np.ndarray, 
                         sample_id: str = "sample",
                         actual_value: Optional[float] = None) -> PredictionExplanation:
        """Explain individual prediction."""
        
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
            
        prediction = self.model.predict(X_sample)[0]
        
        # Calculate feature contributions (simplified approach)
        feature_contributions = self._calculate_feature_contributions(X_sample[0])
        
        # Get top contributing features
        top_features = sorted(
            feature_contributions.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:5]
        
        # Generate explanation text
        explanation_text = self._generate_prediction_explanation(
            prediction, actual_value, top_features
        )
        
        return PredictionExplanation(
            sample_id=sample_id,
            prediction=prediction,
            actual_value=actual_value,
            feature_contributions=feature_contributions,
            top_contributing_features=top_features,
            explanation_text=explanation_text
        )
        
    def _generate_global_explanation(self, importance_results: List[FeatureImportanceResult],
                                   performance: Dict[str, float]) -> str:
        """Generate global model explanation."""
        
        # Get top features by different methods
        builtin_top = [r for r in importance_results if r.importance_type == 'builtin'][:5]
        perm_top = [r for r in importance_results if r.importance_type == 'permutation'][:5]
        
        explanation = f"""
Model Performance Summary:
- R² Score: {performance['r2']:.3f} (explains {performance['r2']*100:.1f}% of variance)
- Mean Absolute Error: {performance['mae']:.3f}
- Root Mean Square Error: {performance['rmse']:.3f}

Key Findings:
"""
        
        if builtin_top:
            explanation += f"""
Most Important Features (Built-in Importance):
"""
            for i, result in enumerate(builtin_top, 1):
                explanation += f"{i}. {result.feature_name}: {result.importance_score:.4f}\n"
                
        if perm_top:
            explanation += f"""
Most Important Features (Permutation Importance):
"""
            for i, result in enumerate(perm_top, 1):
                explanation += f"{i}. {result.feature_name}: {result.importance_score:.4f}\n"
                
        # Model insights
        if performance['r2'] > 0.8:
            explanation += "\nModel shows excellent predictive performance."
        elif performance['r2'] > 0.6:
            explanation += "\nModel shows good predictive performance."
        elif performance['r2'] > 0.4:
            explanation += "\nModel shows moderate predictive performance."
        else:
            explanation += "\nModel shows limited predictive performance - consider feature engineering or different algorithms."
            
        return explanation.strip()
        
    def _analyze_feature_interactions(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Analyze feature interactions (simplified)."""
        
        interactions = {}
        
        # For tree-based models, we can analyze feature interactions through tree structure
        if hasattr(self.model, 'estimators_'):
            # Random Forest - analyze feature co-occurrence in splits
            feature_cooccurrence = np.zeros((len(self.feature_names), len(self.feature_names)))
            
            for estimator in self.model.estimators_[:10]:  # Sample first 10 trees
                tree = estimator.tree_
                for node in range(tree.node_count):
                    if tree.children_left[node] != tree.children_right[node]:  # Not a leaf
                        feature_idx = tree.feature[node]
                        # Find parent node feature
                        for parent in range(tree.node_count):
                            if (tree.children_left[parent] == node or 
                                tree.children_right[parent] == node):
                                if tree.feature[parent] != feature_idx:
                                    feature_cooccurrence[feature_idx][tree.feature[parent]] += 1
                                    
            # Convert to feature name pairs
            for i in range(len(self.feature_names)):
                for j in range(i+1, len(self.feature_names)):
                    interaction_strength = feature_cooccurrence[i][j] + feature_cooccurrence[j][i]
                    if interaction_strength > 0:
                        pair_name = f"{self.feature_names[i]} × {self.feature_names[j]}"
                        interactions[pair_name] = float(interaction_strength)
                        
        return interactions
        
    def _calculate_feature_contributions(self, x_sample: np.ndarray) -> Dict[str, float]:
        """Calculate feature contributions for a single prediction."""
        
        contributions = {}
        
        # For tree-based models, use decision path
        if hasattr(self.model, 'decision_path'):
            try:
                decision_path = self.model.decision_path(x_sample.reshape(1, -1))
                
                # Simplified contribution calculation
                base_prediction = np.mean(self.model.predict(np.zeros((1, len(x_sample)))))
                
                for i, feature_name in enumerate(self.feature_names):
                    # Create modified sample with feature zeroed out
                    modified_sample = x_sample.copy()
                    modified_sample[i] = 0
                    
                    modified_prediction = self.model.predict(modified_sample.reshape(1, -1))[0]
                    original_prediction = self.model.predict(x_sample.reshape(1, -1))[0]
                    
                    contribution = original_prediction - modified_prediction
                    contributions[feature_name] = contribution
                    
            except Exception:
                # Fallback: use feature importance as proxy
                if hasattr(self.model, 'feature_importances_'):
                    for i, feature_name in enumerate(self.feature_names):
                        importance = self.model.feature_importances_[i]
                        feature_value = x_sample[i]
                        contributions[feature_name] = importance * feature_value
                        
        else:
            # For linear models, contributions are weights * feature values
            if hasattr(self.model, 'coef_'):
                for i, feature_name in enumerate(self.feature_names):
                    contribution = self.model.coef_[i] * x_sample[i]
                    contributions[feature_name] = contribution
                    
        return contributions
        
    def _generate_prediction_explanation(self, prediction: float,
                                       actual_value: Optional[float],
                                       top_features: List[Tuple[str, float]]) -> str:
        """Generate explanation text for individual prediction."""
        
        explanation = f"Predicted value: ${prediction:.2f}"
        
        if actual_value is not None:
            error = abs(prediction - actual_value)
            error_pct = error / actual_value * 100 if actual_value != 0 else 0
            explanation += f"\nActual value: ${actual_value:.2f}"
            explanation += f"\nAbsolute error: ${error:.2f} ({error_pct:.1f}%)"
            
        explanation += f"\n\nTop contributing factors:"
        
        for i, (feature, contribution) in enumerate(top_features, 1):
            direction = "increases" if contribution > 0 else "decreases"
            explanation += f"\n{i}. {feature}: {direction} prediction by ${abs(contribution):.2f}"
            
        return explanation


def create_feature_importance_dashboard(explanation: ModelExplanation,
                                      save_path: Optional[str] = None) -> plt.Figure:
    """Create a dashboard visualizing feature importance."""
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Feature Importance Dashboard - {explanation.model_name}', fontsize=16)
    
    # Extract data for plotting
    builtin_features = [f for f in explanation.feature_importances if f.importance_type == 'builtin']
    perm_features = [f for f in explanation.feature_importances if f.importance_type == 'permutation']
    
    # Plot 1: Built-in Feature Importance
    if builtin_features:
        top_builtin = sorted(builtin_features, key=lambda x: x.importance_score, reverse=True)[:15]
        names = [f.feature_name for f in top_builtin]
        scores = [f.importance_score for f in top_builtin]
        
        axes[0, 0].barh(range(len(names)), scores)
        axes[0, 0].set_yticks(range(len(names)))
        axes[0, 0].set_yticklabels(names)
        axes[0, 0].set_xlabel('Importance Score')
        axes[0, 0].set_title('Built-in Feature Importance (Top 15)')
        axes[0, 0].invert_yaxis()
        
    # Plot 2: Permutation Feature Importance
    if perm_features:
        top_perm = sorted(perm_features, key=lambda x: x.importance_score, reverse=True)[:15]
        names = [f.feature_name for f in top_perm]
        scores = [f.importance_score for f in top_perm]
        errors = [f.confidence_interval[1] - f.importance_score if f.confidence_interval else 0 
                 for f in top_perm]
        
        axes[0, 1].barh(range(len(names)), scores, xerr=errors)
        axes[0, 1].set_yticks(range(len(names)))
        axes[0, 1].set_yticklabels(names)
        axes[0, 1].set_xlabel('Importance Score')
        axes[0, 1].set_title('Permutation Feature Importance (Top 15)')
        axes[0, 1].invert_yaxis()
        
    # Plot 3: Model Performance Metrics
    performance = explanation.model_performance
    metrics = ['R²', 'MAE', 'RMSE']
    values = [performance['r2'], performance['mae'], performance['rmse']]
    
    bars = axes[1, 0].bar(metrics, values)
    axes[1, 0].set_title('Model Performance Metrics')
    axes[1, 0].set_ylabel('Score')
    
    # Color bars based on performance
    for i, (bar, value) in enumerate(zip(bars, values)):
        if i == 0:  # R²
            color = 'green' if value > 0.7 else 'orange' if value > 0.5 else 'red'
        else:  # MAE, RMSE (lower is better)
            color = 'green' if value < 50 else 'orange' if value < 100 else 'red'
        bar.set_color(color)
        
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
    # Plot 4: Feature Importance Comparison (if multiple methods available)
    importance_types = list(set(f.importance_type for f in explanation.feature_importances))
    
    if len(importance_types) > 1:
        # Create comparison of top features across methods
        feature_comparison = {}
        
        for imp_type in importance_types:
            features = [f for f in explanation.feature_importances if f.importance_type == imp_type]
            top_features = sorted(features, key=lambda x: x.importance_score, reverse=True)[:10]
            
            for f in top_features:
                if f.feature_name not in feature_comparison:
                    feature_comparison[f.feature_name] = {}
                feature_comparison[f.feature_name][imp_type] = f.importance_score
                
        # Convert to DataFrame for plotting
        comp_df = pd.DataFrame.from_dict(feature_comparison, orient='index').fillna(0)
        
        # Plot heatmap
        if not comp_df.empty:
            sns.heatmap(comp_df.T, annot=True, fmt='.3f', cmap='viridis', ax=axes[1, 1])
            axes[1, 1].set_title('Feature Importance Comparison')
            axes[1, 1].set_xlabel('Features')
            axes[1, 1].set_ylabel('Importance Method')
    else:
        # Just show a text summary
        axes[1, 1].text(0.5, 0.5, f'Single importance method used:\n{importance_types[0]}', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Importance Method Summary')
        axes[1, 1].axis('off')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance dashboard saved to {save_path}")
        
    return fig


def demonstrate_explainability_tools():
    """Demonstrate the explainability tools."""
    
    print("=== ML Model Explainability Tools Demonstration ===\n")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    
    # Generate synthetic features based on CPT cost estimation
    cpt_codes = np.random.randint(10000, 99999, n_samples)
    price_base = np.random.uniform(5, 500, n_samples)
    complexity_scores = np.random.uniform(0, 1, n_samples)
    word_counts = np.random.randint(1, 8, n_samples)
    is_imaging = np.random.binomial(1, 0.3, n_samples)
    price_tiers = np.random.randint(1, 8, n_samples)
    
    # Create feature matrix
    X = np.column_stack([
        cpt_codes / 100000,  # Normalized CPT codes
        price_base / 500,    # Normalized base prices
        complexity_scores,
        word_counts / 8,
        is_imaging,
        price_tiers / 8
    ])
    
    feature_names = [
        'cpt_code_normalized',
        'price_base_normalized', 
        'complexity_score',
        'word_count_normalized',
        'is_imaging',
        'price_tier_normalized'
    ]
    
    # Create target with realistic relationships
    y = (
        price_base * (1 + complexity_scores * 0.5) +  # Base price adjusted by complexity
        is_imaging * 50 +                             # Imaging adds cost
        word_counts * 5 +                             # More complex names cost more
        np.random.normal(0, 20, n_samples)            # Add noise
    )
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target range: ${y.min():.2f} - ${y.max():.2f}")
    
    # Train model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create explainer
    explainer = ModelExplainer(model, feature_names)
    
    # Explain the model
    print(f"\n=== Generating Model Explanation ===")
    explanation = explainer.explain_model(X, y, ['builtin', 'permutation', 'correlation'])
    
    # Display results
    print(f"\nModel: {explanation.model_name}")
    print(f"Performance Metrics:")
    for metric, value in explanation.model_performance.items():
        print(f"  {metric.upper()}: {value:.3f}")
        
    print(f"\nTop 5 Most Important Features (Built-in):")
    builtin_features = [f for f in explanation.feature_importances if f.importance_type == 'builtin']
    top_builtin = sorted(builtin_features, key=lambda x: x.importance_score, reverse=True)[:5]
    
    for i, feature in enumerate(top_builtin, 1):
        print(f"  {i}. {feature.feature_name}: {feature.importance_score:.4f}")
        
    print(f"\nTop 5 Most Important Features (Permutation):")
    perm_features = [f for f in explanation.feature_importances if f.importance_type == 'permutation']
    top_perm = sorted(perm_features, key=lambda x: x.importance_score, reverse=True)[:5]
    
    for i, feature in enumerate(top_perm, 1):
        ci_text = f" (±{feature.confidence_interval[1] - feature.importance_score:.4f})" if feature.confidence_interval else ""
        print(f"  {i}. {feature.feature_name}: {feature.importance_score:.4f}{ci_text}")
        
    # Explain individual predictions
    print(f"\n=== Individual Prediction Explanations ===")
    
    # Select interesting samples
    sample_indices = [
        np.argmax(y),      # Most expensive
        np.argmin(y),      # Least expensive  
        len(y) // 2        # Median
    ]
    
    for i, idx in enumerate(sample_indices):
        sample_description = ["Most Expensive", "Least Expensive", "Median Price"][i]
        
        pred_explanation = explainer.explain_prediction(
            X[idx], 
            sample_id=f"sample_{idx}",
            actual_value=y[idx]
        )
        
        print(f"\n{sample_description} Sample:")
        print(f"  Prediction: ${pred_explanation.prediction:.2f}")
        print(f"  Actual: ${pred_explanation.actual_value:.2f}")
        print(f"  Top Contributing Features:")
        
        for j, (feature, contribution) in enumerate(pred_explanation.top_contributing_features[:3], 1):
            direction = "↑" if contribution > 0 else "↓"
            print(f"    {j}. {feature}: {direction} ${abs(contribution):.2f}")
            
    # Feature importance comparison
    print(f"\n=== Feature Importance Method Comparison ===")
    analyzer = FeatureImportanceAnalyzer(model)
    analyzer.importance_results_ = explanation.feature_importances
    
    comparison_df = analyzer.compare_importance_methods()
    if not comparison_df.empty:
        print(f"Feature importance correlation between methods:")
        correlation_methods = [col for col in comparison_df.columns if not col.endswith('_rank')]
        if len(correlation_methods) > 1:
            corr_matrix = comparison_df[correlation_methods].corr()
            print(corr_matrix)
            
    print(f"\n=== Global Model Explanation ===")
    print(explanation.global_explanation)
    
    print(f"\n=== Explainability Tools Ready for Production ===")


if __name__ == "__main__":
    demonstrate_explainability_tools()