#!/usr/bin/env python3
"""
ML-Enhanced Cost Estimator Integration
Integrates machine learning feature engineering with the existing cost estimation system
"""

import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import structlog

from .cost_estimator import CostEstimator, CptCost
from .ml_preprocessing import MLPreprocessingPipeline, create_default_preprocessing_pipeline
from .ml_feature_engineering import FeatureConfig
from .ml_feature_store import SQLiteFeatureStore, FeatureCache
from .ml_explainability import ModelExplainer, ModelExplanation
from .ml_validation import run_comprehensive_validation

logger = structlog.get_logger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class MLModelConfig:
    """Configuration for ML cost estimation models."""
    
    model_type: str = "random_forest"  # "random_forest", "gradient_boosting", "linear", "ridge"
    model_params: Dict[str, Any] = None
    
    # Training configuration
    test_size: float = 0.2
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    random_state: int = 42
    
    # Feature engineering configuration
    feature_config: Optional[FeatureConfig] = None
    
    # Model selection and validation
    enable_model_selection: bool = True
    performance_threshold: float = 0.6  # Minimum R² score
    
    def __post_init__(self):
        if self.model_params is None:
            if self.model_type == "random_forest":
                self.model_params = {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": self.random_state
                }
            elif self.model_type == "gradient_boosting":
                self.model_params = {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": self.random_state
                }
            elif self.model_type == "linear":
                self.model_params = {}
            elif self.model_type == "ridge":
                self.model_params = {"alpha": 1.0}
            else:
                self.model_params = {}
                
        if self.feature_config is None:
            self.feature_config = FeatureConfig()


@dataclass
class MLModelPerformance:
    """Performance metrics for ML cost estimation model."""
    
    model_name: str
    r2_score: float
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_absolute_percentage_error: float
    cross_validation_scores: List[float]
    prediction_confidence: float
    training_samples: int
    feature_count: int
    training_time: float
    
    @property
    def performance_grade(self) -> str:
        """Get performance grade based on R² score."""
        if self.r2_score >= 0.9:
            return "Excellent"
        elif self.r2_score >= 0.8:
            return "Very Good"
        elif self.r2_score >= 0.7:
            return "Good"
        elif self.r2_score >= 0.6:
            return "Fair"
        elif self.r2_score >= 0.4:
            return "Poor"
        else:
            return "Very Poor"


class MLCostEstimator(CostEstimator):
    """ML-enhanced cost estimator that integrates with the existing system."""
    
    def __init__(self, 
                 cost_table: Dict[str, CptCost],
                 ml_config: Optional[MLModelConfig] = None,
                 model_path: Optional[str] = None,
                 enable_ml_fallback: bool = True,
                 confidence_threshold: float = 0.7):
        
        super().__init__(cost_table)
        
        self.ml_config = ml_config or MLModelConfig()
        self.model_path = model_path
        self.enable_ml_fallback = enable_ml_fallback
        self.confidence_threshold = confidence_threshold
        
        # ML components
        self.ml_model: Optional[BaseEstimator] = None
        self.preprocessing_pipeline: Optional[MLPreprocessingPipeline] = None
        self.feature_store = SQLiteFeatureStore()
        self.feature_cache = FeatureCache()
        self.model_explainer: Optional[ModelExplainer] = None
        
        # Performance tracking
        self.model_performance: Optional[MLModelPerformance] = None
        self.prediction_cache: Dict[str, Tuple[float, float]] = {}  # test_name -> (prediction, confidence)
        
        # Load pre-trained model if available
        if self.model_path and os.path.exists(self.model_path):
            self.load_ml_model(self.model_path)
            
    def train_ml_model(self, training_data: pd.DataFrame, 
                      target_column: str = 'price',
                      validation_data: Optional[pd.DataFrame] = None) -> MLModelPerformance:
        """Train ML model for cost estimation."""
        
        logger.info("Starting ML model training for cost estimation")
        
        # Validate training data
        if training_data.empty:
            raise ValueError("Training data cannot be empty")
            
        required_columns = ['test_name', 'cpt_code', 'price']
        missing_columns = [col for col in required_columns if col not in training_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in training data: {missing_columns}")
            
        # Prepare features and target
        X = training_data[['test_name', 'cpt_code', 'price']].copy()
        y = training_data[target_column].copy()
        
        # Split data for training and testing
        if validation_data is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.ml_config.test_size,
                random_state=self.ml_config.random_state
            )
        else:
            X_train, y_train = X, y
            X_test = validation_data[['test_name', 'cpt_code', 'price']].copy()
            y_test = validation_data[target_column].copy()
            
        # Initialize preprocessing pipeline
        self.preprocessing_pipeline = create_default_preprocessing_pipeline()
        
        # Fit preprocessing pipeline
        start_time = datetime.now()
        X_train_processed = self.preprocessing_pipeline.fit_transform(X_train, y_train)
        X_test_processed = self.preprocessing_pipeline.transform(X_test)
        
        # Initialize ML model
        self.ml_model = self._create_ml_model()
        
        # Train model
        self.ml_model.fit(X_train_processed, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate model
        y_pred_train = self.ml_model.predict(X_train_processed)
        y_pred_test = self.ml_model.predict(X_test_processed)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(
            y_test, y_pred_test, X_train_processed, y_train, training_time
        )
        
        self.model_performance = performance
        
        # Initialize model explainer
        if hasattr(self.preprocessing_pipeline.feature_engineer, 'get_feature_names'):
            feature_names = self.preprocessing_pipeline.feature_engineer.get_feature_names()
        else:
            feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
            
        self.model_explainer = ModelExplainer(self.ml_model, feature_names)
        
        logger.info(f"ML model training complete. Performance: R²={performance.r2_score:.3f}, MAE={performance.mean_absolute_error:.2f}")
        
        return performance
        
    def _create_ml_model(self) -> BaseEstimator:
        """Create ML model based on configuration."""
        
        model_type = self.ml_config.model_type
        params = self.ml_config.model_params
        
        if model_type == "random_forest":
            return RandomForestRegressor(**params)
        elif model_type == "gradient_boosting":
            return GradientBoostingRegressor(**params)
        elif model_type == "linear":
            return LinearRegression(**params)
        elif model_type == "ridge":
            return Ridge(**params)
        else:
            logger.warning(f"Unknown model type {model_type}, using RandomForestRegressor")
            return RandomForestRegressor(n_estimators=100, random_state=42)
            
    def _calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     X_train: np.ndarray, y_train: np.ndarray,
                                     training_time: float) -> MLModelPerformance:
        """Calculate comprehensive performance metrics."""
        
        # Basic metrics
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.ml_model, X_train, y_train,
            cv=self.ml_config.cross_validation_folds,
            scoring='r2'
        )
        
        # Prediction confidence (based on prediction interval)
        prediction_errors = np.abs(y_true - y_pred)
        confidence = 1 - (np.std(prediction_errors) / np.mean(y_true))
        confidence = max(0, min(1, confidence))
        
        return MLModelPerformance(
            model_name=f"{self.ml_config.model_type}_{type(self.ml_model).__name__}",
            r2_score=r2,
            mean_absolute_error=mae,
            root_mean_squared_error=rmse,
            mean_absolute_percentage_error=mape,
            cross_validation_scores=cv_scores.tolist(),
            prediction_confidence=confidence,
            training_samples=len(X_train),
            feature_count=X_train.shape[1],
            training_time=training_time
        )
        
    def predict_ml_cost(self, test_name: str, cpt_code: Optional[str] = None) -> Tuple[float, float]:
        """Predict cost using ML model with confidence score."""
        
        if not self.ml_model or not self.preprocessing_pipeline:
            raise ValueError("ML model not trained. Call train_ml_model() first.")
            
        # Check cache first
        cache_key = f"{test_name}_{cpt_code or 'None'}"
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
            
        # Prepare input data
        input_data = pd.DataFrame({
            'test_name': [test_name],
            'cpt_code': [cpt_code or '00000'],
            'price': [0.0]  # Placeholder, will be ignored
        })
        
        try:
            # Preprocess input
            X_processed = self.preprocessing_pipeline.transform(input_data)
            
            # Make prediction
            prediction = self.ml_model.predict(X_processed)[0]
            
            # Calculate confidence (simplified approach)
            confidence = self.model_performance.prediction_confidence if self.model_performance else 0.5
            
            # Adjust confidence based on feature coverage
            if hasattr(self.preprocessing_pipeline, 'get_preprocessing_stats'):
                stats = self.preprocessing_pipeline.get_preprocessing_stats()
                if stats.get('final_samples', 1) == 0:
                    confidence *= 0.5  # Lower confidence for edge cases
                    
            # Cache result
            result = (float(prediction), float(confidence))
            self.prediction_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.warning(f"ML prediction failed for {test_name}: {e}")
            # Return fallback prediction with low confidence
            fallback_price = np.mean(list(self.cost_table.values())) if self.cost_table else 50.0
            return (fallback_price, 0.1)
            
    def estimate_cost(self, test_name: str) -> float:
        """Enhanced cost estimation with ML fallback."""
        
        # First try the original lookup-based approach
        try:
            cost_info = self.lookup_cost(test_name)
            return cost_info.price
        except KeyError:
            pass
            
        # Try ML prediction if available and enabled
        if self.enable_ml_fallback and self.ml_model:
            try:
                prediction, confidence = self.predict_ml_cost(test_name)
                
                # Only use ML prediction if confidence is sufficient
                if confidence >= self.confidence_threshold:
                    logger.debug(f"Using ML prediction for {test_name}: ${prediction:.2f} (confidence: {confidence:.2f})")
                    return prediction
                else:
                    logger.debug(f"ML prediction confidence too low ({confidence:.2f}) for {test_name}")
                    
            except Exception as e:
                logger.warning(f"ML prediction failed for {test_name}: {e}")
                
        # Fallback to parent class implementation (LLM-based)
        return super().estimate_cost(test_name)
        
    def estimate(self, test_name: str) -> Tuple[float, str]:
        """Enhanced estimation with prediction source tracking."""
        
        # Track prediction source
        prediction_source = "lookup"
        
        # Try lookup first
        try:
            cost_info = self.lookup_cost(test_name)
            return cost_info.price, cost_info.category
        except KeyError:
            pass
            
        # Try ML prediction
        if self.enable_ml_fallback and self.ml_model:
            try:
                prediction, confidence = self.predict_ml_cost(test_name)
                
                if confidence >= self.confidence_threshold:
                    prediction_source = "ml_model"
                    
                    # Try to determine category from similar tests
                    category = self._infer_category_from_ml(test_name)
                    
                    logger.debug(f"ML prediction for {test_name}: ${prediction:.2f} (confidence: {confidence:.2f})")
                    return prediction, category
                    
            except Exception as e:
                logger.warning(f"ML prediction failed for {test_name}: {e}")
                
        # Fallback to parent implementation
        prediction_source = "llm_fallback"
        price, category = super().estimate(test_name)
        
        return price, category
        
    def _infer_category_from_ml(self, test_name: str) -> str:
        """Infer test category using ML features."""
        
        # Simple category inference based on test name keywords
        test_lower = test_name.lower()
        
        if any(word in test_lower for word in ['blood', 'serum', 'plasma', 'culture']):
            return 'laboratory'
        elif any(word in test_lower for word in ['xray', 'x-ray', 'ct', 'mri', 'ultrasound']):
            return 'imaging'
        elif any(word in test_lower for word in ['ecg', 'ekg', 'echo', 'stress']):
            return 'cardiology'
        elif any(word in test_lower for word in ['biopsy', 'endoscopy', 'surgery']):
            return 'procedure'
        else:
            return 'unknown'
            
    def explain_prediction(self, test_name: str, cpt_code: Optional[str] = None) -> Dict[str, Any]:
        """Explain ML prediction for a given test."""
        
        if not self.model_explainer:
            return {"error": "Model explainer not available"}
            
        try:
            # Prepare input data
            input_data = pd.DataFrame({
                'test_name': [test_name],
                'cpt_code': [cpt_code or '00000'],
                'price': [0.0]  # Placeholder
            })
            
            # Preprocess input
            X_processed = self.preprocessing_pipeline.transform(input_data)
            
            # Get prediction explanation
            explanation = self.model_explainer.explain_prediction(
                X_processed[0], 
                sample_id=test_name
            )
            
            return {
                'test_name': test_name,
                'prediction': explanation.prediction,
                'top_features': explanation.top_contributing_features,
                'explanation_text': explanation.explanation_text
            }
            
        except Exception as e:
            return {"error": f"Explanation failed: {e}"}
            
    def get_model_performance(self) -> Optional[Dict[str, Any]]:
        """Get current ML model performance metrics."""
        
        if not self.model_performance:
            return None
            
        return asdict(self.model_performance)
        
    def save_ml_model(self, filepath: str):
        """Save trained ML model and preprocessing pipeline."""
        
        if not self.ml_model or not self.preprocessing_pipeline:
            raise ValueError("No ML model to save. Train model first.")
            
        model_data = {
            'ml_model': self.ml_model,
            'preprocessing_pipeline': self.preprocessing_pipeline,
            'ml_config': self.ml_config,
            'model_performance': self.model_performance,
            'feature_names': (self.model_explainer.feature_names 
                            if self.model_explainer else []),
            'saved_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"ML model saved to {filepath}")
        
    def load_ml_model(self, filepath: str):
        """Load pre-trained ML model and preprocessing pipeline."""
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            self.ml_model = model_data['ml_model']
            self.preprocessing_pipeline = model_data['preprocessing_pipeline']
            self.ml_config = model_data.get('ml_config', MLModelConfig())
            self.model_performance = model_data.get('model_performance')
            
            # Recreate model explainer if feature names available
            feature_names = model_data.get('feature_names', [])
            if feature_names and self.ml_model:
                self.model_explainer = ModelExplainer(self.ml_model, feature_names)
                
            logger.info(f"ML model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load ML model from {filepath}: {e}")
            raise
            
    def validate_ml_model(self, validation_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate ML model performance on new data."""
        
        if not self.ml_model or not self.preprocessing_pipeline:
            raise ValueError("No ML model available for validation")
            
        logger.info("Running ML model validation")
        
        # Run comprehensive validation
        validation_results = run_comprehensive_validation(
            validation_data[['test_name', 'cpt_code', 'price']], 
            validation_data['price'] if 'price' in validation_data.columns else None
        )
        
        # Add ML-specific validation
        try:
            X_val = validation_data[['test_name', 'cpt_code', 'price']].copy()
            y_val = validation_data['price'].copy()
            
            # Preprocess validation data
            X_val_processed = self.preprocessing_pipeline.transform(X_val)
            
            # Make predictions
            predictions = self.ml_model.predict(X_val_processed)
            
            # Calculate validation metrics
            val_r2 = r2_score(y_val, predictions)
            val_mae = mean_absolute_error(y_val, predictions)
            val_rmse = np.sqrt(mean_squared_error(y_val, predictions))
            
            validation_results['ml_specific_validation'] = {
                'validation_r2': val_r2,
                'validation_mae': val_mae,
                'validation_rmse': val_rmse,
                'performance_degradation': {
                    'r2_change': val_r2 - (self.model_performance.r2_score if self.model_performance else 0),
                    'mae_change': val_mae - (self.model_performance.mean_absolute_error if self.model_performance else 0)
                }
            }
            
        except Exception as e:
            validation_results['ml_specific_validation'] = {'error': str(e)}
            
        return validation_results
        
    def auto_retrain(self, new_data: pd.DataFrame, 
                    performance_threshold: Optional[float] = None) -> bool:
        """Automatically retrain model if performance degrades."""
        
        threshold = performance_threshold or self.ml_config.performance_threshold
        
        # Validate current performance
        validation_results = self.validate_ml_model(new_data)
        
        ml_validation = validation_results.get('ml_specific_validation', {})
        current_r2 = ml_validation.get('validation_r2', 0)
        
        # Check if retraining is needed
        if current_r2 < threshold:
            logger.info(f"Model performance below threshold ({current_r2:.3f} < {threshold:.3f}). Retraining...")
            
            try:
                # Combine existing training data with new data if available
                combined_data = new_data  # In practice, you'd combine with historical data
                
                # Retrain model
                new_performance = self.train_ml_model(combined_data)
                
                logger.info(f"Model retrained. New performance: R²={new_performance.r2_score:.3f}")
                return True
                
            except Exception as e:
                logger.error(f"Auto-retraining failed: {e}")
                return False
                
        return False


def create_ml_enhanced_cost_estimator(cost_table: Dict[str, CptCost],
                                    training_data: Optional[pd.DataFrame] = None,
                                    ml_config: Optional[MLModelConfig] = None) -> MLCostEstimator:
    """Create ML-enhanced cost estimator with optional training."""
    
    ml_config = ml_config or MLModelConfig()
    estimator = MLCostEstimator(cost_table, ml_config)
    
    if training_data is not None and not training_data.empty:
        logger.info("Training ML model during estimator creation")
        estimator.train_ml_model(training_data)
        
    return estimator


def demonstrate_ml_cost_estimator():
    """Demonstrate the ML-enhanced cost estimator."""
    
    print("=== ML-Enhanced Cost Estimator Demonstration ===\n")
    
    # Create sample cost table (from existing data)
    cost_table = {
        'complete blood count': CptCost('85027', 9.0, 'laboratory'),
        'comprehensive metabolic panel': CptCost('80053', 14.0, 'laboratory'),
        'mri brain with contrast': CptCost('70553', 400.0, 'imaging'),
        'chest x-ray 2 views': CptCost('71046', 30.0, 'imaging'),
        'electrocardiogram': CptCost('93000', 10.0, 'cardiology')
    }
    
    # Generate training data
    np.random.seed(42)
    training_samples = []
    
    base_tests = [
        ('complete blood count', '85027', 9.0),
        ('comprehensive metabolic panel', '80053', 14.0),
        ('mri brain with contrast', '70553', 400.0),
        ('chest x-ray 2 views', '71046', 30.0),
        ('electrocardiogram', '93000', 10.0),
        ('thyroid stimulating hormone', '84443', 20.0),
        ('lipid panel', '80061', 15.0),
        ('ct head without contrast', '70450', 100.0),
        ('holter monitor 24 hour', '93224', 200.0),
        ('basic metabolic panel', '80048', 13.0)
    ]
    
    # Create variations of each test
    for test_name, cpt_code, base_price in base_tests:
        for i in range(50):  # 50 variations per test
            price_variation = np.random.normal(1, 0.15)  # 15% price variation
            price = max(1, base_price * price_variation)
            
            training_samples.append({
                'test_name': test_name,
                'cpt_code': cpt_code,
                'price': price
            })
            
    training_data = pd.DataFrame(training_samples)
    
    print(f"Generated training data: {len(training_data)} samples")
    print(f"Price range: ${training_data['price'].min():.2f} - ${training_data['price'].max():.2f}")
    
    # Create ML-enhanced cost estimator
    ml_config = MLModelConfig(
        model_type="random_forest",
        model_params={"n_estimators": 50, "random_state": 42}
    )
    
    estimator = create_ml_enhanced_cost_estimator(
        cost_table=cost_table,
        training_data=training_data,
        ml_config=ml_config
    )
    
    # Display model performance
    performance = estimator.get_model_performance()
    if performance:
        print(f"\n=== ML Model Performance ===")
        print(f"Model: {performance['model_name']}")
        print(f"Performance Grade: {estimator.model_performance.performance_grade}")
        print(f"R² Score: {performance['r2_score']:.3f}")
        print(f"Mean Absolute Error: ${performance['mean_absolute_error']:.2f}")
        print(f"Training Samples: {performance['training_samples']}")
        print(f"Features Generated: {performance['feature_count']}")
        
    # Test predictions
    print(f"\n=== Cost Estimation Comparisons ===")
    
    test_cases = [
        'complete blood count',           # Known test
        'comprehensive metabolic panel',  # Known test
        'advanced lipid panel',          # Unknown test (similar to known)
        'cardiac stress test',           # Completely unknown test
        'mri spine with contrast'        # Unknown but similar to known
    ]
    
    for test_name in test_cases:
        # Traditional lookup
        try:
            traditional_cost = super(MLCostEstimator, estimator).estimate_cost(test_name)
            traditional_source = "lookup"
        except:
            traditional_cost = 50.0  # Default fallback
            traditional_source = "fallback"
            
        # ML-enhanced estimation
        ml_cost, category = estimator.estimate(test_name)
        
        # ML prediction with confidence
        if estimator.ml_model:
            try:
                ml_prediction, confidence = estimator.predict_ml_cost(test_name)
                ml_source = f"ML (conf: {confidence:.2f})"
            except:
                ml_prediction = ml_cost
                ml_source = "ML (fallback)"
        else:
            ml_prediction = ml_cost
            ml_source = "No ML"
            
        print(f"\nTest: {test_name}")
        print(f"  Traditional: ${traditional_cost:.2f} ({traditional_source})")
        print(f"  ML-Enhanced: ${ml_cost:.2f} ({category})")
        print(f"  ML Direct: ${ml_prediction:.2f} ({ml_source})")
        
        # Get prediction explanation
        explanation = estimator.explain_prediction(test_name)
        if 'top_features' in explanation:
            print(f"  Top Contributing Features:")
            for feature, contribution in explanation['top_features'][:3]:
                direction = "↑" if contribution > 0 else "↓"
                print(f"    - {feature}: {direction} ${abs(contribution):.2f}")
                
    # Demonstrate model validation
    print(f"\n=== Model Validation ===")
    
    # Create validation data (different from training)
    validation_samples = []
    for test_name, cpt_code, base_price in base_tests[:5]:  # Use subset for validation
        for i in range(20):
            price_variation = np.random.normal(1, 0.2)  # Different variation
            price = max(1, base_price * price_variation)
            
            validation_samples.append({
                'test_name': test_name,
                'cpt_code': cpt_code,
                'price': price
            })
            
    validation_data = pd.DataFrame(validation_samples)
    
    validation_results = estimator.validate_ml_model(validation_data)
    
    ml_validation = validation_results.get('ml_specific_validation', {})
    if 'validation_r2' in ml_validation:
        print(f"Validation R² Score: {ml_validation['validation_r2']:.3f}")
        print(f"Validation MAE: ${ml_validation['validation_mae']:.2f}")
        
        perf_change = ml_validation.get('performance_degradation', {})
        r2_change = perf_change.get('r2_change', 0)
        if r2_change < -0.1:
            print(f"⚠️  Model performance degraded by {abs(r2_change):.3f}")
        else:
            print(f"✅ Model performance stable (change: {r2_change:.3f})")
            
    print(f"\n=== ML-Enhanced Cost Estimator Ready ===")
    print(f"Features:")
    print(f"  - Seamless integration with existing cost estimator")
    print(f"  - ML fallback for unknown tests")
    print(f"  - Confidence-based prediction selection")
    print(f"  - Model performance monitoring")
    print(f"  - Prediction explanations")
    print(f"  - Automatic model validation and retraining")


if __name__ == "__main__":
    demonstrate_ml_cost_estimator()