#!/usr/bin/env python3
"""
ML-Compatible Preprocessing Pipeline Components
Scikit-learn compatible transformers for CPT cost estimation
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.feature_selection import SelectKBest, SelectFromModel, VarianceThreshold
from sklearn.feature_selection import f_regression, mutual_info_regression, chi2
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.exceptions import NotFittedError

import structlog

from .ml_feature_engineering import FeatureConfig, ComprehensiveFeatureEngineering

logger = structlog.get_logger(__name__)


class CPTDataValidator(BaseEstimator, TransformerMixin):
    """Validate and clean CPT cost estimation data."""
    
    def __init__(self, 
                 min_price: float = 0.0,
                 max_price: float = 10000.0,
                 required_cpt_format: bool = True,
                 remove_duplicates: bool = True):
        self.min_price = min_price
        self.max_price = max_price
        self.required_cpt_format = required_cpt_format
        self.remove_duplicates = remove_duplicates
        self.validation_stats_ = {}
        
    def fit(self, X, y=None):
        self.validation_stats_ = {}
        return self
        
    def transform(self, X):
        """Validate and clean the input data."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        df = X.copy()
        original_shape = df.shape
        
        # Validate required columns
        required_cols = ['test_name', 'cpt_code', 'price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Clean test names
        df['test_name'] = df['test_name'].fillna('').astype(str).str.strip()
        df = df[df['test_name'] != '']
        
        # Clean CPT codes
        df['cpt_code'] = df['cpt_code'].fillna('').astype(str).str.strip()
        
        if self.required_cpt_format:
            # Validate CPT code format (5 digits or HCPCS)
            cpt_pattern = r'^(\d{5}|[A-Z]\d{4})$'
            valid_cpt_mask = df['cpt_code'].str.match(cpt_pattern, na=False)
            df = df[valid_cpt_mask]
            
        # Clean and validate prices
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])
        df = df[(df['price'] >= self.min_price) & (df['price'] <= self.max_price)]
        
        # Remove duplicates if requested
        if self.remove_duplicates:
            df = df.drop_duplicates(subset=['test_name', 'cpt_code'])
            
        # Store validation statistics
        self.validation_stats_ = {
            'original_rows': original_shape[0],
            'cleaned_rows': len(df),
            'rows_removed': original_shape[0] - len(df),
            'removal_percentage': (original_shape[0] - len(df)) / original_shape[0] * 100,
            'price_range': [df['price'].min(), df['price'].max()] if not df.empty else [0, 0],
            'unique_cpt_codes': df['cpt_code'].nunique() if not df.empty else 0,
            'unique_test_names': df['test_name'].nunique() if not df.empty else 0
        }
        
        if df.empty:
            logger.warning("All rows removed during validation")
        else:
            logger.info(f"Data validation complete: {self.validation_stats_['cleaned_rows']} rows remaining")
            
        return df


class AdvancedImputer(BaseEstimator, TransformerMixin):
    """Advanced missing value imputation with multiple strategies."""
    
    def __init__(self, 
                 numerical_strategy: str = 'median',
                 categorical_strategy: str = 'most_frequent',
                 use_knn: bool = False,
                 knn_neighbors: int = 5,
                 use_iterative: bool = False):
        
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.use_knn = use_knn
        self.knn_neighbors = knn_neighbors
        self.use_iterative = use_iterative
        
        self.numerical_imputer_ = None
        self.categorical_imputer_ = None
        self.feature_types_ = {}
        
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Identify feature types
        self.feature_types_ = {
            'numerical': X.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical': X.select_dtypes(exclude=[np.number]).columns.tolist()
        }
        
        # Initialize imputers based on configuration
        if self.use_iterative and self.feature_types_['numerical']:
            self.numerical_imputer_ = IterativeImputer(random_state=42)
        elif self.use_knn and self.feature_types_['numerical']:
            self.numerical_imputer_ = KNNImputer(n_neighbors=self.knn_neighbors)
        else:
            self.numerical_imputer_ = SimpleImputer(strategy=self.numerical_strategy)
            
        self.categorical_imputer_ = SimpleImputer(strategy=self.categorical_strategy)
        
        # Fit imputers
        if self.feature_types_['numerical']:
            self.numerical_imputer_.fit(X[self.feature_types_['numerical']])
            
        if self.feature_types_['categorical']:
            self.categorical_imputer_.fit(X[self.feature_types_['categorical']])
            
        return self
        
    def transform(self, X):
        check_is_fitted(self)
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_imputed = X.copy()
        
        # Impute numerical features
        if self.feature_types_['numerical'] and self.numerical_imputer_:
            numerical_cols = [col for col in self.feature_types_['numerical'] if col in X.columns]
            if numerical_cols:
                X_imputed[numerical_cols] = self.numerical_imputer_.transform(X[numerical_cols])
                
        # Impute categorical features  
        if self.feature_types_['categorical'] and self.categorical_imputer_:
            categorical_cols = [col for col in self.feature_types_['categorical'] if col in X.columns]
            if categorical_cols:
                X_imputed[categorical_cols] = self.categorical_imputer_.transform(X[categorical_cols])
                
        return X_imputed


class OutlierDetector(BaseEstimator, TransformerMixin):
    """Detect and handle outliers using multiple methods."""
    
    def __init__(self, 
                 method: str = 'iqr',
                 threshold: float = 3.0,
                 action: str = 'remove',  # 'remove', 'cap', 'flag'
                 feature_subset: Optional[List[str]] = None):
        
        self.method = method
        self.threshold = threshold
        self.action = action
        self.feature_subset = feature_subset
        
        self.outlier_bounds_ = {}
        self.outlier_stats_ = {}
        
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Determine features to check for outliers
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.feature_subset:
            features_to_check = [f for f in self.feature_subset if f in numerical_features]
        else:
            features_to_check = numerical_features
            
        # Calculate outlier bounds for each feature
        for feature in features_to_check:
            data = X[feature].dropna()
            
            if self.method == 'iqr':
                q25, q75 = np.percentile(data, [25, 75])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                
            elif self.method == 'zscore':
                mean, std = data.mean(), data.std()
                lower_bound = mean - self.threshold * std
                upper_bound = mean + self.threshold * std
                
            elif self.method == 'modified_zscore':
                median = data.median()
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad
                threshold_mad = self.threshold
                lower_bound = data[modified_z_scores > -threshold_mad].min()
                upper_bound = data[modified_z_scores < threshold_mad].max()
                
            else:
                raise ValueError(f"Unknown outlier detection method: {self.method}")
                
            self.outlier_bounds_[feature] = {
                'lower': lower_bound,
                'upper': upper_bound
            }
            
        return self
        
    def transform(self, X):
        check_is_fitted(self)
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_processed = X.copy()
        outliers_detected = 0
        
        for feature, bounds in self.outlier_bounds_.items():
            if feature in X_processed.columns:
                lower, upper = bounds['lower'], bounds['upper']
                
                # Identify outliers
                outlier_mask = (X_processed[feature] < lower) | (X_processed[feature] > upper)
                outliers_count = outlier_mask.sum()
                outliers_detected += outliers_count
                
                if self.action == 'remove':
                    X_processed = X_processed[~outlier_mask]
                elif self.action == 'cap':
                    X_processed[feature] = X_processed[feature].clip(lower=lower, upper=upper)
                elif self.action == 'flag':
                    X_processed[f'{feature}_is_outlier'] = outlier_mask
                    
        self.outlier_stats_ = {
            'total_outliers_detected': outliers_detected,
            'features_checked': list(self.outlier_bounds_.keys()),
            'action_taken': self.action
        }
        
        if outliers_detected > 0:
            logger.info(f"Detected {outliers_detected} outliers using {self.method} method")
            
        return X_processed


class SmartColumnTransformer(BaseEstimator, TransformerMixin):
    """Intelligent column transformer that automatically handles different feature types."""
    
    def __init__(self, 
                 numerical_transformer: str = 'standard',
                 categorical_transformer: str = 'onehot',
                 text_transformer: str = 'tfidf',
                 handle_unknown: str = 'ignore',
                 max_categories: int = 20):
        
        self.numerical_transformer = numerical_transformer
        self.categorical_transformer = categorical_transformer  
        self.text_transformer = text_transformer
        self.handle_unknown = handle_unknown
        self.max_categories = max_categories
        
        self.column_transformer_ = None
        self.feature_names_ = []
        self.feature_types_ = {}
        
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Automatically detect feature types
        numerical_features = []
        categorical_features = []
        text_features = []
        
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                numerical_features.append(col)
            elif X[col].dtype == 'object':
                # Distinguish between categorical and text
                unique_ratio = X[col].nunique() / len(X)
                avg_length = X[col].astype(str).str.len().mean()
                
                if unique_ratio > 0.5 or avg_length > 20:
                    text_features.append(col)
                else:
                    categorical_features.append(col)
            else:
                categorical_features.append(col)
                
        # Filter categorical features with too many categories
        categorical_features = [
            col for col in categorical_features 
            if X[col].nunique() <= self.max_categories
        ]
        
        self.feature_types_ = {
            'numerical': numerical_features,
            'categorical': categorical_features,
            'text': text_features
        }
        
        # Build transformers
        transformers = []
        
        if numerical_features:
            if self.numerical_transformer == 'standard':
                num_transformer = StandardScaler()
            elif self.numerical_transformer == 'minmax':
                num_transformer = MinMaxScaler()
            elif self.numerical_transformer == 'robust':
                num_transformer = RobustScaler()
            else:
                num_transformer = StandardScaler()
                
            transformers.append(('numerical', num_transformer, numerical_features))
            
        if categorical_features:
            if self.categorical_transformer == 'onehot':
                cat_transformer = OneHotEncoder(
                    handle_unknown=self.handle_unknown,
                    sparse_output=False
                )
            elif self.categorical_transformer == 'ordinal':
                cat_transformer = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )
            else:
                cat_transformer = OneHotEncoder(
                    handle_unknown=self.handle_unknown,
                    sparse_output=False
                )
                
            transformers.append(('categorical', cat_transformer, categorical_features))
            
        if text_features:
            # Text features handled by feature engineering pipeline
            # For now, just pass through
            transformers.append(('text', 'passthrough', text_features))
            
        if not transformers:
            # If no features, create a dummy transformer
            transformers.append(('passthrough', 'passthrough', []))
            
        # Create column transformer
        self.column_transformer_ = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        self.column_transformer_.fit(X)
        
        # Generate feature names
        self.feature_names_ = self._get_feature_names()
        
        return self
        
    def transform(self, X):
        check_is_fitted(self, 'column_transformer_')
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        return self.column_transformer_.transform(X)
        
    def _get_feature_names(self):
        """Generate feature names for transformed columns."""
        feature_names = []
        
        for name, transformer, features in self.column_transformer_.transformers_:
            if name == 'remainder':
                continue
                
            if hasattr(transformer, 'get_feature_names_out'):
                names = transformer.get_feature_names_out(features)
            elif name == 'categorical' and hasattr(transformer, 'categories_'):
                names = []
                for i, feature in enumerate(features):
                    categories = transformer.categories_[i]
                    names.extend([f"{feature}_{cat}" for cat in categories])
            else:
                names = features
                
            feature_names.extend(names)
            
        return feature_names
        
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        return self.feature_names_


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Advanced feature selection with multiple methods."""
    
    def __init__(self, 
                 method: str = 'variance_threshold',
                 k_best: int = None,
                 threshold: float = None,
                 estimator=None,
                 percentile: int = None):
        
        self.method = method
        self.k_best = k_best
        self.threshold = threshold
        self.estimator = estimator
        self.percentile = percentile
        
        self.selector_ = None
        self.selected_features_ = []
        
    def fit(self, X, y=None):
        if self.method == 'variance_threshold':
            threshold = self.threshold or 0.0
            self.selector_ = VarianceThreshold(threshold=threshold)
            
        elif self.method == 'k_best':
            k = self.k_best or 10
            score_func = f_regression if y is not None else mutual_info_regression
            self.selector_ = SelectKBest(score_func=score_func, k=k)
            
        elif self.method == 'model_based':
            if self.estimator is None:
                from sklearn.ensemble import RandomForestRegressor
                self.estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            self.selector_ = SelectFromModel(self.estimator, threshold=self.threshold)
            
        elif self.method == 'percentile':
            percentile = self.percentile or 50
            score_func = f_regression if y is not None else mutual_info_regression
            from sklearn.feature_selection import SelectPercentile
            self.selector_ = SelectPercentile(score_func=score_func, percentile=percentile)
            
        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")
            
        self.selector_.fit(X, y)
        
        # Store selected feature indices
        if hasattr(self.selector_, 'get_support'):
            self.selected_features_ = np.where(self.selector_.get_support())[0].tolist()
        else:
            self.selected_features_ = list(range(X.shape[1]))
            
        logger.info(f"Feature selection: {len(self.selected_features_)} features selected from {X.shape[1]}")
        
        return self
        
    def transform(self, X):
        check_is_fitted(self, 'selector_')
        return self.selector_.transform(X)


class MLPreprocessingPipeline(BaseEstimator, TransformerMixin):
    """Complete preprocessing pipeline for CPT cost estimation ML models."""
    
    def __init__(self, 
                 feature_config: Optional[FeatureConfig] = None,
                 validation_config: Optional[Dict] = None,
                 preprocessing_config: Optional[Dict] = None):
        
        self.feature_config = feature_config or FeatureConfig()
        self.validation_config = validation_config or {}
        self.preprocessing_config = preprocessing_config or {}
        
        # Initialize pipeline components
        self.validator = CPTDataValidator(**self.validation_config)
        self.feature_engineer = ComprehensiveFeatureEngineering(self.feature_config)
        
        # Advanced preprocessing components
        self.imputer = AdvancedImputer(
            **self.preprocessing_config.get('imputation', {})
        )
        self.outlier_detector = OutlierDetector(
            **self.preprocessing_config.get('outlier_detection', {})
        )
        self.column_transformer = SmartColumnTransformer(
            **self.preprocessing_config.get('column_transformation', {})
        )
        
        # Feature selection
        if self.feature_config.enable_feature_selection:
            self.feature_selector = FeatureSelector(
                method=self.feature_config.feature_selection_method,
                k_best=self.feature_config.max_features
            )
        else:
            self.feature_selector = None
            
        # Pipeline state
        self.fitted_ = False
        self.feature_names_ = []
        self.preprocessing_stats_ = {}
        
    def fit(self, X, y=None):
        """Fit the complete preprocessing pipeline."""
        
        logger.info("Starting ML preprocessing pipeline fit")
        
        # Step 1: Data validation and cleaning
        X_clean = self.validator.fit_transform(X)
        
        if X_clean.empty:
            raise ValueError("No valid data remaining after validation")
            
        # Step 2: Feature engineering
        self.feature_engineer.fit(X_clean, y)
        X_features = self.feature_engineer.transform(X_clean)
        
        # Convert to DataFrame for easier handling
        feature_names = [f"feature_{i}" for i in range(X_features.shape[1])]
        X_features_df = pd.DataFrame(X_features, columns=feature_names)
        
        # Step 3: Handle missing values
        X_imputed = self.imputer.fit_transform(X_features_df)
        
        # Step 4: Outlier detection and handling
        X_no_outliers = self.outlier_detector.fit_transform(X_imputed)
        
        # Step 5: Column transformation (scaling, encoding)
        X_transformed = self.column_transformer.fit_transform(X_no_outliers)
        
        # Step 6: Feature selection (if enabled)
        if self.feature_selector is not None:
            X_selected = self.feature_selector.fit_transform(X_transformed, y)
        else:
            X_selected = X_transformed
            
        # Store preprocessing statistics
        self.preprocessing_stats_ = {
            'original_samples': len(X),
            'final_samples': X_selected.shape[0] if hasattr(X_selected, 'shape') else len(X_selected),
            'original_features': X.shape[1],
            'final_features': X_selected.shape[1] if hasattr(X_selected, 'shape') else len(X_selected[0]) if X_selected else 0,
            'validation_stats': self.validator.validation_stats_,
            'outlier_stats': self.outlier_detector.outlier_stats_,
            'feature_engineering_stats': {
                'generated_features': len(self.feature_engineer.get_feature_names()),
                'feature_types': len(self.feature_engineer.get_feature_metadata())
            }
        }
        
        self.fitted_ = True
        logger.info(f"Preprocessing pipeline fit complete: {self.preprocessing_stats_}")
        
        return self
        
    def transform(self, X):
        """Transform data through the complete preprocessing pipeline."""
        
        if not self.fitted_:
            raise NotFittedError("Pipeline must be fitted before transform")
            
        # Apply the same transformations as in fit
        X_clean = self.validator.transform(X)
        
        if X_clean.empty:
            logger.warning("No valid data remaining after validation in transform")
            return np.array([]).reshape(0, -1)
            
        X_features = self.feature_engineer.transform(X_clean)
        
        # Convert to DataFrame
        feature_names = [f"feature_{i}" for i in range(X_features.shape[1])]
        X_features_df = pd.DataFrame(X_features, columns=feature_names)
        
        X_imputed = self.imputer.transform(X_features_df)
        X_no_outliers = self.outlier_detector.transform(X_imputed)
        X_transformed = self.column_transformer.transform(X_no_outliers)
        
        if self.feature_selector is not None:
            X_final = self.feature_selector.transform(X_transformed)
        else:
            X_final = X_transformed
            
        return X_final
        
    def fit_transform(self, X, y=None):
        """Fit the pipeline and transform the data."""
        return self.fit(X, y).transform(X)
        
    def get_preprocessing_stats(self):
        """Get preprocessing pipeline statistics."""
        return self.preprocessing_stats_.copy()
        
    def save_pipeline(self, filepath: str):
        """Save the fitted pipeline to disk."""
        if not self.fitted_:
            raise NotFittedError("Pipeline must be fitted before saving")
            
        pipeline_data = {
            'pipeline': self,
            'feature_config': self.feature_config,
            'validation_config': self.validation_config,
            'preprocessing_config': self.preprocessing_config,
            'preprocessing_stats': self.preprocessing_stats_,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
            
        logger.info(f"Pipeline saved to {filepath}")
        
    @classmethod
    def load_pipeline(cls, filepath: str):
        """Load a fitted pipeline from disk."""
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
            
        logger.info(f"Pipeline loaded from {filepath}")
        return pipeline_data['pipeline']


def create_default_preprocessing_pipeline() -> MLPreprocessingPipeline:
    """Create a preprocessing pipeline with sensible defaults."""
    
    feature_config = FeatureConfig(
        include_text_features=True,
        include_hierarchical_features=True,
        include_interaction_features=True,
        max_tfidf_features=50,
        handle_missing_values=True,
        remove_outliers=True,
        outlier_method='iqr',
        enable_feature_selection=False
    )
    
    validation_config = {
        'min_price': 0.0,
        'max_price': 10000.0,
        'required_cpt_format': True,
        'remove_duplicates': True
    }
    
    preprocessing_config = {
        'imputation': {
            'numerical_strategy': 'median',
            'categorical_strategy': 'most_frequent',
            'use_knn': False
        },
        'outlier_detection': {
            'method': 'iqr',
            'action': 'cap',
            'threshold': 1.5
        },
        'column_transformation': {
            'numerical_transformer': 'standard',
            'categorical_transformer': 'onehot',
            'handle_unknown': 'ignore'
        }
    }
    
    return MLPreprocessingPipeline(
        feature_config=feature_config,
        validation_config=validation_config,
        preprocessing_config=preprocessing_config
    )


def demonstrate_preprocessing_pipeline():
    """Demonstrate the preprocessing pipeline."""
    
    # Create sample data with various issues
    sample_data = pd.DataFrame({
        'test_name': [
            'complete blood count',
            'comprehensive metabolic panel', 
            'mri brain with contrast',
            '',  # Empty name
            '24 hour holter monitor',
            'thyroid stimulating hormone',
            'lipid panel fasting',
            'ct head without contrast',
            'electrocardiogram 12 lead',
            'basic metabolic panel'
        ],
        'cpt_code': [
            '85027', '80053', '70553', '71046', '93224',
            'ABCDE',  # Invalid format
            '80061', '70450', '93000', '80048'
        ],
        'price': [9.0, 14.0, 400.0, np.nan, 200.0, 20.0, 15.0, 100.0, 10.0, 13.0]
    })
    
    print("=== ML Preprocessing Pipeline Demonstration ===\n")
    print(f"Original data shape: {sample_data.shape}")
    print("Original data:")
    print(sample_data)
    print("\nData issues:")
    print(f"- Missing prices: {sample_data['price'].isna().sum()}")
    print(f"- Empty test names: {(sample_data['test_name'] == '').sum()}")
    print(f"- Invalid CPT codes: {(~sample_data['cpt_code'].str.match(r'^\\d{5}$', na=False)).sum()}")
    
    # Create and fit pipeline
    pipeline = create_default_preprocessing_pipeline()
    
    try:
        X_processed = pipeline.fit_transform(sample_data)
        
        print(f"\n=== Preprocessing Results ===")
        print(f"Processed data shape: {X_processed.shape}")
        
        stats = pipeline.get_preprocessing_stats()
        print(f"\nProcessing Statistics:")
        print(f"- Original samples: {stats['original_samples']}")
        print(f"- Final samples: {stats['final_samples']}")
        print(f"- Original features: {stats['original_features']}")
        print(f"- Final features: {stats['final_features']}")
        
        print(f"\nValidation Results:")
        val_stats = stats['validation_stats']
        print(f"- Rows removed: {val_stats['rows_removed']}")
        print(f"- Removal percentage: {val_stats['removal_percentage']:.1f}%")
        print(f"- Unique CPT codes: {val_stats['unique_cpt_codes']}")
        
        print(f"\nFeature Engineering:")
        fe_stats = stats['feature_engineering_stats']
        print(f"- Generated features: {fe_stats['generated_features']}")
        
        print("\n=== Pipeline Ready for ML Model Training ===")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")


if __name__ == "__main__":
    demonstrate_preprocessing_pipeline()