#!/usr/bin/env python3
"""
Comprehensive Feature Engineering Pipeline for ML-based Cost Estimation
Production-ready system for transforming CPT code and pricing data into ML features
"""

import re
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

import structlog

logger = structlog.get_logger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline."""
    
    # Feature generation settings
    include_text_features: bool = True
    include_hierarchical_features: bool = True
    include_interaction_features: bool = True
    include_temporal_features: bool = False
    
    # Text processing settings
    max_tfidf_features: int = 100
    min_word_freq: int = 2
    ngram_range: Tuple[int, int] = (1, 2)
    
    # Preprocessing settings
    handle_missing_values: bool = True
    remove_outliers: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation"
    outlier_threshold: float = 3.0
    
    # Feature selection settings
    enable_feature_selection: bool = False
    max_features: Optional[int] = None
    feature_selection_method: str = "f_regression"
    
    # Scaling settings
    numerical_scaler: str = "standard"  # "standard", "minmax", "robust"
    categorical_encoding: str = "onehot"  # "onehot", "label", "target"
    
    # Caching settings
    enable_caching: bool = True
    cache_dir: Optional[str] = None


class CPTHierarchyExtractor(BaseEstimator, TransformerMixin):
    """Extract hierarchical features from CPT codes."""
    
    def __init__(self, include_category_maps: bool = True):
        self.include_category_maps = include_category_maps
        self.cpt_categories = {
            "evaluation_management": ("99", "E&M Services"),
            "anesthesia": ("00", "Anesthesia"),
            "surgery": ("10", "Surgery"),
            "radiology": ("70", "Radiology"),
            "pathology_lab": ("80", "Pathology and Laboratory"),
            "medicine": ("90", "Medicine")
        }
        
        # Detailed category mappings
        self.detailed_categories = {
            range(0, 1000): "anesthesia",
            range(10000, 70000): "surgery", 
            range(70000, 80000): "radiology",
            range(80000, 90000): "pathology_lab",
            range(90000, 100000): "medicine"
        }
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)
            
        features = []
        for cpt_code in X:
            features.append(self._extract_cpt_features(str(cpt_code)))
            
        return pd.DataFrame(features)
    
    def _extract_cpt_features(self, cpt_code: str) -> Dict[str, Any]:
        """Extract hierarchical features from a single CPT code."""
        features = {}
        
        # Clean and validate CPT code
        cpt_clean = re.sub(r'[^0-9]', '', cpt_code)
        if not cpt_clean:
            cpt_clean = "00000"
            
        # Ensure 5 digits
        cpt_clean = cpt_clean.zfill(5)[:5]
        cpt_numeric = int(cpt_clean)
        
        # Basic numeric features
        features['cpt_numeric'] = cpt_numeric
        features['cpt_first_digit'] = int(cpt_clean[0])
        features['cpt_first_two_digits'] = int(cpt_clean[:2])
        features['cpt_first_three_digits'] = int(cpt_clean[:3])
        
        # Category classification
        features['cpt_major_category'] = self._get_major_category(cpt_numeric)
        features['cpt_subcategory'] = cpt_clean[:3]
        
        # Special code indicators
        features['is_add_on_code'] = "+" in cpt_code
        features['has_modifier'] = "-" in cpt_code
        features['is_unlisted_code'] = cpt_clean.endswith("99")
        
        # Category binary flags
        if self.include_category_maps:
            for category in self.cpt_categories.keys():
                features[f'is_{category}'] = features['cpt_major_category'] == category
                
        return features
    
    def _get_major_category(self, cpt_numeric: int) -> str:
        """Determine major category from CPT numeric value."""
        for code_range, category in self.detailed_categories.items():
            if cpt_numeric in code_range:
                return category
        return "other"


class TestNameFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract semantic and linguistic features from test names."""
    
    def __init__(self, include_semantic_features: bool = True, 
                 include_complexity_features: bool = True):
        self.include_semantic_features = include_semantic_features
        self.include_complexity_features = include_complexity_features
        
        # Complexity indicators
        self.complexity_keywords = {
            "simple": ["basic", "simple", "routine", "standard", "screening", "single"],
            "moderate": ["comprehensive", "detailed", "extended", "multiple", "panel"],
            "complex": ["complex", "complicated", "extensive", "advanced", "specialized", "complete"]
        }
        
        # Body system mappings  
        self.body_systems = {
            "cardiovascular": ["heart", "cardiac", "vascular", "artery", "vein", "blood vessel", "ecg", "ekg"],
            "respiratory": ["lung", "breath", "respiratory", "pulmonary", "chest", "oxygen"],
            "neurological": ["brain", "nerve", "neuro", "spine", "cranial", "cognitive"],
            "gastrointestinal": ["stomach", "intestine", "colon", "liver", "digestive", "gastro"],
            "musculoskeletal": ["bone", "joint", "muscle", "skeletal", "orthopedic", "fracture"],
            "endocrine": ["hormone", "thyroid", "diabetes", "metabolic", "endocrine", "insulin"],
            "renal": ["kidney", "renal", "urinary", "bladder", "urine"],
            "hematologic": ["blood", "hematology", "anemia", "bleeding", "platelet"],
            "immunologic": ["immune", "antibody", "antigen", "allergy", "infection"]
        }
        
        # Test type indicators
        self.test_types = {
            "imaging": ["xray", "x-ray", "ct", "mri", "ultrasound", "scan", "imaging", "nuclear"],
            "laboratory": ["blood", "urine", "culture", "panel", "test", "assay", "level"],
            "procedure": ["biopsy", "endoscopy", "surgery", "removal", "insertion", "injection"],
            "diagnostic": ["ecg", "ekg", "eeg", "emg", "monitor", "study", "evaluation"],
            "therapeutic": ["therapy", "treatment", "injection", "infusion", "rehabilitation"]
        }
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)
            
        features = []
        for test_name in X:
            features.append(self._extract_name_features(str(test_name)))
            
        return pd.DataFrame(features)
    
    def _extract_name_features(self, test_name: str) -> Dict[str, Any]:
        """Extract comprehensive features from test name."""
        features = {}
        test_lower = test_name.lower().strip()
        words = test_lower.split()
        
        # Basic linguistic features
        features['name_length'] = len(test_name)
        features['name_word_count'] = len(words)
        features['name_char_count'] = len(test_name.replace(" ", ""))
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['word_diversity'] = len(set(words)) / len(words) if words else 0
        
        # Character and format features
        features['has_numbers'] = bool(re.search(r'\d', test_name))
        features['has_special_chars'] = bool(re.search(r'[^a-zA-Z0-9\s]', test_name))
        features['has_abbreviation'] = any(len(w) <= 3 and w.isupper() for w in test_name.split())
        features['has_parentheses'] = "(" in test_name
        features['has_slash'] = "/" in test_name
        features['has_hyphen'] = "-" in test_name
        features['has_comma'] = "," in test_name
        
        # Complexity indicators
        if self.include_complexity_features:
            for complexity, keywords in self.complexity_keywords.items():
                features[f'complexity_{complexity}'] = self._has_keywords(test_lower, keywords)
            
            features['estimated_complexity_score'] = self._calculate_complexity_score(test_lower)
            features['has_contrast'] = "contrast" in test_lower
            features['is_bilateral'] = any(word in test_lower for word in ["bilateral", "both", "dual"])
            features['is_complete'] = "complete" in test_lower or "comprehensive" in test_lower
            features['is_limited'] = "limited" in test_lower or "partial" in test_lower
            features['requires_prep'] = any(word in test_lower for word in ["prep", "preparation", "fasting"])
            
        # Semantic features
        if self.include_semantic_features:
            features['test_type'] = self._classify_test_type(test_lower)
            features['primary_body_system'] = self._identify_body_system(test_lower)
            
            # Body system binary indicators
            for system in self.body_systems.keys():
                features[f'involves_{system}'] = self._has_keywords(test_lower, self.body_systems[system])
                
            # Test type binary indicators  
            for test_type in self.test_types.keys():
                features[f'is_{test_type}'] = self._has_keywords(test_lower, self.test_types[test_type])
                
            # Clinical context
            features['is_emergency'] = self._is_emergency_test(test_lower)
            features['is_preventive'] = self._is_preventive_test(test_lower)
            features['is_panel'] = "panel" in test_lower
            features['is_culture'] = "culture" in test_lower
            features['requires_fasting'] = self._requires_fasting(test_lower)
            
        # First and last word features
        features['first_word'] = words[0] if words else ""
        features['last_word'] = words[-1] if words else ""
        
        return features
    
    def _has_keywords(self, text: str, keywords: List[str]) -> bool:
        """Check if text contains any keywords."""
        return any(keyword in text for keyword in keywords)
    
    def _calculate_complexity_score(self, test_name: str) -> float:
        """Calculate complexity score from 0 to 1."""
        score = 0.0
        
        # Base complexity from keywords
        if self._has_keywords(test_name, self.complexity_keywords["complex"]):
            score += 0.4
        elif self._has_keywords(test_name, self.complexity_keywords["moderate"]):
            score += 0.2
        elif self._has_keywords(test_name, self.complexity_keywords["simple"]):
            score -= 0.1
            
        # Additional complexity indicators
        if "comprehensive" in test_name:
            score += 0.2
        if "with contrast" in test_name:
            score += 0.15
        if "complete" in test_name:
            score += 0.1
        if len(test_name.split()) > 5:
            score += 0.05
        if "24 hour" in test_name or "continuous" in test_name:
            score += 0.1
            
        return min(max(score, 0.0), 1.0)
    
    def _classify_test_type(self, test_name: str) -> str:
        """Classify test into primary type."""
        for test_type, keywords in self.test_types.items():
            if self._has_keywords(test_name, keywords):
                return test_type
        return "other"
    
    def _identify_body_system(self, test_name: str) -> str:
        """Identify primary body system."""
        for system, keywords in self.body_systems.items():
            if self._has_keywords(test_name, keywords):
                return system
        return "general"
        
    def _is_emergency_test(self, test_name: str) -> bool:
        """Check if test is typically emergency."""
        emergency_keywords = ["stat", "emergency", "urgent", "rapid", "troponin", "d-dimer", "critical"]
        return self._has_keywords(test_name, emergency_keywords)
        
    def _is_preventive_test(self, test_name: str) -> bool:
        """Check if test is preventive/screening."""
        preventive_keywords = ["screening", "preventive", "annual", "routine", "wellness", "check"]
        return self._has_keywords(test_name, preventive_keywords)
        
    def _requires_fasting(self, test_name: str) -> bool:
        """Check if test requires fasting."""
        fasting_tests = ["glucose", "lipid", "cholesterol", "triglyceride", "metabolic panel", "fasting"]
        return self._has_keywords(test_name, fasting_tests)


class PricingFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from pricing data with statistical and comparative analysis."""
    
    def __init__(self, include_percentile_features: bool = True,
                 include_outlier_detection: bool = True):
        self.include_percentile_features = include_percentile_features
        self.include_outlier_detection = include_outlier_detection
        self.price_stats = {}
        
    def fit(self, X, y=None):
        if isinstance(X, pd.Series):
            prices = X.values
        else:
            prices = np.array(X).flatten()
            
        # Calculate statistical measures
        self.price_stats = {
            'mean': np.mean(prices),
            'median': np.median(prices),
            'std': np.std(prices),
            'min': np.min(prices),
            'max': np.max(prices),
            'q25': np.percentile(prices, 25),
            'q75': np.percentile(prices, 75),
            'iqr': np.percentile(prices, 75) - np.percentile(prices, 25)
        }
        
        return self
        
    def transform(self, X):
        if isinstance(X, pd.Series):
            prices = X.values
        else:
            prices = np.array(X).flatten()
            
        features = []
        for price in prices:
            features.append(self._extract_price_features(float(price)))
            
        return pd.DataFrame(features)
    
    def _extract_price_features(self, price: float) -> Dict[str, Any]:
        """Extract comprehensive pricing features."""
        features = {}
        
        # Basic transformations
        features['price'] = price
        features['price_log'] = np.log1p(price) 
        features['price_sqrt'] = np.sqrt(max(price, 0))
        features['price_squared'] = price ** 2
        features['price_reciprocal'] = 1 / (price + 1e-8)
        
        # Price categorization
        features['price_tier'] = self._get_price_tier(price)
        features['price_magnitude'] = int(np.log10(max(price, 1)))
        features['is_round_number'] = price == int(price)
        features['price_last_digit'] = int(price) % 10
        features['is_common_price'] = price in [5, 10, 15, 20, 25, 30, 50, 100]
        
        # Statistical features
        if self.price_stats:
            features['price_z_score'] = (price - self.price_stats['mean']) / (self.price_stats['std'] + 1e-8)
            features['price_relative_to_median'] = price / (self.price_stats['median'] + 1e-8)
            features['price_relative_to_mean'] = price / (self.price_stats['mean'] + 1e-8)
            
            if self.include_percentile_features:
                # Calculate percentile position
                features['price_percentile'] = self._calculate_percentile(price)
                features['is_above_median'] = price > self.price_stats['median']
                features['is_above_q75'] = price > self.price_stats['q75']
                features['is_below_q25'] = price < self.price_stats['q25']
                
            if self.include_outlier_detection:
                features['is_price_outlier_zscore'] = abs(features['price_z_score']) > 3
                features['is_price_outlier_iqr'] = self._is_outlier_iqr(price)
                features['distance_from_median'] = abs(price - self.price_stats['median'])
                
        return features
    
    def _get_price_tier(self, price: float) -> int:
        """Categorize price into tiers."""
        if price < 10:
            return 1
        elif price < 25:
            return 2
        elif price < 50:
            return 3
        elif price < 100:
            return 4
        elif price < 250:
            return 5
        elif price < 500:
            return 6
        elif price < 1000:
            return 7
        else:
            return 8
            
    def _calculate_percentile(self, price: float) -> float:
        """Calculate price percentile position."""
        if not self.price_stats:
            return 50.0
        
        # Approximate percentile using normal distribution assumption
        z_score = (price - self.price_stats['mean']) / (self.price_stats['std'] + 1e-8)
        # Convert z-score to percentile (approximate)
        from scipy.stats import norm
        try:
            percentile = norm.cdf(z_score) * 100
            return min(max(percentile, 0), 100)
        except:
            return 50.0
    
    def _is_outlier_iqr(self, price: float) -> bool:
        """Detect outliers using IQR method."""
        if not self.price_stats:
            return False
            
        lower_bound = self.price_stats['q25'] - 1.5 * self.price_stats['iqr']
        upper_bound = self.price_stats['q75'] + 1.5 * self.price_stats['iqr']
        return price < lower_bound or price > upper_bound


class InteractionFeatureGenerator(BaseEstimator, TransformerMixin):
    """Generate interaction features between different feature groups."""
    
    def __init__(self, max_interactions: int = 20):
        self.max_interactions = max_interactions
        self.feature_columns = []
        
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_columns = X.columns.tolist()
        return self
        
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return X
            
        df = X.copy()
        interaction_features = {}
        
        # Price-complexity interactions
        if 'price' in df.columns and 'estimated_complexity_score' in df.columns:
            interaction_features['price_complexity_ratio'] = (
                df['price'] / (df['estimated_complexity_score'] + 0.1)
            )
            interaction_features['price_complexity_product'] = (
                df['price'] * df['estimated_complexity_score']
            )
            
        # CPT-price interactions
        if 'cpt_numeric' in df.columns and 'price' in df.columns:
            interaction_features['cpt_price_ratio'] = df['cpt_numeric'] / (df['price'] + 1)
            interaction_features['price_per_cpt_thousand'] = df['price'] / (df['cpt_numeric'] / 1000 + 1)
            
        # Name-price interactions
        if 'name_word_count' in df.columns and 'price' in df.columns:
            interaction_features['price_per_word'] = df['price'] / (df['name_word_count'] + 1)
            
        # Category-price interactions
        if 'price_tier' in df.columns and 'cpt_major_category' in df.columns:
            for category in df['cpt_major_category'].unique():
                mask = df['cpt_major_category'] == category
                interaction_features[f'is_{category}_high_price'] = (
                    mask & (df['price_tier'] >= 6)
                )
                
        # Add interaction features to dataframe
        for name, values in list(interaction_features.items())[:self.max_interactions]:
            df[name] = values
            
        return df


class ComprehensiveFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Complete feature engineering pipeline for CPT cost estimation.
    
    This transformer orchestrates all feature extraction components and provides
    a single interface for transforming raw CPT data into ML-ready features.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.fitted = False
        
        # Initialize component transformers
        self.cpt_extractor = CPTHierarchyExtractor()
        self.name_extractor = TestNameFeatureExtractor(
            include_semantic_features=True,
            include_complexity_features=True
        )
        self.price_extractor = PricingFeatureExtractor()
        self.interaction_generator = InteractionFeatureGenerator()
        
        # Text processing components
        if self.config.include_text_features:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config.max_tfidf_features,
                min_df=self.config.min_word_freq,
                ngram_range=self.config.ngram_range,
                lowercase=True,
                stop_words='english'
            )
        
        # Feature preprocessing components
        self.numerical_scaler = self._get_scaler(self.config.numerical_scaler)
        self.imputer = SimpleImputer(strategy='median')
        
        # Feature selection
        if self.config.enable_feature_selection:
            self.feature_selector = SelectKBest(
                score_func=f_regression,
                k=self.config.max_features or 'all'
            )
            
        # Cache for feature names and metadata
        self.feature_names_ = []
        self.feature_metadata_ = {}
        
    def fit(self, X, y=None):
        """
        Fit the feature engineering pipeline.
        
        Parameters
        ----------
        X : DataFrame with columns ['test_name', 'cpt_code', 'price']
        y : array-like, optional
            Target values for supervised feature selection
        """
        
        logger.info("Starting feature engineering fit process")
        
        # Validate input data
        X = self._validate_input(X)
        
        # Fit individual extractors
        self.cpt_extractor.fit(X['cpt_code'])
        self.name_extractor.fit(X['test_name'])
        self.price_extractor.fit(X['price'])
        
        # Generate initial features
        features_df = self._extract_all_features(X)
        
        # Fit text processing if enabled
        if self.config.include_text_features:
            self.tfidf_vectorizer.fit(X['test_name'])
            
        # Fit interaction generator
        self.interaction_generator.fit(features_df)
        
        # Get final feature set with interactions
        if self.config.include_interaction_features:
            features_df = self.interaction_generator.transform(features_df)
            
        # Separate numerical and categorical features
        numerical_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = features_df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Fit preprocessing components
        if numerical_features:
            numerical_data = features_df[numerical_features]
            
            # Handle missing values
            if self.config.handle_missing_values:
                numerical_data = self.imputer.fit_transform(numerical_data)
            
            # Scale numerical features
            self.numerical_scaler.fit(numerical_data)
            
        # Fit feature selection if enabled
        if self.config.enable_feature_selection and y is not None:
            final_features = self._prepare_final_features(features_df, X)
            self.feature_selector.fit(final_features, y)
            
        # Store feature metadata
        self.feature_names_ = features_df.columns.tolist()
        self.feature_metadata_ = self._generate_feature_metadata(features_df)
        
        self.fitted = True
        logger.info(f"Feature engineering fit complete. Generated {len(self.feature_names_)} features")
        
        return self
        
    def transform(self, X):
        """
        Transform input data into ML-ready features.
        
        Parameters
        ----------
        X : DataFrame with columns ['test_name', 'cpt_code', 'price']
        
        Returns
        -------
        numpy.ndarray
            Feature matrix ready for ML model training/prediction
        """
        
        if not self.fitted:
            raise ValueError("Feature engineering pipeline must be fitted before transform")
            
        # Validate input data
        X = self._validate_input(X)
        
        # Extract all features
        features_df = self._extract_all_features(X)
        
        # Add interaction features
        if self.config.include_interaction_features:
            features_df = self.interaction_generator.transform(features_df)
            
        # Prepare final feature matrix
        final_features = self._prepare_final_features(features_df, X)
        
        # Apply feature selection if enabled
        if self.config.enable_feature_selection:
            final_features = self.feature_selector.transform(final_features)
            
        return final_features
        
    def _validate_input(self, X):
        """Validate and clean input data."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        required_columns = ['test_name', 'cpt_code', 'price']
        missing_columns = [col for col in required_columns if col not in X.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Clean and validate data
        X = X.copy()
        X['test_name'] = X['test_name'].fillna('').astype(str)
        X['cpt_code'] = X['cpt_code'].fillna('').astype(str)
        X['price'] = pd.to_numeric(X['price'], errors='coerce').fillna(0)
        
        return X
        
    def _extract_all_features(self, X):
        """Extract features from all components."""
        feature_dfs = []
        
        # CPT code features
        cpt_features = self.cpt_extractor.transform(X['cpt_code'])
        feature_dfs.append(cpt_features)
        
        # Test name features
        name_features = self.name_extractor.transform(X['test_name'])
        feature_dfs.append(name_features)
        
        # Pricing features
        price_features = self.price_extractor.transform(X['price'])
        feature_dfs.append(price_features)
        
        # Combine all features
        combined_features = pd.concat(feature_dfs, axis=1)
        
        return combined_features
        
    def _prepare_final_features(self, features_df, original_X):
        """Prepare final feature matrix with preprocessing."""
        # Separate feature types
        numerical_features = features_df.select_dtypes(include=[np.number])
        categorical_features = features_df.select_dtypes(exclude=[np.number])
        
        feature_arrays = []
        
        # Process numerical features
        if not numerical_features.empty:
            numerical_data = numerical_features.values
            
            # Handle missing values
            if self.config.handle_missing_values:
                numerical_data = self.imputer.transform(numerical_data)
                
            # Remove outliers if configured
            if self.config.remove_outliers:
                numerical_data = self._remove_outliers(numerical_data)
                
            # Scale features
            numerical_data = self.numerical_scaler.transform(numerical_data)
            feature_arrays.append(numerical_data)
            
        # Process categorical features
        if not categorical_features.empty and self.config.categorical_encoding == "onehot":
            # Simple label encoding for now (OneHot would expand dimensions significantly)
            categorical_encoded = []
            for col in categorical_features.columns:
                le = LabelEncoder()
                try:
                    encoded = le.fit_transform(categorical_features[col].astype(str))
                    categorical_encoded.append(encoded.reshape(-1, 1))
                except:
                    # Fallback for problematic columns
                    categorical_encoded.append(np.zeros((len(categorical_features), 1)))
                    
            if categorical_encoded:
                feature_arrays.append(np.hstack(categorical_encoded))
                
        # Add text features if enabled
        if self.config.include_text_features:
            text_features = self.tfidf_vectorizer.transform(original_X['test_name'])
            feature_arrays.append(text_features.toarray())
            
        # Combine all feature arrays
        if feature_arrays:
            final_features = np.hstack(feature_arrays)
        else:
            final_features = np.array([]).reshape(len(features_df), 0)
            
        return final_features
        
    def _get_scaler(self, scaler_type: str):
        """Get scaler instance by type."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': StandardScaler()  # Placeholder
        }
        return scalers.get(scaler_type, StandardScaler())
        
    def _remove_outliers(self, data):
        """Remove outliers from numerical data."""
        if self.config.outlier_method == "zscore":
            z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
            mask = (z_scores < self.config.outlier_threshold).all(axis=1)
            return data[mask] if mask.any() else data
        elif self.config.outlier_method == "iqr":
            q25 = np.percentile(data, 25, axis=0)
            q75 = np.percentile(data, 75, axis=0)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            mask = ((data >= lower_bound) & (data <= upper_bound)).all(axis=1)
            return data[mask] if mask.any() else data
        else:
            return data
            
    def _generate_feature_metadata(self, features_df):
        """Generate metadata about features for interpretability."""
        metadata = {}
        
        for col in features_df.columns:
            col_data = features_df[col]
            metadata[col] = {
                'type': str(col_data.dtype),
                'missing_count': col_data.isna().sum(),
                'missing_percentage': col_data.isna().mean() * 100,
                'unique_values': col_data.nunique() if col_data.dtype == 'object' else None,
                'mean': col_data.mean() if pd.api.types.is_numeric_dtype(col_data) else None,
                'std': col_data.std() if pd.api.types.is_numeric_dtype(col_data) else None,
                'description': self._get_feature_description(col)
            }
            
        return metadata
        
    def _get_feature_description(self, feature_name: str) -> str:
        """Get human-readable description of feature."""
        descriptions = {
            'price': 'Original test price',
            'price_log': 'Natural log of price + 1',
            'cpt_numeric': 'Numeric value of CPT code',
            'estimated_complexity_score': 'Estimated complexity from 0-1',
            'name_word_count': 'Number of words in test name',
            'cpt_major_category': 'Major CPT category classification',
            'price_tier': 'Price category from 1-8',
            'is_imaging': 'Whether test involves imaging',
            'is_laboratory': 'Whether test is a laboratory test'
        }
        return descriptions.get(feature_name, f"Feature: {feature_name}")
        
    def get_feature_names(self):
        """Get list of feature names."""
        return self.feature_names_.copy()
        
    def get_feature_metadata(self):
        """Get metadata about all features."""
        return self.feature_metadata_.copy()


def create_ml_pipeline(config: Optional[FeatureConfig] = None) -> Pipeline:
    """
    Create a complete ML preprocessing pipeline.
    
    Parameters
    ----------
    config : FeatureConfig, optional
        Configuration for feature engineering
        
    Returns
    -------
    sklearn.pipeline.Pipeline
        Complete preprocessing pipeline
    """
    
    config = config or FeatureConfig()
    
    # Create feature engineering step
    feature_engineering = ComprehensiveFeatureEngineering(config)
    
    # Create complete pipeline
    pipeline = Pipeline([
        ('feature_engineering', feature_engineering)
    ])
    
    return pipeline


def demonstrate_feature_engineering():
    """Demonstrate the feature engineering pipeline with sample data."""
    
    # Create sample data
    sample_data = pd.DataFrame({
        'test_name': [
            'complete blood count',
            'comprehensive metabolic panel', 
            'mri brain with contrast',
            'chest x-ray 2 views',
            '24 hour holter monitor',
            'thyroid stimulating hormone',
            'lipid panel fasting',
            'ct head without contrast',
            'electrocardiogram 12 lead',
            'basic metabolic panel'
        ],
        'cpt_code': [
            '85027', '80053', '70553', '71046', '93224',
            '84443', '80061', '70450', '93000', '80048'
        ],
        'price': [9.0, 14.0, 400.0, 30.0, 200.0, 20.0, 15.0, 100.0, 10.0, 13.0]
    })
    
    print("=== ML Feature Engineering Demonstration ===\n")
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample data:\n{sample_data}\n")
    
    # Create and configure pipeline
    config = FeatureConfig(
        include_text_features=True,
        include_interaction_features=True,
        max_tfidf_features=50,
        enable_feature_selection=False
    )
    
    # Initialize feature engineering
    feature_engineer = ComprehensiveFeatureEngineering(config)
    
    # Fit and transform
    feature_engineer.fit(sample_data)
    features = feature_engineer.transform(sample_data)
    
    print(f"Generated feature matrix shape: {features.shape}")
    print(f"Number of features: {len(feature_engineer.get_feature_names())}")
    
    # Display feature metadata
    metadata = feature_engineer.get_feature_metadata()
    print(f"\nFeature categories:")
    
    feature_types = {}
    for name, meta in metadata.items():
        ftype = meta['type']
        if ftype not in feature_types:
            feature_types[ftype] = []
        feature_types[ftype].append(name)
        
    for ftype, names in feature_types.items():
        print(f"  {ftype}: {len(names)} features")
        
    # Show sample features
    print(f"\nSample features (first 10):")
    feature_names = feature_engineer.get_feature_names()
    for i, name in enumerate(feature_names[:10]):
        if name in metadata:
            print(f"  {name}: {metadata[name]['description']}")
            
    print(f"\nFeature engineering pipeline ready for ML model training!")


if __name__ == "__main__":
    demonstrate_feature_engineering()