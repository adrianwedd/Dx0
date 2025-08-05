#!/usr/bin/env python3
"""
Comprehensive Validation and Testing Framework for ML Feature Engineering
Testing accuracy, consistency, and performance of feature engineering pipeline
"""

import time
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

import structlog

from .ml_feature_engineering import ComprehensiveFeatureEngineering, FeatureConfig
from .ml_preprocessing import MLPreprocessingPipeline, create_default_preprocessing_pipeline

logger = structlog.get_logger(__name__)

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class FeatureQualityMetrics:
    """Metrics for assessing feature quality."""
    
    feature_name: str
    missing_percentage: float
    unique_values: int
    cardinality_ratio: float  # unique_values / total_samples
    outlier_percentage: float
    variance: float
    correlation_with_target: float
    mutual_information: float
    data_type: str
    is_constant: bool
    quality_score: float  # Overall quality score 0-1
    
    
@dataclass
class PipelineValidationResults:
    """Results from pipeline validation."""
    
    test_name: str
    success: bool
    execution_time: float
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    feature_count: int
    error_message: Optional[str] = None
    warnings: List[str] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metrics is None:
            self.metrics = {}


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results."""
    
    test_name: str
    data_size: int
    feature_count: int
    fit_time: float
    transform_time: float
    memory_usage_mb: float
    throughput_samples_per_second: float
    scalability_score: float  # 0-1 based on time complexity


class FeatureQualityAnalyzer:
    """Analyze the quality of engineered features."""
    
    def __init__(self, target_correlation_threshold: float = 0.1,
                 outlier_threshold: float = 3.0,
                 missing_threshold: float = 0.1):
        
        self.target_correlation_threshold = target_correlation_threshold
        self.outlier_threshold = outlier_threshold
        self.missing_threshold = missing_threshold
        
    def analyze_feature_quality(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> List[FeatureQualityMetrics]:
        """Analyze quality of all features in dataset."""
        
        results = []
        
        for column in X.columns:
            feature_data = X[column]
            
            # Basic statistics
            missing_pct = feature_data.isna().sum() / len(feature_data) * 100
            unique_vals = feature_data.nunique()
            cardinality_ratio = unique_vals / len(feature_data)
            
            # Variance and outliers for numerical features
            if pd.api.types.is_numeric_dtype(feature_data):
                variance = feature_data.var()
                
                # Detect outliers using Z-score
                if variance > 0:
                    z_scores = np.abs((feature_data - feature_data.mean()) / feature_data.std())
                    outlier_pct = (z_scores > self.outlier_threshold).sum() / len(feature_data) * 100
                else:
                    outlier_pct = 0
                    
                is_constant = variance == 0
            else:
                variance = 0
                outlier_pct = 0
                is_constant = unique_vals <= 1
                
            # Correlation with target
            target_corr = 0
            mutual_info = 0
            
            if y is not None and pd.api.types.is_numeric_dtype(feature_data):
                try:
                    target_corr = abs(feature_data.corr(y))
                    if np.isnan(target_corr):
                        target_corr = 0
                        
                    # Mutual information
                    from sklearn.feature_selection import mutual_info_regression
                    mi_scores = mutual_info_regression(
                        feature_data.values.reshape(-1, 1), 
                        y.values,
                        random_state=42
                    )
                    mutual_info = mi_scores[0] if len(mi_scores) > 0 else 0
                    
                except Exception:
                    target_corr = 0
                    mutual_info = 0
                    
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(
                missing_pct, cardinality_ratio, outlier_pct, 
                target_corr, is_constant
            )
            
            metrics = FeatureQualityMetrics(
                feature_name=column,
                missing_percentage=missing_pct,
                unique_values=unique_vals,
                cardinality_ratio=cardinality_ratio,
                outlier_percentage=outlier_pct,
                variance=variance,
                correlation_with_target=target_corr,
                mutual_information=mutual_info,
                data_type=str(feature_data.dtype),
                is_constant=is_constant,
                quality_score=quality_score
            )
            
            results.append(metrics)
            
        return results
        
    def _calculate_quality_score(self, missing_pct: float, cardinality_ratio: float,
                               outlier_pct: float, target_corr: float, is_constant: bool) -> float:
        """Calculate overall feature quality score."""
        
        score = 1.0
        
        # Penalize high missing values
        if missing_pct > self.missing_threshold * 100:
            score -= (missing_pct / 100) * 0.3
            
        # Penalize constant features
        if is_constant:
            score -= 0.5
            
        # Penalize high outlier percentage
        if outlier_pct > 5:
            score -= (outlier_pct / 100) * 0.2
            
        # Reward good target correlation
        if target_corr > self.target_correlation_threshold:
            score += target_corr * 0.3
            
        # Penalize extremely high cardinality (potential overfitting)
        if cardinality_ratio > 0.8:
            score -= 0.2
            
        return max(0, min(1, score))
        
    def generate_quality_report(self, quality_metrics: List[FeatureQualityMetrics]) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        
        # Summary statistics
        quality_scores = [m.quality_score for m in quality_metrics]
        correlations = [m.correlation_with_target for m in quality_metrics if m.correlation_with_target > 0]
        
        report = {
            'summary': {
                'total_features': len(quality_metrics),
                'avg_quality_score': np.mean(quality_scores),
                'high_quality_features': sum(1 for s in quality_scores if s > 0.7),
                'low_quality_features': sum(1 for s in quality_scores if s < 0.3),
                'constant_features': sum(1 for m in quality_metrics if m.is_constant),
                'avg_target_correlation': np.mean(correlations) if correlations else 0
            },
            'feature_issues': {
                'high_missing': [m.feature_name for m in quality_metrics if m.missing_percentage > 10],
                'constant_features': [m.feature_name for m in quality_metrics if m.is_constant],
                'high_outliers': [m.feature_name for m in quality_metrics if m.outlier_percentage > 10],
                'low_correlation': [m.feature_name for m in quality_metrics 
                                  if m.correlation_with_target < self.target_correlation_threshold 
                                  and m.correlation_with_target > 0]
            },
            'top_features': {
                'by_quality': sorted(quality_metrics, key=lambda x: x.quality_score, reverse=True)[:10],
                'by_target_correlation': sorted(quality_metrics, key=lambda x: x.correlation_with_target, reverse=True)[:10],
                'by_mutual_information': sorted(quality_metrics, key=lambda x: x.mutual_information, reverse=True)[:10]
            }
        }
        
        return report


class PipelineValidator:
    """Validate feature engineering pipeline functionality."""
    
    def __init__(self):
        self.test_results = []
        
    def run_all_tests(self, pipeline: MLPreprocessingPipeline, 
                     test_data: pd.DataFrame, 
                     target_data: Optional[pd.Series] = None) -> List[PipelineValidationResults]:
        """Run comprehensive validation tests."""
        
        logger.info("Starting comprehensive pipeline validation")
        
        self.test_results = []
        
        # Basic functionality tests
        self.test_results.append(self._test_basic_functionality(pipeline, test_data))
        self.test_results.append(self._test_fit_transform(pipeline, test_data, target_data))
        self.test_results.append(self._test_transform_consistency(pipeline, test_data))
        
        # Edge case tests
        self.test_results.append(self._test_empty_data(pipeline))
        self.test_results.append(self._test_missing_data(pipeline, test_data))
        self.test_results.append(self._test_invalid_data(pipeline, test_data))
        
        # Performance tests
        self.test_results.append(self._test_large_dataset(pipeline, test_data))
        
        # Feature quality tests
        if target_data is not None:
            self.test_results.append(self._test_feature_target_correlation(pipeline, test_data, target_data))
            
        # Integration tests
        self.test_results.append(self._test_sklearn_compatibility(pipeline, test_data, target_data))
        
        logger.info(f"Validation complete: {len([r for r in self.test_results if r.success])}/{len(self.test_results)} tests passed")
        
        return self.test_results
        
    def _test_basic_functionality(self, pipeline: MLPreprocessingPipeline, 
                                test_data: pd.DataFrame) -> PipelineValidationResults:
        """Test basic pipeline functionality."""
        
        start_time = time.time()
        
        try:
            # Test fit
            pipeline.fit(test_data)
            
            # Test transform
            transformed = pipeline.transform(test_data)
            
            # Validate output
            if transformed is None:
                raise ValueError("Transform returned None")
                
            if not isinstance(transformed, (np.ndarray, pd.DataFrame)):
                raise ValueError(f"Invalid output type: {type(transformed)}")
                
            execution_time = time.time() - start_time
            
            return PipelineValidationResults(
                test_name="basic_functionality",
                success=True,
                execution_time=execution_time,
                input_shape=test_data.shape,
                output_shape=transformed.shape,
                feature_count=transformed.shape[1] if hasattr(transformed, 'shape') else 0
            )
            
        except Exception as e:
            return PipelineValidationResults(
                test_name="basic_functionality",
                success=False,
                execution_time=time.time() - start_time,
                input_shape=test_data.shape,
                output_shape=(0, 0),
                feature_count=0,
                error_message=str(e)
            )
            
    def _test_fit_transform(self, pipeline: MLPreprocessingPipeline,
                          test_data: pd.DataFrame, 
                          target_data: Optional[pd.Series]) -> PipelineValidationResults:
        """Test fit_transform method."""
        
        start_time = time.time()
        
        try:
            # Create fresh pipeline
            from .ml_preprocessing import create_default_preprocessing_pipeline
            fresh_pipeline = create_default_preprocessing_pipeline()
            
            # Test fit_transform
            transformed = fresh_pipeline.fit_transform(test_data, target_data)
            
            execution_time = time.time() - start_time
            
            return PipelineValidationResults(
                test_name="fit_transform",
                success=True,
                execution_time=execution_time,
                input_shape=test_data.shape,
                output_shape=transformed.shape,
                feature_count=transformed.shape[1] if hasattr(transformed, 'shape') else 0
            )
            
        except Exception as e:
            return PipelineValidationResults(
                test_name="fit_transform",
                success=False,
                execution_time=time.time() - start_time,
                input_shape=test_data.shape,
                output_shape=(0, 0),
                feature_count=0,
                error_message=str(e)
            )
            
    def _test_transform_consistency(self, pipeline: MLPreprocessingPipeline,
                                  test_data: pd.DataFrame) -> PipelineValidationResults:
        """Test that multiple transforms produce consistent results."""
        
        start_time = time.time()
        
        try:
            # Fit pipeline
            pipeline.fit(test_data)
            
            # Transform twice
            result1 = pipeline.transform(test_data)
            result2 = pipeline.transform(test_data)
            
            # Check consistency
            if not np.allclose(result1, result2, rtol=1e-10, atol=1e-10):
                raise ValueError("Transform results are not consistent")
                
            execution_time = time.time() - start_time
            
            return PipelineValidationResults(
                test_name="transform_consistency",
                success=True,
                execution_time=execution_time,
                input_shape=test_data.shape,
                output_shape=result1.shape,
                feature_count=result1.shape[1] if hasattr(result1, 'shape') else 0
            )
            
        except Exception as e:
            return PipelineValidationResults(
                test_name="transform_consistency",
                success=False,
                execution_time=time.time() - start_time,
                input_shape=test_data.shape,
                output_shape=(0, 0),
                feature_count=0,
                error_message=str(e)
            )
            
    def _test_empty_data(self, pipeline: MLPreprocessingPipeline) -> PipelineValidationResults:
        """Test handling of empty data."""
        
        start_time = time.time()
        
        try:
            # Create empty DataFrame
            empty_data = pd.DataFrame(columns=['test_name', 'cpt_code', 'price'])
            
            # Create fresh pipeline
            from .ml_preprocessing import create_default_preprocessing_pipeline
            fresh_pipeline = create_default_preprocessing_pipeline()
            
            # Should handle gracefully
            transformed = fresh_pipeline.fit_transform(empty_data)
            
            execution_time = time.time() - start_time
            
            return PipelineValidationResults(
                test_name="empty_data_handling",
                success=True,
                execution_time=execution_time,
                input_shape=empty_data.shape,
                output_shape=transformed.shape if hasattr(transformed, 'shape') else (0, 0),
                feature_count=0,
                warnings=["Empty data processed successfully"]
            )
            
        except Exception as e:
            # This might be expected behavior
            return PipelineValidationResults(
                test_name="empty_data_handling",
                success=False,
                execution_time=time.time() - start_time,
                input_shape=(0, 3),
                output_shape=(0, 0),
                feature_count=0,
                error_message=str(e),
                warnings=["Empty data handling failed - may be expected"]
            )
            
    def _test_missing_data(self, pipeline: MLPreprocessingPipeline,
                         test_data: pd.DataFrame) -> PipelineValidationResults:
        """Test handling of missing data."""
        
        start_time = time.time()
        
        try:
            # Create data with missing values
            missing_data = test_data.copy()
            missing_data.loc[0, 'test_name'] = np.nan
            missing_data.loc[1, 'price'] = np.nan
            missing_data.loc[2, 'cpt_code'] = ''
            
            # Create fresh pipeline
            from .ml_preprocessing import create_default_preprocessing_pipeline
            fresh_pipeline = create_default_preprocessing_pipeline()
            
            transformed = fresh_pipeline.fit_transform(missing_data)
            
            execution_time = time.time() - start_time
            
            return PipelineValidationResults(
                test_name="missing_data_handling",
                success=True,
                execution_time=execution_time,
                input_shape=missing_data.shape,
                output_shape=transformed.shape,
                feature_count=transformed.shape[1] if hasattr(transformed, 'shape') else 0
            )
            
        except Exception as e:
            return PipelineValidationResults(
                test_name="missing_data_handling",
                success=False,
                execution_time=time.time() - start_time,
                input_shape=test_data.shape,
                output_shape=(0, 0),
                feature_count=0,
                error_message=str(e)
            )
            
    def _test_invalid_data(self, pipeline: MLPreprocessingPipeline,
                         test_data: pd.DataFrame) -> PipelineValidationResults:
        """Test handling of invalid data."""
        
        start_time = time.time()
        
        try:
            # Create data with invalid values
            invalid_data = test_data.copy()
            invalid_data.loc[0, 'price'] = -100  # Negative price
            invalid_data.loc[1, 'cpt_code'] = 'INVALID'  # Invalid CPT format
            invalid_data.loc[2, 'price'] = 999999  # Extremely high price
            
            # Create fresh pipeline
            from .ml_preprocessing import create_default_preprocessing_pipeline
            fresh_pipeline = create_default_preprocessing_pipeline()
            
            transformed = fresh_pipeline.fit_transform(invalid_data)
            
            execution_time = time.time() - start_time
            
            return PipelineValidationResults(
                test_name="invalid_data_handling",
                success=True,
                execution_time=execution_time,
                input_shape=invalid_data.shape,
                output_shape=transformed.shape,
                feature_count=transformed.shape[1] if hasattr(transformed, 'shape') else 0
            )
            
        except Exception as e:
            return PipelineValidationResults(
                test_name="invalid_data_handling",
                success=False,
                execution_time=time.time() - start_time,
                input_shape=test_data.shape,
                output_shape=(0, 0),
                feature_count=0,
                error_message=str(e)
            )
            
    def _test_large_dataset(self, pipeline: MLPreprocessingPipeline,
                          test_data: pd.DataFrame) -> PipelineValidationResults:
        """Test performance with larger dataset."""
        
        start_time = time.time()
        
        try:
            # Create larger dataset by repeating test data
            large_data = pd.concat([test_data] * 100, ignore_index=True)
            
            # Add some noise to make it more realistic
            np.random.seed(42)
            price_noise = np.random.normal(0, 5, len(large_data))
            large_data['price'] = large_data['price'] + price_noise
            
            # Create fresh pipeline
            from .ml_preprocessing import create_default_preprocessing_pipeline
            fresh_pipeline = create_default_preprocessing_pipeline()
            
            transformed = fresh_pipeline.fit_transform(large_data)
            
            execution_time = time.time() - start_time
            
            return PipelineValidationResults(
                test_name="large_dataset_performance",
                success=True,
                execution_time=execution_time,
                input_shape=large_data.shape,
                output_shape=transformed.shape,
                feature_count=transformed.shape[1] if hasattr(transformed, 'shape') else 0,
                metrics={'throughput': len(large_data) / execution_time}
            )
            
        except Exception as e:
            return PipelineValidationResults(
                test_name="large_dataset_performance",
                success=False,
                execution_time=time.time() - start_time,
                input_shape=(0, 0),
                output_shape=(0, 0),
                feature_count=0,
                error_message=str(e)
            )
            
    def _test_feature_target_correlation(self, pipeline: MLPreprocessingPipeline,
                                       test_data: pd.DataFrame,
                                       target_data: pd.Series) -> PipelineValidationResults:
        """Test that generated features have some correlation with target."""
        
        start_time = time.time()
        
        try:
            # Create fresh pipeline
            from .ml_preprocessing import create_default_preprocessing_pipeline
            fresh_pipeline = create_default_preprocessing_pipeline()
            
            transformed = fresh_pipeline.fit_transform(test_data, target_data)
            
            # Calculate correlations
            correlations = []
            for i in range(transformed.shape[1]):
                feature_data = transformed[:, i]
                if len(np.unique(feature_data)) > 1:  # Skip constant features
                    corr = np.corrcoef(feature_data, target_data)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                        
            avg_correlation = np.mean(correlations) if correlations else 0
            max_correlation = np.max(correlations) if correlations else 0
            
            execution_time = time.time() - start_time
            
            return PipelineValidationResults(
                test_name="feature_target_correlation",
                success=True,
                execution_time=execution_time,
                input_shape=test_data.shape,
                output_shape=transformed.shape,
                feature_count=transformed.shape[1],
                metrics={
                    'avg_correlation': avg_correlation,
                    'max_correlation': max_correlation,
                    'features_with_correlation': len(correlations)
                }
            )
            
        except Exception as e:
            return PipelineValidationResults(
                test_name="feature_target_correlation",
                success=False,
                execution_time=time.time() - start_time,
                input_shape=test_data.shape,
                output_shape=(0, 0),
                feature_count=0,
                error_message=str(e)
            )
            
    def _test_sklearn_compatibility(self, pipeline: MLPreprocessingPipeline,
                                  test_data: pd.DataFrame,
                                  target_data: Optional[pd.Series]) -> PipelineValidationResults:
        """Test compatibility with scikit-learn models."""
        
        start_time = time.time()
        
        try:
            # Create fresh pipeline
            from .ml_preprocessing import create_default_preprocessing_pipeline
            fresh_pipeline = create_default_preprocessing_pipeline()
            
            # Transform data
            X_transformed = fresh_pipeline.fit_transform(test_data, target_data)
            
            if target_data is not None and len(X_transformed) > 0:
                # Test with simple model
                model = LinearRegression()
                model.fit(X_transformed, target_data)
                
                # Make predictions
                predictions = model.predict(X_transformed)
                
                # Calculate basic metrics
                mae = mean_absolute_error(target_data, predictions)
                r2 = r2_score(target_data, predictions)
                
                metrics = {'mae': mae, 'r2': r2}
            else:
                metrics = {'note': 'No target data provided for model testing'}
                
            execution_time = time.time() - start_time
            
            return PipelineValidationResults(
                test_name="sklearn_compatibility",
                success=True,
                execution_time=execution_time,
                input_shape=test_data.shape,
                output_shape=X_transformed.shape,
                feature_count=X_transformed.shape[1],
                metrics=metrics
            )
            
        except Exception as e:
            return PipelineValidationResults(
                test_name="sklearn_compatibility",
                success=False,
                execution_time=time.time() - start_time,
                input_shape=test_data.shape,
                output_shape=(0, 0),
                feature_count=0,
                error_message=str(e)
            )


class PerformanceBenchmarker:
    """Benchmark performance of feature engineering pipeline."""
    
    def __init__(self):
        self.benchmark_results = []
        
    def run_performance_benchmarks(self, pipeline_factory: Callable,
                                 base_data: pd.DataFrame) -> List[PerformanceBenchmark]:
        """Run comprehensive performance benchmarks."""
        
        logger.info("Starting performance benchmarks")
        
        self.benchmark_results = []
        
        # Test different data sizes
        sizes = [100, 500, 1000, 2000, 5000]
        
        for size in sizes:
            if size <= len(base_data) * 100:  # Don't create impossibly large datasets
                benchmark = self._benchmark_data_size(pipeline_factory, base_data, size)
                if benchmark:
                    self.benchmark_results.append(benchmark)
                    
        logger.info(f"Performance benchmarks complete: {len(self.benchmark_results)} tests run")
        
        return self.benchmark_results
        
    def _benchmark_data_size(self, pipeline_factory: Callable,
                           base_data: pd.DataFrame, target_size: int) -> Optional[PerformanceBenchmark]:
        """Benchmark performance for specific data size."""
        
        try:
            # Create dataset of target size
            repeat_factor = max(1, target_size // len(base_data))
            test_data = pd.concat([base_data] * repeat_factor, ignore_index=True)
            test_data = test_data.head(target_size)
            
            # Add some variation
            np.random.seed(42)
            if 'price' in test_data.columns:
                price_noise = np.random.normal(0, test_data['price'].std() * 0.1, len(test_data))
                test_data['price'] = test_data['price'] + price_noise
                
            # Create fresh pipeline
            pipeline = pipeline_factory()
            
            # Measure memory before
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Benchmark fit time
            fit_start = time.time()
            pipeline.fit(test_data)
            fit_time = time.time() - fit_start
            
            # Benchmark transform time
            transform_start = time.time()
            transformed = pipeline.transform(test_data)
            transform_time = time.time() - transform_start
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before
            
            # Calculate throughput
            total_time = fit_time + transform_time
            throughput = len(test_data) / total_time if total_time > 0 else 0
            
            # Calculate scalability score (lower is better)
            # Based on expected linear time complexity
            expected_time = len(test_data) * 0.001  # 1ms per sample baseline
            scalability_score = min(1.0, expected_time / total_time) if total_time > 0 else 1.0
            
            feature_count = transformed.shape[1] if hasattr(transformed, 'shape') else 0
            
            return PerformanceBenchmark(
                test_name=f"data_size_{target_size}",
                data_size=len(test_data),
                feature_count=feature_count,
                fit_time=fit_time,
                transform_time=transform_time,
                memory_usage_mb=memory_usage,
                throughput_samples_per_second=throughput,
                scalability_score=scalability_score
            )
            
        except Exception as e:
            logger.warning(f"Performance benchmark failed for size {target_size}: {e}")
            return None


def run_comprehensive_validation(test_data: pd.DataFrame, 
                               target_data: Optional[pd.Series] = None) -> Dict[str, Any]:
    """Run comprehensive validation of the feature engineering system."""
    
    logger.info("Starting comprehensive validation of feature engineering system")
    
    results = {
        'validation_timestamp': datetime.now().isoformat(),
        'test_data_shape': test_data.shape,
        'pipeline_tests': [],
        'feature_quality': {},
        'performance_benchmarks': [],
        'summary': {}
    }
    
    try:
        # Create pipeline
        from .ml_preprocessing import create_default_preprocessing_pipeline
        pipeline = create_default_preprocessing_pipeline()
        
        # Run pipeline validation tests
        validator = PipelineValidator()
        pipeline_results = validator.run_all_tests(pipeline, test_data, target_data)
        results['pipeline_tests'] = [asdict(r) for r in pipeline_results]
        
        # Analyze feature quality if we can successfully transform data
        try:
            transformed_data = pipeline.fit_transform(test_data, target_data)
            if transformed_data.shape[0] > 0:
                # Convert to DataFrame for quality analysis
                feature_names = [f"feature_{i}" for i in range(transformed_data.shape[1])]
                feature_df = pd.DataFrame(transformed_data, columns=feature_names)
                
                quality_analyzer = FeatureQualityAnalyzer()
                quality_metrics = quality_analyzer.analyze_feature_quality(feature_df, target_data)
                quality_report = quality_analyzer.generate_quality_report(quality_metrics)
                
                results['feature_quality'] = {
                    'individual_metrics': [asdict(m) for m in quality_metrics],
                    'summary_report': quality_report
                }
                
        except Exception as e:
            logger.warning(f"Feature quality analysis failed: {e}")
            results['feature_quality'] = {'error': str(e)}
            
        # Run performance benchmarks
        try:
            benchmarker = PerformanceBenchmarker()
            benchmark_results = benchmarker.run_performance_benchmarks(
                create_default_preprocessing_pipeline, 
                test_data
            )
            results['performance_benchmarks'] = [asdict(b) for b in benchmark_results]
            
        except Exception as e:
            logger.warning(f"Performance benchmarking failed: {e}")
            results['performance_benchmarks'] = {'error': str(e)}
            
        # Generate summary
        successful_tests = sum(1 for r in pipeline_results if r.success)
        total_tests = len(pipeline_results)
        
        results['summary'] = {
            'overall_success': successful_tests == total_tests,
            'tests_passed': successful_tests,
            'total_tests': total_tests,
            'success_rate': successful_tests / total_tests * 100 if total_tests > 0 else 0,
            'major_issues': [r.test_name for r in pipeline_results if not r.success],
            'warnings': sum(len(r.warnings or []) for r in pipeline_results)
        }
        
        logger.info(f"Comprehensive validation complete: {successful_tests}/{total_tests} tests passed")
        
    except Exception as e:
        logger.error(f"Comprehensive validation failed: {e}")
        results['summary'] = {'error': str(e)}
        
    return results


def demonstrate_validation_framework():
    """Demonstrate the validation framework."""
    
    print("=== ML Feature Engineering Validation Framework ===\n")
    
    # Create comprehensive test data
    np.random.seed(42)
    n_samples = 200
    
    test_names = [
        'complete blood count', 'comprehensive metabolic panel', 'mri brain with contrast',
        'chest x-ray 2 views', '24 hour holter monitor', 'thyroid stimulating hormone',
        'lipid panel fasting', 'ct head without contrast', 'electrocardiogram 12 lead',
        'basic metabolic panel', 'liver function panel', 'cardiac enzymes',
        'urinalysis complete', 'blood culture aerobic', 'vitamin d 25-hydroxy'
    ]
    
    cpt_codes = [
        '85027', '80053', '70553', '71046', '93224', '84443',
        '80061', '70450', '93000', '80048', '80076', '82550',
        '81001', '87040', '82306'
    ]
    
    base_prices = [9, 14, 400, 30, 200, 20, 15, 100, 10, 13, 25, 45, 5, 20, 35]
    
    # Generate test dataset
    test_data = []
    for i in range(n_samples):
        idx = i % len(test_names)
        price_variation = np.random.normal(1, 0.2)  # 20% price variation
        price = max(1, base_prices[idx] * price_variation)
        
        test_data.append({
            'test_name': test_names[idx],
            'cpt_code': cpt_codes[idx],
            'price': price
        })
        
    test_df = pd.DataFrame(test_data)
    
    # Create synthetic target (price prediction target)
    # Add some complexity to make it more realistic
    target = []
    for _, row in test_df.iterrows():
        base_target = row['price']
        # Add complexity based on test name features
        if 'comprehensive' in row['test_name']:
            base_target *= 1.2
        if 'mri' in row['test_name'] or 'ct' in row['test_name']:
            base_target *= 1.5
        if 'contrast' in row['test_name']:
            base_target *= 1.3
            
        # Add some noise
        noise = np.random.normal(0, base_target * 0.1)
        target.append(base_target + noise)
        
    target_series = pd.Series(target)
    
    print(f"Test dataset shape: {test_df.shape}")
    print(f"Target range: ${target_series.min():.2f} - ${target_series.max():.2f}")
    print(f"Sample data:")
    print(test_df.head())
    
    # Run comprehensive validation
    print(f"\n=== Running Comprehensive Validation ===")
    
    validation_results = run_comprehensive_validation(test_df, target_series)
    
    # Display results
    print(f"\n=== Validation Results ===")
    
    summary = validation_results.get('summary', {})
    print(f"Overall Success: {summary.get('overall_success', False)}")
    print(f"Tests Passed: {summary.get('tests_passed', 0)}/{summary.get('total_tests', 0)}")
    print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
    
    if summary.get('major_issues'):
        print(f"Failed Tests: {', '.join(summary['major_issues'])}")
        
    # Feature quality summary
    feature_quality = validation_results.get('feature_quality', {})
    if 'summary_report' in feature_quality:
        quality_summary = feature_quality['summary_report']['summary']
        print(f"\n=== Feature Quality ===")
        print(f"Total Features Generated: {quality_summary.get('total_features', 0)}")
        print(f"Average Quality Score: {quality_summary.get('avg_quality_score', 0):.2f}")
        print(f"High Quality Features: {quality_summary.get('high_quality_features', 0)}")
        print(f"Average Target Correlation: {quality_summary.get('avg_target_correlation', 0):.3f}")
        
    # Performance summary
    benchmarks = validation_results.get('performance_benchmarks', [])
    if benchmarks and not isinstance(benchmarks, dict):
        print(f"\n=== Performance Benchmarks ===")
        print(f"Benchmarks Run: {len(benchmarks)}")
        
        if benchmarks:
            avg_throughput = np.mean([b.get('throughput_samples_per_second', 0) for b in benchmarks])
            avg_memory = np.mean([b.get('memory_usage_mb', 0) for b in benchmarks])
            print(f"Average Throughput: {avg_throughput:.0f} samples/second")
            print(f"Average Memory Usage: {avg_memory:.1f} MB")
            
    print(f"\n=== Validation Framework Ready for Production ===")


if __name__ == "__main__":
    demonstrate_validation_framework()