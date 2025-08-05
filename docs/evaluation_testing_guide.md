# Evaluation Testing Guide for Dx0

This comprehensive guide covers the evaluation testing suite for the Dx0 diagnostic system, including metrics calculation, statistical analysis, performance benchmarking, and quality assurance methodologies.

## Overview

The Dx0 evaluation testing suite provides comprehensive validation of diagnostic system performance through:

- **Diagnostic accuracy metrics** (precision, recall, sensitivity, specificity)
- **Statistical analysis** (significance testing, confidence intervals, effect sizes)
- **Cost-benefit analysis** (cost-effectiveness ratios, Pareto frontiers)
- **Performance benchmarking** (throughput, latency, resource usage)
- **Integration testing** (Judge-CostEstimator integration, ensemble methods)
- **Longitudinal tracking** (performance drift detection, trend analysis)

## Test Modules

### 1. Core Evaluation Metrics (`test_evaluation_metrics.py`)

#### Diagnostic Accuracy Testing
Tests comprehensive diagnostic accuracy metrics including:

```python
# Example usage
from tests.test_evaluation_metrics import TestDiagnosticAccuracyMetrics

test_class = TestDiagnosticAccuracyMetrics()
test_class.test_precision_recall_calculation()
test_class.test_sensitivity_specificity_calculation()
test_class.test_multiclass_accuracy_metrics()
```

**Key Metrics Tested:**
- Precision, Recall, F1-score
- Sensitivity, Specificity  
- Multiclass accuracy metrics
- Confusion matrix generation
- ROC AUC calculation

#### Performance Metrics Testing
Validates diagnostic performance across different dimensions:

```python
# Cost-effectiveness analysis
test_class.test_cost_effectiveness_calculation()
test_class.test_time_efficiency_metrics()
test_class.test_diagnostic_consistency_metrics()
```

**Performance Dimensions:**
- Cost-effectiveness ratios
- Time efficiency metrics
- Diagnostic consistency across sessions
- Resource utilization patterns

### 2. Statistical Analysis (`test_statistical_analysis.py`)

#### Significance Testing
Comprehensive statistical significance testing methods:

```python
# Statistical tests
from tests.test_statistical_analysis import TestSignificanceTesting

test_class = TestSignificanceTesting()
test_class.test_t_test_implementation()
test_class.test_chi_square_test()
test_class.test_mann_whitney_u_test()
```

**Statistical Tests Covered:**
- Independent and paired t-tests
- Chi-square tests for categorical data
- Mann-Whitney U test (nonparametric)
- Kruskal-Wallis test (multiple groups)
- Permutation tests with validation

#### Confidence Intervals and Effect Sizes
Robust estimation of confidence intervals and effect sizes:

```python
# Confidence intervals
test_class = TestConfidenceIntervals()
test_class.test_mean_confidence_interval()
test_class.test_bootstrap_confidence_interval()

# Effect sizes
test_class = TestEffectSizeCalculations()
test_class.test_cohens_d_calculation()
test_class.test_hedges_g_calculation()
```

**Methods Included:**
- Parametric and bootstrap confidence intervals
- Cohen's d, Hedges' g, Glass's delta
- R-squared and Cramér's V
- Multiple comparison corrections (Bonferroni, FDR, Holm)

### 3. Evaluation Frameworks (`test_evaluation_frameworks.py`)

#### Diagnostic Accuracy Framework
Comprehensive framework for accuracy assessment:

```python
from tests.test_evaluation_frameworks import DiagnosticAccuracyFramework

framework = DiagnosticAccuracyFramework()
metrics = framework.calculate_accuracy_metrics(true_labels, pred_labels, positive_class)
complexity_metrics = framework.evaluate_by_complexity(evaluation_results)
```

**Framework Features:**
- Multi-class diagnostic accuracy
- Complexity-stratified evaluation
- ROC AUC calculation
- Matthews correlation coefficient

#### Cost-Benefit Analysis Framework
Advanced cost-benefit analysis capabilities:

```python
from tests.test_evaluation_frameworks import CostBenefitAnalysisFramework

framework = CostBenefitAnalysisFramework()
cost_metrics = framework.calculate_cost_metrics(session_results)
incremental_analysis = framework.calculate_incremental_analysis(baseline, intervention)
pareto_strategies = framework.pareto_frontier_analysis(strategies)
```

**Analysis Features:**
- Incremental cost-effectiveness ratios (ICER)
- Net monetary benefit calculations
- Pareto efficiency analysis
- Budget impact modeling

### 4. Performance Testing (`test_evaluation_performance.py`)

#### Performance Benchmarking
Comprehensive performance validation under various conditions:

```python
from tests.test_evaluation_performance import TestEvaluationPerformance

test_class = TestEvaluationPerformance()
test_class.test_large_scale_evaluation_performance()
test_class.test_concurrent_evaluation_performance()
test_class.test_memory_efficiency_under_load()
```

**Performance Areas:**
- Single evaluation latency
- Batch processing throughput
- Memory usage optimization
- CPU efficiency metrics
- Scalability with test count

#### Integration Testing
Validation of component integration:

```python
test_class = TestEvaluationIntegration()
test_class.test_judge_cost_estimator_integration()
test_class.test_ensemble_evaluation_integration()
test_class.test_real_world_workflow_integration()
```

**Integration Scenarios:**
- Judge-CostEstimator coordination
- Ensemble voting systems
- Asynchronous evaluation workflows
- Error handling and recovery

### 5. Test Data Generation (`test_evaluation_data.py`)

#### Realistic Test Case Generation
Generates comprehensive test datasets:

```python
from tests.test_evaluation_data import TestDataGenerator

generator = TestDataGenerator(seed=42)
test_cases = generator.generate_diagnostic_cases(n_cases=100)
evaluation_results = generator.generate_evaluation_results(test_cases)
benchmark_data = generator.generate_performance_benchmark_data("large")
```

**Data Generation Features:**
- Realistic diagnostic cases with complexity levels
- Ground truth labels with known properties
- Statistical validation datasets
- Performance benchmarking scenarios
- Edge case generation

## Running the Tests

### Quick Start

Run all evaluation tests:
```bash
pytest tests/test_evaluation_*.py -v
```

Run specific test categories:
```bash
# Metrics testing only
pytest tests/test_evaluation_metrics.py -v

# Statistical analysis only  
pytest tests/test_statistical_analysis.py -v

# Performance testing only
pytest tests/test_evaluation_performance.py -v
```

### Test Configuration

Configure test parameters via environment variables:
```bash
export EVALUATION_SEED=42
export PERFORMANCE_TIMEOUT=300
export MEMORY_LIMIT_MB=1000

pytest tests/test_evaluation_*.py
```

### Parallel Execution

Run tests in parallel for faster execution:
```bash
pytest tests/test_evaluation_*.py -n auto --dist worksteal
```

## Understanding Test Results

### Diagnostic Accuracy Results

Example accuracy metrics output:
```python
DiagnosticAccuracyMetrics(
    sensitivity=0.85,           # 85% of positive cases correctly identified
    specificity=0.90,           # 90% of negative cases correctly identified  
    positive_predictive_value=0.89,  # 89% precision for positive predictions
    negative_predictive_value=0.87,  # 87% precision for negative predictions
    accuracy=0.875,             # 87.5% overall accuracy
    f1_score=0.87,              # Harmonic mean of precision and recall
    balanced_accuracy=0.875,    # Average of sensitivity and specificity
    matthews_correlation=0.75   # Matthews correlation coefficient
)
```

### Statistical Analysis Results

Example statistical test output:
```python
StatisticalTestResult(
    statistic=2.85,             # Test statistic value
    p_value=0.004,              # Probability of observing this difference by chance
    effect_size=0.67,           # Cohen's d effect size (medium effect)
    confidence_interval=(0.02, 0.18),  # 95% CI for the difference
    test_name="independent_t_test",
    interpretation="statistically_significant"
)
```

### Performance Metrics Results

Example performance benchmark:
```python
PerformanceMetrics(
    execution_time=2.45,        # Seconds to complete operation
    memory_usage_mb=156.7,      # Peak memory usage in MB
    cpu_percent=78.3,           # CPU utilization percentage
    throughput_ops_per_sec=89.2, # Operations processed per second
    peak_memory_mb=189.4        # Peak memory during operation
)
```

## Best Practices

### Test Design

1. **Use Fixed Seeds**: Always use fixed random seeds for reproducible results
2. **Test Edge Cases**: Include edge cases like empty data, extreme values
3. **Validate Assumptions**: Check statistical assumptions before applying tests
4. **Monitor Performance**: Set performance thresholds and monitor regularly

### Statistical Analysis

1. **Check Sample Sizes**: Ensure adequate power for statistical tests
2. **Correct for Multiple Comparisons**: Apply appropriate corrections
3. **Report Effect Sizes**: Include practical significance alongside statistical significance
4. **Use Appropriate Tests**: Match statistical tests to data characteristics

### Performance Testing

1. **Isolate Components**: Test individual components and integrated systems
2. **Use Realistic Data**: Test with data representative of production workloads
3. **Monitor Resources**: Track memory, CPU, and I/O usage
4. **Test Failure Scenarios**: Validate error handling and recovery

## Troubleshooting

### Common Issues

#### Test Failures Due to Random Variation
```python
# Problem: Tests fail due to random statistical variation
# Solution: Use larger sample sizes or wider tolerance ranges

# Bad
assert p_value < 0.05  # Might fail due to randomness

# Good  
assert p_value < 0.05 or abs(p_value - 0.05) < 0.01  # Allow small tolerance
```

#### Memory Issues in Performance Tests
```python
# Problem: Tests consume too much memory
# Solution: Use smaller datasets or implement streaming

# Bad
large_dataset = generate_cases(100000)  # May exhaust memory

# Good
for batch in generate_cases_in_batches(1000, total=100000):
    process_batch(batch)
```

#### Statistical Test Sensitivity
```python
# Problem: Tests too sensitive to small differences
# Solution: Set appropriate thresholds based on clinical significance

# Bad
assert abs(accuracy_diff) > 0.001  # Too sensitive

# Good
assert abs(accuracy_diff) > 0.05  # Clinically meaningful difference
```

### Performance Optimization

#### Optimize Test Execution
```bash
# Use markers to run specific test categories
pytest -m "not slow" tests/test_evaluation_*.py

# Run tests in parallel
pytest tests/test_evaluation_*.py -n 4

# Skip performance tests during development
pytest tests/test_evaluation_*.py -k "not performance"
```

#### Memory Management
```python
# Clean up resources in tests
def test_large_evaluation():
    try:
        # Test logic here
        pass
    finally:
        gc.collect()  # Force garbage collection
```

## Integration with CI/CD

### GitHub Actions Configuration

```yaml
name: Evaluation Tests
on: [push, pull_request]

jobs:
  evaluation-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-xdist
    
    - name: Run evaluation tests
      run: |
        pytest tests/test_evaluation_*.py -v --tb=short
        
    - name: Run performance tests
      run: |
        pytest tests/test_evaluation_performance.py -v --tb=short
      if: github.event_name == 'push'  # Only on push, not PR
```

### Performance Monitoring

Set up continuous performance monitoring:
```python
# performance_monitor.py
def monitor_evaluation_performance():
    """Monitor evaluation system performance over time."""
    
    # Run benchmark suite
    benchmark_results = run_performance_benchmarks()
    
    # Compare against baselines
    regression_detected = check_performance_regression(benchmark_results)
    
    if regression_detected:
        alert_team("Performance regression detected in evaluation system")
    
    # Store results for trending
    store_performance_metrics(benchmark_results)
```

## Advanced Usage

### Custom Metrics Implementation

Implement custom evaluation metrics:
```python
from tests.test_evaluation_metrics import TestDiagnosticAccuracyMetrics

class CustomMetricsTest(TestDiagnosticAccuracyMetrics):
    def test_custom_diagnostic_score(self):
        """Test custom diagnostic scoring metric."""
        
        def custom_score(true_labels, pred_labels, confidence_scores):
            # Custom scoring logic
            return weighted_accuracy_score
        
        # Test implementation
        score = custom_score(self.true_labels, self.pred_labels, self.confidences)
        assert 0.0 <= score <= 1.0
```

### Statistical Power Analysis

Perform power analysis for study design:
```python
from tests.test_statistical_analysis import TestAdvancedStatisticalMethods

def calculate_required_sample_size(effect_size, alpha=0.05, power=0.80):
    """Calculate required sample size for detecting effect."""
    
    test_class = TestAdvancedStatisticalMethods()
    return test_class._power_analysis_two_groups(effect_size, alpha, power)

# Example usage
n_required = calculate_required_sample_size(effect_size=0.5)
print(f"Required sample size: {n_required}")
```

### Longitudinal Analysis

Track performance over time:
```python
from tests.test_evaluation_frameworks import LongitudinalTrackingFramework

tracker = LongitudinalTrackingFramework(storage_path="performance_history.json")

# Record performance metrics
tracker.record_performance("system_v1", {"accuracy": 0.85, "cost": 400.0})

# Detect performance drift
drift_analysis = tracker.detect_performance_drift("system_v1", "accuracy", threshold=0.05)

# Generate trend analysis
trends = tracker.generate_trend_analysis("system_v1", ["accuracy", "cost"])
```

## Conclusion

The Dx0 evaluation testing suite provides comprehensive validation of diagnostic system performance across multiple dimensions. By following this guide and utilizing the provided test frameworks, you can ensure robust, reliable, and statistically sound evaluation of diagnostic systems.

For additional support or questions, refer to the test source code documentation or contact the development team.