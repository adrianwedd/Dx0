# Judge Agent Testing Guide

This guide provides comprehensive documentation for testing the Judge Agent component in the Dx0 project. The Judge Agent is responsible for evaluating diagnostic reasoning and providing scoring based on similarity between candidate diagnoses and ground truth.

## Table of Contents

1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Running Tests](#running-tests)
4. [Test Categories](#test-categories)
5. [Test Data and Fixtures](#test-data-and-fixtures)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Adding New Tests](#adding-new-tests)
8. [Troubleshooting](#troubleshooting)

## Overview

The Judge Agent testing suite provides comprehensive coverage for:
- **Core functionality**: Scoring algorithms, input validation, error handling
- **Validation**: Accuracy against ground truth, consistency, statistical properties
- **Performance**: Load testing, concurrency, memory usage, scalability
- **Integration**: Compatibility with Evaluator, LLM clients, and other components

### Judge Agent Architecture

The Judge Agent consists of:
- `Judge` class: Main scoring component
- `Judgement` dataclass: Structured result container
- LLM integration: Uses configurable LLM clients for scoring
- Scoring rubric: 5-point Likert scale (1=incorrect, 5=exact match)

## Test Structure

```
tests/
├── test_judge.py                    # Enhanced unit tests
├── test_judge_validation.py         # Scoring validation tests
├── test_judge_performance.py        # Performance and integration tests
├── judge_test_data.py              # Test data fixtures and datasets
└── JUDGE_TESTING_GUIDE.md          # This documentation
```

### Test File Organization

- **`test_judge.py`**: Enhanced version of original tests with comprehensive unit testing
- **`test_judge_validation.py`**: Validation tests for accuracy and consistency
- **`test_judge_performance.py`**: Performance, concurrency, and integration tests
- **`judge_test_data.py`**: Centralized test data and fixtures

## Running Tests

### Prerequisites

Ensure you have the test dependencies installed:
```bash
pip install -r requirements-dev.txt
```

### Running All Judge Tests

```bash
# Run all Judge-related tests
pytest tests/test_judge*.py -v

# Run specific test file
pytest tests/test_judge_validation.py -v

# Run with coverage
pytest tests/test_judge*.py --cov=sdb.judge --cov-report=html
```

### Running Specific Test Categories

```bash
# Unit tests only
pytest tests/test_judge.py -v

# Validation tests only
pytest tests/test_judge_validation.py -v

# Performance tests only
pytest tests/test_judge_performance.py -v

# Run tests by marker (if implemented)
pytest -m "performance" -v
pytest -m "integration" -v
```

### Running Tests with Different Configurations

```bash
# Run with specific timeout
pytest tests/test_judge_performance.py --timeout=60

# Run in parallel (if pytest-xdist installed)
pytest tests/test_judge*.py -n 4

# Run with specific log level
pytest tests/test_judge*.py --log-level=DEBUG
```

## Test Categories

### 1. Unit Tests (`test_judge.py`)

#### TestJudgeConstruction
- Judge initialization with default and custom parameters
- Rubric, model, and client configuration
- Prompt loading validation

#### TestJudgeScoring
- Score range validation (1-5)
- Explanation mapping completeness
- Input whitespace and empty string handling
- Edge case diagnosis formats

#### TestJudgmentDataClass
- Judgement object creation and equality
- Data validation and serialization

#### TestErrorHandling
- LLM client failure fallback
- Unparseable response handling
- Exception recovery and graceful degradation
- Score extraction from various response formats

#### TestScoringConsistency
- Identical input consistency
- Symmetric scoring analysis
- Statistical distribution validation
- Edge case diagnosis handling

#### TestPerformanceCharacteristics
- Concurrent evaluation safety
- Memory usage stability
- Response time consistency

#### TestIntegrationScenarios
- Evaluator class integration
- Different model compatibility
- LLM client interaction patterns

#### TestValidationTests
- Ground truth validation cases
- Diagnostic scenario accuracy
- Medical domain-specific testing

#### TestRegressionPrevention
- Regex extraction edge cases
- Prompt injection resistance
- Unicode character handling

### 2. Validation Tests (`test_judge_validation.py`)

#### TestScoringAccuracy
- Ground truth validation against known cases
- Complex medical scenario handling
- Domain-specific accuracy (cardiology, pulmonology, etc.)
- Edge case graceful handling

#### TestScoringConsistency
- Perfect consistency with deterministic clients
- Consistency under controlled noise
- Cross-session consistency validation

#### TestStatisticalValidation
- Score distribution properties
- Scoring variance analysis
- Correlation between similarity and scores

#### TestRegressionValidation
- Known regression case prevention
- Unicode handling validation
- Whitespace handling consistency

#### TestBenchmarkValidation
- Accuracy benchmark compliance
- Consistency benchmark adherence
- Performance benchmark validation

#### TestValidationReporting
- Metrics collection and analysis
- Comprehensive validation report generation

### 3. Performance Tests (`test_judge_performance.py`)

#### TestJudgePerformance
- Single evaluation performance timing
- Batch evaluation efficiency
- Performance under increasing load
- Memory usage stability analysis

#### TestJudgeConcurrency
- Thread safety validation
- Concurrent load handling
- Consistency under concurrent access

#### TestJudgeIntegration
- Evaluator component integration
- Multiple model configuration support
- LLM client caching integration
- Error propagation handling

#### TestJudgeScalability
- Linear scalability validation
- Resource usage scaling analysis

#### TestJudgeReliability
- Fault tolerance with client failures
- Graceful degradation under stress

## Test Data and Fixtures

### Test Data Categories

The `judge_test_data.py` module provides structured test datasets:

#### Ground Truth Cases
- **Exact matches**: Identical diagnoses (Score: 5)
- **Clinical synonyms**: Heart attack ↔ Myocardial infarction (Score: 4-5)
- **Partial matches**: Chest pain ↔ Myocardial infarction (Score: 2-4)
- **Poor matches**: Broken arm ↔ Myocardial infarction (Score: 1-2)

#### Complex Scenarios
- Medical abbreviations: STEMI ↔ ST-elevation myocardial infarction
- Evolving terminology: Congestive heart failure ↔ Heart failure with reduced ejection fraction
- Formal vs. common terms: Cerebrovascular accident ↔ Stroke

#### Edge Cases
- Empty inputs
- Very long diagnoses
- Differential diagnosis notation
- Special characters and Unicode

#### Domain-Specific Cases
- **Cardiology**: A-fib, V-tach, MI
- **Pulmonology**: COPD, ARDS
- **Endocrinology**: DKA, HHS

### Using Test Data

```python
from tests.judge_test_data import (
    GROUND_TRUTH_CASES,
    get_test_cases_by_category,
    get_test_cases_by_difficulty,
    export_test_cases_json
)

# Get specific categories
cardiology_cases = get_test_cases_by_category("cardiology")
hard_cases = get_test_cases_by_difficulty("hard")

# Export for external validation
export_test_cases_json("judge_validation_cases.json")
```

### Mock Clients

The test suite includes specialized mock clients:

- **`DummyClient`**: Predictable scoring based on diagnosis pairs
- **`ValidationClient`**: Configurable scoring logic for validation
- **`ConsistencyTestClient`**: Controlled noise for consistency testing
- **`PerformanceTestClient`**: Configurable delays for performance testing
- **`LoadTestClient`**: Load-dependent response times
- **`StatisticalClient`**: Known score distributions

## Performance Benchmarks

### Current Benchmarks

| Metric | Benchmark | Description |
|--------|-----------|-------------|
| Accuracy | ≥85% | Percentage of test cases scoring within expected range |
| Consistency | ≥95% | Consistency rate for identical inputs |
| Performance | ≤2.0s | Maximum time per evaluation |
| Concurrent Success | 100% | Success rate for concurrent evaluations |

### Benchmark Validation

```python
from tests.judge_test_data import get_benchmark_expectations

benchmarks = get_benchmark_expectations()
print(f"Accuracy threshold: {benchmarks['accuracy_threshold']:.1%}")
print(f"Performance threshold: {benchmarks['performance_threshold']}s")
```

### Performance Test Examples

```bash
# Run performance benchmarks
pytest tests/test_judge_performance.py::TestJudgePerformance::test_single_evaluation_performance -v

# Run concurrency tests
pytest tests/test_judge_performance.py::TestJudgeConcurrency -v

# Run scalability tests
pytest tests/test_judge_performance.py::TestJudgeScalability -v
```

## Adding New Tests

### Adding Unit Tests

1. **Choose appropriate test class** based on functionality being tested
2. **Follow naming convention**: `test_descriptive_name`
3. **Use appropriate mock clients** for predictable behavior
4. **Include edge cases** and error conditions
5. **Add assertions** for both positive and negative cases

Example:
```python
def test_new_scoring_feature(self):
    """Test description of new feature."""
    judge = Judge({}, client=DummyClient())
    
    # Test normal case
    result = judge.evaluate("test diagnosis", "test truth")
    assert result.score in range(1, 6)
    
    # Test edge case
    result = judge.evaluate("", "test truth")
    assert result.score == 1  # Expected fallback
```

### Adding Validation Tests

1. **Define test cases** with expected score ranges
2. **Use ValidationClient** with appropriate scoring logic
3. **Test multiple scenarios** within the same category
4. **Include statistical validation** where appropriate

Example:
```python
def test_new_medical_domain(self):
    """Test accuracy for new medical domain."""
    test_cases = [
        DiagnosticTestCase(
            diagnosis="domain-specific term",
            truth="equivalent term",
            expected_score_min=4,
            expected_score_max=5,
            category="new_domain",
            description="Domain-specific test",
            difficulty="medium"
        )
    ]
    
    judge = Judge({}, client=ValidationClient())
    
    for case in test_cases:
        result = judge.evaluate(case.diagnosis, case.truth)
        assert case.expected_score_min <= result.score <= case.expected_score_max
```

### Adding Performance Tests

1. **Use appropriate performance client** with realistic delays
2. **Measure relevant metrics** (time, memory, throughput)
3. **Compare against benchmarks**
4. **Test both normal and stress conditions**

Example:
```python
def test_new_performance_scenario(self):
    """Test performance under new conditions."""
    judge = Judge({}, client=PerformanceTestClient(response_delay=0.001))
    
    start_time = time.perf_counter()
    
    # Perform test scenario
    for i in range(100):
        result = judge.evaluate(f"test_{i}", f"truth_{i}")
        assert result.score == 3
    
    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / 100
    
    benchmark = get_benchmark_expectations()["performance_threshold"]
    assert avg_time <= benchmark
```

### Adding Test Data

1. **Add to appropriate dataset** in `judge_test_data.py`
2. **Include all required fields** for DiagnosticTestCase
3. **Choose appropriate category and difficulty**
4. **Validate expected score ranges**

Example:
```python
NEW_DOMAIN_CASES = [
    DiagnosticTestCase(
        diagnosis="New condition",
        truth="Equivalent condition", 
        expected_score_min=4,
        expected_score_max=5,
        category="new_domain",
        description="New domain test case",
        difficulty="medium"
    ),
]
```

## Troubleshooting

### Common Test Failures

#### 1. Score Range Assertions
**Problem**: `AssertionError: Score 6 out of range`
**Solution**: Check mock client responses and score extraction logic

#### 2. Consistency Failures
**Problem**: `Inconsistent results: [3, 4, 3, 3, 4]`
**Solution**: Ensure mock clients return deterministic responses for identical inputs

#### 3. Performance Benchmark Failures
**Problem**: `Average time 2.5s exceeds benchmark 2.0s`
**Solution**: Check for inefficient test setup or adjust benchmark expectations

#### 4. Integration Test Failures
**Problem**: Mock objects not behaving as expected
**Solution**: Verify mock specifications match actual component interfaces

### Debugging Tips

1. **Use verbose output**: `pytest -v -s`
2. **Add debug prints**: Include temporary print statements in tests
3. **Run single tests**: Isolate failing tests with specific pytest selection
4. **Check mock configurations**: Verify mock clients return expected responses
5. **Validate test data**: Ensure test cases have realistic expected ranges

### Test Environment Issues

#### Missing Dependencies
```bash
# Install missing test dependencies
pip install pytest pytest-cov pytest-mock
```

#### Import Errors
```bash
# Ensure project is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/test_judge*.py
```

#### Timeout Issues
```bash
# Adjust timeout for slow tests
pytest tests/test_judge_performance.py --timeout=120
```

### Performance Debugging

For performance test failures:

1. **Profile individual operations**:
```python
import cProfile
cProfile.run('judge.evaluate("test", "truth")')
```

2. **Monitor memory usage**:
```python
import tracemalloc
tracemalloc.start()
# ... run tests ...
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
```

3. **Check for resource leaks**:
```bash
# Run with memory profiling
python -m memory_profiler tests/test_judge_performance.py
```

## Best Practices

### Test Development

1. **Write descriptive test names** that explain what is being tested
2. **Use appropriate assertions** with meaningful error messages
3. **Test both success and failure paths**
4. **Keep tests independent** and avoid shared state
5. **Use fixtures appropriately** for setup and teardown

### Mock Usage

1. **Use specific mock clients** designed for each test category
2. **Configure mocks realistically** to match actual component behavior
3. **Verify mock interactions** when testing integration scenarios
4. **Reset mock state** between tests to avoid interference

### Performance Testing

1. **Establish baseline measurements** before making changes
2. **Test under realistic conditions** that match production usage
3. **Include both synthetic and realistic workloads**
4. **Monitor multiple metrics** (time, memory, throughput)
5. **Document performance expectations** and benchmark rationale

### Validation Testing

1. **Use diverse test cases** covering different medical domains
2. **Include edge cases** and boundary conditions
3. **Validate statistical properties** of scoring distributions
4. **Test regression prevention** for known issues
5. **Document expected behaviors** and scoring rationale

---

This testing guide provides comprehensive coverage for validating the Judge Agent component. Regular execution of these tests ensures the reliability, performance, and accuracy of the diagnostic scoring system in the Dx0 project.