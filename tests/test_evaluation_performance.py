"""Performance and integration tests for the Dx0 evaluation system.

This module provides comprehensive performance testing for evaluation system
components under different scenarios, integration testing with Judge Agent and
CostEstimator, large-scale evaluation runs, and memory usage validation.
"""

import pytest
import asyncio
import time
import psutil
import gc
import numpy as np
from typing import List, Dict, Any, Callable
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import multiprocessing

from sdb.evaluation import Evaluator, SessionResult, batch_evaluate, async_batch_evaluate
from sdb.judge import Judge, Judgement
from sdb.cost_estimator import CostEstimator, CptCost
from sdb.ensemble import DiagnosisResult, WeightedVoter
from sdb.llm_client import LLMClient, OpenAIClient


@dataclass
class PerformanceMetrics:
    """Performance metrics for evaluation operations."""
    
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    throughput_ops_per_sec: float
    peak_memory_mb: float


class MockLLMClient(LLMClient):
    """Mock LLM client for performance testing."""
    
    def __init__(self, response_time: float = 0.1, response: str = "4"):
        """Initialize mock LLM client.
        
        Parameters
        ----------
        response_time : float
            Simulated response time in seconds.
        response : str
            Mock response to return.
        """
        self.response_time = response_time
        self.response = response
        self.call_count = 0
        self.total_time = 0.0
    
    def chat(self, messages: List[Dict[str, str]], model: str) -> str:
        """Mock chat method with simulated latency."""
        self.call_count += 1
        start_time = time.time()
        time.sleep(self.response_time)
        end_time = time.time()
        self.total_time += (end_time - start_time)
        return self.response


class MockCostEstimatorFast:
    """Fast mock cost estimator for performance testing."""
    
    def __init__(self, base_cost: float = 100.0):
        """Initialize fast mock cost estimator."""
        self.base_cost = base_cost
        self.call_count = 0
    
    def estimate_cost(self, test_name: str) -> float:
        """Fast cost estimation."""
        self.call_count += 1
        return self.base_cost


class TestEvaluationPerformance:
    """Test suite for evaluation system performance."""
    
    def test_single_evaluation_performance(self):
        """Test performance of single evaluation operation."""
        judge = Judge(rubric={}, client=MockLLMClient(response_time=0.01))
        cost_estimator = MockCostEstimatorFast()
        evaluator = Evaluator(judge, cost_estimator)
        
        # Measure performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        result = evaluator.evaluate(
            diagnosis="pneumonia",
            truth="bacterial pneumonia",
            tests=[f"test_{i}" for i in range(10)],
            visits=1,
            duration=30.0
        )
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        # Performance assertions
        assert execution_time < 1.0  # Should complete within 1 second
        assert memory_delta < 10.0   # Should not use more than 10MB additional memory
        assert result.score > 0      # Should produce valid result
    
    def test_batch_evaluation_performance(self):
        """Test performance of batch evaluation operations."""
        def mock_run_case(case_id: str) -> Dict[str, str]:
            time.sleep(0.01)  # Simulate work
            return {"id": case_id, "diagnosis": "flu", "cost": "300.0"}
        
        case_ids = [f"case_{i}" for i in range(100)]
        
        # Measure performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        results = batch_evaluate(case_ids, mock_run_case, concurrency=10)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        execution_time = end_time - start_time
        throughput = len(results) / execution_time
        
        # Performance assertions
        assert len(results) == 100
        assert execution_time < 5.0     # Should complete within 5 seconds with concurrency
        assert throughput > 20          # Should process at least 20 cases per second
        assert end_memory - start_memory < 50.0  # Memory usage should be reasonable
    
    def test_large_scale_evaluation_performance(self):
        """Test performance with large-scale evaluation runs."""
        judge = Judge(rubric={}, client=MockLLMClient(response_time=0.001))
        cost_estimator = MockCostEstimatorFast()
        evaluator = Evaluator(judge, cost_estimator)
        
        # Generate large test dataset
        test_cases = []
        for i in range(1000):
            test_cases.append({
                "diagnosis": f"diagnosis_{i % 10}",  # 10 different diagnoses
                "truth": f"truth_{i % 10}",
                "tests": [f"test_{j}" for j in range(i % 5 + 1)],  # Variable test counts
                "visits": (i % 3) + 1
            })
        
        # Measure performance
        performance_metrics = self._measure_performance(
            lambda: [evaluator.evaluate(**case) for case in test_cases]
        )
        
        # Performance assertions
        assert performance_metrics.execution_time < 30.0  # Should complete within 30 seconds
        assert performance_metrics.throughput_ops_per_sec > 30  # At least 30 evaluations per second
        assert performance_metrics.peak_memory_mb < 200.0  # Peak memory under 200MB
    
    def test_concurrent_evaluation_performance(self):
        """Test performance under concurrent evaluation scenarios."""
        judge = Judge(rubric={}, client=MockLLMClient(response_time=0.01))
        cost_estimator = MockCostEstimatorFast()
        
        def single_evaluation():
            evaluator = Evaluator(judge, cost_estimator)
            return evaluator.evaluate("flu", "influenza", ["cbc"], visits=1)
        
        # Test concurrent evaluations
        num_concurrent = 20
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(single_evaluation) for _ in range(num_concurrent)]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance assertions
        assert len(results) == num_concurrent
        assert execution_time < 5.0  # Should handle concurrency efficiently
        assert all(isinstance(r, SessionResult) for r in results)  # All results valid
    
    def test_memory_efficiency_under_load(self):
        """Test memory efficiency under sustained load."""
        judge = Judge(rubric={}, client=MockLLMClient(response_time=0.001))
        cost_estimator = MockCostEstimatorFast()
        evaluator = Evaluator(judge, cost_estimator)
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_measurements = []
        
        # Perform many evaluations and track memory
        for i in range(500):
            result = evaluator.evaluate("flu", "flu", ["test"], visits=1)
            
            if i % 50 == 0:  # Sample memory every 50 evaluations
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory - initial_memory)
                
                # Force garbage collection
                gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Memory efficiency assertions
        assert memory_growth < 50.0  # Memory growth should be limited
        assert max(memory_measurements) < 100.0  # Peak memory usage reasonable
        
        # Check for memory leaks (memory should not keep growing)
        if len(memory_measurements) > 2:
            late_avg = np.mean(memory_measurements[-3:])
            early_avg = np.mean(memory_measurements[:3])
            growth_rate = (late_avg - early_avg) / len(memory_measurements)
            assert growth_rate < 0.1  # Memory growth rate should be minimal
    
    def test_cpu_efficiency_evaluation(self):
        """Test CPU efficiency of evaluation operations."""
        judge = Judge(rubric={}, client=MockLLMClient(response_time=0.001))
        cost_estimator = MockCostEstimatorFast()
        evaluator = Evaluator(judge, cost_estimator)
        
        # Monitor CPU usage during evaluation
        cpu_samples = []
        
        def monitor_cpu():
            for _ in range(10):  # Sample for 1 second
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
        
        # Start CPU monitoring in background
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Perform evaluations
        for i in range(100):
            evaluator.evaluate("flu", "flu", ["test"], visits=1)
        
        monitor_thread.join()
        
        avg_cpu = np.mean(cpu_samples)
        max_cpu = max(cpu_samples)
        
        # CPU efficiency assertions
        assert avg_cpu < 80.0  # Average CPU usage should be reasonable
        assert max_cpu < 95.0  # Peak CPU usage should not saturate system
    
    def test_scalability_with_test_count(self):
        """Test scalability with varying numbers of tests per evaluation."""
        judge = Judge(rubric={}, client=MockLLMClient(response_time=0.001))
        cost_estimator = MockCostEstimatorFast()
        evaluator = Evaluator(judge, cost_estimator)
        
        test_counts = [1, 10, 50, 100, 500]
        execution_times = []
        
        for test_count in test_counts:
            tests = [f"test_{i}" for i in range(test_count)]
            
            start_time = time.time()
            result = evaluator.evaluate("flu", "flu", tests, visits=1)
            end_time = time.time()
            
            execution_times.append(end_time - start_time)
            assert result.total_cost > 0  # Sanity check
        
        # Scalability assertions
        # Execution time should scale reasonably with test count
        time_per_test = [t / c for t, c in zip(execution_times, test_counts)]
        
        # Time per test should be relatively consistent (linear scaling)
        assert max(time_per_test) / min(time_per_test) < 5.0  # Within 5x factor
    
    def _measure_performance(self, operation: Callable) -> PerformanceMetrics:
        """Measure comprehensive performance metrics for an operation."""
        process = psutil.Process()
        
        # Initial measurements
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024
        start_cpu_times = process.cpu_times()
        
        # Track peak memory during operation
        peak_memory = start_memory
        
        def memory_monitor():
            nonlocal peak_memory
            while not monitor_stop.is_set():
                current_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                time.sleep(0.01)
        
        monitor_stop = threading.Event()
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.start()
        
        try:
            # Execute operation
            result = operation()
            operation_count = len(result) if hasattr(result, '__len__') else 1
        finally:
            monitor_stop.set()
            monitor_thread.join()
        
        # Final measurements
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024
        end_cpu_times = process.cpu_times()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        cpu_time = (end_cpu_times.user - start_cpu_times.user + 
                   end_cpu_times.system - start_cpu_times.system)
        cpu_percent = (cpu_time / execution_time) * 100 if execution_time > 0 else 0
        throughput = operation_count / execution_time if execution_time > 0 else 0
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_percent=cpu_percent,
            throughput_ops_per_sec=throughput,
            peak_memory_mb=peak_memory
        )


class TestEvaluationIntegration:
    """Test suite for evaluation system integration."""
    
    def test_judge_cost_estimator_integration(self):
        """Test integration between Judge and CostEstimator."""
        # Create real Judge instance with mock LLM
        mock_client = MockLLMClient(response_time=0.01, response="4")
        judge = Judge(rubric={}, client=mock_client)
        
        # Create CostEstimator with test data
        cost_table = {
            "cbc": CptCost("85025", 50.0, "laboratory"),
            "chest_xray": CptCost("71010", 200.0, "imaging")
        }
        cost_estimator = CostEstimator(cost_table)
        
        # Create evaluator with real components
        evaluator = Evaluator(judge, cost_estimator)
        
        # Test evaluation with integration
        result = evaluator.evaluate(
            diagnosis="pneumonia",
            truth="bacterial pneumonia",
            tests=["cbc", "chest_xray"],
            visits=2,
            duration=45.0
        )
        
        # Integration assertions
        assert result.score == 4  # From mock LLM response
        assert result.total_cost == 850.0  # 2*300 (visits) + 50 + 200 (tests)
        assert result.correct is True  # Score >= threshold
        assert result.duration == 45.0
        
        # Verify components were called
        assert mock_client.call_count == 1
        assert cost_estimator.cost_table["cbc"].price == 50.0
    
    def test_ensemble_evaluation_integration(self):
        """Test integration with ensemble voting systems."""
        # Create multiple evaluators with different configurations
        judges = [
            Judge(rubric={}, client=MockLLMClient(response="5")),
            Judge(rubric={}, client=MockLLMClient(response="4")),
            Judge(rubric={}, client=MockLLMClient(response="3"))
        ]
        
        cost_estimator = MockCostEstimatorFast(base_cost=100.0)
        
        evaluators = [Evaluator(judge, cost_estimator) for judge in judges]
        
        # Evaluate same case with all evaluators
        diagnosis = "pneumonia"
        truth = "bacterial pneumonia"
        tests = ["cbc"]
        
        results = []
        for i, evaluator in enumerate(evaluators):
            result = evaluator.evaluate(diagnosis, truth, tests, visits=1)
            diagnosis_result = DiagnosisResult(
                diagnosis=diagnosis,
                confidence=result.score / 5.0,  # Normalize to 0-1
                cost=result.total_cost,
                run_id=f"evaluator_{i}"
            )
            results.append(diagnosis_result)
        
        # Test weighted voting
        voter = WeightedVoter()
        weights = {"evaluator_0": 1.5, "evaluator_1": 1.0, "evaluator_2": 0.5}
        
        winning_diagnosis = voter.vote(results, weights=weights)
        
        # Integration assertions
        assert winning_diagnosis == diagnosis
        assert len(results) == 3
        assert all(r.diagnosis == diagnosis for r in results)
    
    def test_async_evaluation_integration(self):
        """Test asynchronous evaluation integration."""
        async def async_run_case(case_id: str) -> Dict[str, str]:
            # Simulate async diagnostic process
            await asyncio.sleep(0.01)
            
            # Mock evaluation result
            judge = Judge(rubric={}, client=MockLLMClient(response="4"))
            cost_estimator = MockCostEstimatorFast()
            evaluator = Evaluator(judge, cost_estimator)
            
            result = evaluator.evaluate("flu", "influenza", ["cbc"], visits=1)
            
            return {
                "id": case_id,
                "score": str(result.score),
                "cost": str(result.total_cost),
                "correct": str(result.correct)
            }
        
        # Test async batch evaluation
        case_ids = [f"case_{i}" for i in range(20)]
        
        start_time = time.time()
        results = asyncio.run(async_batch_evaluate(case_ids, async_run_case, concurrency=5))
        end_time = time.time()
        
        # Integration assertions
        assert len(results) == 20
        assert all("id" in result for result in results)
        assert all("score" in result for result in results)
        assert end_time - start_time < 2.0  # Should benefit from concurrency
    
    def test_real_world_workflow_integration(self):
        """Test integration in realistic diagnostic workflow."""
        # Setup realistic components
        judge = Judge(rubric={}, client=MockLLMClient(response="4"))
        
        cost_table = {
            "cbc": CptCost("85025", 50.0, "laboratory"),
            "bmp": CptCost("80048", 75.0, "laboratory"),
            "chest_xray": CptCost("71010", 200.0, "imaging"),
            "ct_chest": CptCost("71250", 800.0, "imaging"),
            "ecg": CptCost("93000", 150.0, "cardiac")
        }
        cost_estimator = CostEstimator(cost_table)
        evaluator = Evaluator(judge, cost_estimator, correct_threshold=4)
        
        # Simulate realistic diagnostic cases
        diagnostic_cases = [
            {
                "case_id": "chest_pain_001",
                "diagnosis": "myocardial infarction",
                "truth": "acute myocardial infarction",
                "tests": ["cbc", "bmp", "ecg", "chest_xray"],
                "visits": 2,
                "complexity": "high"
            },
            {
                "case_id": "respiratory_002", 
                "diagnosis": "pneumonia",
                "truth": "community-acquired pneumonia",
                "tests": ["cbc", "chest_xray"],
                "visits": 1,
                "complexity": "moderate"
            },
            {
                "case_id": "routine_003",
                "diagnosis": "viral upper respiratory infection",
                "truth": "viral URI",
                "tests": ["cbc"],
                "visits": 1,
                "complexity": "low"
            }
        ]
        
        # Process cases and collect metrics
        session_results = []
        case_complexity_performance = {}
        
        for case in diagnostic_cases:
            result = evaluator.evaluate(
                diagnosis=case["diagnosis"],
                truth=case["truth"],
                tests=case["tests"],
                visits=case["visits"],
                duration=30.0 + len(case["tests"]) * 5.0  # Realistic duration
            )
            
            session_results.append(result)
            
            # Track performance by complexity
            complexity = case["complexity"]
            if complexity not in case_complexity_performance:
                case_complexity_performance[complexity] = []
            case_complexity_performance[complexity].append(result.correct)
        
        # Calculate aggregate metrics
        total_cost = sum(r.total_cost for r in session_results)
        accuracy = sum(1 for r in session_results if r.correct) / len(session_results)
        avg_duration = sum(r.duration for r in session_results) / len(session_results)
        
        # Workflow integration assertions
        assert len(session_results) == 3
        assert total_cost > 0
        assert 0.0 <= accuracy <= 1.0
        assert avg_duration > 0
        
        # Complexity-based performance
        for complexity, results in case_complexity_performance.items():
            complexity_accuracy = sum(results) / len(results)
            assert 0.0 <= complexity_accuracy <= 1.0
    
    def test_error_handling_integration(self):
        """Test error handling in integrated evaluation scenarios."""
        # Create components that may fail
        class FailingJudge:
            def __init__(self, fail_rate: float = 0.5):
                self.fail_rate = fail_rate
                self.call_count = 0
            
            def evaluate(self, diagnosis: str, truth: str) -> Judgement:
                self.call_count += 1
                if self.call_count % int(1/self.fail_rate) == 0:
                    raise Exception("Judge evaluation failed")
                return Judgement(score=4, explanation="OK")
        
        class FailingCostEstimator:
            def __init__(self, fail_rate: float = 0.3):
                self.fail_rate = fail_rate
                self.call_count = 0
            
            def estimate_cost(self, test_name: str) -> float:
                self.call_count += 1
                if self.call_count % int(1/self.fail_rate) == 0:
                    raise Exception("Cost estimation failed")
                return 100.0
        
        # Test error recovery
        failing_judge = FailingJudge(fail_rate=0.5)
        failing_cost_estimator = FailingCostEstimator(fail_rate=0.3)
        evaluator = Evaluator(failing_judge, failing_cost_estimator)
        
        # Attempt multiple evaluations
        successful_evaluations = 0
        failed_evaluations = 0
        
        for i in range(20):
            try:
                result = evaluator.evaluate("flu", "influenza", ["test"], visits=1)
                successful_evaluations += 1
                assert isinstance(result, SessionResult)
            except Exception:
                failed_evaluations += 1
        
        # Error handling assertions
        assert successful_evaluations > 0  # Some should succeed
        assert failed_evaluations > 0     # Some should fail as expected
        assert successful_evaluations + failed_evaluations == 20


class TestEvaluationStressTest:
    """Stress tests for evaluation system under extreme conditions."""
    
    def test_high_concurrency_stress(self):
        """Test evaluation system under high concurrency stress."""
        judge = Judge(rubric={}, client=MockLLMClient(response_time=0.001))
        cost_estimator = MockCostEstimatorFast()
        
        def stress_evaluation():
            evaluator = Evaluator(judge, cost_estimator)
            results = []
            for i in range(50):
                result = evaluator.evaluate(f"diagnosis_{i}", f"truth_{i}", ["test"], visits=1)
                results.append(result)
            return results
        
        # Launch many concurrent stress tests
        num_threads = multiprocessing.cpu_count() * 2  # 2x CPU cores
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(stress_evaluation) for _ in range(num_threads)]
            all_results = []
            
            for future in as_completed(futures):
                try:
                    results = future.result(timeout=10.0)  # 10 second timeout
                    all_results.extend(results)
                except Exception as e:
                    pytest.fail(f"Stress test failed: {e}")
        
        end_time = time.time()
        
        # Stress test assertions
        expected_results = num_threads * 50
        assert len(all_results) == expected_results
        assert end_time - start_time < 15.0  # Should complete within reasonable time
        assert all(isinstance(r, SessionResult) for r in all_results)
    
    def test_memory_pressure_stress(self):
        """Test evaluation system under memory pressure."""
        judge = Judge(rubric={}, client=MockLLMClient(response_time=0.001))
        cost_estimator = MockCostEstimatorFast()
        evaluator = Evaluator(judge, cost_estimator)
        
        # Create memory pressure with large test lists
        large_test_lists = []
        for i in range(100):
            test_list = [f"test_{j}_{i}" for j in range(1000)]  # 1000 tests per case
            large_test_lists.append(test_list)
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = initial_memory
        
        # Monitor memory during stress test
        def memory_monitor():
            nonlocal peak_memory
            while not monitor_stop.is_set():
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                time.sleep(0.1)
        
        monitor_stop = threading.Event()
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.start()
        
        try:
            # Perform evaluations under memory pressure
            results = []
            for i, test_list in enumerate(large_test_lists):
                result = evaluator.evaluate(f"diagnosis_{i}", f"truth_{i}", test_list, visits=1)
                results.append(result)
                
                # Periodic garbage collection
                if i % 10 == 0:
                    gc.collect()
        
        finally:
            monitor_stop.set()
            monitor_thread.join()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Memory stress assertions
        assert len(results) == 100
        assert peak_memory - initial_memory < 500.0  # Memory growth under 500MB
        assert final_memory - initial_memory < 100.0  # Final memory growth limited
    
    def test_extended_runtime_stress(self):
        """Test evaluation system stability over extended runtime."""
        judge = Judge(rubric={}, client=MockLLMClient(response_time=0.01))
        cost_estimator = MockCostEstimatorFast()
        evaluator = Evaluator(judge, cost_estimator)
        
        # Run evaluations for extended period
        start_time = time.time()
        runtime_limit = 30.0  # 30 seconds
        evaluation_count = 0
        errors = []
        
        while time.time() - start_time < runtime_limit:
            try:
                result = evaluator.evaluate(
                    f"diagnosis_{evaluation_count}",
                    f"truth_{evaluation_count}",
                    [f"test_{evaluation_count % 10}"],
                    visits=1
                )
                evaluation_count += 1
                
                # Validate result integrity
                assert isinstance(result, SessionResult)
                assert result.score > 0
                assert result.total_cost > 0
                
            except Exception as e:
                errors.append(str(e))
        
        end_time = time.time()
        actual_runtime = end_time - start_time
        throughput = evaluation_count / actual_runtime
        
        # Extended runtime assertions
        assert evaluation_count > 100  # Should complete many evaluations
        assert len(errors) < evaluation_count * 0.01  # Error rate < 1%
        assert throughput > 10  # Maintain reasonable throughput
        assert actual_runtime >= runtime_limit * 0.95  # Actually ran for expected time


# Pytest fixtures for performance testing
@pytest.fixture
def performance_test_data():
    """Provide test data optimized for performance testing."""
    return {
        "diagnoses": [f"diagnosis_{i}" for i in range(100)],
        "truths": [f"truth_{i}" for i in range(100)],
        "test_lists": [[f"test_{j}" for j in range(i % 10 + 1)] for i in range(100)]
    }


@pytest.fixture
def mock_components():
    """Provide mock components optimized for performance testing."""
    return {
        "fast_judge": Judge(rubric={}, client=MockLLMClient(response_time=0.001)),
        "fast_cost_estimator": MockCostEstimatorFast(base_cost=100.0)
    }


if __name__ == "__main__":
    pytest.main([__file__])