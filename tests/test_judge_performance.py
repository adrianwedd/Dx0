"""
Performance and integration tests for Judge Agent.

This module provides comprehensive performance testing, load testing,
concurrency testing, and integration testing for the Judge Agent component.
"""

import pytest
import time
import threading
import asyncio
import statistics
import gc
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch
import tracemalloc

from sdb.judge import Judge, Judgement
from sdb.evaluation import Evaluator, SessionResult
from sdb.cost_estimator import CostEstimator
from judge_test_data import (
    PERFORMANCE_TEST_CASES,
    get_benchmark_expectations,
)


class PerformanceTestClient:
    """Mock client for performance testing with configurable delays."""
    
    def __init__(self, response_delay: float = 0.001, response: str = "3"):
        self.response_delay = response_delay
        self.response = response
        self.call_count = 0
        self.total_delay = 0.0
        self.lock = threading.Lock()
    
    def chat(self, messages, model):
        with self.lock:
            self.call_count += 1
        
        time.sleep(self.response_delay)
        
        with self.lock:
            self.total_delay += self.response_delay
        
        return self.response
    
    def get_stats(self):
        with self.lock:
            return {
                "call_count": self.call_count,
                "total_delay": self.total_delay,
                "avg_delay": self.total_delay / max(1, self.call_count)
            }


class LoadTestClient:
    """Mock client for load testing with realistic response patterns."""
    
    def __init__(self, base_delay: float = 0.1, load_factor: float = 1.0):
        self.base_delay = base_delay
        self.load_factor = load_factor
        self.concurrent_calls = 0
        self.max_concurrent = 0
        self.total_calls = 0
        self.lock = threading.Lock()
    
    def chat(self, messages, model):
        with self.lock:
            self.concurrent_calls += 1
            self.total_calls += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent_calls)
        
        try:
            # Simulate load-dependent delay
            actual_delay = self.base_delay * (1 + (self.concurrent_calls - 1) * self.load_factor)
            time.sleep(actual_delay)
            return "3"
        finally:
            with self.lock:
                self.concurrent_calls -= 1
    
    def get_load_stats(self):
        with self.lock:
            return {
                "total_calls": self.total_calls,
                "max_concurrent": self.max_concurrent,
                "current_concurrent": self.concurrent_calls
            }


class MemoryTrackingClient:
    """Mock client for memory usage testing."""
    
    def __init__(self, response: str = "3"):
        self.response = response
        self.call_count = 0
        self.memory_snapshots = []
    
    def chat(self, messages, model):
        self.call_count += 1
        
        # Take memory snapshot periodically
        if self.call_count % 10 == 0:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            self.memory_snapshots.append({
                "call_count": self.call_count,
                "rss": memory_info.rss,
                "vms": memory_info.vms,
            })
        
        return self.response


class TestJudgePerformance:
    """Test Judge performance characteristics."""
    
    def test_single_evaluation_performance(self):
        """Test performance of single evaluation calls."""
        judge = Judge({}, client=PerformanceTestClient(response_delay=0.001))
        
        # Warm up
        judge.evaluate("warm up", "warm up")
        
        # Measure performance
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = judge.evaluate("test diagnosis", "test truth")
            end = time.perf_counter()
            
            times.append(end - start)
            assert result.score == 3
        
        # Analyze performance
        avg_time = statistics.mean(times)
        max_time = max(times)
        min_time = min(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        
        benchmark = get_benchmark_expectations()["performance_threshold"]
        
        assert avg_time <= benchmark, \
            f"Average time {avg_time:.3f}s exceeds benchmark {benchmark}s"
        assert max_time <= benchmark * 2, \
            f"Max time {max_time:.3f}s exceeds 2x benchmark"
        assert std_dev <= avg_time * 0.5, \
            f"High variance in performance: std_dev={std_dev:.3f}s"
    
    def test_batch_evaluation_performance(self):
        """Test performance of batch evaluations."""
        judge = Judge({}, client=PerformanceTestClient(response_delay=0.001))
        
        test_cases = PERFORMANCE_TEST_CASES[:50]  # Use subset for speed
        
        start_time = time.perf_counter()
        
        results = []
        for diagnosis, truth in test_cases:
            result = judge.evaluate(diagnosis, truth)
            results.append(result)
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time_per_eval = total_time / len(test_cases)
        
        # Verify all evaluations completed successfully
        assert len(results) == len(test_cases)
        assert all(1 <= r.score <= 5 for r in results)
        
        # Performance assertions
        benchmark = get_benchmark_expectations()["performance_threshold"]
        assert avg_time_per_eval <= benchmark, \
            f"Batch average time {avg_time_per_eval:.3f}s exceeds benchmark"
    
    def test_performance_under_load(self):
        """Test Judge performance under increasing load."""
        load_levels = [1, 5, 10, 20]
        performance_results = {}
        
        for load_level in load_levels:
            judge = Judge({}, client=PerformanceTestClient(response_delay=0.001))
            
            start_time = time.perf_counter()
            
            # Simulate load with multiple quick evaluations
            for i in range(load_level):
                result = judge.evaluate(f"diagnosis_{i}", f"truth_{i}")
                assert result.score == 3
            
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            avg_time = total_time / load_level
            
            performance_results[load_level] = avg_time
        
        # Performance should not degrade significantly with load
        baseline_perf = performance_results[1]
        for load_level, avg_time in performance_results.items():
            if load_level > 1:
                degradation_factor = avg_time / baseline_perf
                assert degradation_factor <= 2.0, \
                    f"Performance degraded {degradation_factor:.2f}x at load {load_level}"
    
    def test_memory_usage_stability(self):
        """Test that Judge doesn't accumulate memory over time."""
        if not hasattr(sys, 'gettotalrefcount'):  # Only available in debug builds
            pytest.skip("Memory tracking requires debug Python build")
        
        judge = Judge({}, client=MemoryTrackingClient())
        
        tracemalloc.start()
        
        initial_snapshot = tracemalloc.take_snapshot()
        
        # Run many evaluations
        for i in range(100):
            result = judge.evaluate(f"diagnosis_{i}", f"truth_{i}")
            assert result.score == 3
            
            # Force garbage collection periodically
            if i % 20 == 19:
                gc.collect()
        
        final_snapshot = tracemalloc.take_snapshot()
        
        # Compare memory usage
        top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
        
        # Check for significant memory growth
        total_growth = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        
        # Allow some growth but not excessive (less than 1MB for 100 evaluations)
        assert total_growth < 1024 * 1024, \
            f"Excessive memory growth: {total_growth} bytes"
        
        tracemalloc.stop()


class TestJudgeConcurrency:
    """Test Judge behavior under concurrent access."""
    
    def test_thread_safety_basic(self):
        """Test basic thread safety of Judge evaluations."""
        judge = Judge({}, client=PerformanceTestClient(response_delay=0.01))
        
        results = []
        errors = []
        num_threads = 10
        evaluations_per_thread = 5
        
        def worker(thread_id):
            try:
                thread_results = []
                for i in range(evaluations_per_thread):
                    result = judge.evaluate(f"diagnosis_{thread_id}_{i}", f"truth_{thread_id}_{i}")
                    thread_results.append(result.score)
                results.extend(thread_results)
            except Exception as e:
                errors.append(e)
        
        # Create and start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors in concurrent execution: {errors}"
        assert len(results) == num_threads * evaluations_per_thread
        assert all(score == 3 for score in results)
        
        # Verify benchmark
        benchmark = get_benchmark_expectations()["concurrent_success_rate"]
        success_rate = len(results) / (num_threads * evaluations_per_thread)
        assert success_rate >= benchmark, \
            f"Concurrent success rate {success_rate:.2%} below benchmark"
    
    def test_concurrent_load_handling(self):
        """Test Judge behavior under high concurrent load."""
        judge = Judge({}, client=LoadTestClient(base_delay=0.01, load_factor=0.1))
        
        num_workers = 20
        tasks_per_worker = 3
        
        def worker(worker_id):
            results = []
            for i in range(tasks_per_worker):
                result = judge.evaluate(f"diag_{worker_id}_{i}", f"truth_{worker_id}_{i}")
                results.append(result.score)
                time.sleep(0.001)  # Small delay between evaluations
            return results
        
        # Use ThreadPoolExecutor for better control
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(worker, i) for i in range(num_workers)]
            
            # Collect results
            all_results = []
            for future in as_completed(futures):
                try:
                    worker_results = future.result(timeout=30)  # 30 second timeout
                    all_results.extend(worker_results)
                except Exception as e:
                    pytest.fail(f"Worker failed: {e}")
        
        # Verify all tasks completed successfully
        expected_total = num_workers * tasks_per_worker
        assert len(all_results) == expected_total, \
            f"Expected {expected_total} results, got {len(all_results)}"
        
        # Verify all scores are valid
        assert all(1 <= score <= 5 for score in all_results)
    
    def test_concurrent_consistency(self):
        """Test that concurrent evaluations of same input are consistent."""
        judge = Judge({}, client=PerformanceTestClient(response_delay=0.01))
        
        test_case = ("consistent test", "consistent truth")
        num_concurrent = 10
        
        def evaluate_once():
            return judge.evaluate(test_case[0], test_case[1]).score
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(evaluate_once) for _ in range(num_concurrent)]
            scores = [future.result() for future in as_completed(futures)]
        
        # All scores should be identical for same input
        unique_scores = set(scores)
        assert len(unique_scores) == 1, \
            f"Inconsistent concurrent results: {scores}"
        assert list(unique_scores)[0] == 3  # Expected score from mock client


class TestJudgeIntegration:
    """Test Judge integration with other system components."""
    
    def test_integration_with_evaluator(self):
        """Test Judge integration with Evaluator component."""
        # Create mock cost estimator
        cost_estimator = Mock(spec=CostEstimator)
        cost_estimator.estimate_cost.return_value = 100.0
        
        # Create Judge and Evaluator
        judge = Judge({}, client=PerformanceTestClient(response_delay=0.001, response="4"))
        evaluator = Evaluator(judge, cost_estimator, correct_threshold=4)
        
        # Test evaluation
        result = evaluator.evaluate(
            diagnosis="test diagnosis",
            truth="test truth",
            tests=["CBC", "BMP", "CMP"],
            visits=2,
            duration=300.0
        )
        
        # Verify integration
        assert isinstance(result, SessionResult)
        assert result.score == 4
        assert result.correct is True  # Score 4 >= threshold 4
        assert result.total_cost > 0  # Should include visit fees and test costs
        assert result.duration == 300.0
        
        # Verify cost estimator was called for each test
        assert cost_estimator.estimate_cost.call_count == 3
    
    def test_integration_with_different_models(self):
        """Test Judge integration with different model configurations."""
        models = ["gpt-4", "gpt-3.5-turbo", "custom-model"]
        
        for model in models:
            judge = Judge({}, model=model, client=PerformanceTestClient(response="3"))
            
            result = judge.evaluate("test", "truth")
            
            assert result.score == 3
            assert judge.model == model
    
    def test_integration_with_llm_client_caching(self):
        """Test Judge integration with LLM client caching."""
        # Create client with caching enabled
        from sdb.llm_client import FileCache
        import tempfile
        import os
        
        # Use temporary file for cache
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            cache_path = tmp.name
        
        try:
            # Create client with file cache
            class CachingTestClient:
                def __init__(self, cache_path):
                    self.cache = FileCache(cache_path, max_size=10)
                    self.call_count = 0
                
                def chat(self, messages, model):
                    import json
                    key = json.dumps({"model": model, "messages": messages}, sort_keys=True)
                    
                    # Check cache first
                    cached = self.cache.get(key)
                    if cached is not None:
                        return cached
                    
                    # Simulate LLM call
                    self.call_count += 1
                    response = "3"
                    
                    # Cache response
                    self.cache.set(key, response)
                    return response
            
            client = CachingTestClient(cache_path)
            judge = Judge({}, client=client)
            
            # First evaluation - should call LLM
            result1 = judge.evaluate("test diagnosis", "test truth")
            assert result1.score == 3
            assert client.call_count == 1
            
            # Second evaluation with same input - should use cache
            result2 = judge.evaluate("test diagnosis", "test truth")
            assert result2.score == 3
            assert client.call_count == 1  # No additional call
            
            # Different input - should call LLM again
            result3 = judge.evaluate("different diagnosis", "different truth")
            assert result3.score == 3
            assert client.call_count == 2
        
        finally:
            # Clean up
            if os.path.exists(cache_path):
                os.unlink(cache_path)
    
    def test_integration_error_propagation(self):
        """Test how Judge integrates with error handling in broader system."""
        from sdb.evaluation import Evaluator, EvaluationError
        
        # Create client that occasionally fails
        class UnreliableClient:
            def __init__(self):
                self.call_count = 0
            
            def chat(self, messages, model):
                self.call_count += 1
                if self.call_count % 3 == 0:  # Fail every 3rd call
                    return None
                return "3"
        
        cost_estimator = Mock(spec=CostEstimator)
        cost_estimator.estimate_cost.return_value = 50.0
        
        judge = Judge({}, client=UnreliableClient())
        evaluator = Evaluator(judge, cost_estimator)
        
        # Test multiple evaluations
        results = []
        for i in range(5):
            result = evaluator.evaluate(
                diagnosis=f"diagnosis_{i}",
                truth=f"truth_{i}",
                tests=["CBC"],
                visits=1
            )
            results.append(result)
        
        # Judge should handle LLM failures gracefully (fallback to score 1)
        assert len(results) == 5
        for result in results:
            assert 1 <= result.score <= 5
            # Failed LLM calls should result in score 1
            if result.score == 1:
                assert result.correct is False  # Below threshold


class TestJudgeScalability:
    """Test Judge scalability characteristics."""
    
    def test_linear_scalability(self):
        """Test that Judge performance scales linearly with load."""
        judge = Judge({}, client=PerformanceTestClient(response_delay=0.001))
        
        load_sizes = [10, 20, 50, 100]
        times = []
        
        for load_size in load_sizes:
            start_time = time.perf_counter()
            
            for i in range(load_size):
                result = judge.evaluate(f"diagnosis_{i}", f"truth_{i}")
                assert result.score == 3
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            times.append((load_size, total_time))
        
        # Check for linear scaling
        for i in range(1, len(times)):
            prev_load, prev_time = times[i-1]
            curr_load, curr_time = times[i]
            
            expected_time_ratio = curr_load / prev_load
            actual_time_ratio = curr_time / prev_time
            
            # Allow some variance but should be roughly linear
            assert 0.5 <= actual_time_ratio / expected_time_ratio <= 2.0, \
                f"Non-linear scaling detected: {actual_time_ratio:.2f} vs expected {expected_time_ratio:.2f}"
    
    def test_resource_usage_scaling(self):
        """Test resource usage under different scales."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        judge = Judge({}, client=PerformanceTestClient(response_delay=0.001))
        
        initial_memory = process.memory_info().rss
        
        # Run evaluations in batches
        batch_sizes = [10, 50, 100]
        memory_usage = []
        
        for batch_size in batch_sizes:
            # Run batch
            for i in range(batch_size):
                result = judge.evaluate(f"test_{i}", f"truth_{i}")
                assert result.score == 3
            
            # Force garbage collection
            gc.collect()
            
            # Measure memory
            current_memory = process.memory_info().rss
            memory_growth = current_memory - initial_memory
            memory_usage.append((batch_size, memory_growth))
        
        # Memory growth should be bounded
        for batch_size, memory_growth in memory_usage:
            # Allow reasonable memory growth (less than 10MB per 100 evaluations)
            max_allowed_growth = (batch_size / 100) * 10 * 1024 * 1024
            assert memory_growth <= max_allowed_growth, \
                f"Excessive memory growth: {memory_growth} bytes for {batch_size} evaluations"


class TestJudgeReliability:
    """Test Judge reliability and fault tolerance."""
    
    def test_fault_tolerance_client_failures(self):
        """Test Judge behavior when LLM client fails intermittently."""
        class FlakyClient:
            def __init__(self, failure_rate=0.2):
                self.failure_rate = failure_rate
                self.call_count = 0
                import random
                self.random = random.Random(42)
            
            def chat(self, messages, model):
                self.call_count += 1
                if self.random.random() < self.failure_rate:
                    return None  # Simulate failure
                return "3"
        
        judge = Judge({}, client=FlakyClient(failure_rate=0.3))
        
        # Run many evaluations
        results = []
        failures = 0
        
        for i in range(50):
            result = judge.evaluate(f"diagnosis_{i}", f"truth_{i}")
            results.append(result.score)
            
            if result.score == 1:  # Indicates fallback due to failure
                failures += 1
        
        # Should handle failures gracefully
        assert len(results) == 50
        assert all(1 <= score <= 5 for score in results)
        
        # Failure rate should be roughly as expected
        expected_failures = 50 * 0.3
        assert abs(failures - expected_failures) <= 10, \
            f"Unexpected failure handling: {failures} vs expected ~{expected_failures}"
    
    def test_graceful_degradation(self):
        """Test Judge graceful degradation under stress."""
        class StressedClient:
            def __init__(self):
                self.call_count = 0
                self.stress_threshold = 20
            
            def chat(self, messages, model):
                self.call_count += 1
                
                if self.call_count > self.stress_threshold:
                    # Simulate degraded performance
                    time.sleep(0.1)  # Increased latency
                    if self.call_count % 5 == 0:
                        return None  # Occasional failures
                
                return "3"
        
        judge = Judge({}, client=StressedClient())
        
        # Run evaluations past stress threshold
        start_time = time.perf_counter()
        results = []
        
        for i in range(30):
            result = judge.evaluate(f"diagnosis_{i}", f"truth_{i}")
            results.append(result.score)
        
        end_time = time.perf_counter()
        
        # Should complete all evaluations despite stress
        assert len(results) == 30
        assert all(1 <= score <= 5 for score in results)
        
        # Performance should degrade but remain functional
        avg_time = (end_time - start_time) / 30
        assert avg_time <= 0.2, f"Excessive degradation: {avg_time:.3f}s per evaluation"