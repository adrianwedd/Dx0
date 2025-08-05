"""
Performance and integration tests for CostEstimator component.

This module focuses on performance benchmarking, load testing, and integration
testing with external systems and large datasets.
"""

import csv
import json
import os
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import Mock, patch, MagicMock
import statistics

import pytest

from sdb.cost_estimator import CostEstimator, CptCost
from sdb import cost_estimator as ce_mod
try:
    from .cost_estimator_test_data import (
        ALL_TEST_ENTRIES, generate_large_test_dataset, create_temp_csv_file,
        get_benchmark_expectations, MOCK_CMS_RESPONSE_DATA
    )
except ImportError:
    from cost_estimator_test_data import (
        ALL_TEST_ENTRIES, generate_large_test_dataset, create_temp_csv_file,
        get_benchmark_expectations, MOCK_CMS_RESPONSE_DATA
    )


class TestCostEstimatorPerformance:
    """Performance testing for cost estimator operations."""
    
    @pytest.fixture
    def small_dataset(self):
        """Small dataset for basic performance testing."""
        entries = ALL_TEST_ENTRIES[:20]  # 20 entries
        csv_path = create_temp_csv_file(entries)
        estimator = CostEstimator.load_from_csv(csv_path)
        os.unlink(csv_path)
        return estimator
    
    @pytest.fixture
    def medium_dataset(self):
        """Medium dataset for moderate performance testing."""
        entries = generate_large_test_dataset(1000)  # 1K entries
        csv_path = create_temp_csv_file(entries)
        estimator = CostEstimator.load_from_csv(csv_path)
        os.unlink(csv_path)
        return estimator
    
    @pytest.fixture
    def large_dataset(self):
        """Large dataset for stress testing."""
        entries = generate_large_test_dataset(10000)  # 10K entries
        csv_path = create_temp_csv_file(entries)
        estimator = CostEstimator.load_from_csv(csv_path)
        os.unlink(csv_path)
        return estimator
    
    def test_single_lookup_performance(self, medium_dataset):
        """Test single lookup performance across different dataset sizes."""
        estimator = medium_dataset
        
        # Warm up
        estimator.lookup_cost("test_0000")
        
        # Measure single lookup performance
        times = []
        for i in range(100):
            test_name = f"test_{i:04d}"
            start_time = time.perf_counter()
            estimator.lookup_cost(test_name)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = statistics.mean(times)
        median_time = statistics.median(times)
        max_time = max(times)
        
        benchmark = get_benchmark_expectations()
        
        assert avg_time < benchmark["lookup_performance_threshold"], \
            f"Average lookup time {avg_time:.6f}s exceeds threshold"
        assert median_time < benchmark["lookup_performance_threshold"], \
            f"Median lookup time {median_time:.6f}s exceeds threshold"
        assert max_time < benchmark["lookup_performance_threshold"] * 10, \
            f"Max lookup time {max_time:.6f}s is excessive"
    
    def test_bulk_lookup_performance(self, large_dataset):
        """Test bulk lookup performance."""
        estimator = large_dataset
        
        # Test bulk lookups
        test_names = [f"test_{i:04d}" for i in range(1000)]
        
        start_time = time.perf_counter()
        results = []
        for name in test_names:
            try:
                result = estimator.lookup_cost(name)
                results.append(result)
            except KeyError:
                results.append(None)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time_per_lookup = total_time / len(test_names)
        
        benchmark = get_benchmark_expectations()
        
        assert total_time < benchmark["bulk_load_threshold"], \
            f"Bulk lookup time {total_time:.2f}s exceeds threshold"
        assert avg_time_per_lookup < benchmark["lookup_performance_threshold"], \
            f"Average bulk lookup time {avg_time_per_lookup:.6f}s exceeds threshold"
        
        # Verify results
        successful_lookups = [r for r in results if r is not None]
        assert len(successful_lookups) >= len(test_names) * 0.8, \
            "Should find at least 80% of test entries"
    
    def test_estimate_function_performance(self, medium_dataset, monkeypatch):
        """Test performance of estimate function with LLM fallbacks."""
        estimator = medium_dataset
        
        # Mock LLM to avoid external calls
        monkeypatch.setattr(ce_mod, "lookup_cpt", lambda name: None)
        
        # Test estimate function (will use average fallback)
        test_names = ["unknown_test_" + str(i) for i in range(100)]
        
        start_time = time.perf_counter()
        results = []
        for name in test_names:
            price, category = estimator.estimate(name)
            results.append((price, category))
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time_per_estimate = total_time / len(test_names)
        
        # Should be fast since it's just computing averages
        assert avg_time_per_estimate < 0.001, \
            f"Average estimate time {avg_time_per_estimate:.6f}s is too slow"
        
        # Verify all estimates returned reasonable values
        assert all(price > 0 for price, category in results)
        assert all(category == "unknown" for price, category in results)
    
    def test_concurrent_lookup_performance(self, medium_dataset):
        """Test performance under concurrent access."""
        estimator = medium_dataset
        
        def worker_function(worker_id, num_lookups):
            times = []
            for i in range(num_lookups):
                test_name = f"test_{(worker_id * num_lookups + i) % 1000:04d}"
                start_time = time.perf_counter()
                try:
                    estimator.lookup_cost(test_name)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                except KeyError:
                    pass
            return times
        
        num_workers = 10
        lookups_per_worker = 100
        
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(worker_function, i, lookups_per_worker)
                for i in range(num_workers)
            ]
            
            all_times = []
            for future in as_completed(futures):
                all_times.extend(future.result())
        end_time = time.perf_counter()
        
        total_concurrent_time = end_time - start_time
        total_lookups = num_workers * lookups_per_worker
        
        if all_times:
            avg_lookup_time = statistics.mean(all_times)
            max_lookup_time = max(all_times)
            
            benchmark = get_benchmark_expectations()
            
            # Concurrent performance might be slightly slower
            assert avg_lookup_time < benchmark["lookup_performance_threshold"] * 2, \
                f"Concurrent average lookup time {avg_lookup_time:.6f}s exceeds threshold"
            assert max_lookup_time < benchmark["lookup_performance_threshold"] * 10, \
                f"Concurrent max lookup time {max_lookup_time:.6f}s is excessive"
        
        # Overall throughput check
        throughput = total_lookups / total_concurrent_time
        assert throughput > 1000, f"Concurrent throughput {throughput:.1f} lookups/sec is too low"
    
    def test_memory_usage_scaling(self, large_dataset):
        """Test memory usage patterns with large datasets."""
        estimator = large_dataset
        
        # Basic checks for memory efficiency
        table_size = len(estimator.cost_table)
        alias_size = len(estimator.aliases)
        
        assert table_size == 10000, "Expected 10K entries in large dataset"
        assert alias_size >= 0, "Aliases should be non-negative"
        
        # Test that repeated operations don't cause memory leaks
        initial_alias_count = len(estimator.aliases)
        
        # Perform operations that might accumulate memory
        for i in range(100):
            estimator.add_aliases({f"temp_alias_{i}": f"test_{i:04d}"})
            
            # Perform some lookups
            for j in range(10):
                try:
                    estimator.lookup_cost(f"test_{j:04d}")
                except KeyError:
                    pass
        
        # Check that alias count increased predictably
        final_alias_count = len(estimator.aliases)
        expected_increase = 100  # We added 100 aliases
        
        assert final_alias_count >= initial_alias_count + expected_increase, \
            "Aliases were not added correctly"
        assert final_alias_count <= initial_alias_count + expected_increase + 10, \
            "Unexpected alias growth detected"
    
    def test_csv_loading_performance(self, tmp_path):
        """Test CSV loading performance with various file sizes."""
        file_sizes = [100, 1000, 10000]
        
        for size in file_sizes:
            # Generate test data
            entries = generate_large_test_dataset(size)
            csv_path = tmp_path / f"perf_test_{size}.csv"
            
            # Create CSV file
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["test_name", "cpt_code", "price", "category"])
                for entry in entries:
                    writer.writerow([entry.test_name, entry.cpt_code, entry.price, entry.category])
            
            # Measure loading time
            start_time = time.perf_counter()
            estimator = CostEstimator.load_from_csv(str(csv_path))
            end_time = time.perf_counter()
            
            load_time = end_time - start_time
            entries_per_second = size / load_time if load_time > 0 else float('inf')
            
            # Performance expectations scale with file size
            max_time = min(5.0, size / 1000)  # Max 5 seconds, or 1ms per entry
            assert load_time < max_time, \
                f"Loading {size} entries took {load_time:.2f}s (expected < {max_time:.2f}s)"
            
            # Verify loaded correctly
            assert len(estimator.cost_table) == size
            
            print(f"Loaded {size} entries in {load_time:.3f}s ({entries_per_second:.0f} entries/sec)")


class TestConcurrencyAndThreadSafety:
    """Test concurrency and thread safety of cost estimator operations."""
    
    @pytest.fixture
    def shared_estimator(self):
        """Shared estimator for concurrency testing."""
        entries = generate_large_test_dataset(1000)
        csv_path = create_temp_csv_file(entries)
        estimator = CostEstimator.load_from_csv(csv_path)
        os.unlink(csv_path)
        return estimator
    
    def test_concurrent_read_operations(self, shared_estimator):
        """Test concurrent read operations for thread safety."""
        estimator = shared_estimator
        
        def read_worker(worker_id, iterations):
            results = []
            for i in range(iterations):
                test_name = f"test_{(worker_id * iterations + i) % 1000:04d}"
                try:
                    result = estimator.lookup_cost(test_name)
                    results.append((test_name, result.price, result.category))
                except KeyError:
                    results.append((test_name, None, None))
            return results
        
        num_workers = 20
        iterations_per_worker = 50
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(read_worker, i, iterations_per_worker)
                for i in range(num_workers)
            ]
            
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())
        
        # Verify all operations completed successfully
        assert len(all_results) == num_workers * iterations_per_worker
        
        # Verify data consistency across threads
        successful_results = [r for r in all_results if r[1] is not None]
        assert len(successful_results) > 0
        
        # Check that same test names return same results
        result_dict = {}
        for test_name, price, category in successful_results:
            if test_name not in result_dict:
                result_dict[test_name] = (price, category)
            else:
                # Same test should return same price and category
                assert result_dict[test_name] == (price, category), \
                    f"Inconsistent results for {test_name}"
    
    def test_concurrent_write_operations(self, shared_estimator):
        """Test concurrent write operations (alias additions)."""
        estimator = shared_estimator
        
        def write_worker(worker_id, iterations):
            for i in range(iterations):
                alias_name = f"worker_{worker_id}_alias_{i}"
                canonical_name = f"test_{i % 100:04d}"
                estimator.add_aliases({alias_name: canonical_name})
            return worker_id
        
        num_workers = 10
        iterations_per_worker = 20
        total_aliases = num_workers * iterations_per_worker
        
        initial_alias_count = len(estimator.aliases)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(write_worker, i, iterations_per_worker)
                for i in range(num_workers)
            ]
            
            # Wait for all workers to complete
            completed_workers = []
            for future in as_completed(futures):
                completed_workers.append(future.result())
        
        final_alias_count = len(estimator.aliases)
        
        # Should have added all aliases
        assert final_alias_count >= initial_alias_count + total_aliases, \
            f"Expected at least {total_aliases} new aliases, got {final_alias_count - initial_alias_count}"
        
        # Verify some of the added aliases work
        for worker_id in range(min(5, num_workers)):  # Test first 5 workers
            alias_name = f"worker_{worker_id}_alias_0"
            if alias_name in estimator.aliases:
                # Should be able to lookup through alias
                canonical = estimator.aliases[alias_name]
                result = estimator.lookup_cost(alias_name)
                assert result is not None
    
    def test_mixed_concurrent_operations(self, shared_estimator):
        """Test mixed concurrent read/write operations."""
        estimator = shared_estimator
        
        def reader_worker(worker_id):
            read_count = 0
            for i in range(100):
                test_name = f"test_{i:04d}"
                try:
                    estimator.lookup_cost(test_name)
                    read_count += 1
                except KeyError:
                    pass
            return ("reader", worker_id, read_count)
        
        def writer_worker(worker_id):
            write_count = 0
            for i in range(20):
                alias_name = f"mixed_worker_{worker_id}_alias_{i}"
                canonical_name = f"test_{i % 10:04d}"
                estimator.add_aliases({alias_name: canonical_name})
                write_count += 1
            return ("writer", worker_id, write_count)
        
        # Start mixed workload
        with ThreadPoolExecutor(max_workers=15) as executor:
            # Submit reader tasks
            reader_futures = [executor.submit(reader_worker, i) for i in range(10)]
            # Submit writer tasks
            writer_futures = [executor.submit(writer_worker, i) for i in range(5)]
            
            all_futures = reader_futures + writer_futures
            results = []
            
            for future in as_completed(all_futures):
                results.append(future.result())
        
        # Analyze results
        reader_results = [r for r in results if r[0] == "reader"]
        writer_results = [r for r in results if r[0] == "writer"]
        
        assert len(reader_results) == 10, "All reader workers should complete"
        assert len(writer_results) == 5, "All writer workers should complete"
        
        # Readers should have found some entries
        total_reads = sum(r[2] for r in reader_results)
        assert total_reads > 0, "Readers should have found some entries"
        
        # Writers should have added some aliases
        total_writes = sum(r[2] for r in writer_results)
        assert total_writes == 5 * 20, "All writes should have succeeded"
    
    def test_process_based_concurrency(self, tmp_path):
        """Test process-based concurrency (for true parallelism)."""
        # Create a shared CSV file
        entries = generate_large_test_dataset(1000)
        csv_path = tmp_path / "process_test.csv"
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["test_name", "cpt_code", "price", "category"])
            for entry in entries:
                writer.writerow([entry.test_name, entry.cpt_code, entry.price, entry.category])
        
        def process_worker(csv_file_path, worker_id, num_lookups):
            """Worker function that runs in separate process."""
            from sdb.cost_estimator import CostEstimator
            
            # Each process loads its own estimator
            estimator = CostEstimator.load_from_csv(csv_file_path)
            
            successful_lookups = 0
            for i in range(num_lookups):
                test_name = f"test_{(worker_id * num_lookups + i) % 1000:04d}"
                try:
                    result = estimator.lookup_cost(test_name)
                    if result:
                        successful_lookups += 1
                except KeyError:
                    pass
            
            return worker_id, successful_lookups
        
        num_processes = 4
        lookups_per_process = 100
        
        start_time = time.perf_counter()
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(process_worker, str(csv_path), i, lookups_per_process)
                for i in range(num_processes)
            ]
            
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        total_lookups = num_processes * lookups_per_process
        
        # Verify all processes completed
        assert len(results) == num_processes
        
        # Verify successful lookups
        total_successful = sum(r[1] for r in results)
        success_rate = total_successful / total_lookups
        
        assert success_rate > 0.8, f"Success rate {success_rate:.1%} too low"
        
        # Performance check
        throughput = total_lookups / total_time
        assert throughput > 500, f"Process-based throughput {throughput:.1f} lookups/sec too low"


class TestIntegrationWithExternalSystems:
    """Test integration with external systems and APIs."""
    
    def test_cms_api_integration_simulation(self, tmp_path, monkeypatch):
        """Simulate integration with CMS pricing API."""
        def mock_cms_fetch(*args, **kwargs):
            """Mock CMS API response."""
            class MockResponse:
                def __init__(self):
                    self.status_code = 200
                    self.content = b'''cpt_code,rate
85027,18.25
80048,14.75
71046,68.50
70450,365.00'''
                
                def raise_for_status(self):
                    pass
            
            return MockResponse()
        
        # Create initial cost table
        entries = ALL_TEST_ENTRIES[:5]
        csv_path = create_temp_csv_file(entries)
        
        # Create CMS pricing file for coverage validation
        cms_csv_path = tmp_path / "cms_pricing.csv"
        with open(cms_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["cpt_code"])
            for entry in entries:
                writer.writerow([entry.cpt_code])
        
        # Mock external HTTP request
        with patch('sdb.http_utils.get_client') as mock_client:
            mock_client.return_value.get.return_value = mock_cms_fetch()
            
            # Test loading with CMS validation
            estimator = CostEstimator.load_from_csv(
                csv_path,
                cms_pricing_path=str(cms_csv_path),
                coverage_threshold=0.8
            )
            
            assert estimator.match_rate >= 0.8
            assert len(estimator.cost_table) > 0
        
        os.unlink(csv_path)
    
    def test_llm_integration_performance(self, monkeypatch):
        """Test LLM integration performance characteristics."""
        # Create estimator with limited data to force LLM lookups
        cost_data = {
            "known_test": CptCost("85027", 25.0, "lab")
        }
        estimator = CostEstimator(cost_data)
        
        # Mock LLM with realistic response times
        call_count = 0
        def mock_llm_with_delay(test_name):
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)  # Simulate 100ms API call
            
            # Return CPT codes for some tests
            responses = {
                "blood work": "85027",
                "metabolic panel": "80048",
                "x-ray": "71046"
            }
            return responses.get(test_name.lower())
        
        monkeypatch.setattr(ce_mod, "lookup_cpt", mock_llm_with_delay)
        
        # Test LLM lookup performance
        test_names = ["blood work", "metabolic panel", "x-ray", "unknown_test"]
        
        start_time = time.perf_counter()
        results = []
        for name in test_names:
            price = estimator.estimate_cost(name)
            results.append((name, price))
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        
        # Should have made LLM calls for unknown tests
        assert call_count > 0, "Should have made LLM calls"
        
        # Time should reflect LLM call overhead
        expected_min_time = call_count * 0.09  # Allow some variance
        assert total_time >= expected_min_time, \
            f"Total time {total_time:.2f}s seems too fast for {call_count} LLM calls"
        
        # But shouldn't be excessively slow
        assert total_time < call_count * 0.5, \
            f"Total time {total_time:.2f}s too slow for {call_count} LLM calls"
        
        # Verify some successful lookups and caching
        successful_llm_lookups = sum(1 for name, price in results if price == 25.0)
        assert successful_llm_lookups > 0, "Should have some successful LLM lookups"
        
        # Verify aliases were cached
        assert len(estimator.aliases) > 0, "Should have cached some aliases"
    
    def test_external_system_failure_handling(self, tmp_path, monkeypatch):
        """Test handling of external system failures."""
        # Create basic cost table
        entries = ALL_TEST_ENTRIES[:3]
        csv_path = create_temp_csv_file(entries)
        
        # Simulate various external system failures
        failure_scenarios = [
            ConnectionError("Network unreachable"),
            TimeoutError("Request timed out"),
            ValueError("Invalid response format"),
            Exception("Unexpected error")
        ]
        
        for error in failure_scenarios:
            def failing_llm_lookup(test_name):
                raise error
            
            monkeypatch.setattr(ce_mod, "lookup_cpt", failing_llm_lookup)
            
            # Load estimator (should succeed despite LLM failures)
            estimator = CostEstimator.load_from_csv(csv_path)
            
            # Known lookups should still work
            result = estimator.lookup_cost(entries[0].test_name)
            assert result.price == entries[0].price
            
            # Unknown lookups should fallback gracefully
            unknown_price = estimator.estimate_cost("unknown_test")
            assert unknown_price > 0, "Should fallback to average price"
            assert isinstance(unknown_price, (int, float))
        
        os.unlink(csv_path)
    
    def test_data_freshness_and_caching(self, tmp_path):
        """Test data freshness detection and caching strategies."""
        # Create initial CSV file
        csv_path = tmp_path / "freshness_test.csv"
        
        initial_data = [
            {"test_name": "test_a", "cpt_code": "85027", "price": "25.00"},
            {"test_name": "test_b", "cpt_code": "80048", "price": "15.00"},
        ]
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
            writer.writeheader()
            writer.writerows(initial_data)
        
        # Load estimator
        estimator1 = CostEstimator.load_from_csv(str(csv_path))
        assert estimator1.lookup_cost("test_a").price == 25.00
        
        # Modify CSV file (simulate data update)
        time.sleep(0.1)  # Ensure different timestamp
        updated_data = [
            {"test_name": "test_a", "cpt_code": "85027", "price": "30.00"},  # Updated price
            {"test_name": "test_b", "cpt_code": "80048", "price": "15.00"},
            {"test_name": "test_c", "cpt_code": "71046", "price": "89.00"},  # New entry
        ]
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
            writer.writeheader()
            writer.writerows(updated_data)
        
        # Load new estimator (should pick up changes)
        estimator2 = CostEstimator.load_from_csv(str(csv_path))
        assert estimator2.lookup_cost("test_a").price == 30.00  # Updated price
        assert estimator2.lookup_cost("test_c").price == 89.00   # New entry
        
        # Original estimator should still have old data
        assert estimator1.lookup_cost("test_a").price == 25.00
        
        # New entry shouldn't exist in old estimator
        with pytest.raises(KeyError):
            estimator1.lookup_cost("test_c")


class TestLoadTesting:
    """Load testing and stress testing for cost estimator."""
    
    def test_high_volume_lookup_load(self, tmp_path):
        """Test performance under high-volume lookup loads."""
        # Create large dataset
        entries = generate_large_test_dataset(50000)  # 50K entries
        csv_path = tmp_path / "load_test.csv"
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["test_name", "cpt_code", "price", "category"])
            for entry in entries:
                writer.writerow([entry.test_name, entry.cpt_code, entry.price, entry.category])
        
        # Load estimator
        load_start = time.perf_counter()
        estimator = CostEstimator.load_from_csv(str(csv_path))
        load_end = time.perf_counter()
        
        load_time = load_end - load_start
        assert load_time < 30.0, f"Loading 50K entries took {load_time:.2f}s (too slow)"
        
        # High-volume lookup test
        num_lookups = 10000
        test_names = [f"test_{i:05d}" for i in range(num_lookups)]
        
        lookup_start = time.perf_counter()
        successful_lookups = 0
        for name in test_names:
            try:
                result = estimator.lookup_cost(name)
                if result:
                    successful_lookups += 1
            except KeyError:
                pass
        lookup_end = time.perf_counter()
        
        lookup_time = lookup_end - lookup_start
        throughput = num_lookups / lookup_time
        
        assert throughput > 5000, f"Lookup throughput {throughput:.0f} lookups/sec too low"
        assert successful_lookups > num_lookups * 0.8, "Should find most entries"
        
        print(f"Load test: {successful_lookups}/{num_lookups} lookups in {lookup_time:.2f}s "
              f"({throughput:.0f} lookups/sec)")
    
    def test_memory_stress_test(self, tmp_path):
        """Test behavior under memory stress conditions."""
        # Create multiple large estimators to stress memory
        estimators = []
        
        for i in range(5):  # Create 5 estimators
            entries = generate_large_test_dataset(10000)  # 10K entries each
            csv_path = tmp_path / f"memory_test_{i}.csv"
            
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["test_name", "cpt_code", "price", "category"])
                for entry in entries:
                    writer.writerow([entry.test_name, entry.cpt_code, entry.price, entry.category])
            
            estimator = CostEstimator.load_from_csv(str(csv_path))
            estimators.append(estimator)
        
        # Verify all estimators work
        for i, estimator in enumerate(estimators):
            assert len(estimator.cost_table) == 10000
            
            # Test some lookups on each
            for j in range(10):
                test_name = f"test_{j:04d}"
                try:
                    result = estimator.lookup_cost(test_name)
                    assert result is not None
                except KeyError:
                    pass  # Some entries might not exist
        
        # Add aliases to stress memory further
        for i, estimator in enumerate(estimators):
            for j in range(1000):
                alias_name = f"stress_alias_{i}_{j}"
                canonical_name = f"test_{j % 100:04d}"
                estimator.add_aliases({alias_name: canonical_name})
        
        # Verify system is still responsive
        total_aliases = sum(len(est.aliases) for est in estimators)
        assert total_aliases >= 5000, "Should have added many aliases"
        
        # Test lookups still work
        for estimator in estimators[:2]:  # Test first 2 estimators
            for j in range(10):
                alias_name = f"stress_alias_0_{j}"
                if alias_name in estimator.aliases:
                    result = estimator.lookup_cost(alias_name) 
                    assert result is not None
    
    def test_concurrent_load_stress(self, tmp_path):
        """Test system under concurrent load stress."""
        # Create shared dataset
        entries = generate_large_test_dataset(5000)
        csv_path = tmp_path / "concurrent_stress.csv"
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["test_name", "cpt_code", "price", "category"])
            for entry in entries:
                writer.writerow([entry.test_name, entry.cpt_code, entry.price, entry.category])
        
        estimator = CostEstimator.load_from_csv(str(csv_path))
        
        def stress_worker(worker_id, duration_seconds):
            """Worker that performs operations for specified duration."""
            end_time = time.time() + duration_seconds
            operations = 0
            
            while time.time() < end_time:
                # Mix of different operations
                if operations % 10 == 0:
                    # Add alias
                    alias_name = f"stress_{worker_id}_{operations}"
                    canonical_name = f"test_{operations % 1000:04d}"
                    estimator.add_aliases({alias_name: canonical_name})
                else:
                    # Lookup operation
                    test_name = f"test_{operations % 5000:04d}"
                    try:
                        estimator.lookup_cost(test_name)
                    except KeyError:
                        pass
                
                operations += 1
            
            return worker_id, operations
        
        # Run stress test for 5 seconds with 20 workers
        duration = 5.0
        num_workers = 20
        
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(stress_worker, i, duration)
                for i in range(num_workers)
            ]
            
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        end_time = time.perf_counter()
        
        actual_duration = end_time - start_time
        total_operations = sum(r[1] for r in results)
        
        # Verify stress test completed
        assert len(results) == num_workers, "All workers should complete"
        assert actual_duration >= duration * 0.9, "Should run for requested duration"
        
        # Performance metrics
        operations_per_second = total_operations / actual_duration
        assert operations_per_second > 10000, \
            f"Operations rate {operations_per_second:.0f} ops/sec too low under stress"
        
        print(f"Stress test: {total_operations} operations in {actual_duration:.2f}s "
              f"({operations_per_second:.0f} ops/sec)")


if __name__ == "__main__":
    # Run basic performance benchmark
    pytest.main([__file__ + "::TestCostEstimatorPerformance::test_single_lookup_performance", "-v"])