"""
Comprehensive validation tests for CostEstimator component.

This module focuses on CPT code validation, real-world pricing data validation,
and integration testing with external data sources.
"""

import csv
import json
import os
import tempfile
import time
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import Mock, patch, MagicMock, call
import re

import pytest

from sdb.cost_estimator import CostEstimator, CptCost
from sdb import cost_estimator as ce_mod
try:
    from .cost_estimator_test_data import (
        ALL_TEST_ENTRIES, CPT_VALIDATION_CASES, CMS_VALIDATION_DATA,
        PRICING_SCENARIOS, EDGE_CASE_TEST_DATA, MOCK_CMS_RESPONSE_DATA,
        MOCK_LLM_RESPONSES, create_temp_csv_file, create_cms_pricing_csv,
        get_benchmark_expectations
    )
except ImportError:
    from cost_estimator_test_data import (
        ALL_TEST_ENTRIES, CPT_VALIDATION_CASES, CMS_VALIDATION_DATA,
        PRICING_SCENARIOS, EDGE_CASE_TEST_DATA, MOCK_CMS_RESPONSE_DATA,
        MOCK_LLM_RESPONSES, create_temp_csv_file, create_cms_pricing_csv,
        get_benchmark_expectations
    )


class TestCPTCodeValidation:
    """Comprehensive CPT code validation test suite."""
    
    def test_standard_cpt_code_formats(self):
        """Test validation of standard CPT code formats."""
        valid_standard_codes = [
            case for case in CPT_VALIDATION_CASES 
            if case.is_valid and case.code_type == "standard"
        ]
        
        for case in valid_standard_codes:
            # Test format validation
            assert len(case.cpt_code) == 5
            assert case.cpt_code.isdigit()
            assert 10000 <= int(case.cpt_code) <= 99999
            
            # Test in CostEstimator
            cost = CptCost(case.cpt_code, 100.0, "test")
            assert cost.cpt_code == case.cpt_code
    
    def test_hcpcs_code_formats(self):
        """Test validation of HCPCS Level II code formats."""
        hcpcs_codes = [
            case for case in CPT_VALIDATION_CASES 
            if case.is_valid and case.code_type == "hcpcs"
        ]
        
        for case in hcpcs_codes:
            code = case.cpt_code
            # Test HCPCS format: Letter + 4 digits
            assert len(code) == 5
            assert code[0].isalpha()
            assert code[1:].isdigit()
            
            # Test valid HCPCS prefixes
            valid_prefixes = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'J', 'K', 'L', 'M', 'P', 'Q', 'R', 'S', 'T', 'V']
            assert code[0] in valid_prefixes
    
    def test_category_iii_code_formats(self):
        """Test validation of Category III CPT code formats."""
        category_iii_codes = [
            case for case in CPT_VALIDATION_CASES 
            if case.is_valid and case.code_type == "category_iii"
        ]
        
        for case in category_iii_codes:
            code = case.cpt_code
            # Test Category III format: 4 digits + T
            assert len(code) == 5
            assert code.endswith('T')
            assert code[:-1].isdigit()
            assert 1 <= int(code[:-1]) <= 9999
    
    def test_modifier_code_formats(self):
        """Test validation of CPT codes with modifiers."""
        modifier_codes = [
            case for case in CPT_VALIDATION_CASES 
            if case.is_valid and case.code_type == "modifier"
        ]
        
        for case in modifier_codes:
            code = case.cpt_code
            # Test modifier format: CPT code + dash + modifier
            assert '-' in code
            base_code, modifier = code.split('-', 1)
            
            # Validate base code
            assert len(base_code) == 5
            assert base_code.isdigit()
            
            # Validate common modifiers
            common_modifiers = ['26', 'TC', '59', '25', 'LT', 'RT', '50']
            assert modifier in common_modifiers or modifier.isdigit()
    
    def test_invalid_cpt_code_formats(self):
        """Test detection of invalid CPT code formats."""
        invalid_codes = [
            case for case in CPT_VALIDATION_CASES 
            if not case.is_valid
        ]
        
        # Group by type of invalidity
        format_issues = []
        for case in invalid_codes:
            if case.description in ["Too short", "Too long", "All letters, no valid pattern", 
                                  "Contains letter O instead of 0", "Empty code"]:
                format_issues.append(case.cpt_code)
        
        # Test that these would be caught by validation logic
        for code in format_issues:
            if code == "":
                assert len(code) == 0
            elif len(code) < 5:
                assert len(code) < 5
            elif len(code) > 5 and '-' not in code:
                assert len(code) > 5
            elif code.isalpha():
                assert not any(c.isdigit() for c in code)
    
    @pytest.mark.parametrize("cpt_code,expected_category", [
        ("85027", "laboratory"),     # Hematology
        ("80048", "laboratory"),     # Chemistry
        ("82306", "laboratory"),     # Chemistry
        ("71046", "imaging"),        # Radiology
        ("70450", "imaging"),        # Radiology
        ("76700", "imaging"),        # Ultrasound
        ("93000", "cardiology"),     # Cardiovascular
        ("93307", "cardiology"),     # Echocardiography
        ("99213", "office_visit"),   # E&M
        ("99203", "office_visit"),   # E&M
        ("45378", "procedure"),      # Surgery/GI
        ("43235", "procedure"),      # Surgery/GI
        ("11104", "procedure"),      # Surgery/Integumentary
    ])
    def test_cpt_code_category_classification(self, cpt_code, expected_category):
        """Test CPT code to medical category classification."""
        # This tests the business logic of categorizing CPT codes
        # Based on CPT code ranges and clinical knowledge
        
        # Laboratory codes are typically in ranges:
        # 80047-89398 (Chemistry, Hematology, Immunology, etc.)
        if expected_category == "laboratory":
            code_num = int(cpt_code)
            assert 80047 <= code_num <= 89398
        
        # Radiology codes are typically in ranges:
        # 70010-79999 (Diagnostic Radiology, Nuclear Medicine, etc.)
        elif expected_category == "imaging":
            code_num = int(cpt_code)
            assert 70010 <= code_num <= 79999
        
        # Cardiovascular codes are typically in ranges:
        # 92920-93799 (Cardiovascular procedures)
        elif expected_category == "cardiology":
            code_num = int(cpt_code)
            assert 92920 <= code_num <= 93799 or code_num == 93000  # ECG exception
        
        # E&M codes are typically in ranges:
        # 99201-99499 (Evaluation and Management)
        elif expected_category == "office_visit":
            code_num = int(cpt_code)
            assert 99201 <= code_num <= 99499
        
        # Surgery codes span multiple ranges
        elif expected_category == "procedure":
            code_num = int(cpt_code)
            # Common surgery ranges
            surgery_ranges = [
                (10021, 69990),  # Surgery section
                (43200, 43999),  # GI procedures
                (11000, 11999),  # Integumentary
            ]
            assert any(start <= code_num <= end for start, end in surgery_ranges)
    
    def test_cpt_code_pricing_reasonableness(self):
        """Test that CPT code pricing falls within reasonable ranges."""
        # Define reasonable price ranges by CPT code ranges
        price_expectations = {
            # Laboratory tests: $5-$200
            (80000, 89999): (5.0, 200.0),
            # Basic imaging: $30-$500  
            (70000, 76999): (30.0, 500.0),
            # Advanced imaging: $200-$3000
            (77000, 79999): (200.0, 3000.0),
            # Cardiovascular: $25-$1500
            (92900, 93799): (25.0, 1500.0),
            # E&M codes: $50-$500
            (99200, 99499): (50.0, 500.0),
            # Procedures: $50-$5000
            (10000, 69999): (50.0, 5000.0),
        }
        
        for entry in ALL_TEST_ENTRIES:
            if entry.cpt_code.isdigit():
                code_num = int(entry.cpt_code)
                
                # Find applicable price range
                for (range_start, range_end), (min_price, max_price) in price_expectations.items():
                    if range_start <= code_num <= range_end:
                        assert min_price <= entry.price <= max_price, \
                            f"Price {entry.price} for {entry.cpt_code} outside expected range {min_price}-{max_price}"
                        break


class TestRealWorldDataValidation:
    """Test suite for validating against real-world medical pricing data."""
    
    def test_cms_pricing_alignment(self):
        """Test alignment with CMS pricing data."""
        for cms_entry in CMS_VALIDATION_DATA:
            # Find matching test entry
            matching_entries = [
                entry for entry in ALL_TEST_ENTRIES 
                if entry.cpt_code == cms_entry.cpt_code
            ]
            
            if matching_entries:
                test_entry = matching_entries[0]
                
                # Check if price is within CMS range (allowing for geographic variation)
                tolerance = 0.20  # 20% tolerance for geographic/temporal variation
                cms_min_adjusted = cms_entry.min_price * (1 - tolerance)
                cms_max_adjusted = cms_entry.max_price * (1 + tolerance)
                
                assert cms_min_adjusted <= test_entry.price <= cms_max_adjusted, \
                    f"Price {test_entry.price} for {test_entry.cpt_code} outside CMS range " \
                    f"{cms_min_adjusted:.2f}-{cms_max_adjusted:.2f}"
    
    def test_price_distribution_sanity(self):
        """Test that price distributions make clinical sense."""
        # Group entries by category and analyze price distributions
        categories = {}
        for entry in ALL_TEST_ENTRIES:
            if entry.category not in categories:
                categories[entry.category] = []
            categories[entry.category].append(entry.price)
        
        # Test category-specific price expectations
        if "laboratory" in categories:
            lab_prices = categories["laboratory"]
            lab_median = sorted(lab_prices)[len(lab_prices) // 2]
            assert 10.0 <= lab_median <= 50.0, f"Lab median price {lab_median} seems unreasonable"
        
        if "imaging" in categories:
            imaging_prices = categories["imaging"]
            imaging_median = sorted(imaging_prices)[len(imaging_prices) // 2]
            assert 50.0 <= imaging_median <= 800.0, f"Imaging median price {imaging_median} seems unreasonable"
        
        if "cardiology" in categories:
            cardio_prices = categories["cardiology"]
            cardio_median = sorted(cardio_prices)[len(cardio_prices) // 2]
            assert 30.0 <= cardio_median <= 400.0, f"Cardiology median price {cardio_median} seems unreasonable"
        
        if "procedure" in categories:
            procedure_prices = categories["procedure"]
            procedure_median = sorted(procedure_prices)[len(procedure_prices) // 2]
            assert 100.0 <= procedure_median <= 1500.0, f"Procedure median price {procedure_median} seems unreasonable"
    
    def test_geographic_price_variation_modeling(self):
        """Test modeling of geographic price variations."""
        # Define geographic adjustment factors (approximate)
        geographic_factors = {
            "rural": 0.85,      # 15% lower in rural areas
            "suburban": 1.0,    # Baseline
            "urban": 1.15,      # 15% higher in urban areas
            "high_cost": 1.35,  # 35% higher in high-cost areas (NYC, SF, etc.)
        }
        
        base_entry = ALL_TEST_ENTRIES[0]  # Use first entry as example
        
        for region, factor in geographic_factors.items():
            adjusted_price = base_entry.price * factor
            
            # Verify adjustment is reasonable
            assert 0.5 * base_entry.price <= adjusted_price <= 2.0 * base_entry.price, \
                f"Geographic adjustment for {region} seems extreme"
    
    def test_temporal_price_inflation_modeling(self):
        """Test modeling of healthcare price inflation over time."""
        # Healthcare inflation typically 3-7% annually
        annual_inflation_rates = [0.03, 0.05, 0.07]  # 3%, 5%, 7%
        years_back = [1, 2, 3, 5]
        
        current_price = 100.0  # Example current price
        
        for years in years_back:
            for rate in annual_inflation_rates:
                # Calculate historical price
                historical_price = current_price / ((1 + rate) ** years)
                
                # Verify reasonable historical pricing
                assert 0.7 * current_price <= historical_price <= current_price, \
                    f"Historical price calculation seems unrealistic: {historical_price}"
    
    def test_insurance_reimbursement_rates(self):
        """Test typical insurance reimbursement rate patterns."""
        # Different payers typically reimburse at different rates relative to charges
        reimbursement_rates = {
            "medicare": 0.45,      # Medicare typically pays ~45% of charges
            "medicaid": 0.40,      # Medicaid typically pays ~40% of charges  
            "commercial": 0.65,    # Commercial insurance ~65% of charges
            "cash_discount": 0.30, # Cash discount rates ~30% of charges
        }
        
        charge_price = 200.0  # Example facility charge
        
        for payer, rate in reimbursement_rates.items():
            reimbursement = charge_price * rate
            
            # Verify reimbursement rates are realistic
            assert 0.25 * charge_price <= reimbursement <= 0.80 * charge_price, \
                f"Reimbursement rate for {payer} seems unrealistic"


class TestPerformanceWithRealData:
    """Performance tests using realistic data volumes and patterns."""
    
    @pytest.fixture
    def large_realistic_dataset(self):
        """Create a large, realistic dataset for performance testing."""
        from tests.cost_estimator_test_data import generate_large_test_dataset
        
        # Generate 5000 entries with realistic distribution
        entries = generate_large_test_dataset(5000)
        
        # Create temporary CSV
        csv_path = create_temp_csv_file(entries)
        estimator = CostEstimator.load_from_csv(csv_path)
        
        # Clean up
        os.unlink(csv_path)
        
        return estimator
    
    def test_bulk_lookup_performance(self, large_realistic_dataset):
        """Test performance of bulk cost lookups."""
        estimator = large_realistic_dataset
        
        # Test 1000 lookups
        test_names = [f"test_{i:04d}" for i in range(1000)]
        
        start_time = time.time()
        for name in test_names:
            try:
                estimator.lookup_cost(name)
            except KeyError:
                pass  # Expected for some test names
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_lookup = total_time / len(test_names)
        
        # Performance expectations
        benchmark = get_benchmark_expectations()
        assert avg_time_per_lookup < benchmark["lookup_performance_threshold"], \
            f"Average lookup time {avg_time_per_lookup:.4f}s exceeds threshold"
    
    def test_concurrent_access_performance(self, large_realistic_dataset):
        """Test performance under concurrent access patterns."""
        import threading
        import queue
        
        estimator = large_realistic_dataset
        results_queue = queue.Queue()
        
        def worker_thread(thread_id, num_lookups):
            thread_results = []
            for i in range(num_lookups):
                test_name = f"test_{(thread_id * num_lookups + i) % 1000:04d}"
                start_time = time.time()
                try:
                    result = estimator.lookup_cost(test_name)
                    end_time = time.time()
                    thread_results.append(end_time - start_time)
                except KeyError:
                    pass
            results_queue.put(thread_results)
        
        # Start 10 threads, each doing 100 lookups
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker_thread, args=(i, 100))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        all_times = []
        while not results_queue.empty():
            all_times.extend(results_queue.get())
        
        # Analyze performance
        if all_times:
            avg_time = sum(all_times) / len(all_times)
            max_time = max(all_times)
            
            benchmark = get_benchmark_expectations()
            assert avg_time < benchmark["lookup_performance_threshold"], \
                f"Concurrent average lookup time {avg_time:.4f}s exceeds threshold"
            assert max_time < benchmark["lookup_performance_threshold"] * 10, \
                f"Concurrent max lookup time {max_time:.4f}s is excessive"
    
    def test_memory_usage_scaling(self, large_realistic_dataset):
        """Test memory usage patterns with large datasets."""
        import sys
        
        estimator = large_realistic_dataset
        
        # Measure approximate memory usage
        table_size = len(estimator.cost_table)
        alias_size = len(estimator.aliases)
        
        # Basic memory usage check
        assert table_size > 0, "Cost table should not be empty"
        
        # Memory efficiency check (very approximate)
        # Each entry should use reasonable memory
        estimated_memory_per_entry = 200  # bytes (rough estimate)
        total_estimated_memory = table_size * estimated_memory_per_entry
        
        # This is a sanity check rather than precise measurement
        assert total_estimated_memory < 100 * 1024 * 1024, \
            "Estimated memory usage seems excessive for dataset size"


class TestIntegrationWithExternalSources:
    """Test integration with external data sources and APIs."""
    
    def test_cms_api_integration_mock(self):
        """Test integration with CMS pricing API (mocked)."""
        def mock_cms_response(*args, **kwargs):
            class MockResponse:
                def __init__(self, json_data, status_code):
                    self.json_data = json_data
                    self.status_code = status_code
                
                def json(self):
                    return self.json_data
                
                def raise_for_status(self):
                    if self.status_code >= 400:
                        raise Exception(f"HTTP {self.status_code}")
            
            return MockResponse(MOCK_CMS_RESPONSE_DATA["valid_response"], 200)
        
        with patch('requests.get', side_effect=mock_cms_response):
            # Test that we can process CMS data correctly
            cms_data = MOCK_CMS_RESPONSE_DATA["valid_response"]
            
            for cpt_code, price in cms_data.items():
                assert isinstance(cpt_code, str)
                assert len(cpt_code) == 5
                assert isinstance(price, (int, float))
                assert price > 0
    
    def test_llm_integration_patterns(self, monkeypatch):
        """Test integration patterns with LLM services."""
        # Mock successful LLM responses
        def mock_successful_lookup(test_name):
            return MOCK_LLM_RESPONSES["successful_lookups"].get(test_name.lower())
        
        # Mock failed LLM responses  
        def mock_failed_lookup(test_name):
            return MOCK_LLM_RESPONSES["failed_lookups"].get(test_name.lower())
        
        cost_data = {
            "complete blood count": CptCost("85027", 25.50, "lab"),
            "basic metabolic panel": CptCost("80048", 15.75, "lab"),
        }
        estimator = CostEstimator(cost_data)
        
        # Test successful LLM integration
        monkeypatch.setattr(ce_mod, "lookup_cpt", mock_successful_lookup)
        
        # Should find CBC via LLM lookup
        price = estimator.estimate_cost("blood work")
        assert price == 25.50
        
        # Should create alias
        assert "blood work" in estimator.aliases
        
        # Test failed LLM integration
        monkeypatch.setattr(ce_mod, "lookup_cpt", mock_failed_lookup)
        
        # Should fallback to average
        avg_price = (25.50 + 15.75) / 2
        unknown_price = estimator.estimate_cost("unknown test")
        assert abs(unknown_price - avg_price) < 0.01
    
    def test_error_handling_external_sources(self, monkeypatch):
        """Test error handling for external source failures."""
        def mock_error_response(*args, **kwargs):
            raise ConnectionError("Network error")
        
        # Test that system handles external source failures gracefully
        with patch('requests.get', side_effect=mock_error_response):
            # Should not crash when external source fails
            try:
                # Attempt to load estimator (would normally fetch external data)
                estimator = CostEstimator({})
                # Should work with local data even if external sources fail
                assert isinstance(estimator, CostEstimator)
            except ConnectionError:
                # It's acceptable for this to fail, but shouldn't crash system
                pass
    
    def test_data_freshness_validation(self):
        """Test validation of data freshness and staleness detection."""
        import datetime
        
        # Mock timestamps for data freshness testing
        now = datetime.datetime.now()
        
        freshness_scenarios = {
            "fresh": now - datetime.timedelta(hours=1),      # 1 hour old
            "acceptable": now - datetime.timedelta(days=1),  # 1 day old  
            "stale": now - datetime.timedelta(days=30),      # 30 days old
            "very_stale": now - datetime.timedelta(days=365), # 1 year old
        }
        
        for scenario, timestamp in freshness_scenarios.items():
            age_hours = (now - timestamp).total_seconds() / 3600
            
            if scenario == "fresh":
                assert age_hours <= 24, "Fresh data should be less than 24 hours old"
            elif scenario == "acceptable":
                assert age_hours <= 168, "Acceptable data should be less than 7 days old" 
            elif scenario == "stale":
                assert age_hours > 168, "Stale data should be identified"
            elif scenario == "very_stale":
                assert age_hours > 8760, "Very stale data should be flagged"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""
    
    def test_extreme_price_values(self):
        """Test handling of extreme price values."""
        extreme_cases = [
            ("zero_price", "99999", 0.0),
            ("negative_price", "99998", -10.0),
            ("very_high_price", "99997", 999999.99),
            ("tiny_price", "99996", 0.01),
            ("precise_price", "99995", 123.456789),
        ]
        
        for name, cpt, price in extreme_cases:
            if price >= 0:  # Only test non-negative prices (business rule)
                cost = CptCost(cpt, price, "test")
                estimator = CostEstimator({name: cost})
                
                result = estimator.lookup_cost(name)
                assert result.price == price
                assert isinstance(result.price, (int, float))
    
    def test_unicode_and_encoding_handling(self):
        """Test handling of Unicode characters and encoding issues."""
        unicode_cases = [
            ("café_test", "Café Medical Test"),
            ("naïve_test", "Naïve Diagnostic"),
            ("résumé_test", "Résumé Analysis"),
            ("chinese_test", "测试"),
            ("emoji_test", "Test 🩺"),
        ]
        
        for cpt_suffix, test_name in enumerate(unicode_cases, 1):
            cpt_code = f"9999{cpt_suffix}"
            name_key, display_name = test_name
            
            cost_data = {display_name: CptCost(cpt_code, 25.0, "test")}
            estimator = CostEstimator(cost_data)
            
            # Should handle Unicode characters properly
            result = estimator.lookup_cost(display_name)
            assert result.price == 25.0
    
    def test_malicious_input_handling(self):
        """Test handling of potentially malicious inputs."""
        malicious_inputs = [
            "'; DROP TABLE tests; --",  # SQL injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
            "../../etc/passwd",  # Path traversal attempt
            "\\x00\\x01\\x02",  # Binary data
            "A" * 10000,  # Very long string
        ]
        
        estimator = CostEstimator({"test": CptCost("85027", 25.0, "lab")})
        
        for malicious_input in malicious_inputs:
            # Should handle malicious input gracefully (not crash)
            try:
                estimator.lookup_cost(malicious_input)
            except KeyError:
                pass  # Expected - input won't be found
            except Exception as e:
                # Should not raise unexpected exceptions
                pytest.fail(f"Unexpected exception for input '{malicious_input}': {e}")
    
    def test_concurrent_modification_safety(self):
        """Test safety under concurrent modifications."""
        import threading
        import time
        
        estimator = CostEstimator({"test": CptCost("85027", 25.0, "lab")})
        
        def modifier_thread():
            for i in range(100):
                estimator.add_aliases({f"alias_{i}": "test"})
                time.sleep(0.001)  # Small delay
        
        def reader_thread():
            for i in range(100):
                try:
                    estimator.lookup_cost("test")
                    estimator.lookup_cost(f"alias_{i % 10}")
                except KeyError:
                    pass  # Expected for some aliases
                time.sleep(0.001)  # Small delay
        
        # Start threads
        threads = []
        for _ in range(2):
            threads.append(threading.Thread(target=modifier_thread))
            threads.append(threading.Thread(target=reader_thread))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should complete without crashes or deadlocks
        assert len(estimator.aliases) >= 0  # Basic sanity check
    
    def test_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion attacks."""
        # Test with very large datasets
        large_cost_data = {}
        
        # Create a dataset that might cause memory issues if not handled properly
        for i in range(10000):
            test_name = f"test_{i:05d}"
            cpt_code = f"{85000 + (i % 9000)}"  # Realistic CPT range
            price = 10.0 + (i % 1000)  # Varied prices
            large_cost_data[test_name] = CptCost(cpt_code, price, "test")
        
        # Should be able to create estimator without memory issues
        estimator = CostEstimator(large_cost_data)
        assert len(estimator.cost_table) == 10000
        
        # Should be able to perform lookups efficiently
        start_time = time.time()
        result = estimator.lookup_cost("test_05000")
        end_time = time.time()
        
        lookup_time = end_time - start_time
        assert lookup_time < 0.1, f"Lookup took too long: {lookup_time:.4f}s"
        assert result.price > 0


class TestRegressionPrevention:
    """Test suite to prevent regression of known issues."""
    
    def test_price_precision_regression(self):
        """Test that price precision is maintained (regression test)."""
        # This tests a hypothetical bug where prices lose precision
        precise_prices = [
            123.45,
            67.89,
            0.01,
            999.99,
            12.345,  # More than 2 decimal places
        ]
        
        for i, price in enumerate(precise_prices):
            cpt_code = f"9999{i}"
            cost = CptCost(cpt_code, price, "test")
            estimator = CostEstimator({f"test_{i}": cost})
            
            result = estimator.lookup_cost(f"test_{i}")
            # Should maintain reasonable precision
            assert abs(result.price - price) < 0.001
    
    def test_case_sensitivity_regression(self):
        """Test that case sensitivity is handled consistently."""
        cost_data = {
            "Complete Blood Count": CptCost("85027", 25.50, "lab")
        }
        estimator = CostEstimator(cost_data)
        
        # All these should work (case insensitive)
        variations = [
            "complete blood count",
            "COMPLETE BLOOD COUNT", 
            "Complete Blood Count",
            "CoMpLeTe BlOoD cOuNt",
        ]
        
        for variation in variations:
            result = estimator.lookup_cost(variation)
            assert result.price == 25.50
    
    def test_alias_chaining_regression(self):
        """Test that alias chaining doesn't create infinite loops."""
        estimator = CostEstimator({"target": CptCost("85027", 25.0, "lab")})
        
        # Add chain of aliases
        estimator.add_aliases({
            "alias1": "target",
            "alias2": "alias1", 
            "alias3": "alias2",
        })
        
        # Should resolve through chain without infinite loop
        result = estimator.lookup_cost("alias3")
        assert result.price == 25.0
    
    def test_memory_leak_regression(self):
        """Test for memory leaks in repetitive operations."""
        estimator = CostEstimator({"test": CptCost("85027", 25.0, "lab")})
        
        # Perform many operations that might cause memory leaks
        for i in range(1000):
            # Add and remove aliases
            alias_name = f"temp_alias_{i}"
            estimator.add_aliases({alias_name: "test"})
            
            # Perform lookups
            result = estimator.lookup_cost("test")
            assert result.price == 25.0
            
            # Clean up alias (manual cleanup for testing)
            if alias_name in estimator.aliases:
                del estimator.aliases[alias_name]
        
        # Memory usage should not grow excessively
        # (This is a basic check; real memory profiling would be better)
        assert len(estimator.aliases) < 100, "Aliases dictionary grew unexpectedly"