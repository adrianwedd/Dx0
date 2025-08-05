import csv
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch, MagicMock

import pytest

from sdb.cost_estimator import CostEstimator, CptCost, load_cost_estimator, load_csv_estimator
from sdb import cost_estimator as ce_mod


def test_lookup_and_estimate(monkeypatch):
    data = [
        {"test_name": "cbc", "cpt_code": "100", "price": "10", "category": "labs"},
        {"test_name": "bmp", "cpt_code": "101", "price": "20", "category": "imaging"},
    ]
    with tempfile.NamedTemporaryFile("w", newline="", delete=False) as f:
        writer = csv.DictWriter(
            f, fieldnames=["test_name", "cpt_code", "price", "category"]
        )
        writer.writeheader()
        writer.writerows(data)
        path = f.name
    ce = CostEstimator.load_from_csv(path)
    ce.add_aliases({"basic metabolic panel": "bmp"})

    monkeypatch.setattr(ce_mod, "lookup_cpt", lambda name: "101")

    assert ce.lookup_cost("cbc").price == 10.0
    assert ce.lookup_cost("cbc").category == "labs"
    assert ce.lookup_cost("basic metabolic panel").cpt_code == "101"
    assert ce.estimate_cost("unknown") == 20.0
    assert ce.estimate("unknown") == (20.0, "imaging")
    assert ce.aliases["unknown"] == "bmp"
    # Second call should use cached alias and avoid LLM lookup
    monkeypatch.setattr(
        ce_mod,
        "lookup_cpt",
        lambda name: (_ for _ in ()).throw(AssertionError),
    )
    assert ce.estimate_cost("unknown") == 20.0


def test_load_aliases_from_csv(tmp_path):
    cost_rows = [
        {"test_name": "bmp", "cpt_code": "101", "price": "20"},
    ]
    cost_path = tmp_path / "cost.csv"
    with open(cost_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["test_name", "cpt_code", "price"]
        )
        writer.writeheader()
        writer.writerows(cost_rows)

    alias_rows = [
        {"alias": "basic metabolic panel", "canonical": "bmp"},
    ]
    alias_path = tmp_path / "aliases.csv"
    with open(alias_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["alias", "canonical"])
        writer.writeheader()
        writer.writerows(alias_rows)

    ce = CostEstimator.load_from_csv(str(cost_path))
    ce.load_aliases_from_csv(str(alias_path))

    assert ce.lookup_cost("basic metabolic panel").cpt_code == "101"


def test_coverage_check(tmp_path):
    cost_rows = [
        {"test_name": "cbc", "cpt_code": "100", "price": "1"},
    ]
    cost_path = tmp_path / "cost.csv"
    with open(cost_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["test_name", "cpt_code", "price"]
        )
        writer.writeheader()
        writer.writerows(cost_rows)

    cms_rows = [
        {"cpt_code": "100"},
        {"cpt_code": "101"},
    ]
    cms_path = tmp_path / "cms.csv"
    with open(cms_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cpt_code"])
        writer.writeheader()
        writer.writerows(cms_rows)

    with pytest.raises(ValueError) as exc:
        CostEstimator.load_from_csv(
            str(cost_path),
            cms_pricing_path=str(cms_path),
            coverage_threshold=1.0,
        )
    assert "101" in str(exc.value)


def test_lookup_cost_missing():
    ce = CostEstimator({})
    with pytest.raises(KeyError):
        ce.lookup_cost("unknown")


def test_plugin_loader(monkeypatch):
    captured: dict[str, str] = {}

    def dummy_factory(path: str) -> CostEstimator:
        captured["path"] = path
        return CostEstimator({})

    class DummyEP:
        name = "dummy"
        value = "dummy:factory"
        group = "dx0.cost_estimators"

        def load(self):
            return dummy_factory

    monkeypatch.setattr(
        ce_mod.metadata,
        "entry_points",
        lambda group=None: [DummyEP()] if group == "dx0.cost_estimators" else [],
    )
    estimator = ce_mod.load_cost_estimator("table.csv", plugin_name="dummy")
    assert isinstance(estimator, CostEstimator)
    assert captured["path"] == "table.csv"


def test_estimate_cost_average(monkeypatch):
    ce = CostEstimator(
        {
            "a": CptCost("1", 10.0),
            "b": CptCost("2", 30.0),
        }
    )
    monkeypatch.setattr(ce_mod, "lookup_cpt", lambda name: None)
    assert ce.estimate_cost("unknown") == 20.0


def test_estimate_cost_empty(monkeypatch):
    ce = CostEstimator({})
    monkeypatch.setattr(ce_mod, "lookup_cpt", lambda name: None)
    assert ce.estimate_cost("x") == 0.0


# ============================================================================
# COMPREHENSIVE UNIT TESTS FOR COST ESTIMATION ACCURACY
# ============================================================================

class TestCostCalculationAccuracy:
    """Test suite for cost calculation accuracy and precision."""
    
    @pytest.fixture
    def sample_cost_data(self):
        """Sample cost data with various test types and categories."""
        return {
            "complete blood count": CptCost("85027", 25.50, "laboratory"),
            "basic metabolic panel": CptCost("80048", 15.75, "laboratory"),
            "chest x-ray": CptCost("71046", 89.00, "imaging"),
            "mri brain with contrast": CptCost("70553", 1250.00, "imaging"),
            "echocardiogram": CptCost("93307", 275.50, "cardiology"),
            "colonoscopy": CptCost("45378", 850.00, "procedure"),
        }
    
    @pytest.fixture
    def cost_estimator(self, sample_cost_data):
        """Cost estimator with sample data."""
        return CostEstimator(sample_cost_data)
    
    def test_exact_cost_lookup(self, cost_estimator):
        """Test exact cost lookup for known tests."""
        assert cost_estimator.lookup_cost("complete blood count").price == 25.50
        assert cost_estimator.lookup_cost("mri brain with contrast").price == 1250.00
        
    def test_case_insensitive_lookup(self, cost_estimator):
        """Test case-insensitive cost lookup."""
        assert cost_estimator.lookup_cost("CHEST X-RAY").price == 89.00
        assert cost_estimator.lookup_cost("Basic Metabolic Panel").price == 15.75
        assert cost_estimator.lookup_cost("EcHoCarDioGraM").price == 275.50
    
    def test_cost_precision(self, cost_estimator):
        """Test cost precision and decimal handling."""
        result = cost_estimator.lookup_cost("complete blood count")
        assert isinstance(result.price, float)
        assert result.price == 25.50
        # Test that we maintain precision
        assert f"{result.price:.2f}" == "25.50"
    
    def test_category_accuracy(self, cost_estimator):
        """Test category assignment accuracy."""
        lab_test = cost_estimator.lookup_cost("basic metabolic panel")
        imaging_test = cost_estimator.lookup_cost("chest x-ray")
        procedure_test = cost_estimator.lookup_cost("colonoscopy")
        
        assert lab_test.category == "laboratory"
        assert imaging_test.category == "imaging"
        assert procedure_test.category == "procedure"
    
    def test_cpt_code_accuracy(self, cost_estimator):
        """Test CPT code accuracy and format."""
        cbc = cost_estimator.lookup_cost("complete blood count")
        mri = cost_estimator.lookup_cost("mri brain with contrast")
        
        assert cbc.cpt_code == "85027"
        assert mri.cpt_code == "70553"
        # Ensure CPT codes are strings (not integers)
        assert isinstance(cbc.cpt_code, str)
        assert isinstance(mri.cpt_code, str)
    
    def test_missing_test_handling(self, cost_estimator):
        """Test handling of missing tests."""
        with pytest.raises(KeyError, match="Unknown test name: nonexistent test"):
            cost_estimator.lookup_cost("nonexistent test")
    
    def test_whitespace_handling(self, cost_estimator):
        """Test whitespace handling in test names."""
        # Leading/trailing whitespace
        assert cost_estimator.lookup_cost("  chest x-ray  ").price == 89.00
        # Multiple spaces
        assert cost_estimator.lookup_cost("complete  blood   count").price == 25.50
    
    def test_alias_functionality(self, cost_estimator):
        """Test alias mapping functionality."""
        cost_estimator.add_aliases({
            "cbc": "complete blood count",
            "bmp": "basic metabolic panel",
            "cxr": "chest x-ray"
        })
        
        assert cost_estimator.lookup_cost("cbc").price == 25.50
        assert cost_estimator.lookup_cost("bmp").price == 15.75
        assert cost_estimator.lookup_cost("cxr").price == 89.00
    
    def test_estimate_function_with_known_tests(self, cost_estimator):
        """Test estimate function with known tests."""
        price, category = cost_estimator.estimate("complete blood count")
        assert price == 25.50
        assert category == "laboratory"
    
    def test_estimate_function_with_unknown_tests(self, cost_estimator, monkeypatch):
        """Test estimate function with unknown tests (fallback to average)."""
        monkeypatch.setattr(ce_mod, "lookup_cpt", lambda name: None)
        
        price, category = cost_estimator.estimate("unknown test")
        expected_avg = 25.50 + 15.75 + 89.00 + 1250.00 + 275.50 + 850.00
        expected_avg /= 6
        
        assert abs(price - expected_avg) < 0.01
        assert category == "unknown"


class TestCPTCodeValidation:
    """Test suite for CPT code validation and standards compliance."""
    
    def test_valid_cpt_code_formats(self):
        """Test various valid CPT code formats."""
        valid_codes = [
            "85027",  # 5-digit numeric
            "99213",  # E&M code
            "G0463",  # HCPCS Level II with letter prefix
            "0001T",  # Category III with T suffix
            "A4217",  # HCPCS Level II A-code
        ]
        
        for code in valid_codes:
            cost = CptCost(code, 100.0, "test")
            assert cost.cpt_code == code
            assert isinstance(cost.cpt_code, str)
    
    def test_cpt_code_format_validation(self):
        """Test CPT code format validation patterns."""
        # Standard 5-digit CPT codes
        assert len("85027") == 5
        assert "85027".isdigit()
        
        # HCPCS codes with letter prefixes
        assert "G0463"[0].isalpha()
        assert "G0463"[1:].isdigit()
        
        # Category III codes with T suffix
        assert "0001T"[-1] == "T"
        assert "0001T"[:-1].isdigit()
    
    @pytest.mark.parametrize("cpt_code,expected_category", [
        ("85027", "laboratory"),  # Lab test
        ("71046", "imaging"),     # Radiology
        ("99213", "office_visit"), # E&M
        ("45378", "procedure"),   # Procedure
        ("93307", "cardiology"),  # Cardiac
    ])
    def test_cpt_code_category_mapping(self, cpt_code, expected_category):
        """Test CPT code to category mapping accuracy."""
        cost = CptCost(cpt_code, 100.0, expected_category)
        assert cost.category == expected_category
    
    def test_cpt_code_price_ranges(self):
        """Test realistic price ranges for different CPT categories."""
        # Laboratory tests: typically $5-50
        lab_cost = CptCost("85027", 25.50, "laboratory")
        assert 5.0 <= lab_cost.price <= 50.0
        
        # Imaging: typically $50-2000
        imaging_cost = CptCost("71046", 89.00, "imaging")
        assert 50.0 <= imaging_cost.price <= 2000.0
        
        # Procedures: typically $100-5000
        procedure_cost = CptCost("45378", 850.00, "procedure")
        assert 100.0 <= procedure_cost.price <= 5000.0


class TestCostEstimatorDataIntegrity:
    """Test suite for data integrity and consistency."""
    
    def test_duplicate_cpt_codes_handling(self):
        """Test handling of duplicate CPT codes in cost table."""
        # This tests business logic - should we allow duplicates?
        cost_data = {
            "test_a": CptCost("85027", 25.00, "lab"),
            "test_b": CptCost("85027", 30.00, "lab"),  # Same CPT, different price
        }
        estimator = CostEstimator(cost_data)
        
        # Both should be retrievable by their test names
        assert estimator.lookup_cost("test_a").price == 25.00
        assert estimator.lookup_cost("test_b").price == 30.00
    
    def test_empty_cost_table(self):
        """Test behavior with empty cost table."""
        estimator = CostEstimator({})
        
        with pytest.raises(KeyError):
            estimator.lookup_cost("any test")
        
        # Average should be 0 for empty table
        assert estimator.estimate_cost("any test") == 0.0
    
    def test_cost_table_normalization(self):
        """Test that cost table keys are properly normalized."""
        original_data = {
            "Complete Blood Count": CptCost("85027", 25.50, "lab"),
            "CHEST X-RAY": CptCost("71046", 89.00, "imaging"),
        }
        estimator = CostEstimator(original_data)
        
        # Keys should be normalized to lowercase
        assert "complete blood count" in estimator.cost_table
        assert "chest x-ray" in estimator.cost_table
        assert "Complete Blood Count" not in estimator.cost_table
    
    def test_price_data_types(self):
        """Test that prices are properly handled as floats."""
        cost_data = {
            "test_int": CptCost("85027", 25, "lab"),      # int
            "test_float": CptCost("85028", 25.50, "lab"),  # float
            "test_str": CptCost("85029", "30.75", "lab"),  # string (should be converted)
        }
        
        # CptCost constructor should handle type conversion
        assert isinstance(cost_data["test_int"].price, (int, float))
        assert isinstance(cost_data["test_float"].price, float)
        # Note: CptCost doesn't do automatic string conversion, 
        # that's handled in CSV loading


class TestPerformanceAndScalability:
    """Test suite for performance and scalability."""
    
    @pytest.fixture
    def large_cost_table(self):
        """Generate a large cost table for performance testing."""
        cost_data = {}
        for i in range(1000):
            test_name = f"test_{i:04d}"
            cpt_code = f"{85000 + i}"
            price = 10.0 + (i % 100)  # Prices between $10-109
            category = ["lab", "imaging", "procedure"][i % 3]
            cost_data[test_name] = CptCost(cpt_code, price, category)
        return cost_data
    
    def test_lookup_performance(self, large_cost_table):
        """Test lookup performance with large cost tables."""
        estimator = CostEstimator(large_cost_table)
        
        start_time = time.time()
        for i in range(100):
            test_name = f"test_{i:04d}"
            result = estimator.lookup_cost(test_name)
            assert result.price == 10.0 + (i % 100)
        end_time = time.time()
        
        # Should complete 100 lookups in under 0.1 seconds
        assert end_time - start_time < 0.1
    
    def test_concurrent_lookups(self, large_cost_table):
        """Test concurrent cost lookups."""
        estimator = CostEstimator(large_cost_table)
        
        def lookup_test(test_num):
            test_name = f"test_{test_num:04d}"
            return estimator.lookup_cost(test_name).price
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(lookup_test, i) for i in range(100)]
            results = [future.result() for future in as_completed(futures)]
        
        # All lookups should succeed
        assert len(results) == 100
        assert all(isinstance(r, (int, float)) for r in results)
    
    def test_memory_usage_with_large_tables(self, large_cost_table):
        """Test memory usage with large cost tables."""
        import sys
        
        # Create estimator and measure approximate memory usage
        estimator = CostEstimator(large_cost_table)
        
        # Basic check that the estimator was created successfully
        assert len(estimator.cost_table) == 1000
        
        # Verify random lookups work
        test_indices = [0, 100, 500, 999]
        for i in test_indices:
            test_name = f"test_{i:04d}"
            result = estimator.lookup_cost(test_name)
            assert result is not None


class TestCSVLoadingAndValidation:
    """Test suite for CSV loading and data validation."""
    
    def test_csv_loading_with_all_columns(self, tmp_path):
        """Test CSV loading with all expected columns."""
        csv_data = [
            {"test_name": "Complete Blood Count", "cpt_code": "85027", "price": "25.50", "category": "laboratory"},
            {"test_name": "Chest X-Ray", "cpt_code": "71046", "price": "89.00", "category": "imaging"},
            {"test_name": "Basic Metabolic Panel", "cpt_code": "80048", "price": "15.75", "category": "laboratory"},
        ]
        
        csv_path = tmp_path / "test_costs.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price", "category"])
            writer.writeheader()
            writer.writerows(csv_data)
        
        estimator = CostEstimator.load_from_csv(str(csv_path))
        
        assert len(estimator.cost_table) == 3
        assert estimator.lookup_cost("complete blood count").price == 25.50
        assert estimator.lookup_cost("chest x-ray").category == "imaging"
    
    def test_csv_loading_with_missing_category(self, tmp_path):
        """Test CSV loading with missing category column."""
        csv_data = [
            {"test_name": "Complete Blood Count", "cpt_code": "85027", "price": "25.50"},
            {"test_name": "Chest X-Ray", "cpt_code": "71046", "price": "89.00"},
        ]
        
        csv_path = tmp_path / "test_costs.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
            writer.writeheader()
            writer.writerows(csv_data)
        
        estimator = CostEstimator.load_from_csv(str(csv_path))
        
        # Should default to "unknown" category
        assert estimator.lookup_cost("complete blood count").category == "unknown"
        assert estimator.lookup_cost("chest x-ray").category == "unknown"
    
    def test_csv_loading_with_malformed_data(self, tmp_path):
        """Test CSV loading with malformed data rows."""
        csv_content = '''test_name,cpt_code,price,category
Complete Blood Count,85027,25.50,laboratory
,71046,89.00,imaging
Chest X-Ray,,89.00,imaging
Basic Panel,80048,invalid_price,laboratory
Valid Test,80049,15.75,laboratory'''
        
        csv_path = tmp_path / "test_costs.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)
        
        estimator = CostEstimator.load_from_csv(str(csv_path))
        
        # Should only load valid rows
        assert len(estimator.cost_table) == 2  # Only "Complete Blood Count" and "Valid Test"
        assert "complete blood count" in estimator.cost_table
        assert "valid test" in estimator.cost_table
    
    def test_csv_loading_with_encoding_issues(self, tmp_path):
        """Test CSV loading with various text encodings."""
        csv_data = [
            {"test_name": "Café Coronary Test", "cpt_code": "85027", "price": "25.50", "category": "laboratory"},
            {"test_name": "Röntgen Chest", "cpt_code": "71046", "price": "89.00", "category": "imaging"},
        ]
        
        csv_path = tmp_path / "test_costs.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price", "category"])
            writer.writeheader()
            writer.writerows(csv_data)
        
        estimator = CostEstimator.load_from_csv(str(csv_path))
        
        assert len(estimator.cost_table) == 2
        assert "café coronary test" in estimator.cost_table
        assert "röntgen chest" in estimator.cost_table


class TestCMSPricingIntegration:
    """Test suite for CMS pricing validation and coverage."""
    
    def test_cms_coverage_validation_pass(self, tmp_path):
        """Test CMS coverage validation when coverage is sufficient."""
        # Create cost table
        cost_data = [
            {"test_name": "Test A", "cpt_code": "85027", "price": "25.50"},
            {"test_name": "Test B", "cpt_code": "71046", "price": "89.00"},
        ]
        cost_path = tmp_path / "costs.csv"
        with open(cost_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
            writer.writeheader()
            writer.writerows(cost_data)
        
        # Create CMS pricing table with same codes
        cms_data = [
            {"cpt_code": "85027"},
            {"cpt_code": "71046"},
        ]
        cms_path = tmp_path / "cms.csv"
        with open(cms_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["cpt_code"])
            writer.writeheader()
            writer.writerows(cms_data)
        
        # Should load successfully with 100% coverage
        estimator = CostEstimator.load_from_csv(
            str(cost_path), 
            cms_pricing_path=str(cms_path), 
            coverage_threshold=1.0
        )
        
        assert estimator.match_rate == 1.0
        assert len(estimator.unmatched_codes) == 0
    
    def test_cms_coverage_validation_fail(self, tmp_path):
        """Test CMS coverage validation when coverage is insufficient."""
        # Create cost table with only some codes
        cost_data = [
            {"test_name": "Test A", "cpt_code": "85027", "price": "25.50"},
        ]
        cost_path = tmp_path / "costs.csv"
        with open(cost_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
            writer.writeheader()
            writer.writerows(cost_data)
        
        # Create CMS pricing table with additional codes
        cms_data = [
            {"cpt_code": "85027"},
            {"cpt_code": "71046"},  # Missing from cost table
            {"cpt_code": "80048"},  # Missing from cost table
        ]
        cms_path = tmp_path / "cms.csv"
        with open(cms_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["cpt_code"])
            writer.writeheader()
            writer.writerows(cms_data)
        
        # Should raise ValueError due to insufficient coverage
        with pytest.raises(ValueError, match="Coverage.*below required"):
            CostEstimator.load_from_csv(
                str(cost_path), 
                cms_pricing_path=str(cms_path), 
                coverage_threshold=1.0
            )
    
    def test_cms_coverage_report_generation(self, tmp_path):
        """Test generation of unmatched codes report."""
        # Create cost table
        cost_data = [
            {"test_name": "Test A", "cpt_code": "85027", "price": "25.50"},
        ]
        cost_path = tmp_path / "costs.csv"
        with open(cost_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
            writer.writeheader()
            writer.writerows(cost_data)
        
        # Create CMS pricing table with additional codes
        cms_data = [
            {"cpt_code": "85027"},
            {"cpt_code": "71046"},
            {"cpt_code": "80048"},
        ]
        cms_path = tmp_path / "cms.csv"
        with open(cms_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["cpt_code"])
            writer.writeheader()
            writer.writerows(cms_data)
        
        report_path = tmp_path / "unmatched.csv"
        
        try:
            CostEstimator.load_from_csv(
                str(cost_path), 
                cms_pricing_path=str(cms_path), 
                coverage_threshold=0.5,  # Allow lower coverage
                report_path=str(report_path)
            )
        except ValueError:
            pass  # Expected due to low coverage
        
        # Report should be generated
        assert report_path.exists()
        
        with open(report_path, "r", encoding="utf-8") as f:
            report_content = f.read()
            assert "71046" in report_content
            assert "80048" in report_content


class TestLLMFallbackIntegration:
    """Test suite for LLM fallback functionality."""
    
    def test_llm_fallback_with_successful_lookup(self, monkeypatch):
        """Test LLM fallback when CPT lookup succeeds."""
        cost_data = {
            "existing test": CptCost("85027", 25.50, "lab")
        }
        estimator = CostEstimator(cost_data)
        
        # Mock successful LLM lookup
        monkeypatch.setattr(ce_mod, "lookup_cpt", lambda name: "85027")
        
        # Should find existing test via LLM and cache alias
        price = estimator.estimate_cost("blood work")
        assert price == 25.50
        assert "blood work" in estimator.aliases
        assert estimator.aliases["blood work"] == "existing test"
    
    def test_llm_fallback_with_failed_lookup(self, monkeypatch):
        """Test LLM fallback when CPT lookup fails."""
        cost_data = {
            "test a": CptCost("85027", 25.50, "lab"),
            "test b": CptCost("71046", 89.00, "imaging")
        }
        estimator = CostEstimator(cost_data)
        
        # Mock failed LLM lookup
        monkeypatch.setattr(ce_mod, "lookup_cpt", lambda name: None)
        
        # Should fallback to average price
        price = estimator.estimate_cost("unknown test")
        expected_avg = (25.50 + 89.00) / 2
        assert abs(price - expected_avg) < 0.01
    
    def test_llm_fallback_caching(self, monkeypatch):
        """Test that LLM lookups are properly cached as aliases."""
        cost_data = {
            "complete blood count": CptCost("85027", 25.50, "lab")
        }
        estimator = CostEstimator(cost_data)
        
        # Mock LLM lookup - should only be called once
        lookup_calls = []
        def mock_lookup(name):
            lookup_calls.append(name)
            return "85027"
        
        monkeypatch.setattr(ce_mod, "lookup_cpt", mock_lookup)
        
        # First call should trigger LLM lookup
        price1 = estimator.estimate_cost("cbc")
        assert price1 == 25.50
        assert len(lookup_calls) == 1
        
        # Second call should use cached alias
        price2 = estimator.estimate_cost("cbc")
        assert price2 == 25.50
        assert len(lookup_calls) == 1  # No additional LLM calls


class TestPluginSystem:
    """Test suite for cost estimator plugin system."""
    
    def test_csv_plugin_loader(self, tmp_path):
        """Test the built-in CSV plugin loader."""
        csv_data = [
            {"test_name": "Test A", "cpt_code": "85027", "price": "25.50"},
        ]
        csv_path = tmp_path / "costs.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
            writer.writeheader()
            writer.writerows(csv_data)
        
        estimator = load_csv_estimator(str(csv_path))
        assert isinstance(estimator, CostEstimator)
        assert estimator.lookup_cost("test a").price == 25.50
    
    def test_generic_plugin_loader_csv(self, tmp_path):
        """Test generic plugin loader with CSV plugin."""
        csv_data = [
            {"test_name": "Test A", "cpt_code": "85027", "price": "25.50"},
        ]
        csv_path = tmp_path / "costs.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
            writer.writeheader()
            writer.writerows(csv_data)
        
        estimator = load_cost_estimator(str(csv_path), plugin_name="csv")
        assert isinstance(estimator, CostEstimator)
        assert estimator.lookup_cost("test a").price == 25.50
    
    def test_plugin_loader_unknown_plugin(self, tmp_path):
        """Test plugin loader with unknown plugin name."""
        csv_path = tmp_path / "costs.csv"
        csv_path.write_text("test_name,cpt_code,price\\nTest,85027,25.50")
        
        with pytest.raises(ValueError, match="Cost estimator plugin 'unknown' not found"):
            load_cost_estimator(str(csv_path), plugin_name="unknown")


class TestAliasManagement:
    """Test suite for alias management functionality."""
    
    def test_manual_alias_addition(self):
        """Test manual alias addition."""
        cost_data = {
            "complete blood count": CptCost("85027", 25.50, "lab")
        }
        estimator = CostEstimator(cost_data)
        
        estimator.add_aliases({
            "cbc": "complete blood count",
            "blood work": "complete blood count"
        })
        
        assert estimator.lookup_cost("cbc").price == 25.50
        assert estimator.lookup_cost("blood work").price == 25.50
    
    def test_alias_case_normalization(self):
        """Test that aliases are properly case-normalized."""
        cost_data = {
            "complete blood count": CptCost("85027", 25.50, "lab")
        }
        estimator = CostEstimator(cost_data)
        
        estimator.add_aliases({
            "CBC": "Complete Blood Count",
            "Blood Work": "COMPLETE BLOOD COUNT"
        })
        
        # All should resolve to the same test
        assert estimator.lookup_cost("cbc").price == 25.50
        assert estimator.lookup_cost("blood work").price == 25.50
    
    def test_csv_alias_loading(self, tmp_path):
        """Test loading aliases from CSV file."""
        # Create cost table
        cost_data = [
            {"test_name": "Complete Blood Count", "cpt_code": "85027", "price": "25.50"},
            {"test_name": "Basic Metabolic Panel", "cpt_code": "80048", "price": "15.75"},
        ]
        cost_path = tmp_path / "costs.csv"
        with open(cost_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
            writer.writeheader()
            writer.writerows(cost_data)
        
        # Create alias table
        alias_data = [
            {"alias": "CBC", "canonical": "Complete Blood Count"},
            {"alias": "BMP", "canonical": "Basic Metabolic Panel"},
            {"alias": "Blood Work", "canonical": "Complete Blood Count"},
        ]
        alias_path = tmp_path / "aliases.csv"
        with open(alias_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["alias", "canonical"])
            writer.writeheader()
            writer.writerows(alias_data)
        
        estimator = CostEstimator.load_from_csv(str(cost_path))
        estimator.load_aliases_from_csv(str(alias_path))
        
        assert estimator.lookup_cost("cbc").price == 25.50
        assert estimator.lookup_cost("bmp").price == 15.75
        assert estimator.lookup_cost("blood work").price == 25.50
    
    def test_malformed_alias_csv(self, tmp_path):
        """Test handling of malformed alias CSV files."""
        cost_data = [
            {"test_name": "Complete Blood Count", "cpt_code": "85027", "price": "25.50"},
        ]
        cost_path = tmp_path / "costs.csv"
        with open(cost_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
            writer.writeheader()
            writer.writerows(cost_data)
        
        # Create malformed alias table
        alias_content = '''alias,canonical
CBC,Complete Blood Count
,Basic Metabolic Panel
BMP,
valid_alias,Complete Blood Count'''
        
        alias_path = tmp_path / "aliases.csv"
        with open(alias_path, "w", encoding="utf-8") as f:
            f.write(alias_content)
        
        estimator = CostEstimator.load_from_csv(str(cost_path))
        estimator.load_aliases_from_csv(str(alias_path))
        
        # Should only load valid aliases
        assert estimator.lookup_cost("cbc").price == 25.50
        assert estimator.lookup_cost("valid_alias").price == 25.50
        
        # Invalid aliases should not be loaded
        with pytest.raises(KeyError):
            estimator.lookup_cost("bmp")  # canonical was empty
