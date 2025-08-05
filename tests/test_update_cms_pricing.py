import csv
from types import SimpleNamespace
import pytest

import scripts.update_cms_pricing as ucp


def test_refresh_pricing(tmp_path, monkeypatch):
    cms_csv = "code,rate\n85027,11.0\n"

    def fake_get(url, timeout=60):
        return SimpleNamespace(
            content=cms_csv.encode(),
            text=cms_csv,
            status_code=200,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(ucp, "requests", SimpleNamespace(get=fake_get))

    path = tmp_path / "cpt.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_name", "cpt_code", "price"],
        )
        writer.writeheader()
        writer.writerow(
            {"test_name": "cbc", "cpt_code": "85027", "price": "9"}
        )

    ucp.refresh_pricing(str(path), "https://example.com/data.csv")

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["price"] == "11.0"


def test_refresh_pricing_low_coverage(tmp_path, monkeypatch):
    cms_csv = "code,rate\n85027,11.0\n"

    def fake_get(url, timeout=60):
        return SimpleNamespace(
            content=cms_csv.encode(),
            text=cms_csv,
            status_code=200,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(ucp, "requests", SimpleNamespace(get=fake_get))

    path = tmp_path / "cpt.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_name", "cpt_code", "price"],
        )
        writer.writeheader()
        writer.writerow({"test_name": "cbc", "cpt_code": "99999", "price": "9"})

    with pytest.raises(ValueError):
        ucp.refresh_pricing(
            str(path),
            "https://example.com/data.csv",
            coverage_threshold=1.0,
        )


# ============================================================================
# COMPREHENSIVE CMS PRICING VALIDATION TESTS
# ============================================================================

class TestCMSPricingValidation:
    """Comprehensive tests for CMS pricing data validation and integration."""
    
    def test_cms_pricing_format_validation(self, monkeypatch):
        """Test validation of CMS pricing data formats."""
        test_formats = [
            # Standard format
            "cpt_code,rate\n85027,25.50\n80048,15.75",
            # Alternative column names
            "hcpcs,payment_rate\n85027,25.50\n80048,15.75",
            # Different price formats
            "cpt_code,rate\n85027,$25.50\n80048,$15.75",
        ]
        
        for i, csv_content in enumerate(test_formats):
            def fake_get(url, timeout=60):
                return SimpleNamespace(
                    content=csv_content.encode('utf-8'),
                    status_code=200,
                    raise_for_status=lambda: None,
                )
            
            monkeypatch.setattr(ucp, "requests", SimpleNamespace(get=fake_get))
            
            prices = ucp._fetch_cms_prices(f"http://test.com/format_{i}.csv")
            
            assert "85027" in prices
            assert "80048" in prices
            assert abs(prices["85027"] - 25.50) < 0.01
            assert abs(prices["80048"] - 15.75) < 0.01
    
    def test_cms_pricing_error_handling(self, monkeypatch):
        """Test error handling for CMS pricing fetches."""
        def fake_get_error(url, timeout=60):
            response = SimpleNamespace(status_code=404, reason="Not Found")
            response.raise_for_status = lambda: Exception("HTTP 404: Not Found")
            response.raise_for_status()
            return response
        
        monkeypatch.setattr(ucp, "requests", SimpleNamespace(get=fake_get_error))
        
        with pytest.raises(Exception, match="404"):
            ucp._fetch_cms_prices("http://test.com/error.csv")
    
    def test_cms_malformed_data_handling(self, monkeypatch):
        """Test handling of malformed CMS data."""
        malformed_data = "cpt_code,rate\n85027,invalid_price\n80048,15.75"
        
        def fake_get(url, timeout=60):
            return SimpleNamespace(
                content=malformed_data.encode('utf-8'),
                status_code=200,
                raise_for_status=lambda: None,
            )
        
        monkeypatch.setattr(ucp, "requests", SimpleNamespace(get=fake_get))
        
        # Should not crash, but might return fewer entries
        prices = ucp._fetch_cms_prices("http://test.com/malformed.csv")
        
        # Should still parse valid entries
        assert "80048" in prices
        assert abs(prices["80048"] - 15.75) < 0.01
    
    def test_cms_pricing_integration_with_cost_estimator(self, tmp_path, monkeypatch):
        """Test integration between CMS pricing and CostEstimator."""
        from sdb.cost_estimator import CostEstimator
        
        # Mock CMS response
        cms_data = "cpt_code,rate\n85027,30.00\n80048,18.00\n71046,95.00"
        
        def fake_get(url, timeout=60):
            return SimpleNamespace(
                content=cms_data.encode('utf-8'),
                status_code=200,
                raise_for_status=lambda: None,
            )
        
        monkeypatch.setattr(ucp, "requests", SimpleNamespace(get=fake_get))
        
        # Create initial pricing file
        initial_data = [
            {"test_name": "Complete Blood Count", "cpt_code": "85027", "price": "25.00"},
            {"test_name": "Basic Metabolic Panel", "cpt_code": "80048", "price": "15.00"},
            {"test_name": "Chest X-Ray", "cpt_code": "71046", "price": "89.00"},
        ]
        
        pricing_path = tmp_path / "pricing.csv"
        with open(pricing_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
            writer.writeheader()
            writer.writerows(initial_data)
        
        # Update pricing from CMS
        ucp.refresh_pricing(str(pricing_path))
        
        # Verify prices were updated
        estimator = CostEstimator.load_from_csv(str(pricing_path))
        
        assert estimator.lookup_cost("complete blood count").price == 30.00
        assert estimator.lookup_cost("basic metabolic panel").price == 18.00
        assert estimator.lookup_cost("chest x-ray").price == 95.00
    
    def test_cms_coverage_validation_comprehensive(self, tmp_path, monkeypatch):
        """Test comprehensive CMS coverage validation."""
        # Mock CMS with more codes than in our pricing table
        cms_codes = [f"8{5000 + i}" for i in range(10)]  # 10 CMS codes
        cms_data = "cpt_code,rate\n" + "\n".join(f"{code},25.00" for code in cms_codes)
        
        def fake_get(url, timeout=60):
            return SimpleNamespace(
                content=cms_data.encode('utf-8'),
                status_code=200,
                raise_for_status=lambda: None,
            )
        
        monkeypatch.setattr(ucp, "requests", SimpleNamespace(get=fake_get))
        
        # Create pricing table with only some of the codes
        our_codes = cms_codes[:5]  # Only first 5 codes
        initial_data = [
            {"test_name": f"test_{code}", "cpt_code": code, "price": "20.00"}
            for code in our_codes
        ]
        
        pricing_path = tmp_path / "coverage_test.csv"
        with open(pricing_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
            writer.writeheader()
            writer.writerows(initial_data)
        
        # Test with high coverage threshold (should fail)
        with pytest.raises(ValueError, match="Coverage.*below required"):
            ucp.refresh_pricing(str(pricing_path), coverage_threshold=0.9)
        
        # Test with lower coverage threshold (should succeed)
        ucp.refresh_pricing(str(pricing_path), coverage_threshold=0.4)
        
        # Verify the estimator was created and prices updated  
        from sdb.cost_estimator import CostEstimator
        estimator = CostEstimator.load_from_csv(str(pricing_path))
        assert len(estimator.cost_table) == 5


class TestCMSPricingRealWorldScenarios:
    """Test real-world scenarios for CMS pricing integration."""
    
    def test_cms_url_generation(self):
        """Test CMS URL generation for different years."""
        import datetime
        
        # Test current year URL
        current_year = datetime.date.today().year
        expected_url = f"https://download.cms.gov/MedicareClinicalLabFeeSched/{str(current_year)[-2:]}CLAB.zip"
        assert ucp.default_cms_url() == expected_url
        
        # Verify URL format is correct
        url = ucp.default_cms_url()
        assert url.startswith("https://download.cms.gov/")
        assert url.endswith("CLAB.zip")
    
    def test_pricing_update_workflow(self, tmp_path, monkeypatch):
        """Test complete pricing update workflow."""
        from sdb.cost_estimator import CostEstimator
        
        # Simulate realistic CMS data
        realistic_cms_data = """cpt_code,rate
85027,18.25
80048,14.75
71046,68.50"""
        
        def fake_get(url, timeout=60):
            return SimpleNamespace(
                content=realistic_cms_data.encode('utf-8'),
                status_code=200,
                raise_for_status=lambda: None,
            )
        
        monkeypatch.setattr(ucp, "requests", SimpleNamespace(get=fake_get))
        
        # Create realistic pricing file
        realistic_data = [
            {"test_name": "Complete Blood Count", "cpt_code": "85027", "price": "25.00"},
            {"test_name": "Basic Metabolic Panel", "cpt_code": "80048", "price": "20.00"},
            {"test_name": "Chest X-Ray 2 Views", "cpt_code": "71046", "price": "75.00"},
        ]
        
        pricing_path = tmp_path / "realistic_pricing.csv"
        with open(pricing_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
            writer.writeheader()
            writer.writerows(realistic_data)
        
        # Perform pricing update
        ucp.refresh_pricing(str(pricing_path), coverage_threshold=0.8)
        
        # Verify update results
        estimator = CostEstimator.load_from_csv(str(pricing_path))
        
        # Check that prices were updated to CMS values
        assert estimator.lookup_cost("complete blood count").price == 18.25
        assert estimator.lookup_cost("basic metabolic panel").price == 14.75
        assert estimator.lookup_cost("chest x-ray 2 views").price == 68.50
        
        # Verify coverage metrics
        assert estimator.match_rate == 1.0  # 100% coverage
        assert len(estimator.unmatched_codes) == 0
