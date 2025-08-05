import csv
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from sdb.cpt_lookup import lookup_cpt, _load_cache, _append_cache, _query_llm
from sdb.metrics import CPT_CACHE_HITS, CPT_LLM_LOOKUPS


def test_cache_hit(tmp_path):
    cache = tmp_path / "cache.csv"
    with open(cache, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code"])
        writer.writeheader()
        writer.writerow({"test_name": "cbc", "cpt_code": "85027"})
    assert lookup_cpt("cbc", cache_path=str(cache)) == "85027"


def test_llm_lookup_and_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache.csv"

    def fake_create(model, messages, max_tokens):
        choice = SimpleNamespace(message=SimpleNamespace(content="12345"))
        return SimpleNamespace(choices=[choice])

    # Mock the new OpenAI v1+ client API
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create)
        )
    )
    
    dummy_openai = SimpleNamespace(
        OpenAI=lambda api_key: fake_client,
        api_key=None,
    )
    import sdb.cpt_lookup as cl

    monkeypatch.setattr(cl, "openai", dummy_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    code = cl.lookup_cpt("bmp", cache_path=str(cache))
    assert code == "12345"
    with open(cache, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["cpt_code"] == "12345"


def test_lookup_metrics(tmp_path):
    cache = tmp_path / "cache.csv"
    with open(cache, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code"])
        writer.writeheader()
        writer.writerow({"test_name": "cbc", "cpt_code": "85027"})

    CPT_CACHE_HITS._value.set(0)
    CPT_LLM_LOOKUPS._value.set(0)

    assert lookup_cpt("cbc", cache_path=str(cache)) == "85027"
    assert CPT_CACHE_HITS._value.get() == 1
    assert CPT_LLM_LOOKUPS._value.get() == 0

    assert lookup_cpt("unknown", cache_path=str(cache)) is None
    assert CPT_CACHE_HITS._value.get() == 1
    assert CPT_LLM_LOOKUPS._value.get() == 1


# ============================================================================
# COMPREHENSIVE CPT LOOKUP VALIDATION TESTS
# ============================================================================

class TestCPTCacheManagement:
    """Test suite for CPT cache management functionality."""
    
    def test_cache_loading_with_various_formats(self, tmp_path):
        """Test cache loading with different CSV formats."""
        # Test with standard format
        cache_data = [
            {"test_name": "complete blood count", "cpt_code": "85027"},
            {"test_name": "basic metabolic panel", "cpt_code": "80048"},
            {"test_name": "chest x-ray", "cpt_code": "71046"},
        ]
        
        cache_path = tmp_path / "standard_cache.csv"
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code"])
            writer.writeheader()
            writer.writerows(cache_data)
        
        cache = _load_cache(str(cache_path))
        assert len(cache) == 3
        assert cache["complete blood count"] == "85027"
        assert cache["basic metabolic panel"] == "80048"
        assert cache["chest x-ray"] == "71046"
    
    def test_cache_loading_with_malformed_data(self, tmp_path):
        """Test cache loading with malformed CSV data."""
        malformed_content = '''test_name,cpt_code
complete blood count,85027
,80048
basic metabolic panel,
chest x-ray,71046,extra_column
invalid_row
valid test,99213'''
        
        cache_path = tmp_path / "malformed_cache.csv"
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(malformed_content)
        
        cache = _load_cache(str(cache_path))
        # Should only load valid rows
        assert len(cache) == 3  # "complete blood count", "chest x-ray", "valid test"
        assert "complete blood count" in cache
        assert "chest x-ray" in cache
        assert "valid test" in cache
    
    def test_cache_loading_with_encoding_issues(self, tmp_path):
        """Test cache loading with Unicode characters."""
        unicode_data = [
            {"test_name": "café coronary test", "cpt_code": "85027"},
            {"test_name": "röntgen examination", "cpt_code": "71046"},
            {"test_name": "naïve analysis", "cpt_code": "80048"},
        ]
        
        cache_path = tmp_path / "unicode_cache.csv"
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code"])
            writer.writeheader()
            writer.writerows(unicode_data)
        
        cache = _load_cache(str(cache_path))
        assert len(cache) == 3
        assert "café coronary test" in cache
        assert "röntgen examination" in cache
        assert "naïve analysis" in cache
    
    def test_cache_append_functionality(self, tmp_path):
        """Test cache append functionality."""
        cache_path = tmp_path / "append_cache.csv"
        
        # Append to non-existent file (should create)
        _append_cache(str(cache_path), "new test", "85027")
        
        cache = _load_cache(str(cache_path))
        assert cache["new test"] == "85027"
        
        # Append to existing file
        _append_cache(str(cache_path), "another test", "80048")
        
        cache = _load_cache(str(cache_path))
        assert len(cache) == 2
        assert cache["new test"] == "85027"
        assert cache["another test"] == "80048"
    
    def test_cache_concurrent_access(self, tmp_path):
        """Test concurrent cache access patterns."""
        cache_path = tmp_path / "concurrent_cache.csv"
        
        # Pre-populate cache
        initial_data = [(f"test_{i}", f"8502{i % 10}") for i in range(100)]
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["test_name", "cpt_code"])
            writer.writerows(initial_data)
        
        def concurrent_lookup(thread_id):
            results = []
            for i in range(10):
                test_name = f"test_{thread_id * 10 + i}"
                result = lookup_cpt(test_name, cache_path=str(cache_path))
                results.append((test_name, result))
            return results
        
        # Run concurrent lookups
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(concurrent_lookup, i) for i in range(10)]
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())
        
        # Verify all lookups succeeded
        assert len(all_results) == 100
        successful_lookups = [r for r in all_results if r[1] is not None]
        assert len(successful_lookups) == 100  # All should be found in cache


class TestLLMIntegration:
    """Test suite for LLM integration functionality."""
    
    def test_llm_query_successful_response(self, monkeypatch):
        """Test successful LLM query responses."""
        def mock_create(model, messages, max_tokens):
            # Extract test name from messages
            test_name = messages[1]["content"].lower()
            
            # Return appropriate CPT code based on test name
            cpt_responses = {
                "complete blood count": "85027",
                "basic metabolic panel": "80048", 
                "chest x-ray": "71046",
                "echocardiogram": "93307",
            }
            
            response_text = cpt_responses.get(test_name, "99999")
            choice = SimpleNamespace(message=SimpleNamespace(content=response_text))
            return SimpleNamespace(choices=[choice])
        
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=mock_create)
            )
        )
        
        dummy_openai = SimpleNamespace(
            OpenAI=lambda api_key: fake_client
        )
        
        import sdb.cpt_lookup as cl
        monkeypatch.setattr(cl, "openai", dummy_openai)
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        
        # Test successful lookups
        assert _query_llm("complete blood count") == "85027"
        assert _query_llm("basic metabolic panel") == "80048"
        assert _query_llm("chest x-ray") == "71046"
