"""
Comprehensive scoring validation tests for Judge Agent.

This module provides validation tests that verify scoring consistency,
accuracy against ground truth data, and statistical properties of the
Judge Agent's scoring system.
"""

import pytest
import time
import statistics
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

from sdb.judge import Judge, Judgement
from judge_test_data import (
    GROUND_TRUTH_CASES,
    COMPLEX_SCENARIOS,
    EDGE_CASE_SCENARIOS,
    DOMAIN_SPECIFIC_CASES,
    UNICODE_TEST_CASES,
    REGRESSION_TEST_CASES,
    PERFORMANCE_TEST_CASES,
    DiagnosticTestCase,
    get_test_cases_by_category,
    get_test_cases_by_difficulty,
    get_benchmark_expectations,
)


class ValidationClient:
    """Mock client that provides validation-focused responses."""
    
    def __init__(self, scoring_logic: Dict[str, int] = None):
        """
        Args:
            scoring_logic: Dict mapping diagnosis-truth pairs to expected scores
        """
        self.scoring_logic = scoring_logic or {}
        self.call_history = []
    
    def chat(self, messages, model):
        """Return scores based on predefined logic or default scoring."""
        content = messages[-1]["content"].lower()
        self.call_history.append(content)
        
        # Extract diagnosis and truth from content
        if "candidate:" in content and "truth:" in content:
            parts = content.split("truth:")
            if len(parts) == 2:
                candidate = parts[0].replace("candidate:", "").strip()
                truth = parts[1].strip()
                
                # Check for exact matches and synonyms
                key = f"{candidate}|{truth}"
                if key in self.scoring_logic:
                    return str(self.scoring_logic[key])
                
                # Default scoring logic based on similarity
                return self._default_scoring(candidate, truth)
        
        return "1"  # Default fallback
    
    def _default_scoring(self, candidate: str, truth: str) -> str:
        """Default scoring logic based on string similarity."""
        candidate_lower = candidate.lower()
        truth_lower = truth.lower()
        
        # Exact match
        if candidate_lower == truth_lower:
            return "5"
        
        # Known medical abbreviations and synonyms
        medical_synonyms = {
            ("atrial fibrillation", "a-fib"): "5",
            ("ventricular tachycardia", "v-tach"): "5",
            ("chronic obstructive pulmonary disease", "copd"): "5",
            ("acute respiratory distress syndrome", "ards"): "5",
            ("diabetic ketoacidosis", "dka"): "5",
            ("hyperosmolar hyperglycemic state", "hhs"): "5",
            ("myocardial infarction", "heart attack"): "5",
            ("myocardial infarction", "mi"): "5",
            ("myocardial infarction", "cardiac event"): "4",
            ("gastroesophageal reflux disease", "gerd"): "5",
            ("type 2 diabetes mellitus", "diabetes mellitus type 2"): "5",
            ("upper respiratory infection", "common cold"): "4",
            ("viral pneumonia", "bacterial pneumonia"): "3",
            ("chest pain", "myocardial infarction"): "3",
        }
        
        # Check medical synonyms (bidirectional)
        for (term1, term2), score in medical_synonyms.items():
            if (candidate_lower == term1 and truth_lower == term2) or \
               (candidate_lower == term2 and truth_lower == term1):
                return score
        
        # High similarity (one contains the other)
        if candidate_lower in truth_lower or truth_lower in candidate_lower:
            return "4"
        
        # Medium similarity (common words)
        candidate_words = set(candidate_lower.split())
        truth_words = set(truth_lower.split())
        common_words = candidate_words.intersection(truth_words)
        
        if len(common_words) >= 2:
            return "3"
        elif len(common_words) == 1:
            return "2"
        else:
            return "1"


class ConsistencyTestClient:
    """Mock client for testing scoring consistency."""
    
    def __init__(self, base_score: int = 3, noise_level: float = 0.0):
        """
        Args:
            base_score: Base score to return
            noise_level: Amount of randomness (0.0 = perfect consistency)
        """
        self.base_score = base_score
        self.noise_level = noise_level
        self.call_count = 0
        import random
        self.random = random.Random(42)  # Fixed seed for reproducibility
    
    def chat(self, messages, model):
        self.call_count += 1
        
        if self.noise_level == 0.0:
            return str(self.base_score)
        
        # Add controlled noise
        noise = self.random.uniform(-self.noise_level, self.noise_level)
        noisy_score = max(1, min(5, int(self.base_score + noise)))
        return str(noisy_score)


class TestScoringAccuracy:
    """Test scoring accuracy against ground truth data."""
    
    def test_ground_truth_validation(self):
        """Test Judge accuracy against ground truth test cases."""
        # Create scoring logic for ground truth cases
        scoring_logic = {}
        for case in GROUND_TRUTH_CASES:
            key = f"{case.diagnosis.lower()}|{case.truth.lower()}"
            # Use middle of expected range for consistent testing
            expected_score = (case.expected_score_min + case.expected_score_max) // 2
            scoring_logic[key] = expected_score
        
        judge = Judge({}, client=ValidationClient(scoring_logic))
        
        correct_scores = 0
        total_cases = len(GROUND_TRUTH_CASES)
        
        for case in GROUND_TRUTH_CASES:
            result = judge.evaluate(case.diagnosis, case.truth)
            
            if case.expected_score_min <= result.score <= case.expected_score_max:
                correct_scores += 1
            else:
                pytest.fail(
                    f"Score {result.score} not in expected range "
                    f"[{case.expected_score_min}, {case.expected_score_max}] "
                    f"for {case.description}: '{case.diagnosis}' vs '{case.truth}'"
                )
        
        accuracy = correct_scores / total_cases
        benchmark = get_benchmark_expectations()["accuracy_threshold"]
        
        assert accuracy >= benchmark, \
            f"Accuracy {accuracy:.2%} below benchmark {benchmark:.2%}"
    
    def test_complex_scenarios_validation(self):
        """Test Judge with complex medical scenarios."""
        judge = Judge({}, client=ValidationClient())
        
        for case in COMPLEX_SCENARIOS:
            result = judge.evaluate(case.diagnosis, case.truth)
            
            # Complex scenarios should generally score reasonably
            assert 1 <= result.score <= 5, \
                f"Invalid score {result.score} for complex case: {case.description}"
            
            # Verify explanation is provided
            assert result.explanation, f"Missing explanation for: {case.description}"
    
    def test_domain_specific_accuracy(self):
        """Test accuracy within specific medical domains."""
        judge = Judge({}, client=ValidationClient())
        
        for domain, cases in DOMAIN_SPECIFIC_CASES.items():
            domain_scores = []
            
            for case in cases:
                result = judge.evaluate(case.diagnosis, case.truth)
                domain_scores.append(result.score)
                
                assert case.expected_score_min <= result.score <= case.expected_score_max, \
                    f"Domain {domain} failed for {case.description}"
            
            # Domain-specific cases should generally score well
            avg_score = statistics.mean(domain_scores)
            assert avg_score >= 3.0, \
                f"Domain {domain} average score {avg_score} too low"
    
    def test_edge_cases_handling(self):
        """Test Judge handles edge cases appropriately."""
        judge = Judge({}, client=ValidationClient())
        
        for case in EDGE_CASE_SCENARIOS:
            result = judge.evaluate(case.diagnosis, case.truth)
            
            assert 1 <= result.score <= 5, \
                f"Invalid score {result.score} for edge case: {case.description}"
            
            # Edge cases should have explanations
            assert result.explanation, f"Missing explanation for edge case: {case.description}"


class TestScoringConsistency:
    """Test scoring consistency and reproducibility."""
    
    def test_perfect_consistency_identical_inputs(self):
        """Test that identical inputs always produce identical scores."""
        judge = Judge({}, client=ConsistencyTestClient(base_score=3, noise_level=0.0))
        
        test_pairs = [
            ("pneumonia", "pneumonia"),
            ("heart attack", "myocardial infarction"),
            ("diabetes", "type 2 diabetes"),
        ]
        
        for diagnosis, truth in test_pairs:
            scores = []
            for _ in range(10):
                result = judge.evaluate(diagnosis, truth)
                scores.append(result.score)
            
            # All scores should be identical
            unique_scores = set(scores)
            assert len(unique_scores) == 1, \
                f"Inconsistent scores for '{diagnosis}' vs '{truth}': {scores}"
    
    def test_consistency_with_noise(self):
        """Test consistency behavior under controlled noise conditions."""
        noise_levels = [0.0, 0.1, 0.5, 1.0]
        
        for noise in noise_levels:
            judge = Judge({}, client=ConsistencyTestClient(base_score=3, noise_level=noise))
            
            scores = []
            for _ in range(50):
                result = judge.evaluate("test diagnosis", "test truth")
                scores.append(result.score)
            
            # Calculate consistency metrics
            most_common_score = Counter(scores).most_common(1)[0][1]
            consistency_rate = most_common_score / len(scores)
            
            if noise == 0.0:
                assert consistency_rate == 1.0, "Perfect consistency expected with no noise"
            else:
                # With noise, consistency should decrease but remain reasonable
                assert consistency_rate >= 0.3, \
                    f"Too much inconsistency with noise {noise}: {consistency_rate:.2%}"
    
    def test_cross_session_consistency(self):
        """Test consistency across different Judge instances."""
        test_cases = [
            ("myocardial infarction", "heart attack"),
            ("diabetes", "diabetes mellitus"),
            ("pneumonia", "lung infection"),
        ]
        
        results_session1 = []
        results_session2 = []
        
        # Session 1
        judge1 = Judge({}, client=ConsistencyTestClient(base_score=4, noise_level=0.0))
        for diagnosis, truth in test_cases:
            result = judge1.evaluate(diagnosis, truth)
            results_session1.append(result.score)
        
        # Session 2 (new instance, same client config)
        judge2 = Judge({}, client=ConsistencyTestClient(base_score=4, noise_level=0.0))
        for diagnosis, truth in test_cases:
            result = judge2.evaluate(diagnosis, truth)
            results_session2.append(result.score)
        
        # Results should be identical across sessions
        assert results_session1 == results_session2, \
            "Cross-session inconsistency detected"


class TestStatisticalValidation:
    """Test statistical properties of scoring system."""
    
    def test_score_distribution_properties(self):
        """Test that score distributions have expected statistical properties."""
        # Create client with known distribution
        score_counts = {1: 10, 2: 15, 3: 30, 4: 25, 5: 20}  # Expected distribution
        
        class DistributionClient:
            def __init__(self):
                self.call_count = 0
                import random
                self.random = random.Random(42)
            
            def chat(self, messages, model):
                # Generate scores according to distribution
                rand_val = self.random.randint(1, 100)
                if rand_val <= 10:
                    return "1"
                elif rand_val <= 25:
                    return "2"
                elif rand_val <= 55:
                    return "3"
                elif rand_val <= 80:
                    return "4"
                else:
                    return "5"
        
        judge = Judge({}, client=DistributionClient())
        
        scores = []
        for i in range(100):
            result = judge.evaluate(f"diagnosis_{i}", f"truth_{i}")
            scores.append(result.score)
        
        # Test statistical properties
        score_counter = Counter(scores)
        
        # Should see all score values
        assert len(score_counter) == 5, f"Missing score values: {sorted(score_counter.keys())}"
        
        # Distribution should be reasonable
        for score in range(1, 6):
            count = score_counter[score]
            assert count > 0, f"Score {score} never observed"
        
        # Mean should be in reasonable range
        mean_score = statistics.mean(scores)
        assert 2.0 <= mean_score <= 4.0, f"Mean score {mean_score} outside expected range"
    
    def test_scoring_variance_analysis(self):
        """Test variance in scoring for similar inputs."""
        similar_pairs = [
            [
                ("myocardial infarction", "heart attack"),
                ("myocardial infarction", "MI"),
                ("myocardial infarction", "cardiac event"),
            ],
            [
                ("pneumonia", "lung infection"),
                ("pneumonia", "respiratory infection"),
                ("pneumonia", "chest infection"),
            ]
        ]
        
        judge = Judge({}, client=ValidationClient())
        
        for pair_group in similar_pairs:
            scores = []
            for diagnosis, truth in pair_group:
                result = judge.evaluate(diagnosis, truth)
                scores.append(result.score)
            
            # Similar pairs should have low variance
            if len(scores) > 1:
                variance = statistics.variance(scores)
                assert variance <= 2.0, \
                    f"High variance {variance} in similar pairs: {pair_group}"
    
    def test_score_correlation_analysis(self):
        """Test correlation between diagnosis similarity and scores."""
        # Create test cases with known similarity levels
        test_cases = [
            # High similarity cases (should score 4-5)
            ("exact match", "exact match"),
            ("synonym A", "synonym A equivalent"),
            
            # Medium similarity cases (should score 2-4)
            ("related condition", "similar condition"),
            ("symptom", "underlying disease"),
            
            # Low similarity cases (should score 1-2)
            ("completely different", "unrelated condition"),
            ("opposite meaning", "contradictory diagnosis"),
        ]
        
        judge = Judge({}, client=ValidationClient())
        
        high_sim_scores = []
        medium_sim_scores = []
        low_sim_scores = []
        
        for i, (diagnosis, truth) in enumerate(test_cases):
            result = judge.evaluate(diagnosis, truth)
            
            if i < 2:  # High similarity
                high_sim_scores.append(result.score)
            elif i < 4:  # Medium similarity
                medium_sim_scores.append(result.score)
            else:  # Low similarity
                low_sim_scores.append(result.score)
        
        # Check correlation: high similarity → high scores
        if high_sim_scores:
            avg_high = statistics.mean(high_sim_scores)
            assert avg_high >= 3.5, f"High similarity average {avg_high} too low"
        
        if low_sim_scores:
            avg_low = statistics.mean(low_sim_scores)
            assert avg_low <= 2.5, f"Low similarity average {avg_low} too high"


class TestRegressionValidation:
    """Test validation against known regression cases."""
    
    def test_known_regression_cases(self):
        """Test cases that previously caused issues."""
        judge = Judge({}, client=ValidationClient())
        
        for case in REGRESSION_TEST_CASES:
            result = judge.evaluate(case.diagnosis, case.truth)
            
            # Should handle gracefully without errors
            assert 1 <= result.score <= 5, \
                f"Regression case failed: {case.description}"
            
            # Should have explanation
            assert result.explanation, \
                f"Missing explanation for regression case: {case.description}"
    
    def test_unicode_regression_validation(self):
        """Test Unicode handling regression cases."""
        judge = Judge({}, client=ValidationClient())
        
        for case in UNICODE_TEST_CASES:
            result = judge.evaluate(case.diagnosis, case.truth)
            
            assert 1 <= result.score <= 5, \
                f"Unicode regression failed: {case.description}"
            
            # Should handle Unicode gracefully
            assert isinstance(result.explanation, str), \
                f"Invalid explanation type for Unicode case: {case.description}"
    
    def test_whitespace_handling_regression(self):
        """Test whitespace handling regressions."""
        judge = Judge({}, client=ValidationClient())
        
        whitespace_variants = [
            ("diagnosis", "truth"),
            ("  diagnosis  ", "  truth  "),
            ("\ndiagnosis\n", "\ttruth\t"),
            ("diagnosis\r\n", "truth\r\n"),
        ]
        
        # All variants should produce same score
        scores = []
        for diagnosis, truth in whitespace_variants:
            result = judge.evaluate(diagnosis, truth)
            scores.append(result.score)
        
        unique_scores = set(scores)
        assert len(unique_scores) == 1, \
            f"Whitespace handling inconsistency: {scores}"


class TestBenchmarkValidation:
    """Test performance against established benchmarks."""
    
    def test_accuracy_benchmark(self):
        """Test that accuracy meets benchmark standards."""
        judge = Judge({}, client=ValidationClient())
        
        # Use all ground truth cases for comprehensive testing
        correct_predictions = 0
        total_cases = len(GROUND_TRUTH_CASES)
        
        for case in GROUND_TRUTH_CASES:
            result = judge.evaluate(case.diagnosis, case.truth)
            
            if case.expected_score_min <= result.score <= case.expected_score_max:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_cases
        benchmark = get_benchmark_expectations()["accuracy_threshold"]
        
        assert accuracy >= benchmark, \
            f"Accuracy {accuracy:.2%} below benchmark {benchmark:.2%}"
    
    def test_consistency_benchmark(self):
        """Test that consistency meets benchmark standards."""
        judge = Judge({}, client=ConsistencyTestClient(base_score=3, noise_level=0.0))
        
        # Test consistency over multiple runs
        test_case = ("test diagnosis", "test truth")
        scores = []
        
        for _ in range(20):
            result = judge.evaluate(test_case[0], test_case[1])
            scores.append(result.score)
        
        # Calculate consistency rate
        most_common_count = Counter(scores).most_common(1)[0][1]
        consistency_rate = most_common_count / len(scores)
        
        benchmark = get_benchmark_expectations()["consistency_threshold"]
        
        assert consistency_rate >= benchmark, \
            f"Consistency {consistency_rate:.2%} below benchmark {benchmark:.2%}"
    
    def test_performance_benchmark(self):
        """Test that performance meets benchmark standards."""
        judge = Judge({}, client=ValidationClient())
        
        # Measure evaluation time
        start_time = time.perf_counter()
        
        for diagnosis, truth in PERFORMANCE_TEST_CASES[:10]:  # Test subset for speed
            result = judge.evaluate(diagnosis, truth)
            assert 1 <= result.score <= 5  # Basic validation
        
        end_time = time.perf_counter()
        avg_time_per_evaluation = (end_time - start_time) / 10
        
        benchmark = get_benchmark_expectations()["performance_threshold"]
        
        assert avg_time_per_evaluation <= benchmark, \
            f"Average time {avg_time_per_evaluation:.3f}s exceeds benchmark {benchmark}s"


class TestValidationReporting:
    """Test validation reporting and metrics collection."""
    
    def test_validation_metrics_collection(self):
        """Test collection of validation metrics."""
        judge = Judge({}, client=ValidationClient())
        
        metrics = {
            "total_evaluations": 0,
            "score_distribution": defaultdict(int),
            "category_accuracy": defaultdict(list),
            "difficulty_performance": defaultdict(list),
        }
        
        all_test_cases = GROUND_TRUTH_CASES + COMPLEX_SCENARIOS
        
        for case in all_test_cases[:10]:  # Test subset
            result = judge.evaluate(case.diagnosis, case.truth)
            
            metrics["total_evaluations"] += 1
            metrics["score_distribution"][result.score] += 1
            
            # Check if score is within expected range
            is_correct = case.expected_score_min <= result.score <= case.expected_score_max
            metrics["category_accuracy"][case.category].append(is_correct)
            metrics["difficulty_performance"][case.difficulty].append(result.score)
        
        # Validate metrics collection
        assert metrics["total_evaluations"] > 0
        assert len(metrics["score_distribution"]) > 0
        assert len(metrics["category_accuracy"]) > 0
        
        # Calculate category accuracy
        for category, results in metrics["category_accuracy"].items():
            accuracy = sum(results) / len(results)
            assert 0.0 <= accuracy <= 1.0, f"Invalid accuracy for category {category}"
    
    def test_comprehensive_validation_report(self):
        """Test generation of comprehensive validation report."""
        judge = Judge({}, client=ValidationClient())
        
        report = {
            "test_summary": {
                "total_cases": 0,
                "passed_cases": 0,
                "failed_cases": 0,
            },
            "performance_metrics": {
                "avg_evaluation_time": 0.0,
                "total_evaluation_time": 0.0,
            },
            "accuracy_by_category": {},
            "score_distribution": defaultdict(int),
        }
        
        test_cases = GROUND_TRUTH_CASES[:5]  # Small subset for testing
        
        start_time = time.perf_counter()
        
        for case in test_cases:
            case_start = time.perf_counter()
            result = judge.evaluate(case.diagnosis, case.truth)
            case_end = time.perf_counter()
            
            report["test_summary"]["total_cases"] += 1
            report["score_distribution"][result.score] += 1
            
            # Check accuracy
            is_correct = case.expected_score_min <= result.score <= case.expected_score_max
            if is_correct:
                report["test_summary"]["passed_cases"] += 1
            else:
                report["test_summary"]["failed_cases"] += 1
            
            # Track category accuracy
            if case.category not in report["accuracy_by_category"]:
                report["accuracy_by_category"][case.category] = []
            report["accuracy_by_category"][case.category].append(is_correct)
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        report["performance_metrics"]["total_evaluation_time"] = total_time
        report["performance_metrics"]["avg_evaluation_time"] = (
            total_time / report["test_summary"]["total_cases"]
        )
        
        # Validate report structure
        assert report["test_summary"]["total_cases"] > 0
        assert report["test_summary"]["passed_cases"] >= 0
        assert report["test_summary"]["failed_cases"] >= 0
        assert (
            report["test_summary"]["passed_cases"] + 
            report["test_summary"]["failed_cases"] == 
            report["test_summary"]["total_cases"]
        )
        assert report["performance_metrics"]["avg_evaluation_time"] >= 0