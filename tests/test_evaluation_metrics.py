"""Comprehensive evaluation metrics tests for the Dx0 diagnostic system.

This module provides thorough testing of diagnostic accuracy metrics, performance
evaluation, cost-effectiveness analysis, and statistical validation. It ensures
the evaluation system correctly measures diagnostic performance across different
scenarios and provides reliable statistical analysis.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from unittest.mock import Mock, patch

from sdb.evaluation import Evaluator, SessionResult, batch_evaluate
from sdb.judge import Judge, Judgement
from sdb.cost_estimator import CostEstimator, CptCost
from sdb.ensemble import DiagnosisResult, WeightedVoter, cost_adjusted_selection
from sdb.statistics import permutation_test


@dataclass
class MockJudge:
    """Mock judge for testing evaluation metrics."""
    
    def __init__(self, score: int = 4, explanation: str = "Mock judgement"):
        self.score = score
        self.explanation = explanation
    
    def evaluate(self, diagnosis: str, truth: str) -> Judgement:
        return Judgement(score=self.score, explanation=self.explanation)


@dataclass
class MockCostEstimator:
    """Mock cost estimator for testing evaluation metrics."""
    
    def __init__(self, costs: Dict[str, float] = None):
        self.costs = costs or {}
        self.default_cost = 100.0
    
    def estimate_cost(self, test_name: str) -> float:
        return self.costs.get(test_name, self.default_cost)


class TestDiagnosticAccuracyMetrics:
    """Test suite for diagnostic accuracy metrics calculation."""
    
    def test_precision_recall_calculation(self):
        """Test precision and recall calculation for diagnostic accuracy."""
        # Test data: true positives, false positives, false negatives
        true_diagnoses = ["flu", "flu", "pneumonia", "flu", "cold"]
        pred_diagnoses = ["flu", "cold", "pneumonia", "flu", "flu"]
        
        metrics = self._calculate_precision_recall(true_diagnoses, pred_diagnoses, "flu")
        
        # flu: TP=2, FP=1, FN=1
        # Precision = TP/(TP+FP) = 2/3 ≈ 0.667
        # Recall = TP/(TP+FN) = 2/3 ≈ 0.667
        assert abs(metrics["precision"] - 0.667) < 0.001
        assert abs(metrics["recall"] - 0.667) < 0.001
        assert abs(metrics["f1_score"] - 0.667) < 0.001
    
    def test_sensitivity_specificity_calculation(self):
        """Test sensitivity and specificity calculation."""
        # Binary classification: positive = "flu", negative = "not flu"
        true_labels = [1, 1, 0, 1, 0, 0, 1, 0]  # 1=flu, 0=not flu
        pred_labels = [1, 0, 0, 1, 1, 0, 1, 0]  # predictions
        
        metrics = self._calculate_sensitivity_specificity(true_labels, pred_labels)
        
        # TP=3, TN=3, FP=1, FN=1
        # Sensitivity = TP/(TP+FN) = 3/4 = 0.75
        # Specificity = TN/(TN+FP) = 3/4 = 0.75
        assert metrics["sensitivity"] == 0.75
        assert metrics["specificity"] == 0.75
        assert metrics["accuracy"] == 0.75  # (TP+TN)/(TP+TN+FP+FN)
    
    def test_multiclass_accuracy_metrics(self):
        """Test accuracy metrics for multiclass diagnostic scenarios."""
        true_diagnoses = ["flu", "pneumonia", "cold", "flu", "bronchitis", "flu"]
        pred_diagnoses = ["flu", "pneumonia", "flu", "flu", "bronchitis", "cold"]
        
        metrics = self._calculate_multiclass_metrics(true_diagnoses, pred_diagnoses)
        
        # Overall accuracy: 4/6 = 0.667
        assert abs(metrics["overall_accuracy"] - 0.667) < 0.001
        
        # Per-class metrics should be calculated
        assert "flu" in metrics["per_class"]
        assert "pneumonia" in metrics["per_class"]
        assert "precision" in metrics["per_class"]["flu"]
        assert "recall" in metrics["per_class"]["flu"]
    
    def test_confusion_matrix_generation(self):
        """Test confusion matrix generation for diagnostic evaluation."""
        true_diagnoses = ["flu", "flu", "cold", "pneumonia"]
        pred_diagnoses = ["flu", "cold", "cold", "pneumonia"]
        
        confusion_matrix = self._generate_confusion_matrix(true_diagnoses, pred_diagnoses)
        
        # Should have correct dimensions and values
        assert len(confusion_matrix) == 3  # flu, cold, pneumonia
        assert confusion_matrix["flu"]["flu"] == 1  # correct flu prediction
        assert confusion_matrix["flu"]["cold"] == 1  # flu misclassified as cold
        assert confusion_matrix["cold"]["cold"] == 1  # correct cold prediction
        assert confusion_matrix["pneumonia"]["pneumonia"] == 1  # correct pneumonia
    
    def _calculate_precision_recall(self, true_labels: List[str], pred_labels: List[str], 
                                  positive_class: str) -> Dict[str, float]:
        """Calculate precision, recall, and F1-score for a specific class."""
        tp = sum(1 for t, p in zip(true_labels, pred_labels) 
                if t == positive_class and p == positive_class)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) 
                if t != positive_class and p == positive_class)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) 
                if t == positive_class and p != positive_class)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"precision": precision, "recall": recall, "f1_score": f1_score}
    
    def _calculate_sensitivity_specificity(self, true_labels: List[int], 
                                         pred_labels: List[int]) -> Dict[str, float]:
        """Calculate sensitivity, specificity, and accuracy for binary classification."""
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 0)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        return {"sensitivity": sensitivity, "specificity": specificity, "accuracy": accuracy}
    
    def _calculate_multiclass_metrics(self, true_labels: List[str], 
                                    pred_labels: List[str]) -> Dict:
        """Calculate comprehensive metrics for multiclass classification."""
        classes = list(set(true_labels + pred_labels))
        overall_correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
        overall_accuracy = overall_correct / len(true_labels)
        
        per_class_metrics = {}
        for cls in classes:
            metrics = self._calculate_precision_recall(true_labels, pred_labels, cls)
            per_class_metrics[cls] = metrics
        
        return {
            "overall_accuracy": overall_accuracy,
            "per_class": per_class_metrics,
            "classes": classes
        }
    
    def _generate_confusion_matrix(self, true_labels: List[str], 
                                 pred_labels: List[str]) -> Dict[str, Dict[str, int]]:
        """Generate confusion matrix for diagnostic evaluation."""
        classes = sorted(set(true_labels + pred_labels))
        matrix = {true_cls: {pred_cls: 0 for pred_cls in classes} for true_cls in classes}
        
        for true_label, pred_label in zip(true_labels, pred_labels):
            matrix[true_label][pred_label] += 1
        
        return matrix


class TestPerformanceMetrics:
    """Test suite for diagnostic performance metrics."""
    
    def test_diagnostic_quality_scoring(self):
        """Test diagnostic quality scoring based on Judge evaluations."""
        judge = MockJudge(score=4)
        cost_estimator = MockCostEstimator({"cbc": 50.0, "chest_xray": 200.0})
        evaluator = Evaluator(judge, cost_estimator, correct_threshold=4)
        
        result = evaluator.evaluate(
            diagnosis="viral pneumonia",
            truth="bacterial pneumonia", 
            tests=["cbc", "chest_xray"],
            visits=1,
            duration=45.0
        )
        
        assert result.score == 4
        assert result.correct is True
        assert result.total_cost == 550.0  # 300 (visit) + 50 + 200
        assert result.duration == 45.0
    
    def test_cost_effectiveness_calculation(self):
        """Test cost-effectiveness ratio calculation."""
        # High accuracy, low cost scenario
        high_acc_results = [
            SessionResult(total_cost=300.0, score=5, correct=True, duration=30.0),
            SessionResult(total_cost=350.0, score=4, correct=True, duration=25.0),
            SessionResult(total_cost=400.0, score=5, correct=True, duration=35.0),
        ]
        
        # Lower accuracy, higher cost scenario
        low_acc_results = [
            SessionResult(total_cost=800.0, score=3, correct=False, duration=60.0),
            SessionResult(total_cost=750.0, score=2, correct=False, duration=45.0),
            SessionResult(total_cost=900.0, score=4, correct=True, duration=70.0),
        ]
        
        high_eff = self._calculate_cost_effectiveness(high_acc_results)
        low_eff = self._calculate_cost_effectiveness(low_acc_results)
        
        # High accuracy scenario should have better cost-effectiveness
        assert high_eff["accuracy_per_dollar"] > low_eff["accuracy_per_dollar"]
        assert high_eff["cost_per_correct_diagnosis"] < low_eff["cost_per_correct_diagnosis"]
    
    def test_time_efficiency_metrics(self):
        """Test time efficiency metrics for diagnostic sessions."""
        results = [
            SessionResult(total_cost=300.0, score=5, correct=True, duration=20.0),
            SessionResult(total_cost=400.0, score=4, correct=True, duration=30.0),
            SessionResult(total_cost=500.0, score=3, correct=False, duration=45.0),
        ]
        
        efficiency_metrics = self._calculate_time_efficiency(results)
        
        assert "avg_duration" in efficiency_metrics
        assert "duration_per_correct" in efficiency_metrics
        assert "time_cost_ratio" in efficiency_metrics
        assert efficiency_metrics["avg_duration"] == 31.67  # (20+30+45)/3
    
    def test_diagnostic_consistency_metrics(self):
        """Test consistency metrics across multiple diagnostic sessions."""
        # Consistent high performance
        consistent_results = [
            SessionResult(total_cost=300.0, score=5, correct=True, duration=25.0),
            SessionResult(total_cost=320.0, score=5, correct=True, duration=22.0),
            SessionResult(total_cost=310.0, score=4, correct=True, duration=28.0),
        ]
        
        # Inconsistent performance
        inconsistent_results = [
            SessionResult(total_cost=300.0, score=5, correct=True, duration=20.0),
            SessionResult(total_cost=600.0, score=2, correct=False, duration=60.0),
            SessionResult(total_cost=400.0, score=4, correct=True, duration=30.0),
        ]
        
        consistent_metrics = self._calculate_consistency_metrics(consistent_results)
        inconsistent_metrics = self._calculate_consistency_metrics(inconsistent_results)
        
        # Consistent results should have lower variance
        assert consistent_metrics["score_std"] < inconsistent_metrics["score_std"]
        assert consistent_metrics["cost_std"] < inconsistent_metrics["cost_std"]
    
    def _calculate_cost_effectiveness(self, results: List[SessionResult]) -> Dict[str, float]:
        """Calculate cost-effectiveness metrics."""
        total_cost = sum(r.total_cost for r in results)
        correct_count = sum(1 for r in results if r.correct)
        avg_score = sum(r.score for r in results) / len(results)
        
        accuracy_per_dollar = avg_score / (total_cost / len(results)) if total_cost > 0 else 0.0
        cost_per_correct = total_cost / correct_count if correct_count > 0 else float('inf')
        
        return {
            "accuracy_per_dollar": accuracy_per_dollar,
            "cost_per_correct_diagnosis": cost_per_correct,
            "total_cost": total_cost,
            "correct_diagnoses": correct_count
        }
    
    def _calculate_time_efficiency(self, results: List[SessionResult]) -> Dict[str, float]:
        """Calculate time efficiency metrics."""
        durations = [r.duration for r in results]
        correct_durations = [r.duration for r in results if r.correct]
        costs = [r.total_cost for r in results]
        
        avg_duration = sum(durations) / len(durations)
        duration_per_correct = sum(correct_durations) / len(correct_durations) if correct_durations else 0.0
        avg_cost = sum(costs) / len(costs)
        time_cost_ratio = avg_duration / avg_cost if avg_cost > 0 else 0.0
        
        return {
            "avg_duration": round(avg_duration, 2),
            "duration_per_correct": round(duration_per_correct, 2),
            "time_cost_ratio": round(time_cost_ratio, 4)
        }
    
    def _calculate_consistency_metrics(self, results: List[SessionResult]) -> Dict[str, float]:
        """Calculate consistency metrics across sessions."""
        scores = [r.score for r in results]
        costs = [r.total_cost for r in results]
        durations = [r.duration for r in results]
        
        return {
            "score_std": np.std(scores),
            "cost_std": np.std(costs),
            "duration_std": np.std(durations),
            "score_cv": np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0.0
        }


class TestCostBenefitAnalysis:
    """Test suite for cost-benefit analysis in diagnostic evaluation."""
    
    def test_cost_adjusted_diagnosis_selection(self):
        """Test cost-adjusted diagnosis selection algorithm."""
        results = [
            DiagnosisResult("expensive_accurate", confidence=0.9, cost=1000.0),
            DiagnosisResult("cheap_moderate", confidence=0.7, cost=200.0),  
            DiagnosisResult("moderate_accurate", confidence=0.8, cost=500.0)
        ]
        
        # With low cost weight, should prefer accuracy
        low_cost_weight_choice = cost_adjusted_selection(results, cost_weight=0.001)
        assert low_cost_weight_choice == "expensive_accurate"
        
        # With high cost weight, should prefer cost-effectiveness
        high_cost_weight_choice = cost_adjusted_selection(results, cost_weight=0.01)
        assert high_cost_weight_choice == "cheap_moderate"
    
    def test_pareto_frontier_analysis(self):
        """Test Pareto frontier analysis for cost vs accuracy trade-offs."""
        results = [
            (100, 0.6),  # (cost, accuracy)
            (200, 0.7),
            (300, 0.8),
            (250, 0.75),  # Dominated point
            (400, 0.85),
            (500, 0.9)
        ]
        
        pareto_points = self._find_pareto_frontier(results)
        
        # Should exclude dominated point (250, 0.75)
        costs = [p[0] for p in pareto_points]
        accuracies = [p[1] for p in pareto_points]
        
        assert (250, 0.75) not in pareto_points
        assert len(pareto_points) < len(results)
        
        # Pareto frontier should be sorted by cost
        assert costs == sorted(costs)
    
    def test_incremental_cost_effectiveness_ratio(self):
        """Test incremental cost-effectiveness ratio (ICER) calculation."""
        strategies = [
            {"name": "basic", "cost": 300.0, "effectiveness": 0.7},
            {"name": "standard", "cost": 500.0, "effectiveness": 0.8},
            {"name": "comprehensive", "cost": 800.0, "effectiveness": 0.85}
        ]
        
        icers = self._calculate_icers(strategies)
        
        # ICER from basic to standard: (500-300)/(0.8-0.7) = 2000
        # ICER from standard to comprehensive: (800-500)/(0.85-0.8) = 6000
        assert abs(icers["standard"] - 2000.0) < 0.1
        assert abs(icers["comprehensive"] - 6000.0) < 0.1
    
    def test_value_based_ranking(self):
        """Test value-based ranking of diagnostic strategies."""
        strategies = [
            {"name": "A", "cost": 400, "accuracy": 0.85, "time": 30},
            {"name": "B", "cost": 600, "accuracy": 0.90, "time": 45},
            {"name": "C", "cost": 200, "accuracy": 0.70, "time": 15},
        ]
        
        # Weight accuracy highly
        accuracy_weights = {"accuracy": 0.6, "cost": -0.3, "time": -0.1}
        ranking_accuracy = self._rank_strategies(strategies, accuracy_weights)
        assert ranking_accuracy[0]["name"] == "B"  # Highest accuracy
        
        # Weight cost savings highly
        cost_weights = {"accuracy": 0.3, "cost": -0.6, "time": -0.1}
        ranking_cost = self._rank_strategies(strategies, cost_weights)
        assert ranking_cost[0]["name"] == "C"  # Lowest cost
    
    def _find_pareto_frontier(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Find Pareto frontier for cost vs accuracy trade-offs."""
        # Sort by cost
        sorted_points = sorted(points, key=lambda x: x[0])
        pareto_frontier = []
        
        max_accuracy = -1.0
        for cost, accuracy in sorted_points:
            if accuracy > max_accuracy:
                pareto_frontier.append((cost, accuracy))
                max_accuracy = accuracy
        
        return pareto_frontier
    
    def _calculate_icers(self, strategies: List[Dict]) -> Dict[str, float]:
        """Calculate incremental cost-effectiveness ratios."""
        # Sort by effectiveness
        sorted_strategies = sorted(strategies, key=lambda x: x["effectiveness"])
        icers = {}
        
        for i in range(1, len(sorted_strategies)):
            current = sorted_strategies[i]
            previous = sorted_strategies[i-1]
            
            cost_diff = current["cost"] - previous["cost"]
            eff_diff = current["effectiveness"] - previous["effectiveness"]
            
            icer = cost_diff / eff_diff if eff_diff > 0 else float('inf')
            icers[current["name"]] = icer
        
        return icers
    
    def _rank_strategies(self, strategies: List[Dict], weights: Dict[str, float]) -> List[Dict]:
        """Rank strategies based on weighted scoring."""
        scored_strategies = []
        
        for strategy in strategies:
            score = 0.0
            for metric, weight in weights.items():
                if metric in strategy:
                    score += strategy[metric] * weight
            
            strategy_copy = strategy.copy()
            strategy_copy["composite_score"] = score
            scored_strategies.append(strategy_copy)
        
        return sorted(scored_strategies, key=lambda x: x["composite_score"], reverse=True)


class TestEvaluationFrameworks:
    """Test suite for evaluation frameworks and methodologies."""
    
    def test_cross_validation_evaluation(self):
        """Test cross-validation evaluation framework."""
        # Mock data for 10 cases
        case_ids = [f"case_{i}" for i in range(10)]
        
        def mock_run_case(case_id: str) -> Dict[str, str]:
            # Simulate variable performance
            score = 4 if int(case_id.split("_")[1]) % 2 == 0 else 3
            return {
                "id": case_id,
                "score": str(score),
                "cost": "400.0",
                "correct": str(score >= 4)
            }
        
        # Perform 5-fold cross-validation
        cv_results = self._perform_cross_validation(case_ids, mock_run_case, k_folds=5)
        
        assert len(cv_results["fold_results"]) == 5
        assert "mean_accuracy" in cv_results["summary"]
        assert "std_accuracy" in cv_results["summary"]
        assert "mean_cost" in cv_results["summary"]
    
    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval calculation."""
        # Sample accuracy scores
        scores = [0.8, 0.85, 0.75, 0.9, 0.82, 0.88, 0.78, 0.86, 0.84, 0.80]
        
        ci_95 = self._bootstrap_confidence_interval(scores, confidence_level=0.95, n_bootstrap=1000)
        ci_99 = self._bootstrap_confidence_interval(scores, confidence_level=0.99, n_bootstrap=1000)
        
        # 99% CI should be wider than 95% CI
        assert (ci_99["upper"] - ci_99["lower"]) > (ci_95["upper"] - ci_95["lower"])
        
        # Mean should be within both intervals
        mean_score = np.mean(scores)
        assert ci_95["lower"] <= mean_score <= ci_95["upper"]
        assert ci_99["lower"] <= mean_score <= ci_99["upper"]
    
    def test_stratified_evaluation(self):
        """Test stratified evaluation by case complexity or type."""
        cases = [
            {"id": "case_1", "complexity": "simple", "type": "respiratory"},
            {"id": "case_2", "complexity": "complex", "type": "cardiac"},
            {"id": "case_3", "complexity": "simple", "type": "respiratory"},
            {"id": "case_4", "complexity": "complex", "type": "neurological"},
        ]
        
        def mock_run_case(case: Dict) -> Dict[str, float]:
            # Complex cases have lower accuracy
            base_accuracy = 0.9 if case["complexity"] == "simple" else 0.7
            return {"accuracy": base_accuracy, "cost": 400.0}
        
        stratified_results = self._perform_stratified_evaluation(cases, mock_run_case, "complexity")
        
        assert "simple" in stratified_results
        assert "complex" in stratified_results
        assert stratified_results["simple"]["mean_accuracy"] > stratified_results["complex"]["mean_accuracy"]
    
    def test_learning_curve_analysis(self):
        """Test learning curve analysis for diagnostic system improvement."""
        # Simulate improving performance with more training data
        training_sizes = [10, 20, 50, 100, 200]
        
        def simulate_performance(train_size: int) -> Dict[str, float]:
            # Performance improves with more data but plateaus
            accuracy = min(0.95, 0.6 + 0.3 * np.log(train_size) / np.log(200))
            return {"accuracy": accuracy, "cost": 400.0}
        
        learning_curve = self._generate_learning_curve(training_sizes, simulate_performance)
        
        assert len(learning_curve) == len(training_sizes)
        
        # Performance should generally improve with more data
        accuracies = [result["accuracy"] for result in learning_curve]
        assert accuracies[0] < accuracies[-1]  # First < Last
    
    def _perform_cross_validation(self, case_ids: List[str], run_case_func, k_folds: int = 5) -> Dict:
        """Perform k-fold cross-validation evaluation."""
        fold_size = len(case_ids) // k_folds
        fold_results = []
        
        for i in range(k_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < k_folds - 1 else len(case_ids)
            
            test_cases = case_ids[start_idx:end_idx]
            results = [run_case_func(case_id) for case_id in test_cases]
            
            fold_accuracy = sum(1 for r in results if r["correct"] == "True") / len(results)
            fold_cost = sum(float(r["cost"]) for r in results) / len(results)
            
            fold_results.append({
                "fold": i,
                "accuracy": fold_accuracy,
                "cost": fold_cost,
                "n_cases": len(test_cases)
            })
        
        # Calculate summary statistics
        accuracies = [fold["accuracy"] for fold in fold_results]
        costs = [fold["cost"] for fold in fold_results]
        
        return {
            "fold_results": fold_results,
            "summary": {
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
                "mean_cost": np.mean(costs),
                "std_cost": np.std(costs),
                "n_folds": k_folds
            }
        }
    
    def _bootstrap_confidence_interval(self, data: List[float], confidence_level: float = 0.95, 
                                     n_bootstrap: int = 1000) -> Dict[str, float]:
        """Calculate bootstrap confidence interval."""
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            "lower": np.percentile(bootstrap_means, lower_percentile),
            "upper": np.percentile(bootstrap_means, upper_percentile),
            "mean": np.mean(bootstrap_means),
            "confidence_level": confidence_level
        }
    
    def _perform_stratified_evaluation(self, cases: List[Dict], run_case_func, 
                                     stratify_by: str) -> Dict:
        """Perform stratified evaluation by a given attribute."""
        strata = {}
        
        # Group cases by stratification attribute
        for case in cases:
            stratum = case[stratify_by]
            if stratum not in strata:
                strata[stratum] = []
            strata[stratum].append(case)
        
        # Evaluate each stratum
        stratified_results = {}
        for stratum, stratum_cases in strata.items():
            results = [run_case_func(case) for case in stratum_cases]
            
            accuracies = [r["accuracy"] for r in results]
            costs = [r["cost"] for r in results]
            
            stratified_results[stratum] = {
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
                "mean_cost": np.mean(costs),
                "n_cases": len(stratum_cases)
            }
        
        return stratified_results
    
    def _generate_learning_curve(self, training_sizes: List[int], 
                               performance_func) -> List[Dict[str, float]]:
        """Generate learning curve data."""
        learning_curve = []
        
        for train_size in training_sizes:
            performance = performance_func(train_size)
            performance["training_size"] = train_size
            learning_curve.append(performance)
        
        return learning_curve


class TestEvaluationValidation:
    """Test suite for evaluation system validation and quality assurance."""
    
    def test_evaluator_input_validation(self):
        """Test input validation for evaluation components."""
        judge = MockJudge(score=4)
        cost_estimator = MockCostEstimator()
        evaluator = Evaluator(judge, cost_estimator)
        
        # Test negative visits validation
        with pytest.raises(Exception):  # Should raise EvaluationError
            evaluator.evaluate("flu", "flu", [], visits=-1)
        
        # Test negative duration validation
        with pytest.raises(Exception):  # Should raise EvaluationError
            evaluator.evaluate("flu", "flu", [], duration=-10.0)
    
    def test_edge_case_handling(self):
        """Test handling of edge cases in evaluation."""
        judge = MockJudge(score=1)
        cost_estimator = MockCostEstimator({})
        evaluator = Evaluator(judge, cost_estimator)
        
        # Test with empty test list
        result = evaluator.evaluate("flu", "flu", [], visits=0)
        assert result.total_cost == 0.0
        assert result.score == 1
        
        # Test with very long test list
        many_tests = [f"test_{i}" for i in range(100)]
        result = evaluator.evaluate("flu", "flu", many_tests, visits=1)
        assert result.total_cost == 300.0 + 100 * 100.0  # Visit fee + test costs
    
    def test_statistical_significance_validation(self):
        """Test statistical significance calculations for evaluation results."""
        # Two groups with different performance
        group_a_scores = [0.85, 0.88, 0.82, 0.87, 0.84, 0.86, 0.83, 0.89, 0.85, 0.87]
        group_b_scores = [0.75, 0.78, 0.72, 0.77, 0.74, 0.76, 0.73, 0.79, 0.75, 0.77]
        
        # Perform permutation test
        p_value = permutation_test(group_a_scores, group_b_scores, num_rounds=1000, seed=42)
        
        # Should detect significant difference
        assert p_value < 0.05
        
        # Test with identical groups
        identical_scores = [0.8] * 10
        p_value_identical = permutation_test(identical_scores, identical_scores, num_rounds=1000, seed=42)
        
        # Should not detect significant difference
        assert p_value_identical > 0.05
    
    def test_evaluation_reproducibility(self):
        """Test reproducibility of evaluation results."""
        judge = MockJudge(score=4)
        cost_estimator = MockCostEstimator({"test1": 100.0, "test2": 200.0})
        evaluator = Evaluator(judge, cost_estimator)
        
        # Run same evaluation multiple times
        results = []
        for _ in range(5):
            result = evaluator.evaluate(
                "pneumonia", "pneumonia", ["test1", "test2"], visits=1, duration=30.0
            )
            results.append(result)
        
        # All results should be identical
        for result in results[1:]:
            assert result.score == results[0].score
            assert result.total_cost == results[0].total_cost
            assert result.correct == results[0].correct
            assert result.duration == results[0].duration
    
    def test_performance_under_scale(self):
        """Test evaluation system performance under scale."""
        import time
        
        judge = MockJudge(score=4)
        cost_estimator = MockCostEstimator()
        evaluator = Evaluator(judge, cost_estimator)
        
        # Test batch evaluation performance
        large_case_list = [f"case_{i}" for i in range(100)]
        
        def quick_run_case(case_id: str) -> Dict[str, str]:
            return {"id": case_id, "result": "evaluated"}
        
        start_time = time.time()
        results = batch_evaluate(large_case_list, quick_run_case, concurrency=10)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert len(results) == 100
        assert (end_time - start_time) < 10.0  # Should complete within 10 seconds
    
    def test_memory_usage_validation(self):
        """Test memory usage during large-scale evaluation."""
        import sys
        
        judge = MockJudge(score=4)
        cost_estimator = MockCostEstimator()
        evaluator = Evaluator(judge, cost_estimator)
        
        # Create large dataset
        large_test_list = [f"test_{i}" for i in range(1000)]
        
        # Monitor memory usage
        initial_size = sys.getsizeof(evaluator)
        
        # Perform evaluation
        result = evaluator.evaluate("flu", "flu", large_test_list, visits=1)
        
        final_size = sys.getsizeof(evaluator)
        
        # Evaluator shouldn't grow significantly in memory
        assert final_size - initial_size < 10000  # Less than 10KB growth
        assert result.total_cost > 0  # Should still produce valid results


# Pytest fixtures for test data
@pytest.fixture
def sample_evaluation_results():
    """Provide sample evaluation results for testing."""
    return [
        SessionResult(total_cost=300.0, score=5, correct=True, duration=25.0),
        SessionResult(total_cost=450.0, score=4, correct=True, duration=35.0),
        SessionResult(total_cost=200.0, score=3, correct=False, duration=20.0),
        SessionResult(total_cost=600.0, score=4, correct=True, duration=45.0),
        SessionResult(total_cost=350.0, score=2, correct=False, duration=30.0),
    ]


@pytest.fixture
def sample_diagnosis_results():
    """Provide sample diagnosis results for ensemble testing."""
    return [
        DiagnosisResult("pneumonia", confidence=0.85, cost=500.0, run_id="expert_1"),
        DiagnosisResult("bronchitis", confidence=0.75, cost=300.0, run_id="expert_2"),
        DiagnosisResult("pneumonia", confidence=0.90, cost=600.0, run_id="expert_3"),
        DiagnosisResult("flu", confidence=0.60, cost=200.0, run_id="expert_4"),
    ]


if __name__ == "__main__":
    pytest.main([__file__])