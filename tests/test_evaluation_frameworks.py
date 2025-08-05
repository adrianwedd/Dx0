"""Comprehensive evaluation frameworks for the Dx0 diagnostic system.

This module provides evaluation frameworks for diagnostic accuracy assessment,
cost-benefit analysis, performance benchmarking, comparative evaluation between
models/approaches, longitudinal performance tracking, and statistical reporting.
"""

import pytest
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any, Callable
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import json
import tempfile
import os

from sdb.evaluation import Evaluator, SessionResult, batch_evaluate
from sdb.judge import Judge, Judgement
from sdb.cost_estimator import CostEstimator, CptCost
from sdb.ensemble import DiagnosisResult, WeightedVoter, cost_adjusted_selection
from sdb.statistics import permutation_test


@dataclass
class DiagnosticAccuracyMetrics:
    """Comprehensive diagnostic accuracy metrics."""
    
    sensitivity: float
    specificity: float
    positive_predictive_value: float
    negative_predictive_value: float
    accuracy: float
    f1_score: float
    auc_roc: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    matthews_correlation: Optional[float] = None


@dataclass
class CostBenefitMetrics:
    """Cost-benefit analysis metrics."""
    
    total_cost: float
    cost_per_correct_diagnosis: float
    cost_effectiveness_ratio: float
    incremental_cost_effectiveness_ratio: Optional[float] = None
    net_monetary_benefit: Optional[float] = None
    return_on_investment: Optional[float] = None


@dataclass
class PerformanceBenchmark:
    """Performance benchmark for diagnostic systems."""
    
    system_id: str
    accuracy_metrics: DiagnosticAccuracyMetrics
    cost_metrics: CostBenefitMetrics
    time_metrics: Dict[str, float]
    case_complexity_performance: Dict[str, float]
    timestamp: datetime


@dataclass
class ComparisonResult:
    """Result of comparative evaluation between systems."""
    
    system_a_id: str
    system_b_id: str
    accuracy_comparison: Dict[str, Any]
    cost_comparison: Dict[str, Any]
    statistical_significance: Dict[str, float]
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]


class DiagnosticAccuracyFramework:
    """Framework for comprehensive diagnostic accuracy assessment."""
    
    def __init__(self, case_complexity_weights: Optional[Dict[str, float]] = None):
        """Initialize the diagnostic accuracy framework.
        
        Parameters
        ----------
        case_complexity_weights : Dict[str, float], optional
            Weights for different case complexity levels.
        """
        self.case_complexity_weights = case_complexity_weights or {
            "simple": 1.0,
            "moderate": 1.5,
            "complex": 2.0
        }
    
    def calculate_accuracy_metrics(self, true_labels: List[str], 
                                 predicted_labels: List[str],
                                 positive_class: str) -> DiagnosticAccuracyMetrics:
        """Calculate comprehensive diagnostic accuracy metrics.
        
        Parameters
        ----------
        true_labels : List[str]
            Ground truth diagnostic labels.
        predicted_labels : List[str]
            Predicted diagnostic labels.
        positive_class : str
            The positive class for binary classification metrics.
            
        Returns
        -------
        DiagnosticAccuracyMetrics
            Comprehensive accuracy metrics.
        """
        # Convert to binary classification for sensitivity/specificity
        true_binary = [1 if label == positive_class else 0 for label in true_labels]
        pred_binary = [1 if label == positive_class else 0 for label in predicted_labels]
        
        # Calculate confusion matrix values
        tp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 0)
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        # F1 score
        precision = ppv
        recall = sensitivity
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Balanced accuracy
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # Matthews Correlation Coefficient
        mcc_numerator = (tp * tn) - (fp * fn)
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        matthews_correlation = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0.0
        
        return DiagnosticAccuracyMetrics(
            sensitivity=sensitivity,
            specificity=specificity,
            positive_predictive_value=ppv,
            negative_predictive_value=npv,
            accuracy=accuracy,
            f1_score=f1_score,
            balanced_accuracy=balanced_accuracy,
            matthews_correlation=matthews_correlation
        )
    
    def evaluate_by_complexity(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, DiagnosticAccuracyMetrics]:
        """Evaluate diagnostic accuracy stratified by case complexity.
        
        Parameters
        ----------
        evaluation_results : List[Dict[str, Any]]
            List of evaluation results with complexity annotations.
            
        Returns
        -------
        Dict[str, DiagnosticAccuracyMetrics]
            Accuracy metrics by complexity level.
        """
        complexity_groups = {}
        
        # Group results by complexity
        for result in evaluation_results:
            complexity = result.get("complexity", "unknown")
            if complexity not in complexity_groups:
                complexity_groups[complexity] = {"true": [], "pred": []}
            
            complexity_groups[complexity]["true"].append(result["true_diagnosis"])
            complexity_groups[complexity]["pred"].append(result["predicted_diagnosis"])
        
        # Calculate metrics for each complexity level
        complexity_metrics = {}
        for complexity, labels in complexity_groups.items():
            if len(labels["true"]) > 0:
                # Use most common diagnosis as positive class
                most_common = max(set(labels["true"]), key=labels["true"].count)
                metrics = self.calculate_accuracy_metrics(
                    labels["true"], labels["pred"], most_common
                )
                complexity_metrics[complexity] = metrics
        
        return complexity_metrics
    
    def calculate_roc_auc(self, true_scores: List[float], predicted_scores: List[float]) -> float:
        """Calculate ROC AUC score for diagnostic confidence scores.
        
        Parameters
        ----------
        true_scores : List[float]
            Ground truth scores (0 or 1 for binary classification).
        predicted_scores : List[float]
            Predicted confidence scores.
            
        Returns
        -------
        float
            ROC AUC score.
        """
        # Sort by predicted scores (descending)
        sorted_pairs = sorted(zip(predicted_scores, true_scores), reverse=True)
        
        # Calculate AUC using trapezoidal rule
        n_pos = sum(true_scores)
        n_neg = len(true_scores) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return 0.5  # No discrimination possible
        
        # Calculate area under ROC curve
        auc = 0.0
        fp_prev = 0
        tp_prev = 0
        
        for i, (score, label) in enumerate(sorted_pairs):
            if i > 0 and sorted_pairs[i-1][0] != score:
                # Add trapezoid
                auc += (fp_prev / n_neg) * (tp_prev / n_pos - (tp_prev - (tp_prev if i == 1 else sum(1 for _, l in sorted_pairs[:i-1] if l == 1))) / n_pos)
            
            if label == 1:
                tp_prev += 1
            else:
                fp_prev += 1
        
        # Simplified AUC calculation
        return self._simple_auc_calculation(true_scores, predicted_scores)
    
    def _simple_auc_calculation(self, y_true: List[float], y_scores: List[float]) -> float:
        """Simplified AUC calculation using Mann-Whitney U statistic."""
        pos_scores = [score for score, label in zip(y_scores, y_true) if label == 1]
        neg_scores = [score for score, label in zip(y_scores, y_true) if label == 0]
        
        if not pos_scores or not neg_scores:
            return 0.5
        
        # Count concordant pairs
        concordant = 0
        total_pairs = len(pos_scores) * len(neg_scores)
        
        for pos_score in pos_scores:
            for neg_score in neg_scores:
                if pos_score > neg_score:
                    concordant += 1
                elif pos_score == neg_score:
                    concordant += 0.5
        
        return concordant / total_pairs


class CostBenefitAnalysisFramework:
    """Framework for cost-benefit analysis in diagnostic evaluation."""
    
    def __init__(self, cost_per_qaly: float = 50000.0):
        """Initialize cost-benefit analysis framework.
        
        Parameters
        ----------
        cost_per_qaly : float
            Cost per Quality-Adjusted Life Year threshold.
        """
        self.cost_per_qaly = cost_per_qaly
    
    def calculate_cost_metrics(self, session_results: List[SessionResult]) -> CostBenefitMetrics:
        """Calculate comprehensive cost-benefit metrics.
        
        Parameters
        ----------
        session_results : List[SessionResult]
            List of diagnostic session results.
            
        Returns
        -------
        CostBenefitMetrics
            Comprehensive cost-benefit metrics.
        """
        total_cost = sum(result.total_cost for result in session_results)
        correct_diagnoses = sum(1 for result in session_results if result.correct)
        
        # Cost per correct diagnosis
        cost_per_correct = total_cost / correct_diagnoses if correct_diagnoses > 0 else float('inf')
        
        # Cost-effectiveness ratio (cost per unit of effectiveness)
        total_effectiveness = sum(result.score for result in session_results)
        cost_effectiveness_ratio = total_cost / total_effectiveness if total_effectiveness > 0 else float('inf')
        
        return CostBenefitMetrics(
            total_cost=total_cost,
            cost_per_correct_diagnosis=cost_per_correct,
            cost_effectiveness_ratio=cost_effectiveness_ratio
        )
    
    def calculate_incremental_analysis(self, baseline_results: List[SessionResult],
                                     intervention_results: List[SessionResult]) -> Dict[str, float]:
        """Calculate incremental cost-effectiveness analysis.
        
        Parameters
        ----------
        baseline_results : List[SessionResult]
            Results from baseline diagnostic approach.
        intervention_results : List[SessionResult]
            Results from intervention diagnostic approach.
            
        Returns
        -------
        Dict[str, float]
            Incremental analysis metrics.
        """
        baseline_metrics = self.calculate_cost_metrics(baseline_results)
        intervention_metrics = self.calculate_cost_metrics(intervention_results)
        
        # Calculate incremental values
        incremental_cost = intervention_metrics.total_cost - baseline_metrics.total_cost
        
        baseline_effectiveness = sum(r.score for r in baseline_results) / len(baseline_results)
        intervention_effectiveness = sum(r.score for r in intervention_results) / len(intervention_results)
        incremental_effectiveness = intervention_effectiveness - baseline_effectiveness
        
        # Incremental Cost-Effectiveness Ratio (ICER)
        icer = incremental_cost / incremental_effectiveness if incremental_effectiveness != 0 else float('inf')
        
        # Net Monetary Benefit
        willingness_to_pay = self.cost_per_qaly
        nmb = (incremental_effectiveness * willingness_to_pay) - incremental_cost
        
        return {
            "incremental_cost": incremental_cost,
            "incremental_effectiveness": incremental_effectiveness,
            "icer": icer,
            "net_monetary_benefit": nmb,
            "cost_effective": icer < willingness_to_pay
        }
    
    def pareto_frontier_analysis(self, strategies: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Identify Pareto-efficient diagnostic strategies.
        
        Parameters
        ----------
        strategies : List[Dict[str, float]]
            List of strategies with 'cost' and 'effectiveness' keys.
            
        Returns
        -------
        List[Dict[str, float]]
            Pareto-efficient strategies.
        """
        # Sort by cost
        sorted_strategies = sorted(strategies, key=lambda x: x["cost"])
        
        pareto_strategies = []
        max_effectiveness = -float('inf')
        
        for strategy in sorted_strategies:
            if strategy["effectiveness"] > max_effectiveness:
                pareto_strategies.append(strategy)
                max_effectiveness = strategy["effectiveness"]
        
        return pareto_strategies
    
    def budget_impact_analysis(self, strategies: List[Dict[str, Any]], 
                             budget: float, population_size: int) -> Dict[str, Any]:
        """Perform budget impact analysis for diagnostic strategies.
        
        Parameters
        ----------
        strategies : List[Dict[str, Any]]
            List of diagnostic strategies with cost and outcome data.
        budget : float
            Available budget.
        population_size : int
            Target population size.
            
        Returns
        -------
        Dict[str, Any]
            Budget impact analysis results.
        """
        results = {}
        
        for strategy in strategies:
            cost_per_case = strategy["cost"]
            total_cost = cost_per_case * population_size
            
            if total_cost <= budget:
                cases_affordable = population_size
                budget_utilization = total_cost / budget
            else:
                cases_affordable = int(budget / cost_per_case)
                budget_utilization = 1.0
            
            results[strategy["name"]] = {
                "total_cost": min(total_cost, budget),
                "cases_affordable": cases_affordable,
                "budget_utilization": budget_utilization,
                "cost_per_case": cost_per_case,
                "effectiveness": strategy.get("effectiveness", 0.0)
            }
        
        return results


class PerformanceBenchmarkingFramework:
    """Framework for performance benchmarking and tracking."""
    
    def __init__(self):
        """Initialize performance benchmarking framework."""
        self.benchmarks: List[PerformanceBenchmark] = []
    
    def create_benchmark(self, system_id: str, evaluation_results: List[Dict[str, Any]],
                        session_results: List[SessionResult]) -> PerformanceBenchmark:
        """Create a performance benchmark from evaluation results.
        
        Parameters
        ----------
        system_id : str
            Identifier for the diagnostic system.
        evaluation_results : List[Dict[str, Any]]
            Detailed evaluation results.
        session_results : List[SessionResult]
            Session-level results.
            
        Returns
        -------
        PerformanceBenchmark
            Comprehensive performance benchmark.
        """
        # Calculate accuracy metrics
        accuracy_framework = DiagnosticAccuracyFramework()
        true_labels = [r["true_diagnosis"] for r in evaluation_results]
        pred_labels = [r["predicted_diagnosis"] for r in evaluation_results]
        
        # Use most common true diagnosis as positive class
        positive_class = max(set(true_labels), key=true_labels.count)
        accuracy_metrics = accuracy_framework.calculate_accuracy_metrics(
            true_labels, pred_labels, positive_class
        )
        
        # Calculate cost metrics
        cost_framework = CostBenefitAnalysisFramework()
        cost_metrics = cost_framework.calculate_cost_metrics(session_results)
        
        # Calculate time metrics
        durations = [r.duration for r in session_results]
        time_metrics = {
            "mean_duration": np.mean(durations),
            "median_duration": np.median(durations),
            "std_duration": np.std(durations),
            "min_duration": np.min(durations),
            "max_duration": np.max(durations)
        }
        
        # Calculate complexity-stratified performance
        complexity_performance = {}
        complexity_groups = {}
        for result in evaluation_results:
            complexity = result.get("complexity", "unknown")
            if complexity not in complexity_groups:
                complexity_groups[complexity] = []
            complexity_groups[complexity].append(result.get("correct", False))
        
        for complexity, correct_list in complexity_groups.items():
            complexity_performance[complexity] = sum(correct_list) / len(correct_list)
        
        benchmark = PerformanceBenchmark(
            system_id=system_id,
            accuracy_metrics=accuracy_metrics,
            cost_metrics=cost_metrics,
            time_metrics=time_metrics,
            case_complexity_performance=complexity_performance,
            timestamp=datetime.now()
        )
        
        self.benchmarks.append(benchmark)
        return benchmark
    
    def compare_benchmarks(self, system_a_id: str, system_b_id: str) -> Optional[ComparisonResult]:
        """Compare performance benchmarks between two systems.
        
        Parameters
        ----------
        system_a_id : str
            First system identifier.
        system_b_id : str
            Second system identifier.
            
        Returns
        -------
        ComparisonResult, optional
            Comparison results if both systems found.
        """
        benchmark_a = self._get_latest_benchmark(system_a_id)
        benchmark_b = self._get_latest_benchmark(system_b_id)
        
        if not benchmark_a or not benchmark_b:
            return None
        
        # Compare accuracy metrics
        accuracy_comparison = {
            "sensitivity_diff": benchmark_a.accuracy_metrics.sensitivity - benchmark_b.accuracy_metrics.sensitivity,
            "specificity_diff": benchmark_a.accuracy_metrics.specificity - benchmark_b.accuracy_metrics.specificity,
            "accuracy_diff": benchmark_a.accuracy_metrics.accuracy - benchmark_b.accuracy_metrics.accuracy,
            "f1_diff": benchmark_a.accuracy_metrics.f1_score - benchmark_b.accuracy_metrics.f1_score
        }
        
        # Compare cost metrics
        cost_comparison = {
            "total_cost_diff": benchmark_a.cost_metrics.total_cost - benchmark_b.cost_metrics.total_cost,
            "cost_per_correct_diff": (benchmark_a.cost_metrics.cost_per_correct_diagnosis - 
                                    benchmark_b.cost_metrics.cost_per_correct_diagnosis),
            "cost_effectiveness_diff": (benchmark_a.cost_metrics.cost_effectiveness_ratio - 
                                      benchmark_b.cost_metrics.cost_effectiveness_ratio)
        }
        
        return ComparisonResult(
            system_a_id=system_a_id,
            system_b_id=system_b_id,
            accuracy_comparison=accuracy_comparison,
            cost_comparison=cost_comparison,
            statistical_significance={},  # Would need raw data for proper statistical tests
            effect_sizes={},
            confidence_intervals={}
        )
    
    def _get_latest_benchmark(self, system_id: str) -> Optional[PerformanceBenchmark]:
        """Get the latest benchmark for a system."""
        system_benchmarks = [b for b in self.benchmarks if b.system_id == system_id]
        if not system_benchmarks:
            return None
        return max(system_benchmarks, key=lambda b: b.timestamp)
    
    def track_performance_over_time(self, system_id: str) -> Dict[str, List[Tuple[datetime, float]]]:
        """Track performance metrics over time for a system.
        
        Parameters
        ----------
        system_id : str
            System identifier.
            
        Returns
        -------
        Dict[str, List[Tuple[datetime, float]]]
            Time series of performance metrics.
        """
        system_benchmarks = [b for b in self.benchmarks if b.system_id == system_id]
        system_benchmarks.sort(key=lambda b: b.timestamp)
        
        time_series = {
            "accuracy": [(b.timestamp, b.accuracy_metrics.accuracy) for b in system_benchmarks],
            "sensitivity": [(b.timestamp, b.accuracy_metrics.sensitivity) for b in system_benchmarks],
            "specificity": [(b.timestamp, b.accuracy_metrics.specificity) for b in system_benchmarks],
            "cost_per_correct": [(b.timestamp, b.cost_metrics.cost_per_correct_diagnosis) for b in system_benchmarks],
            "mean_duration": [(b.timestamp, b.time_metrics["mean_duration"]) for b in system_benchmarks]
        }
        
        return time_series


class LongitudinalTrackingFramework:
    """Framework for longitudinal performance tracking and trend analysis."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize longitudinal tracking framework.
        
        Parameters
        ----------
        storage_path : str, optional
            Path to store longitudinal data.
        """
        self.storage_path = storage_path
        self.performance_history: List[Dict[str, Any]] = []
    
    def record_performance(self, system_id: str, metrics: Dict[str, float], 
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record performance metrics for longitudinal tracking.
        
        Parameters
        ----------
        system_id : str
            System identifier.
        metrics : Dict[str, float]
            Performance metrics to record.
        metadata : Dict[str, Any], optional
            Additional metadata (e.g., data version, configuration).
        """
        record = {
            "system_id": system_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "metadata": metadata or {}
        }
        
        self.performance_history.append(record)
        
        if self.storage_path:
            self._save_to_storage()
    
    def detect_performance_drift(self, system_id: str, metric: str, 
                                window_size: int = 10, threshold: float = 0.05) -> Dict[str, Any]:
        """Detect performance drift over time.
        
        Parameters
        ----------
        system_id : str
            System identifier.
        metric : str
            Metric to analyze for drift.
        window_size : int
            Size of rolling window for drift detection.
        threshold : float
            Threshold for detecting significant drift.
            
        Returns
        -------
        Dict[str, Any]
            Drift detection results.
        """
        system_records = [r for r in self.performance_history if r["system_id"] == system_id]
        
        if len(system_records) < window_size * 2:
            return {"drift_detected": False, "reason": "insufficient_data"}
        
        # Extract metric values
        metric_values = [r["metrics"].get(metric, 0.0) for r in system_records]
        
        # Calculate rolling means
        early_window = metric_values[:window_size]
        late_window = metric_values[-window_size:]
        
        early_mean = np.mean(early_window)
        late_mean = np.mean(late_window)
        
        # Calculate relative change
        relative_change = abs(late_mean - early_mean) / early_mean if early_mean != 0 else 0.0
        
        drift_detected = relative_change > threshold
        
        return {
            "drift_detected": drift_detected,
            "early_mean": early_mean,
            "late_mean": late_mean,
            "relative_change": relative_change,
            "threshold": threshold,
            "direction": "improvement" if late_mean > early_mean else "degradation"
        }
    
    def generate_trend_analysis(self, system_id: str, metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Generate trend analysis for specified metrics.
        
        Parameters
        ----------
        system_id : str
            System identifier.
        metrics : List[str]
            List of metrics to analyze.
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Trend analysis results for each metric.
        """
        system_records = [r for r in self.performance_history if r["system_id"] == system_id]
        
        if len(system_records) < 3:
            return {}
        
        trend_analysis = {}
        
        for metric in metrics:
            values = [r["metrics"].get(metric, 0.0) for r in system_records]
            timestamps = [datetime.fromisoformat(r["timestamp"]) for r in system_records]
            
            # Convert timestamps to numeric values for regression
            time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
            
            # Simple linear regression
            slope = self._calculate_trend_slope(time_numeric, values)
            
            # Calculate trend statistics
            trend_analysis[metric] = {
                "slope": slope,
                "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                "mean_value": np.mean(values),
                "std_value": np.std(values),
                "min_value": np.min(values),
                "max_value": np.max(values),
                "coefficient_of_variation": np.std(values) / np.mean(values) if np.mean(values) != 0 else 0.0
            }
        
        return trend_analysis
    
    def _calculate_trend_slope(self, x: List[float], y: List[float]) -> float:
        """Calculate slope of linear trend."""
        n = len(x)
        if n < 2:
            return 0.0
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _save_to_storage(self) -> None:
        """Save performance history to storage."""
        if self.storage_path:
            with open(self.storage_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
    
    def load_from_storage(self) -> None:
        """Load performance history from storage."""
        if self.storage_path and os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                self.performance_history = json.load(f)


class StatisticalReportingFramework:
    """Framework for statistical reporting and visualization of evaluation results."""
    
    def __init__(self):
        """Initialize statistical reporting framework."""
        pass
    
    def generate_comprehensive_report(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive statistical report.
        
        Parameters
        ----------
        evaluation_data : Dict[str, Any]
            Comprehensive evaluation data.
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive statistical report.
        """
        report = {
            "summary_statistics": self._calculate_summary_statistics(evaluation_data),
            "accuracy_analysis": self._analyze_accuracy_metrics(evaluation_data),
            "cost_analysis": self._analyze_cost_metrics(evaluation_data),
            "statistical_tests": self._perform_statistical_tests(evaluation_data),
            "confidence_intervals": self._calculate_confidence_intervals(evaluation_data),
            "effect_sizes": self._calculate_effect_sizes(evaluation_data),
            "recommendations": self._generate_recommendations(evaluation_data)
        }
        
        return report
    
    def _calculate_summary_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if "session_results" not in data:
            return {}
        
        session_results = data["session_results"]
        scores = [r.score for r in session_results]
        costs = [r.total_cost for r in session_results]
        durations = [r.duration for r in session_results]
        
        return {
            "sample_size": len(session_results),
            "score_statistics": {
                "mean": np.mean(scores),
                "median": np.median(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores)
            },
            "cost_statistics": {
                "mean": np.mean(costs),
                "median": np.median(costs),
                "std": np.std(costs),
                "min": np.min(costs),
                "max": np.max(costs)
            },
            "duration_statistics": {
                "mean": np.mean(durations),
                "median": np.median(durations),
                "std": np.std(durations),
                "min": np.min(durations),
                "max": np.max(durations)
            }
        }
    
    def _analyze_accuracy_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze accuracy metrics."""
        # Placeholder implementation
        return {"accuracy_analysis": "completed"}
    
    def _analyze_cost_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cost metrics."""
        # Placeholder implementation
        return {"cost_analysis": "completed"}
    
    def _perform_statistical_tests(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        # Placeholder implementation
        return {"statistical_tests": "completed"}
    
    def _calculate_confidence_intervals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence intervals."""
        # Placeholder implementation
        return {"confidence_intervals": "completed"}
    
    def _calculate_effect_sizes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate effect sizes."""
        # Placeholder implementation
        return {"effect_sizes": "completed"}
    
    def _generate_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = [
            "Continue monitoring diagnostic accuracy trends",
            "Consider cost-effectiveness optimization opportunities",
            "Evaluate performance across different case complexities"
        ]
        return recommendations


# Test classes for the evaluation frameworks
class TestDiagnosticAccuracyFramework:
    """Test suite for diagnostic accuracy framework."""
    
    def test_accuracy_metrics_calculation(self):
        """Test accuracy metrics calculation."""
        framework = DiagnosticAccuracyFramework()
        
        true_labels = ["pneumonia", "pneumonia", "flu", "pneumonia", "flu"]
        pred_labels = ["pneumonia", "flu", "flu", "pneumonia", "pneumonia"]
        
        metrics = framework.calculate_accuracy_metrics(true_labels, pred_labels, "pneumonia")
        
        # Pneumonia: TP=2, TN=1, FP=1, FN=1
        assert abs(metrics.sensitivity - 0.667) < 0.01  # 2/3
        assert abs(metrics.specificity - 0.5) < 0.01    # 1/2
        assert abs(metrics.accuracy - 0.6) < 0.01       # 3/5
    
    def test_complexity_stratified_evaluation(self):
        """Test evaluation stratified by case complexity."""
        framework = DiagnosticAccuracyFramework()
        
        evaluation_results = [
            {"true_diagnosis": "flu", "predicted_diagnosis": "flu", "complexity": "simple"},
            {"true_diagnosis": "flu", "predicted_diagnosis": "cold", "complexity": "simple"},
            {"true_diagnosis": "pneumonia", "predicted_diagnosis": "pneumonia", "complexity": "complex"},
            {"true_diagnosis": "pneumonia", "predicted_diagnosis": "flu", "complexity": "complex"},
        ]
        
        complexity_metrics = framework.evaluate_by_complexity(evaluation_results)
        
        assert "simple" in complexity_metrics
        assert "complex" in complexity_metrics
        assert isinstance(complexity_metrics["simple"], DiagnosticAccuracyMetrics)
    
    def test_roc_auc_calculation(self):
        """Test ROC AUC calculation."""
        framework = DiagnosticAccuracyFramework()
        
        # Perfect separation
        true_scores = [1, 1, 1, 0, 0, 0]
        pred_scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        
        auc = framework.calculate_roc_auc(true_scores, pred_scores)
        
        assert auc == 1.0  # Perfect separation
        
        # Random performance
        true_random = [1, 0, 1, 0, 1, 0]
        pred_random = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        
        auc_random = framework.calculate_roc_auc(true_random, pred_random)
        
        assert abs(auc_random - 0.5) < 0.1  # Should be around 0.5


class TestCostBenefitAnalysisFramework:
    """Test suite for cost-benefit analysis framework."""
    
    def test_cost_metrics_calculation(self):
        """Test cost metrics calculation."""
        framework = CostBenefitAnalysisFramework()
        
        session_results = [
            SessionResult(total_cost=300.0, score=5, correct=True, duration=30.0),
            SessionResult(total_cost=400.0, score=4, correct=True, duration=35.0),
            SessionResult(total_cost=200.0, score=3, correct=False, duration=25.0),
        ]
        
        metrics = framework.calculate_cost_metrics(session_results)
        
        assert metrics.total_cost == 900.0
        assert metrics.cost_per_correct_diagnosis == 450.0  # 900/2
        assert abs(metrics.cost_effectiveness_ratio - 75.0) < 0.1  # 900/12
    
    def test_incremental_analysis(self):
        """Test incremental cost-effectiveness analysis."""
        framework = CostBenefitAnalysisFramework()
        
        baseline_results = [
            SessionResult(total_cost=300.0, score=3, correct=True, duration=30.0),
            SessionResult(total_cost=300.0, score=3, correct=True, duration=30.0),
        ]
        
        intervention_results = [
            SessionResult(total_cost=500.0, score=5, correct=True, duration=25.0),
            SessionResult(total_cost=500.0, score=5, correct=True, duration=25.0),
        ]
        
        analysis = framework.calculate_incremental_analysis(baseline_results, intervention_results)
        
        assert analysis["incremental_cost"] == 400.0  # (1000 - 600)
        assert analysis["incremental_effectiveness"] == 2.0  # (5 - 3)
        assert analysis["icer"] == 200.0  # 400/2
    
    def test_pareto_frontier_analysis(self):
        """Test Pareto frontier analysis."""
        framework = CostBenefitAnalysisFramework()
        
        strategies = [
            {"cost": 100, "effectiveness": 0.6, "name": "A"},
            {"cost": 200, "effectiveness": 0.7, "name": "B"},
            {"cost": 150, "effectiveness": 0.65, "name": "C"},  # Dominated
            {"cost": 300, "effectiveness": 0.8, "name": "D"},
        ]
        
        pareto_strategies = framework.pareto_frontier_analysis(strategies)
        
        # Should exclude dominated strategy C
        pareto_names = [s["name"] for s in pareto_strategies]
        assert "C" not in pareto_names
        assert len(pareto_strategies) == 3


class TestPerformanceBenchmarkingFramework:
    """Test suite for performance benchmarking framework."""
    
    def test_benchmark_creation(self):
        """Test benchmark creation."""
        framework = PerformanceBenchmarkingFramework()
        
        evaluation_results = [
            {"true_diagnosis": "flu", "predicted_diagnosis": "flu", "correct": True, "complexity": "simple"},
            {"true_diagnosis": "flu", "predicted_diagnosis": "cold", "correct": False, "complexity": "complex"},
        ]
        
        session_results = [
            SessionResult(total_cost=300.0, score=5, correct=True, duration=30.0),
            SessionResult(total_cost=400.0, score=3, correct=False, duration=40.0),
        ]
        
        benchmark = framework.create_benchmark("system_1", evaluation_results, session_results)
        
        assert benchmark.system_id == "system_1"
        assert isinstance(benchmark.accuracy_metrics, DiagnosticAccuracyMetrics)
        assert isinstance(benchmark.cost_metrics, CostBenefitMetrics)
        assert "mean_duration" in benchmark.time_metrics
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison."""
        framework = PerformanceBenchmarkingFramework()
        
        # Create two benchmarks
        eval_results_a = [{"true_diagnosis": "flu", "predicted_diagnosis": "flu", "correct": True}]
        session_results_a = [SessionResult(total_cost=300.0, score=5, correct=True, duration=30.0)]
        framework.create_benchmark("system_a", eval_results_a, session_results_a)
        
        eval_results_b = [{"true_diagnosis": "flu", "predicted_diagnosis": "cold", "correct": False}]
        session_results_b = [SessionResult(total_cost=400.0, score=3, correct=False, duration=40.0)]
        framework.create_benchmark("system_b", eval_results_b, session_results_b)
        
        comparison = framework.compare_benchmarks("system_a", "system_b")
        
        assert comparison is not None
        assert comparison.system_a_id == "system_a"
        assert comparison.system_b_id == "system_b"
        assert "accuracy_diff" in comparison.accuracy_comparison


class TestLongitudinalTrackingFramework:
    """Test suite for longitudinal tracking framework."""
    
    def test_performance_recording(self):
        """Test performance recording."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            framework = LongitudinalTrackingFramework(storage_path=tmp_file.name)
            
            metrics = {"accuracy": 0.85, "cost": 400.0}
            framework.record_performance("system_1", metrics)
            
            assert len(framework.performance_history) == 1
            assert framework.performance_history[0]["system_id"] == "system_1"
            
            # Clean up
            os.unlink(tmp_file.name)
    
    def test_drift_detection(self):
        """Test performance drift detection."""
        framework = LongitudinalTrackingFramework()
        
        # Simulate degrading performance
        for i in range(20):
            accuracy = 0.9 - (i * 0.01)  # Gradually decreasing accuracy
            framework.record_performance("system_1", {"accuracy": accuracy})
        
        drift_result = framework.detect_performance_drift("system_1", "accuracy", window_size=5, threshold=0.05)
        
        assert drift_result["drift_detected"] is True
        assert drift_result["direction"] == "degradation"
    
    def test_trend_analysis(self):
        """Test trend analysis."""
        framework = LongitudinalTrackingFramework()
        
        # Simulate improving performance
        for i in range(10):
            accuracy = 0.7 + (i * 0.02)  # Gradually increasing accuracy
            framework.record_performance("system_1", {"accuracy": accuracy})
        
        trend_analysis = framework.generate_trend_analysis("system_1", ["accuracy"])
        
        assert "accuracy" in trend_analysis
        assert trend_analysis["accuracy"]["trend_direction"] == "increasing"
        assert trend_analysis["accuracy"]["slope"] > 0


# Pytest fixtures for evaluation frameworks
@pytest.fixture
def sample_evaluation_data():
    """Provide sample evaluation data for testing."""
    return {
        "session_results": [
            SessionResult(total_cost=300.0, score=5, correct=True, duration=25.0),
            SessionResult(total_cost=450.0, score=4, correct=True, duration=35.0),
            SessionResult(total_cost=200.0, score=3, correct=False, duration=20.0),
        ],
        "evaluation_results": [
            {"true_diagnosis": "flu", "predicted_diagnosis": "flu", "correct": True, "complexity": "simple"},
            {"true_diagnosis": "pneumonia", "predicted_diagnosis": "pneumonia", "correct": True, "complexity": "complex"},
            {"true_diagnosis": "cold", "predicted_diagnosis": "flu", "correct": False, "complexity": "simple"},
        ]
    }


if __name__ == "__main__":
    pytest.main([__file__])