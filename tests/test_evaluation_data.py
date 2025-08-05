"""Test data generation and validation for the Dx0 evaluation system.

This module provides comprehensive test data generation for evaluation testing,
including representative evaluation datasets with ground truth, statistical
validation datasets with known properties, performance benchmarking scenarios,
edge cases for statistical computations, and mock evaluation results.
"""

import pytest
import numpy as np
import pandas as pd
import json
import csv
import tempfile
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from unittest.mock import Mock
import random

from sdb.evaluation import SessionResult
from sdb.judge import Judgement
from sdb.ensemble import DiagnosisResult
from sdb.cost_estimator import CptCost


@dataclass
class TestCase:
    """Comprehensive test case for diagnostic evaluation."""
    
    case_id: str
    patient_age: int
    patient_gender: str
    chief_complaint: str
    history: str
    physical_exam: str
    true_diagnosis: str
    differential_diagnoses: List[str]
    recommended_tests: List[str]
    case_complexity: str  # simple, moderate, complex
    specialty: str
    severity: str  # mild, moderate, severe
    urgency: str  # routine, urgent, emergent


@dataclass
class EvaluationDataset:
    """Complete evaluation dataset with metadata."""
    
    name: str
    description: str
    test_cases: List[TestCase]
    ground_truth_labels: List[str]
    complexity_distribution: Dict[str, int]
    specialty_distribution: Dict[str, int]
    created_at: datetime
    version: str


class TestDataGenerator:
    """Generator for comprehensive test data for evaluation testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize test data generator with random seed."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_diagnostic_cases(self, n_cases: int = 100) -> List[TestCase]:
        """Generate realistic diagnostic test cases.
        
        Parameters
        ----------
        n_cases : int
            Number of test cases to generate.
            
        Returns
        -------
        List[TestCase]
            Generated diagnostic test cases.
        """
        # Common diagnoses with frequencies
        diagnoses = [
            ("pneumonia", 0.15, "respiratory", ["chest_xray", "cbc", "blood_culture"]),
            ("myocardial_infarction", 0.10, "cardiac", ["ecg", "troponin", "chest_xray"]),
            ("diabetes_mellitus", 0.12, "endocrine", ["glucose", "hba1c", "urinalysis"]),
            ("hypertension", 0.08, "cardiovascular", ["bp_monitoring", "ecg", "bmp"]),
            ("influenza", 0.10, "infectious", ["rapid_flu_test", "cbc"]),
            ("pneumothorax", 0.05, "pulmonary", ["chest_xray", "ct_chest"]),
            ("appendicitis", 0.07, "surgical", ["ct_abdomen", "cbc", "urinalysis"]),
            ("stroke", 0.06, "neurological", ["ct_head", "mri_brain", "ecg"]),
            ("copd_exacerbation", 0.08, "pulmonary", ["chest_xray", "abg", "cbc"]),
            ("sepsis", 0.09, "infectious", ["blood_culture", "lactate", "cbc", "bmp"]),
            ("gastroenteritis", 0.10, "gastrointestinal", ["stool_culture", "bmp"])
        ]
        
        # Complexity distributions
        complexity_weights = {"simple": 0.4, "moderate": 0.4, "complex": 0.2}
        severity_weights = {"mild": 0.5, "moderate": 0.3, "severe": 0.2}
        urgency_weights = {"routine": 0.6, "urgent": 0.3, "emergent": 0.1}
        
        test_cases = []
        
        for i in range(n_cases):
            # Select diagnosis based on frequency
            frequencies = [freq for _, freq, _, _ in diagnoses]
            idx = np.random.choice(len(diagnoses), p=frequencies)
            diagnosis, _, specialty, tests = diagnoses[idx]
            
            # Generate case characteristics
            case = TestCase(
                case_id=f"case_{i:04d}",
                patient_age=np.random.randint(18, 90),
                patient_gender=np.random.choice(["M", "F"]),
                chief_complaint=self._generate_chief_complaint(diagnosis),
                history=self._generate_history(diagnosis),
                physical_exam=self._generate_physical_exam(diagnosis),
                true_diagnosis=diagnosis,
                differential_diagnoses=self._generate_differentials(diagnosis, specialty),
                recommended_tests=tests,
                case_complexity=np.random.choice(
                    list(complexity_weights.keys()),
                    p=list(complexity_weights.values())
                ),
                specialty=specialty,
                severity=np.random.choice(
                    list(severity_weights.keys()),
                    p=list(severity_weights.values())
                ),
                urgency=np.random.choice(
                    list(urgency_weights.keys()),
                    p=list(urgency_weights.values())
                )
            )
            
            test_cases.append(case)
        
        return test_cases
    
    def generate_evaluation_results(self, test_cases: List[TestCase],
                                  accuracy_by_complexity: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Generate evaluation results for test cases.
        
        Parameters
        ----------
        test_cases : List[TestCase]
            Test cases to generate results for.
        accuracy_by_complexity : Dict[str, float], optional
            Target accuracy by complexity level.
            
        Returns
        -------
        List[Dict[str, Any]]
            Generated evaluation results.
        """
        if accuracy_by_complexity is None:
            accuracy_by_complexity = {
                "simple": 0.90,
                "moderate": 0.75,
                "complex": 0.60
            }
        
        results = []
        
        for case in test_cases:
            target_accuracy = accuracy_by_complexity.get(case.case_complexity, 0.75)
            
            # Determine if prediction is correct based on target accuracy
            is_correct = np.random.random() < target_accuracy
            
            if is_correct:
                predicted_diagnosis = case.true_diagnosis
                confidence = np.random.uniform(0.8, 0.95)
                score = np.random.choice([4, 5], p=[0.3, 0.7])
            else:
                # Select incorrect diagnosis from differentials
                if case.differential_diagnoses:
                    predicted_diagnosis = np.random.choice(case.differential_diagnoses)
                else:
                    predicted_diagnosis = "unknown"
                confidence = np.random.uniform(0.4, 0.7)
                score = np.random.choice([1, 2, 3], p=[0.2, 0.3, 0.5])
            
            # Generate test utilization
            n_tests_ordered = min(
                len(case.recommended_tests),
                np.random.poisson(len(case.recommended_tests) * 0.8) + 1
            )
            tests_ordered = np.random.choice(
                case.recommended_tests, 
                size=n_tests_ordered, 
                replace=False
            ).tolist()
            
            # Generate timing and cost
            base_duration = 30.0  # minutes
            complexity_multiplier = {"simple": 1.0, "moderate": 1.5, "complex": 2.0}
            duration = base_duration * complexity_multiplier[case.case_complexity]
            duration += np.random.normal(0, duration * 0.2)  # Add noise
            duration = max(10.0, duration)  # Minimum 10 minutes
            
            base_cost = 300.0  # visit cost
            test_costs = {
                "cbc": 50, "bmp": 75, "chest_xray": 200, "ct_chest": 800,
                "ecg": 150, "troponin": 100, "glucose": 25, "hba1c": 60,
                "blood_culture": 120, "urinalysis": 30, "ct_abdomen": 900,
                "ct_head": 1000, "mri_brain": 1500, "abg": 80, "lactate": 40,
                "stool_culture": 90, "rapid_flu_test": 35, "bp_monitoring": 20
            }
            
            total_cost = base_cost + sum(test_costs.get(test, 100) for test in tests_ordered)
            
            # Add cost variation based on complexity and severity
            if case.severity == "severe":
                total_cost *= 1.3
            elif case.severity == "moderate":
                total_cost *= 1.1
            
            result = {
                "case_id": case.case_id,
                "true_diagnosis": case.true_diagnosis,
                "predicted_diagnosis": predicted_diagnosis,
                "confidence": confidence,
                "score": score,
                "correct": is_correct,
                "tests_ordered": tests_ordered,
                "total_cost": total_cost,
                "duration": duration,
                "complexity": case.case_complexity,
                "specialty": case.specialty,
                "severity": case.severity,
                "urgency": case.urgency,
                "patient_age": case.patient_age,
                "patient_gender": case.patient_gender
            }
            
            results.append(result)
        
        return results
    
    def generate_session_results(self, evaluation_results: List[Dict[str, Any]]) -> List[SessionResult]:
        """Convert evaluation results to SessionResult objects.
        
        Parameters
        ----------
        evaluation_results : List[Dict[str, Any]]
            Evaluation results to convert.
            
        Returns
        -------
        List[SessionResult]
            SessionResult objects.
        """
        session_results = []
        
        for result in evaluation_results:
            session_result = SessionResult(
                total_cost=result["total_cost"],
                score=result["score"],
                correct=result["correct"],
                duration=result["duration"]
            )
            session_results.append(session_result)
        
        return session_results
    
    def generate_statistical_validation_data(self, scenario: str) -> Dict[str, List[float]]:
        """Generate data with known statistical properties for validation.
        
        Parameters
        ----------
        scenario : str
            Statistical scenario to generate.
            
        Returns
        -------
        Dict[str, List[float]]
            Generated data with known properties.
        """
        np.random.seed(self.seed)
        
        if scenario == "normal_distribution":
            # Generate normal data with known parameters
            data = {
                "group_a": np.random.normal(loc=0.8, scale=0.1, size=100).tolist(),
                "group_b": np.random.normal(loc=0.75, scale=0.1, size=100).tolist(),
                "properties": {
                    "group_a_mean": 0.8,
                    "group_a_std": 0.1,
                    "group_b_mean": 0.75,
                    "group_b_std": 0.1,
                    "expected_effect_size": 0.5  # Cohen's d
                }
            }
            
        elif scenario == "significant_difference":
            # Generate data with large, detectable difference
            data = {
                "group_a": np.random.normal(loc=0.9, scale=0.05, size=50).tolist(),
                "group_b": np.random.normal(loc=0.7, scale=0.05, size=50).tolist(),
                "properties": {
                    "expected_p_value": "<0.001",
                    "expected_significant": True,
                    "effect_size": "large"
                }
            }
            
        elif scenario == "no_difference":
            # Generate data with no true difference
            common_mean = 0.8
            data = {
                "group_a": np.random.normal(loc=common_mean, scale=0.1, size=100).tolist(),
                "group_b": np.random.normal(loc=common_mean, scale=0.1, size=100).tolist(),
                "properties": {
                    "expected_p_value": ">0.05",
                    "expected_significant": False,
                    "true_difference": 0.0
                }
            }
            
        elif scenario == "skewed_distribution":
            # Generate skewed data
            data = {
                "group_a": np.random.exponential(scale=0.2, size=100).tolist(),
                "group_b": np.random.exponential(scale=0.3, size=100).tolist(),
                "properties": {
                    "distribution": "exponential",
                    "recommend_test": "mann_whitney_u"
                }
            }
            
        elif scenario == "small_sample":
            # Generate small sample data
            data = {
                "group_a": np.random.normal(loc=0.8, scale=0.1, size=10).tolist(),
                "group_b": np.random.normal(loc=0.75, scale=0.1, size=10).tolist(),
                "properties": {
                    "sample_size": "small",
                    "power": "low",
                    "use_exact_tests": True
                }
            }
            
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        return data
    
    def generate_performance_benchmark_data(self, scale: str = "medium") -> Dict[str, Any]:
        """Generate data for performance benchmarking.
        
        Parameters
        ----------
        scale : str
            Scale of benchmark data (small, medium, large, xl).
            
        Returns
        -------
        Dict[str, Any]
            Performance benchmark data.
        """
        scale_configs = {
            "small": {"n_cases": 100, "max_tests": 5, "n_systems": 3},
            "medium": {"n_cases": 1000, "max_tests": 10, "n_systems": 5},
            "large": {"n_cases": 10000, "max_tests": 20, "n_systems": 10},
            "xl": {"n_cases": 100000, "max_tests": 50, "n_systems": 20}
        }
        
        config = scale_configs[scale]
        
        # Generate test cases
        test_cases = self.generate_diagnostic_cases(config["n_cases"])
        
        # Generate evaluation results for multiple systems
        systems_data = {}
        
        for i in range(config["n_systems"]):
            system_id = f"system_{i:02d}"
            
            # Each system has different performance characteristics
            base_accuracy = 0.7 + (i * 0.02)  # Varying performance
            accuracy_by_complexity = {
                "simple": min(0.95, base_accuracy + 0.1),
                "moderate": base_accuracy,
                "complex": max(0.5, base_accuracy - 0.1)
            }
            
            evaluation_results = self.generate_evaluation_results(
                test_cases, accuracy_by_complexity
            )
            
            systems_data[system_id] = {
                "evaluation_results": evaluation_results,
                "session_results": self.generate_session_results(evaluation_results),
                "metadata": {
                    "base_accuracy": base_accuracy,
                    "complexity_handling": accuracy_by_complexity,
                    "system_type": f"type_{i % 3}"  # Different system types
                }
            }
        
        return {
            "scale": scale,
            "config": config,
            "test_cases": test_cases,
            "systems_data": systems_data,
            "benchmark_metadata": {
                "created_at": datetime.now().isoformat(),
                "data_version": "1.0",
                "generator_seed": self.seed
            }
        }
    
    def _generate_chief_complaint(self, diagnosis: str) -> str:
        """Generate realistic chief complaint for diagnosis."""
        complaints = {
            "pneumonia": ["shortness of breath", "chest pain", "cough with fever"],
            "myocardial_infarction": ["chest pain", "left arm pain", "shortness of breath"],
            "diabetes_mellitus": ["increased thirst", "frequent urination", "fatigue"],
            "hypertension": ["headache", "dizziness", "routine checkup"],
            "influenza": ["fever", "body aches", "cough and congestion"],
            "pneumothorax": ["sudden chest pain", "shortness of breath"],
            "appendicitis": ["abdominal pain", "nausea and vomiting"],
            "stroke": ["weakness", "speech difficulty", "confusion"],
            "copd_exacerbation": ["worsening shortness of breath", "increased cough"],
            "sepsis": ["fever", "chills", "confusion"],
            "gastroenteritis": ["nausea and vomiting", "diarrhea", "abdominal pain"]
        }
        
        return np.random.choice(complaints.get(diagnosis, ["general malaise"]))
    
    def _generate_history(self, diagnosis: str) -> str:
        """Generate relevant medical history."""
        histories = {
            "pneumonia": "3-day history of fever, cough, and shortness of breath",
            "myocardial_infarction": "Sudden onset chest pain radiating to left arm",
            "diabetes_mellitus": "6-month history of polyuria, polydipsia, and weight loss",
            "hypertension": "Family history of hypertension, sedentary lifestyle",
            "influenza": "Acute onset fever, myalgia, and respiratory symptoms",
            "pneumothorax": "Sudden onset chest pain while at rest",
            "appendicitis": "Progressive right lower quadrant pain over 24 hours",
            "stroke": "Sudden onset neurological deficits",
            "copd_exacerbation": "Known COPD with worsening symptoms",
            "sepsis": "Progressive illness with fever and altered mental status",
            "gastroenteritis": "Acute onset nausea, vomiting, and diarrhea"
        }
        
        return histories.get(diagnosis, "Nonspecific symptoms")
    
    def _generate_physical_exam(self, diagnosis: str) -> str:
        """Generate relevant physical exam findings."""
        exams = {
            "pneumonia": "Fever, tachypnea, crackles on lung auscultation",
            "myocardial_infarction": "Chest pain, diaphoresis, possible S3 gallop",
            "diabetes_mellitus": "Dehydration, possible fruity breath odor",
            "hypertension": "Elevated blood pressure, possible retinal changes",
            "influenza": "Fever, myalgia, respiratory congestion",
            "pneumothorax": "Decreased breath sounds, hyperresonance to percussion",
            "appendicitis": "Right lower quadrant tenderness, guarding",
            "stroke": "Neurological deficits, altered mental status",
            "copd_exacerbation": "Wheezing, prolonged expiratory phase",
            "sepsis": "Fever, tachycardia, altered mental status",
            "gastroenteritis": "Abdominal tenderness, dehydration"
        }
        
        return exams.get(diagnosis, "Normal physical exam")
    
    def _generate_differentials(self, diagnosis: str, specialty: str) -> List[str]:
        """Generate differential diagnoses."""
        differentials_map = {
            "pneumonia": ["bronchitis", "pulmonary_embolism", "lung_cancer"],
            "myocardial_infarction": ["angina", "pulmonary_embolism", "aortic_dissection"],
            "diabetes_mellitus": ["diabetes_insipidus", "hyperthyroidism", "medication_side_effect"],
            "hypertension": ["white_coat_hypertension", "secondary_hypertension", "medication_noncompliance"],
            "influenza": ["common_cold", "pneumonia", "covid19"],
            "pneumothorax": ["pulmonary_embolism", "pneumonia", "chest_wall_pain"],
            "appendicitis": ["gastroenteritis", "ovarian_cyst", "kidney_stone"],
            "stroke": ["seizure", "hypoglycemia", "migraine"],
            "copd_exacerbation": ["pneumonia", "heart_failure", "pulmonary_embolism"],
            "sepsis": ["pneumonia", "urinary_tract_infection", "meningitis"],
            "gastroenteritis": ["food_poisoning", "inflammatory_bowel_disease", "medication_side_effect"]
        }
        
        return differentials_map.get(diagnosis, ["unknown_condition"])


class EdgeCaseGenerator:
    """Generator for edge cases in statistical computations and evaluation."""
    
    @staticmethod
    def generate_statistical_edge_cases() -> Dict[str, Dict[str, Any]]:
        """Generate edge cases for statistical computations."""
        edge_cases = {
            "empty_data": {
                "group_a": [],
                "group_b": [],
                "expected_error": "insufficient_data"
            },
            
            "single_value": {
                "group_a": [0.8],
                "group_b": [0.7],
                "expected_behavior": "handle_gracefully"
            },
            
            "identical_values": {
                "group_a": [0.8, 0.8, 0.8, 0.8, 0.8],
                "group_b": [0.8, 0.8, 0.8, 0.8, 0.8],
                "expected_p_value": 1.0,
                "expected_effect_size": 0.0
            },
            
            "extreme_outliers": {
                "group_a": [0.1, 0.2, 0.15, 0.18, 10.0],  # 10.0 is extreme outlier
                "group_b": [0.3, 0.25, 0.28, 0.32, 0.29],
                "notes": "should_detect_outlier"
            },
            
            "zero_variance": {
                "group_a": [0.5, 0.5, 0.5, 0.5],
                "group_b": [0.6, 0.6, 0.6, 0.6],
                "expected_std_a": 0.0,
                "expected_std_b": 0.0
            },
            
            "infinite_values": {
                "group_a": [0.8, 0.9, float('inf')],
                "group_b": [0.7, 0.75, 0.8],
                "expected_error": "invalid_values"
            },
            
            "nan_values": {
                "group_a": [0.8, 0.9, float('nan')],
                "group_b": [0.7, 0.75, 0.8],
                "expected_behavior": "handle_missing_data"
            },
            
            "very_large_sample": {
                "group_a": [0.8 + np.random.normal(0, 0.01) for _ in range(100000)],
                "group_b": [0.79 + np.random.normal(0, 0.01) for _ in range(100000)],
                "notes": "small_effect_large_sample"
            },
            
            "highly_skewed": {
                "group_a": [0.01, 0.02, 0.015, 0.98, 0.99, 0.995],
                "group_b": [0.02, 0.025, 0.018, 0.97, 0.98, 0.985],
                "distribution": "bimodal"
            },
            
            "negative_values": {
                "group_a": [-0.1, -0.2, -0.15],
                "group_b": [0.1, 0.2, 0.15],
                "notes": "unusual_for_accuracy_metrics"
            }
        }
        
        return edge_cases
    
    @staticmethod
    def generate_evaluation_edge_cases() -> List[Dict[str, Any]]:
        """Generate edge cases for evaluation system."""
        edge_cases = [
            {
                "name": "zero_cost_evaluation",
                "diagnosis": "flu",
                "truth": "influenza",
                "tests": [],
                "visits": 0,
                "duration": 0.0,
                "expected_cost": 0.0
            },
            
            {
                "name": "extremely_long_test_list",
                "diagnosis": "complex_case",
                "truth": "multiple_diagnoses",
                "tests": [f"test_{i}" for i in range(1000)],
                "visits": 1,
                "duration": 480.0  # 8 hours
            },
            
            {
                "name": "negative_duration",
                "diagnosis": "flu",
                "truth": "flu",
                "tests": ["cbc"],
                "visits": 1,
                "duration": -10.0,
                "expected_error": "invalid_duration"
            },
            
            {
                "name": "negative_visits",
                "diagnosis": "flu",
                "truth": "flu", 
                "tests": ["cbc"],
                "visits": -1,
                "duration": 30.0,
                "expected_error": "invalid_visits"
            },
            
            {
                "name": "empty_strings",
                "diagnosis": "",
                "truth": "",
                "tests": [""],
                "visits": 1,
                "duration": 30.0,
                "notes": "handle_empty_inputs"
            },
            
            {
                "name": "very_long_strings",
                "diagnosis": "x" * 10000,
                "truth": "y" * 10000,
                "tests": ["z" * 1000],
                "visits": 1,
                "duration": 30.0,
                "notes": "handle_large_inputs"
            },
            
            {
                "name": "unicode_characters",
                "diagnosis": "感冒",  # "cold" in Chinese
                "truth": "インフルエンザ",  # "influenza" in Japanese
                "tests": ["血液検査"],  # "blood test" in Japanese
                "visits": 1,
                "duration": 30.0,
                "notes": "handle_unicode"
            },
            
            {
                "name": "special_characters",
                "diagnosis": "diagnosis with special chars: @#$%^&*()",
                "truth": "truth with special chars: <>?:\"{}|[]\\",
                "tests": ["test_with_special_chars_!@#"],
                "visits": 1,
                "duration": 30.0
            }
        ]
        
        return edge_cases


class MockDataProvider:
    """Provider for mock evaluation results for consistent testing."""
    
    @staticmethod
    def get_consistent_session_results(scenario: str = "balanced") -> List[SessionResult]:
        """Get consistent session results for testing.
        
        Parameters
        ----------
        scenario : str
            Scenario type (balanced, high_cost, high_accuracy, etc.).
            
        Returns
        -------
        List[SessionResult]
            Consistent session results.
        """
        if scenario == "balanced":
            return [
                SessionResult(total_cost=300.0, score=4, correct=True, duration=25.0),
                SessionResult(total_cost=450.0, score=5, correct=True, duration=30.0),
                SessionResult(total_cost=200.0, score=3, correct=False, duration=20.0),
                SessionResult(total_cost=600.0, score=4, correct=True, duration=40.0),
                SessionResult(total_cost=350.0, score=2, correct=False, duration=35.0),
            ]
        
        elif scenario == "high_cost":
            return [
                SessionResult(total_cost=1000.0, score=5, correct=True, duration=60.0),
                SessionResult(total_cost=1200.0, score=4, correct=True, duration=65.0),
                SessionResult(total_cost=800.0, score=5, correct=True, duration=55.0),
                SessionResult(total_cost=1500.0, score=4, correct=True, duration=70.0),
                SessionResult(total_cost=900.0, score=5, correct=True, duration=58.0),
            ]
        
        elif scenario == "high_accuracy":
            return [
                SessionResult(total_cost=400.0, score=5, correct=True, duration=30.0),
                SessionResult(total_cost=350.0, score=5, correct=True, duration=28.0),
                SessionResult(total_cost=380.0, score=4, correct=True, duration=32.0),
                SessionResult(total_cost=420.0, score=5, correct=True, duration=35.0),
                SessionResult(total_cost=360.0, score=4, correct=True, duration=29.0),
            ]
        
        elif scenario == "low_accuracy":
            return [
                SessionResult(total_cost=300.0, score=2, correct=False, duration=25.0),
                SessionResult(total_cost=250.0, score=1, correct=False, duration=20.0),
                SessionResult(total_cost=400.0, score=3, correct=False, duration=30.0),
                SessionResult(total_cost=350.0, score=2, correct=False, duration=28.0),
                SessionResult(total_cost=320.0, score=1, correct=False, duration=22.0),
            ]
        
        elif scenario == "variable_performance":
            return [
                SessionResult(total_cost=200.0, score=5, correct=True, duration=15.0),
                SessionResult(total_cost=800.0, score=1, correct=False, duration=60.0),
                SessionResult(total_cost=400.0, score=4, correct=True, duration=30.0),
                SessionResult(total_cost=150.0, score=3, correct=False, duration=12.0),
                SessionResult(total_cost=1000.0, score=5, correct=True, duration=75.0),
            ]
        
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    @staticmethod
    def get_diagnosis_results(scenario: str = "mixed") -> List[DiagnosisResult]:
        """Get consistent diagnosis results for ensemble testing.
        
        Parameters
        ----------
        scenario : str
            Scenario type for diagnosis results.
            
        Returns
        -------
        List[DiagnosisResult]
            Consistent diagnosis results.
        """
        if scenario == "mixed":
            return [
                DiagnosisResult("pneumonia", confidence=0.85, cost=500.0, run_id="expert_1"),
                DiagnosisResult("bronchitis", confidence=0.75, cost=300.0, run_id="expert_2"), 
                DiagnosisResult("pneumonia", confidence=0.90, cost=600.0, run_id="expert_3"),
                DiagnosisResult("flu", confidence=0.60, cost=200.0, run_id="expert_4"),
            ]
        
        elif scenario == "consensus":
            return [
                DiagnosisResult("pneumonia", confidence=0.88, cost=500.0, run_id="expert_1"),
                DiagnosisResult("pneumonia", confidence=0.82, cost=450.0, run_id="expert_2"),
                DiagnosisResult("pneumonia", confidence=0.91, cost=550.0, run_id="expert_3"),
                DiagnosisResult("pneumonia", confidence=0.86, cost=480.0, run_id="expert_4"),
            ]
        
        elif scenario == "disagreement":
            return [
                DiagnosisResult("pneumonia", confidence=0.70, cost=500.0, run_id="expert_1"),
                DiagnosisResult("bronchitis", confidence=0.75, cost=300.0, run_id="expert_2"),
                DiagnosisResult("flu", confidence=0.65, cost=200.0, run_id="expert_3"),
                DiagnosisResult("covid19", confidence=0.80, cost=250.0, run_id="expert_4"),
            ]
        
        else:
            raise ValueError(f"Unknown scenario: {scenario}")


class TestDataValidator:
    """Validator for test data quality and consistency."""
    
    @staticmethod
    def validate_evaluation_dataset(dataset: EvaluationDataset) -> Dict[str, Any]:
        """Validate evaluation dataset quality and consistency.
        
        Parameters
        ----------
        dataset : EvaluationDataset
            Dataset to validate.
            
        Returns
        -------
        Dict[str, Any]
            Validation results.
        """
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "statistics": {}
        }
        
        # Check basic properties
        if not dataset.test_cases:
            validation_results["errors"].append("Dataset contains no test cases")
            validation_results["is_valid"] = False
        
        if len(dataset.test_cases) != len(dataset.ground_truth_labels):
            validation_results["errors"].append(
                "Mismatch between test cases and ground truth labels"
            )
            validation_results["is_valid"] = False
        
        # Check case ID uniqueness
        case_ids = [case.case_id for case in dataset.test_cases]
        if len(case_ids) != len(set(case_ids)):
            validation_results["errors"].append("Duplicate case IDs found")
            validation_results["is_valid"] = False
        
        # Check complexity distribution
        complexity_counts = {}
        for case in dataset.test_cases:
            complexity_counts[case.case_complexity] = complexity_counts.get(case.case_complexity, 0) + 1
        
        if len(complexity_counts) < 2:
            validation_results["warnings"].append("Dataset lacks complexity diversity")
        
        # Check specialty distribution
        specialty_counts = {}
        for case in dataset.test_cases:
            specialty_counts[case.specialty] = specialty_counts.get(case.specialty, 0) + 1
        
        if len(specialty_counts) < 3:
            validation_results["warnings"].append("Dataset lacks specialty diversity")
        
        # Check age distribution
        ages = [case.patient_age for case in dataset.test_cases]
        if max(ages) - min(ages) < 30:
            validation_results["warnings"].append("Limited age range in dataset")
        
        # Statistical summary
        validation_results["statistics"] = {
            "total_cases": len(dataset.test_cases),
            "complexity_distribution": complexity_counts,
            "specialty_distribution": specialty_counts,
            "age_range": (min(ages), max(ages)),
            "gender_distribution": {
                "M": sum(1 for case in dataset.test_cases if case.patient_gender == "M"),
                "F": sum(1 for case in dataset.test_cases if case.patient_gender == "F")
            }
        }
        
        return validation_results
    
    @staticmethod
    def validate_statistical_data(data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Validate statistical test data quality.
        
        Parameters
        ----------
        data : Dict[str, List[float]]
            Statistical data to validate.
            
        Returns
        -------
        Dict[str, Any]
            Validation results.
        """
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "properties": {}
        }
        
        for group_name, values in data.items():
            if group_name == "properties":
                continue
                
            # Check for empty data
            if not values:
                validation_results["errors"].append(f"Group {group_name} is empty")
                validation_results["is_valid"] = False
                continue
            
            # Check for invalid values
            finite_values = [v for v in values if np.isfinite(v)]
            if len(finite_values) != len(values):
                invalid_count = len(values) - len(finite_values)
                validation_results["warnings"].append(
                    f"Group {group_name} contains {invalid_count} non-finite values"
                )
            
            # Calculate properties
            if finite_values:
                validation_results["properties"][group_name] = {
                    "count": len(finite_values),
                    "mean": np.mean(finite_values),
                    "std": np.std(finite_values),
                    "min": np.min(finite_values),
                    "max": np.max(finite_values),
                    "skewness": self._calculate_skewness(finite_values),
                    "has_outliers": self._detect_outliers(finite_values)
                }
        
        return validation_results
    
    @staticmethod
    def _calculate_skewness(data: List[float]) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean([((x - mean) / std) ** 3 for x in data])
        return skewness
    
    @staticmethod
    def _detect_outliers(data: List[float]) -> bool:
        """Detect outliers using IQR method."""
        if len(data) < 4:
            return False
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        return len(outliers) > 0


# Test classes for data generation and validation
class TestDataGeneration:
    """Test suite for test data generation."""
    
    def test_diagnostic_case_generation(self):
        """Test diagnostic case generation."""
        generator = TestDataGenerator(seed=42)
        test_cases = generator.generate_diagnostic_cases(n_cases=50)
        
        assert len(test_cases) == 50
        assert all(isinstance(case, TestCase) for case in test_cases)
        assert all(case.case_id.startswith("case_") for case in test_cases)
        
        # Check diversity
        complexities = set(case.case_complexity for case in test_cases)
        specialties = set(case.specialty for case in test_cases)
        
        assert len(complexities) >= 2
        assert len(specialties) >= 3
    
    def test_evaluation_results_generation(self):
        """Test evaluation results generation."""
        generator = TestDataGenerator(seed=42)
        test_cases = generator.generate_diagnostic_cases(n_cases=20)
        
        accuracy_by_complexity = {"simple": 0.9, "moderate": 0.7, "complex": 0.5}
        results = generator.generate_evaluation_results(test_cases, accuracy_by_complexity)
        
        assert len(results) == 20
        
        # Check accuracy by complexity
        simple_results = [r for r in results if r["complexity"] == "simple"]
        complex_results = [r for r in results if r["complexity"] == "complex"]
        
        if simple_results and complex_results:
            simple_accuracy = sum(r["correct"] for r in simple_results) / len(simple_results)
            complex_accuracy = sum(r["correct"] for r in complex_results) / len(complex_results)
            
            # Simple cases should generally perform better
            assert simple_accuracy >= complex_accuracy - 0.2  # Allow some variance
    
    def test_statistical_validation_data(self):
        """Test statistical validation data generation."""
        generator = TestDataGenerator(seed=42)
        
        # Test normal distribution scenario
        normal_data = generator.generate_statistical_validation_data("normal_distribution")
        
        assert "group_a" in normal_data
        assert "group_b" in normal_data  
        assert "properties" in normal_data
        
        # Check properties
        group_a_mean = np.mean(normal_data["group_a"])
        group_b_mean = np.mean(normal_data["group_b"])
        
        assert abs(group_a_mean - 0.8) < 0.1  # Should be close to target mean
        assert abs(group_b_mean - 0.75) < 0.1
    
    def test_performance_benchmark_data(self):
        """Test performance benchmark data generation."""
        generator = TestDataGenerator(seed=42)
        benchmark_data = generator.generate_performance_benchmark_data("small")
        
        assert benchmark_data["scale"] == "small"
        assert len(benchmark_data["test_cases"]) == 100
        assert len(benchmark_data["systems_data"]) == 3
        
        # Check system data structure
        for system_id, system_data in benchmark_data["systems_data"].items():
            assert "evaluation_results" in system_data
            assert "session_results" in system_data
            assert "metadata" in system_data


class TestEdgeCases:
    """Test suite for edge case handling."""
    
    def test_statistical_edge_cases(self):
        """Test statistical edge case generation."""
        edge_cases = EdgeCaseGenerator.generate_statistical_edge_cases()
        
        assert "empty_data" in edge_cases
        assert "identical_values" in edge_cases
        assert "extreme_outliers" in edge_cases
        
        # Check empty data case
        empty_case = edge_cases["empty_data"]
        assert len(empty_case["group_a"]) == 0
        assert len(empty_case["group_b"]) == 0
        assert "expected_error" in empty_case
    
    def test_evaluation_edge_cases(self):
        """Test evaluation edge case generation."""
        edge_cases = EdgeCaseGenerator.generate_evaluation_edge_cases()
        
        assert len(edge_cases) > 0
        
        # Check zero cost case
        zero_cost_case = next(
            (case for case in edge_cases if case["name"] == "zero_cost_evaluation"), 
            None
        )
        assert zero_cost_case is not None
        assert zero_cost_case["visits"] == 0
        assert len(zero_cost_case["tests"]) == 0


class TestDataValidation:
    """Test suite for data validation."""
    
    def test_evaluation_dataset_validation(self):
        """Test evaluation dataset validation."""
        generator = TestDataGenerator(seed=42)
        test_cases = generator.generate_diagnostic_cases(n_cases=10)
        ground_truth = [case.true_diagnosis for case in test_cases]
        
        dataset = EvaluationDataset(
            name="test_dataset",
            description="Test dataset for validation",
            test_cases=test_cases,
            ground_truth_labels=ground_truth,
            complexity_distribution={},
            specialty_distribution={},
            created_at=datetime.now(),
            version="1.0"
        )
        
        validation_results = TestDataValidator.validate_evaluation_dataset(dataset)
        
        assert "is_valid" in validation_results
        assert "warnings" in validation_results
        assert "errors" in validation_results
        assert "statistics" in validation_results
    
    def test_statistical_data_validation(self):
        """Test statistical data validation."""
        valid_data = {
            "group_a": [0.8, 0.85, 0.75, 0.9, 0.82],
            "group_b": [0.7, 0.72, 0.68, 0.75, 0.71]
        }
        
        validation_results = TestDataValidator.validate_statistical_data(valid_data)
        
        assert validation_results["is_valid"] is True
        assert "group_a" in validation_results["properties"]
        assert "group_b" in validation_results["properties"]


# Pytest fixtures for test data
@pytest.fixture
def test_data_generator():
    """Provide test data generator with fixed seed."""
    return TestDataGenerator(seed=42)


@pytest.fixture
def sample_test_cases():
    """Provide sample test cases for testing."""
    generator = TestDataGenerator(seed=42)
    return generator.generate_diagnostic_cases(n_cases=10)


@pytest.fixture
def mock_session_results():
    """Provide mock session results for testing."""
    return MockDataProvider.get_consistent_session_results("balanced")


if __name__ == "__main__":
    pytest.main([__file__])