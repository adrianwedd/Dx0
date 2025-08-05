#!/usr/bin/env python3
"""
CPT Feature Engineering Framework
Demonstrates feature extraction and engineering for ML-based cost estimation
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
from pathlib import Path


class CPTFeatureEngineer:
    """Comprehensive feature engineering for CPT code and pricing data."""
    
    def __init__(self):
        # CPT category mappings
        self.cpt_categories = {
            "00-09": "Anesthesia",
            "10-19": "Integumentary System",
            "20-29": "Musculoskeletal System",
            "30-39": "Respiratory/Cardiovascular",
            "40-49": "Digestive System",
            "50-59": "Urinary/Male Genital",
            "60-69": "Female Genital/Maternity",
            "70-79": "Radiology",
            "80-89": "Laboratory/Pathology",
            "90-99": "Medicine/E&M"
        }
        
        # Complexity indicators
        self.complexity_keywords = {
            "simple": ["basic", "simple", "routine", "standard", "screening"],
            "moderate": ["comprehensive", "detailed", "extended", "multiple"],
            "complex": ["complex", "complicated", "extensive", "advanced", "specialized"]
        }
        
        # Body system mappings
        self.body_systems = {
            "cardiovascular": ["heart", "cardiac", "vascular", "artery", "vein", "blood vessel"],
            "respiratory": ["lung", "breath", "respiratory", "pulmonary", "chest"],
            "neurological": ["brain", "nerve", "neuro", "spine", "cranial"],
            "gastrointestinal": ["stomach", "intestine", "colon", "liver", "digestive"],
            "musculoskeletal": ["bone", "joint", "muscle", "skeletal", "orthopedic"],
            "endocrine": ["hormone", "thyroid", "diabetes", "metabolic", "endocrine"],
            "renal": ["kidney", "renal", "urinary", "bladder"],
            "hematologic": ["blood", "hematology", "anemia", "bleeding"],
            "immunologic": ["immune", "antibody", "antigen", "allergy"]
        }
        
        # Test type indicators
        self.test_types = {
            "imaging": ["xray", "x-ray", "ct", "mri", "ultrasound", "scan", "imaging"],
            "laboratory": ["blood", "urine", "culture", "panel", "test", "assay"],
            "procedure": ["biopsy", "endoscopy", "surgery", "removal", "insertion"],
            "diagnostic": ["ecg", "ekg", "eeg", "emg", "monitor", "study"],
            "therapeutic": ["therapy", "treatment", "injection", "infusion"]
        }
        
        # Load RVU data (simulated for demo)
        self.rvu_data = self._load_rvu_data()
        
    def _load_rvu_data(self) -> Dict[str, Dict[str, float]]:
        """Load or simulate RVU data for CPT codes."""
        # In production, this would load from CMS RVU file
        # For demo, we'll simulate some values
        return {
            "85027": {"work": 0.0, "facility": 0.45, "malpractice": 0.01, "total": 0.46},
            "80048": {"work": 0.0, "facility": 0.37, "malpractice": 0.01, "total": 0.38},
            "71046": {"work": 0.22, "facility": 0.89, "malpractice": 0.04, "total": 1.15},
            "70450": {"work": 1.27, "facility": 6.82, "malpractice": 0.10, "total": 8.19},
            # Add more as needed
        }
        
    def extract_basic_features(self, test_name: str, cpt_code: str, price: float) -> Dict[str, Any]:
        """Extract basic features from test data."""
        features = {
            # Price transformations
            "price": price,
            "price_log": np.log1p(price),
            "price_sqrt": np.sqrt(price),
            "price_squared": price ** 2,
            
            # CPT code features
            "cpt_numeric": self._extract_cpt_numeric(cpt_code),
            "cpt_category_code": cpt_code[:2] if len(cpt_code) >= 2 else "00",
            "cpt_subcategory_code": cpt_code[:3] if len(cpt_code) >= 3 else "000",
            "is_addon_code": self._is_addon_code(cpt_code),
            "has_modifier": "-" in cpt_code,
            
            # Test name features
            "name_length": len(test_name),
            "name_word_count": len(test_name.split()),
            "name_char_count": len(test_name.replace(" ", "")),
            "has_numbers": bool(re.search(r'\d', test_name)),
            "has_special_chars": bool(re.search(r'[^a-zA-Z0-9\s]', test_name))
        }
        
        return features
        
    def extract_complexity_features(self, test_name: str, cpt_code: str) -> Dict[str, Any]:
        """Extract complexity-related features."""
        test_lower = test_name.lower()
        
        features = {
            "complexity_simple": self._has_keywords(test_lower, self.complexity_keywords["simple"]),
            "complexity_moderate": self._has_keywords(test_lower, self.complexity_keywords["moderate"]),
            "complexity_complex": self._has_keywords(test_lower, self.complexity_keywords["complex"]),
            "estimated_complexity_score": self._estimate_complexity_score(test_name, cpt_code),
            "has_contrast": "contrast" in test_lower,
            "is_bilateral": "bilateral" in test_lower or "both" in test_lower,
            "is_complete": "complete" in test_lower or "comprehensive" in test_lower,
            "is_limited": "limited" in test_lower or "partial" in test_lower,
            "time_indicator": self._extract_time_indicator(test_lower)
        }
        
        return features
        
    def extract_categorical_features(self, test_name: str, cpt_code: str) -> Dict[str, Any]:
        """Extract categorical and classification features."""
        features = {
            "cpt_category": self._get_cpt_category(cpt_code),
            "test_type": self._classify_test_type(test_name),
            "body_system": self._identify_body_system(test_name),
            "is_emergency": self._is_emergency_test(test_name),
            "is_preventive": self._is_preventive_test(test_name),
            "requires_fasting": self._requires_fasting(test_name),
            "is_panel": "panel" in test_name.lower(),
            "is_culture": "culture" in test_name.lower(),
            "is_imaging": self._is_imaging_test(test_name, cpt_code)
        }
        
        return features
        
    def extract_rvu_features(self, cpt_code: str) -> Dict[str, Any]:
        """Extract RVU-based features."""
        rvu_info = self.rvu_data.get(cpt_code, {})
        
        features = {
            "rvu_work": rvu_info.get("work", 0.0),
            "rvu_facility": rvu_info.get("facility", 0.0),
            "rvu_malpractice": rvu_info.get("malpractice", 0.0),
            "rvu_total": rvu_info.get("total", 0.0),
            "has_rvu_data": bool(rvu_info),
            "rvu_work_percentage": (
                rvu_info.get("work", 0) / rvu_info.get("total", 1) 
                if rvu_info.get("total", 0) > 0 else 0
            )
        }
        
        return features
        
    def extract_pricing_features(self, price: float, cpt_code: str, 
                                all_prices: Optional[List[float]] = None) -> Dict[str, Any]:
        """Extract price-related statistical features."""
        features = {
            "price_tier": self._get_price_tier(price),
            "price_magnitude": int(np.log10(max(price, 1))),
            "is_round_number": price == int(price),
            "price_last_digit": int(price) % 10
        }
        
        if all_prices:
            # Calculate relative pricing features
            prices_array = np.array(all_prices)
            features.update({
                "price_percentile": (prices_array <= price).sum() / len(prices_array) * 100,
                "price_z_score": (price - prices_array.mean()) / prices_array.std() if prices_array.std() > 0 else 0,
                "price_relative_to_median": price / np.median(prices_array) if np.median(prices_array) > 0 else 1,
                "is_price_outlier": abs((price - prices_array.mean()) / prices_array.std()) > 3 if prices_array.std() > 0 else False
            })
            
        return features
        
    def extract_semantic_features(self, test_name: str) -> Dict[str, Any]:
        """Extract semantic and linguistic features."""
        words = test_name.lower().split()
        
        features = {
            "has_abbreviation": any(len(w) <= 3 and w.isupper() for w in test_name.split()),
            "has_parentheses": "(" in test_name,
            "has_slash": "/" in test_name,
            "has_hyphen": "-" in test_name,
            "first_word": words[0] if words else "",
            "last_word": words[-1] if words else "",
            "contains_body_part": self._contains_body_part(test_name),
            "contains_measurement": self._contains_measurement(test_name),
            "word_diversity": len(set(words)) / len(words) if words else 0
        }
        
        return features
        
    def create_feature_vector(self, test_name: str, cpt_code: str, price: float,
                            all_prices: Optional[List[float]] = None) -> Dict[str, Any]:
        """Create complete feature vector for ML training."""
        features = {}
        
        # Combine all feature extractors
        features.update(self.extract_basic_features(test_name, cpt_code, price))
        features.update(self.extract_complexity_features(test_name, cpt_code))
        features.update(self.extract_categorical_features(test_name, cpt_code))
        features.update(self.extract_rvu_features(cpt_code))
        features.update(self.extract_pricing_features(price, cpt_code, all_prices))
        features.update(self.extract_semantic_features(test_name))
        
        # Add interaction features
        features.update(self._create_interaction_features(features))
        
        # Add metadata
        features["_metadata"] = {
            "test_name": test_name,
            "cpt_code": cpt_code,
            "original_price": price,
            "feature_version": "1.0",
            "extraction_date": datetime.now().isoformat()
        }
        
        return features
        
    def _extract_cpt_numeric(self, cpt_code: str) -> int:
        """Extract numeric value from CPT code."""
        numeric_part = re.search(r'\d+', cpt_code)
        return int(numeric_part.group()) if numeric_part else 0
        
    def _is_addon_code(self, cpt_code: str) -> bool:
        """Check if CPT code is an add-on code."""
        # Add-on codes typically have specific markers or are in certain ranges
        return "+" in cpt_code or cpt_code.endswith("ZZZ")
        
    def _has_keywords(self, text: str, keywords: List[str]) -> bool:
        """Check if text contains any of the keywords."""
        return any(keyword in text for keyword in keywords)
        
    def _estimate_complexity_score(self, test_name: str, cpt_code: str) -> float:
        """Estimate complexity score based on various factors."""
        score = 0.0
        test_lower = test_name.lower()
        
        # Base score from CPT code range
        cpt_num = self._extract_cpt_numeric(cpt_code)
        if cpt_num > 90000:  # E&M codes tend to be complex
            score += 0.3
        elif cpt_num > 70000:  # Imaging
            score += 0.2
            
        # Complexity keywords
        if self._has_keywords(test_lower, self.complexity_keywords["complex"]):
            score += 0.4
        elif self._has_keywords(test_lower, self.complexity_keywords["moderate"]):
            score += 0.2
            
        # Additional indicators
        if "comprehensive" in test_lower:
            score += 0.1
        if "with contrast" in test_lower:
            score += 0.15
        if len(test_name.split()) > 5:
            score += 0.1
            
        return min(score, 1.0)  # Cap at 1.0
        
    def _extract_time_indicator(self, test_name: str) -> Optional[int]:
        """Extract time duration from test name."""
        time_match = re.search(r'(\d+)\s*(hour|hr|minute|min|day)', test_name)
        if time_match:
            value = int(time_match.group(1))
            unit = time_match.group(2)
            if 'min' in unit:
                return value
            elif 'hour' in unit or 'hr' in unit:
                return value * 60
            elif 'day' in unit:
                return value * 1440
        return None
        
    def _get_cpt_category(self, cpt_code: str) -> str:
        """Get CPT category from code."""
        if not cpt_code or len(cpt_code) < 2:
            return "Unknown"
            
        prefix = cpt_code[:2]
        for range_str, category in self.cpt_categories.items():
            start, end = range_str.split("-")
            if start <= prefix <= end:
                return category
                
        return "Other"
        
    def _classify_test_type(self, test_name: str) -> str:
        """Classify test into major type categories."""
        test_lower = test_name.lower()
        
        for test_type, keywords in self.test_types.items():
            if self._has_keywords(test_lower, keywords):
                return test_type
                
        return "other"
        
    def _identify_body_system(self, test_name: str) -> str:
        """Identify primary body system from test name."""
        test_lower = test_name.lower()
        
        for system, keywords in self.body_systems.items():
            if self._has_keywords(test_lower, keywords):
                return system
                
        return "general"
        
    def _is_emergency_test(self, test_name: str) -> bool:
        """Check if test is typically ordered in emergency settings."""
        emergency_keywords = ["stat", "emergency", "urgent", "rapid", "troponin", "d-dimer"]
        return self._has_keywords(test_name.lower(), emergency_keywords)
        
    def _is_preventive_test(self, test_name: str) -> bool:
        """Check if test is preventive/screening."""
        preventive_keywords = ["screening", "preventive", "annual", "routine", "wellness"]
        return self._has_keywords(test_name.lower(), preventive_keywords)
        
    def _requires_fasting(self, test_name: str) -> bool:
        """Check if test typically requires fasting."""
        fasting_tests = ["glucose", "lipid", "cholesterol", "triglyceride", "metabolic panel"]
        return self._has_keywords(test_name.lower(), fasting_tests)
        
    def _is_imaging_test(self, test_name: str, cpt_code: str) -> bool:
        """Check if test is imaging."""
        return (self._classify_test_type(test_name) == "imaging" or 
                (cpt_code[:2].isdigit() and 70 <= int(cpt_code[:2]) <= 79))
                
    def _get_price_tier(self, price: float) -> int:
        """Categorize price into tiers."""
        if price < 25:
            return 1
        elif price < 50:
            return 2
        elif price < 100:
            return 3
        elif price < 250:
            return 4
        elif price < 500:
            return 5
        elif price < 1000:
            return 6
        else:
            return 7
            
    def _contains_body_part(self, test_name: str) -> bool:
        """Check if test name contains body part reference."""
        body_parts = ["head", "chest", "abdomen", "pelvis", "spine", "brain", "heart", 
                     "lung", "liver", "kidney", "bone", "joint", "blood", "urine"]
        return self._has_keywords(test_name.lower(), body_parts)
        
    def _contains_measurement(self, test_name: str) -> bool:
        """Check if test name contains measurement terms."""
        measurements = ["level", "count", "concentration", "ratio", "percentage", 
                       "volume", "density", "rate"]
        return self._has_keywords(test_name.lower(), measurements)
        
    def _create_interaction_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Create interaction features between existing features."""
        interactions = {}
        
        # Price-complexity interaction
        if "estimated_complexity_score" in features and "price" in features:
            interactions["price_complexity_ratio"] = (
                features["price"] / (features["estimated_complexity_score"] + 0.1)
            )
            
        # RVU-price interaction
        if "rvu_total" in features and "price" in features and features["rvu_total"] > 0:
            interactions["price_per_rvu"] = features["price"] / features["rvu_total"]
            
        # Category-specific price tier
        if "cpt_category" in features and "price_tier" in features:
            interactions[f"price_tier_{features['cpt_category'].lower().replace(' ', '_')}"] = features["price_tier"]
            
        return interactions
        
    def create_feature_report(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate report on feature distributions and quality."""
        all_features = []
        
        # Extract features for all tests
        for item in test_data:
            features = self.create_feature_vector(
                item["test_name"], 
                item["cpt_code"], 
                float(item["price"])
            )
            all_features.append(features)
            
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_features)
        
        # Remove metadata column for statistics
        feature_cols = [col for col in df.columns if not col.startswith("_")]
        
        report = {
            "total_features": len(feature_cols),
            "total_samples": len(df),
            "feature_types": {
                "numeric": [],
                "boolean": [],
                "categorical": []
            },
            "missing_values": {},
            "feature_statistics": {}
        }
        
        # Analyze each feature
        for col in feature_cols:
            # Determine feature type
            if df[col].dtype == bool:
                report["feature_types"]["boolean"].append(col)
            elif df[col].dtype in [int, float, np.int64, np.float64]:
                report["feature_types"]["numeric"].append(col)
                # Calculate statistics for numeric features
                report["feature_statistics"][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median())
                }
            else:
                report["feature_types"]["categorical"].append(col)
                # Value counts for categorical
                report["feature_statistics"][col] = df[col].value_counts().to_dict()
                
            # Check missing values
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                report["missing_values"][col] = {
                    "count": int(missing_count),
                    "percentage": float(missing_count / len(df) * 100)
                }
                
        return report


def demonstrate_feature_engineering():
    """Demonstrate feature engineering on sample data."""
    # Load sample data
    engineer = CPTFeatureEngineer()
    
    # Sample test cases
    sample_tests = [
        {"test_name": "complete blood count", "cpt_code": "85027", "price": "9.0"},
        {"test_name": "comprehensive metabolic panel", "cpt_code": "80053", "price": "14.0"},
        {"test_name": "mri brain with contrast", "cpt_code": "70553", "price": "400.0"},
        {"test_name": "chest x-ray 2 views", "cpt_code": "71046", "price": "30.0"},
        {"test_name": "24 hour holter monitor", "cpt_code": "93224", "price": "200.0"}
    ]
    
    print("CPT Feature Engineering Demonstration")
    print("=" * 60)
    
    for test in sample_tests:
        print(f"\nTest: {test['test_name']}")
        print(f"CPT Code: {test['cpt_code']}")
        print(f"Price: ${test['price']}")
        
        # Extract features
        features = engineer.create_feature_vector(
            test["test_name"],
            test["cpt_code"],
            float(test["price"])
        )
        
        # Display key features
        print("\nKey Features:")
        feature_categories = {
            "Basic": ["price", "cpt_numeric", "name_word_count"],
            "Complexity": ["estimated_complexity_score", "is_complete", "has_contrast"],
            "Category": ["cpt_category", "test_type", "body_system"],
            "Pricing": ["price_tier", "price_log"],
            "RVU": ["rvu_total", "has_rvu_data"]
        }
        
        for category, feature_names in feature_categories.items():
            print(f"\n  {category}:")
            for fname in feature_names:
                if fname in features:
                    print(f"    {fname}: {features[fname]}")
                    
    # Generate feature report
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING REPORT")
    print("=" * 60)
    
    report = engineer.create_feature_report(sample_tests)
    
    print(f"\nTotal Features Generated: {report['total_features']}")
    print(f"Samples Processed: {report['total_samples']}")
    
    print("\nFeature Types:")
    for ftype, features in report["feature_types"].items():
        print(f"  {ftype.capitalize()}: {len(features)} features")
        
    print("\nSample Numeric Feature Statistics:")
    for feature, stats in list(report["feature_statistics"].items())[:5]:
        if isinstance(stats, dict) and "mean" in stats:
            print(f"\n  {feature}:")
            print(f"    Mean: {stats['mean']:.2f}")
            print(f"    Std: {stats['std']:.2f}")
            print(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")


if __name__ == "__main__":
    demonstrate_feature_engineering()