#!/usr/bin/env python3
"""
CPT Data Validation and Analysis Script
Analyzes the current CPT code and pricing data for quality issues and ML readiness
"""

import csv
import json
import re
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
import statistics
from pathlib import Path


class CPTDataValidator:
    """Comprehensive validation and analysis of CPT pricing data."""
    
    def __init__(self, data_path: str = "data/cpt_lookup.csv"):
        self.data_path = Path(data_path)
        self.data = []
        self.validation_results = {
            "total_records": 0,
            "valid_records": 0,
            "issues": defaultdict(list),
            "statistics": {},
            "recommendations": []
        }
        
        # CPT code patterns
        self.cpt_patterns = {
            "standard": re.compile(r"^\d{5}$"),  # 5 digits
            "hcpcs": re.compile(r"^[A-Z]\d{4}$"),  # Letter + 4 digits
            "category_iii": re.compile(r"^\d{4}T$"),  # 4 digits + T
            "modifier": re.compile(r"^\d{5}-\d{2}$")  # 5 digits - 2 digits
        }
        
        # Price validation thresholds
        self.price_thresholds = {
            "min": 0.01,
            "max": 50000.00,
            "outlier_std": 3  # Standard deviations for outlier detection
        }
        
    def load_data(self) -> bool:
        """Load CPT data from CSV file."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.data = list(reader)
                self.validation_results["total_records"] = len(self.data)
                return True
        except Exception as e:
            self.validation_results["issues"]["file_errors"].append(str(e))
            return False
            
    def validate_cpt_code(self, code: str, row_num: int) -> bool:
        """Validate CPT code format."""
        if not code:
            self.validation_results["issues"]["missing_cpt"].append(row_num)
            return False
            
        code = code.strip()
        
        # Check against known patterns
        valid = False
        for pattern_name, pattern in self.cpt_patterns.items():
            if pattern.match(code):
                valid = True
                break
                
        if not valid:
            self.validation_results["issues"]["invalid_cpt_format"].append({
                "row": row_num,
                "code": code
            })
            
        return valid
        
    def validate_price(self, price_str: str, row_num: int) -> Optional[float]:
        """Validate and parse price."""
        try:
            price = float(price_str)
            
            if price < self.price_thresholds["min"]:
                self.validation_results["issues"]["price_too_low"].append({
                    "row": row_num,
                    "price": price
                })
                
            if price > self.price_thresholds["max"]:
                self.validation_results["issues"]["price_too_high"].append({
                    "row": row_num,
                    "price": price
                })
                
            return price
            
        except (ValueError, TypeError):
            self.validation_results["issues"]["invalid_price_format"].append({
                "row": row_num,
                "value": price_str
            })
            return None
            
    def validate_test_name(self, name: str, row_num: int) -> bool:
        """Validate test name."""
        if not name or not name.strip():
            self.validation_results["issues"]["missing_test_name"].append(row_num)
            return False
            
        # Check for suspicious patterns
        if len(name) < 3:
            self.validation_results["issues"]["test_name_too_short"].append({
                "row": row_num,
                "name": name
            })
            
        if len(name) > 100:
            self.validation_results["issues"]["test_name_too_long"].append({
                "row": row_num,
                "name": name
            })
            
        return True
        
    def detect_duplicates(self):
        """Detect duplicate entries and similar names."""
        # Check for exact duplicates
        seen_combos = set()
        name_variations = defaultdict(list)
        
        for i, row in enumerate(self.data):
            combo = (row.get("test_name", "").lower().strip(), 
                    row.get("cpt_code", "").strip())
            
            if combo in seen_combos:
                self.validation_results["issues"]["exact_duplicates"].append({
                    "row": i + 2,
                    "test_name": row.get("test_name"),
                    "cpt_code": row.get("cpt_code")
                })
            seen_combos.add(combo)
            
            # Group by normalized name for variation detection
            normalized = self._normalize_name(row.get("test_name", ""))
            name_variations[normalized].append({
                "row": i + 2,
                "original": row.get("test_name"),
                "cpt": row.get("cpt_code")
            })
            
        # Find potential name variations
        for normalized, entries in name_variations.items():
            if len(entries) > 1:
                unique_names = set(e["original"] for e in entries)
                if len(unique_names) > 1:
                    self.validation_results["issues"]["name_variations"].append({
                        "normalized": normalized,
                        "variations": list(unique_names),
                        "entries": entries
                    })
                    
    def _normalize_name(self, name: str) -> str:
        """Normalize test name for comparison."""
        if not name:
            return ""
            
        # Convert to lowercase and remove extra spaces
        normalized = " ".join(name.lower().split())
        
        # Remove common variations
        replacements = {
            "w/": "with",
            "w/o": "without",
            "&": "and",
            "hrs": "hours",
            "hr": "hour",
            "/": " ",
            "-": " ",
            "  ": " "
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
            
        return normalized.strip()
        
    def analyze_price_distribution(self):
        """Analyze price distribution and detect outliers."""
        prices = []
        price_by_category = defaultdict(list)
        
        for row in self.data:
            try:
                price = float(row.get("price", 0))
                if price > 0:
                    prices.append(price)
                    
                    # Categorize by CPT code range
                    cpt = row.get("cpt_code", "")
                    if cpt and cpt[:2].isdigit():
                        category = f"{cpt[:2]}xxx"
                        price_by_category[category].append(price)
            except:
                pass
                
        if prices:
            # Overall statistics
            self.validation_results["statistics"]["price"] = {
                "count": len(prices),
                "min": min(prices),
                "max": max(prices),
                "mean": statistics.mean(prices),
                "median": statistics.median(prices),
                "stdev": statistics.stdev(prices) if len(prices) > 1 else 0,
                "percentiles": {
                    "25": statistics.quantiles(prices, n=4)[0] if len(prices) > 3 else 0,
                    "75": statistics.quantiles(prices, n=4)[2] if len(prices) > 3 else 0,
                    "90": statistics.quantiles(prices, n=10)[8] if len(prices) > 9 else 0,
                    "95": statistics.quantiles(prices, n=20)[18] if len(prices) > 19 else 0
                }
            }
            
            # Detect outliers
            mean = statistics.mean(prices)
            stdev = statistics.stdev(prices) if len(prices) > 1 else 0
            
            for i, row in enumerate(self.data):
                try:
                    price = float(row.get("price", 0))
                    if stdev > 0 and abs(price - mean) > self.price_thresholds["outlier_std"] * stdev:
                        self.validation_results["issues"]["price_outliers"].append({
                            "row": i + 2,
                            "test_name": row.get("test_name"),
                            "price": price,
                            "z_score": (price - mean) / stdev if stdev > 0 else 0
                        })
                except:
                    pass
                    
    def analyze_cpt_coverage(self):
        """Analyze CPT code coverage by category."""
        cpt_categories = defaultdict(int)
        cpt_codes = set()
        
        for row in self.data:
            cpt = row.get("cpt_code", "").strip()
            if cpt:
                cpt_codes.add(cpt)
                
                # Categorize by first 2 digits
                if len(cpt) >= 2 and cpt[:2].isdigit():
                    category_code = int(cpt[:2])
                    
                    if 70 <= category_code <= 79:
                        category = "Radiology"
                    elif 80 <= category_code <= 89:
                        category = "Laboratory"
                    elif 90 <= category_code <= 99:
                        category = "Medicine/E&M"
                    elif 10 <= category_code <= 69:
                        category = "Surgery"
                    elif 0 <= category_code <= 9:
                        category = "Anesthesia"
                    else:
                        category = "Other"
                        
                    cpt_categories[category] += 1
                elif cpt[0].isalpha():
                    cpt_categories["HCPCS"] += 1
                else:
                    cpt_categories["Other"] += 1
                    
        self.validation_results["statistics"]["cpt_coverage"] = {
            "total_unique_codes": len(cpt_codes),
            "categories": dict(cpt_categories),
            "category_percentages": {
                cat: (count / len(cpt_codes) * 100) if cpt_codes else 0
                for cat, count in cpt_categories.items()
            }
        }
        
    def check_data_consistency(self):
        """Check for data consistency issues."""
        # Check for consistent pricing for same CPT codes
        cpt_prices = defaultdict(list)
        
        for i, row in enumerate(self.data):
            cpt = row.get("cpt_code", "").strip()
            try:
                price = float(row.get("price", 0))
                if cpt and price > 0:
                    cpt_prices[cpt].append({
                        "row": i + 2,
                        "test_name": row.get("test_name"),
                        "price": price
                    })
            except:
                pass
                
        # Find CPT codes with inconsistent pricing
        for cpt, price_data in cpt_prices.items():
            if len(price_data) > 1:
                prices = [p["price"] for p in price_data]
                min_price = min(prices)
                max_price = max(prices)
                
                # Check if price variation is significant (>20%)
                if max_price > min_price * 1.2:
                    self.validation_results["issues"]["inconsistent_pricing"].append({
                        "cpt_code": cpt,
                        "price_range": f"${min_price:.2f} - ${max_price:.2f}",
                        "entries": price_data
                    })
                    
    def generate_ml_readiness_score(self):
        """Calculate ML readiness score based on data quality."""
        scores = {
            "data_volume": min(len(self.data) / 5000 * 100, 100),  # Target: 5000 records
            "data_completeness": 0,
            "data_consistency": 0,
            "feature_richness": 20,  # Only 3 features currently
            "price_quality": 0
        }
        
        # Calculate completeness
        total_fields = len(self.data) * 3  # 3 required fields
        missing_fields = (
            len(self.validation_results["issues"]["missing_test_name"]) +
            len(self.validation_results["issues"]["missing_cpt"]) +
            len(self.validation_results["issues"]["invalid_price_format"])
        )
        scores["data_completeness"] = max(0, (1 - missing_fields / total_fields) * 100) if total_fields > 0 else 0
        
        # Calculate consistency
        total_issues = sum(len(issues) for issues in self.validation_results["issues"].values())
        scores["data_consistency"] = max(0, (1 - total_issues / len(self.data)) * 100) if self.data else 0
        
        # Calculate price quality
        price_issues = (
            len(self.validation_results["issues"]["price_too_low"]) +
            len(self.validation_results["issues"]["price_outliers"]) +
            len(self.validation_results["issues"]["inconsistent_pricing"])
        )
        scores["price_quality"] = max(0, (1 - price_issues / len(self.data)) * 100) if self.data else 0
        
        # Overall score
        overall_score = sum(scores.values()) / len(scores)
        
        self.validation_results["ml_readiness"] = {
            "overall_score": overall_score,
            "component_scores": scores,
            "rating": self._get_rating(overall_score)
        }
        
    def _get_rating(self, score: float) -> str:
        """Convert score to rating."""
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        elif score >= 20:
            return "Poor"
        else:
            return "Critical"
            
    def generate_recommendations(self):
        """Generate specific recommendations based on findings."""
        recommendations = []
        
        # Data volume
        if len(self.data) < 1000:
            recommendations.append({
                "priority": "HIGH",
                "category": "Data Volume",
                "issue": f"Only {len(self.data)} records available",
                "recommendation": "Expand dataset to at least 5,000 CPT codes for viable ML training",
                "impact": "Current volume insufficient for robust ML models"
            })
            
        # Missing data
        if self.validation_results["issues"]["missing_test_name"]:
            recommendations.append({
                "priority": "HIGH",
                "category": "Data Completeness",
                "issue": f"{len(self.validation_results['issues']['missing_test_name'])} missing test names",
                "recommendation": "Fill in missing test names or remove incomplete records",
                "impact": "Incomplete records reduce training data quality"
            })
            
        # Price outliers
        if self.validation_results["issues"]["price_outliers"]:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Price Quality",
                "issue": f"{len(self.validation_results['issues']['price_outliers'])} price outliers detected",
                "recommendation": "Review and validate outlier prices against CMS or commercial rates",
                "impact": "Outliers can skew ML model predictions"
            })
            
        # Duplicates
        if self.validation_results["issues"]["exact_duplicates"]:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Data Quality",
                "issue": f"{len(self.validation_results['issues']['exact_duplicates'])} duplicate entries",
                "recommendation": "Remove duplicate entries to avoid training bias",
                "impact": "Duplicates can lead to overfitting"
            })
            
        # Feature richness
        recommendations.append({
            "priority": "HIGH",
            "category": "Feature Engineering",
            "issue": "Only 3 features available (test_name, cpt_code, price)",
            "recommendation": "Add metadata: category, complexity, RVU values, geographic data",
            "impact": "Limited features restrict model sophistication"
        })
        
        # CPT coverage
        lab_coverage = self.validation_results["statistics"]["cpt_coverage"]["categories"].get("Laboratory", 0)
        total_codes = self.validation_results["statistics"]["cpt_coverage"]["total_unique_codes"]
        
        if total_codes > 0 and lab_coverage / total_codes > 0.6:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "CPT Balance",
                "issue": f"Laboratory codes represent {lab_coverage/total_codes*100:.1f}% of dataset",
                "recommendation": "Balance dataset with more radiology, procedure, and E&M codes",
                "impact": "Imbalanced data may bias predictions toward laboratory pricing"
            })
            
        self.validation_results["recommendations"] = recommendations
        
    def run_validation(self):
        """Run complete validation suite."""
        print("Loading data...")
        if not self.load_data():
            print("Failed to load data!")
            return
            
        print(f"Loaded {len(self.data)} records")
        
        print("Validating records...")
        valid_count = 0
        for i, row in enumerate(self.data):
            row_valid = True
            
            # Validate each field
            if not self.validate_test_name(row.get("test_name", ""), i + 2):
                row_valid = False
                
            if not self.validate_cpt_code(row.get("cpt_code", ""), i + 2):
                row_valid = False
                
            if self.validate_price(row.get("price", ""), i + 2) is None:
                row_valid = False
                
            if row_valid:
                valid_count += 1
                
        self.validation_results["valid_records"] = valid_count
        
        print("Checking for duplicates...")
        self.detect_duplicates()
        
        print("Analyzing price distribution...")
        self.analyze_price_distribution()
        
        print("Analyzing CPT coverage...")
        self.analyze_cpt_coverage()
        
        print("Checking data consistency...")
        self.check_data_consistency()
        
        print("Calculating ML readiness...")
        self.generate_ml_readiness_score()
        
        print("Generating recommendations...")
        self.generate_recommendations()
        
        print("\nValidation complete!")
        
    def generate_report(self, output_path: str = "analysis/validation_report.json"):
        """Generate detailed validation report."""
        report = {
            "metadata": {
                "data_file": str(self.data_path),
                "validation_date": datetime.now().isoformat(),
                "validator_version": "1.0.0"
            },
            "summary": {
                "total_records": self.validation_results["total_records"],
                "valid_records": self.validation_results["valid_records"],
                "validation_rate": (
                    self.validation_results["valid_records"] / 
                    self.validation_results["total_records"] * 100
                ) if self.validation_results["total_records"] > 0 else 0,
                "total_issues": sum(len(issues) for issues in self.validation_results["issues"].values()),
                "ml_readiness_score": self.validation_results.get("ml_readiness", {}).get("overall_score", 0)
            },
            "issues": dict(self.validation_results["issues"]),
            "statistics": self.validation_results["statistics"],
            "ml_readiness": self.validation_results.get("ml_readiness", {}),
            "recommendations": self.validation_results["recommendations"]
        }
        
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"\nReport saved to: {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Records: {report['summary']['total_records']}")
        print(f"Valid Records: {report['summary']['valid_records']} ({report['summary']['validation_rate']:.1f}%)")
        print(f"Total Issues: {report['summary']['total_issues']}")
        print(f"ML Readiness Score: {report['summary']['ml_readiness_score']:.1f}/100")
        print(f"ML Readiness Rating: {self.validation_results.get('ml_readiness', {}).get('rating', 'N/A')}")
        
        if self.validation_results["issues"]:
            print("\nTop Issues:")
            sorted_issues = sorted(
                self.validation_results["issues"].items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:5]
            
            for issue_type, issues in sorted_issues:
                print(f"  - {issue_type}: {len(issues)} occurrences")
                
        if self.validation_results["recommendations"]:
            print("\nTop Recommendations:")
            high_priority = [r for r in self.validation_results["recommendations"] if r["priority"] == "HIGH"]
            for rec in high_priority[:3]:
                print(f"  - [{rec['category']}] {rec['recommendation']}")


if __name__ == "__main__":
    # Run validation
    validator = CPTDataValidator()
    validator.run_validation()
    validator.generate_report()