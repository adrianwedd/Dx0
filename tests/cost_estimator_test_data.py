"""
Test data fixtures and validation datasets for CostEstimator testing.

This module provides comprehensive test datasets for validating cost estimation
accuracy, CPT code handling, and performance across various healthcare scenarios.
"""

import csv
import json
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from decimal import Decimal

from sdb.cost_estimator import CptCost


@dataclass
class TestCostEntry:
    """Structured test case for cost estimation validation."""
    test_name: str
    cpt_code: str
    price: float
    category: str
    description: str
    complexity: str  # simple, moderate, complex
    expected_price_range: Tuple[float, float]


@dataclass 
class CPTValidationCase:
    """Test case for CPT code validation."""
    cpt_code: str
    is_valid: bool
    code_type: str  # standard, hcpcs, category_iii, modifier
    description: str


@dataclass
class PricingScenario:
    """Test scenario for pricing validation."""
    scenario_name: str
    test_entries: List[TestCostEntry]
    expected_coverage: float
    description: str


# ============================================================================
# COMPREHENSIVE TEST COST DATA
# ============================================================================

# Laboratory Tests
LABORATORY_TESTS = [
    TestCostEntry(
        test_name="complete blood count",
        cpt_code="85027",
        price=25.50,
        category="laboratory",
        description="Standard CBC with differential",
        complexity="simple",
        expected_price_range=(20.0, 35.0)
    ),
    TestCostEntry(
        test_name="basic metabolic panel",
        cpt_code="80048",
        price=15.75,
        category="laboratory", 
        description="BMP with glucose, electrolytes, kidney function",
        complexity="simple",
        expected_price_range=(12.0, 25.0)
    ),
    TestCostEntry(
        test_name="comprehensive metabolic panel",
        cpt_code="80053",
        price=18.50,
        category="laboratory",
        description="CMP including liver function tests",
        complexity="simple",
        expected_price_range=(15.0, 30.0)
    ),
    TestCostEntry(
        test_name="lipid panel",
        cpt_code="80061",
        price=22.00,
        category="laboratory",
        description="Total cholesterol, HDL, LDL, triglycerides",
        complexity="simple",
        expected_price_range=(18.0, 35.0)
    ),
    TestCostEntry(
        test_name="hemoglobin a1c",
        cpt_code="83036",
        price=28.75,
        category="laboratory",
        description="Glycated hemoglobin for diabetes monitoring",
        complexity="simple",
        expected_price_range=(20.0, 40.0)
    ),
    TestCostEntry(
        test_name="thyroid stimulating hormone",
        cpt_code="84443",
        price=35.25,
        category="laboratory",
        description="TSH for thyroid function assessment",
        complexity="simple",
        expected_price_range=(25.0, 50.0)
    ),
    TestCostEntry(
        test_name="vitamin d 25-hydroxy",
        cpt_code="82306",
        price=42.50,
        category="laboratory",
        description="25-hydroxyvitamin D measurement",
        complexity="moderate",
        expected_price_range=(35.0, 60.0)
    ),
    TestCostEntry(
        test_name="prostate specific antigen",
        cpt_code="84153",
        price=31.75,
        category="laboratory",
        description="PSA screening for prostate cancer",
        complexity="simple",
        expected_price_range=(25.0, 45.0)
    ),
]

# Imaging Tests
IMAGING_TESTS = [
    TestCostEntry(
        test_name="chest x-ray 2 views",
        cpt_code="71046",
        price=89.00,
        category="imaging",
        description="Chest X-ray PA and lateral views",
        complexity="simple",
        expected_price_range=(60.0, 120.0)
    ),
    TestCostEntry(
        test_name="ct head without contrast",
        cpt_code="70450",
        price=425.00,
        category="imaging",
        description="CT scan of head without IV contrast",
        complexity="moderate",
        expected_price_range=(300.0, 600.0)
    ),
    TestCostEntry(
        test_name="ct head with contrast",
        cpt_code="70460",
        price=525.00,
        category="imaging",
        description="CT scan of head with IV contrast",
        complexity="moderate",
        expected_price_range=(400.0, 750.0)
    ),
    TestCostEntry(
        test_name="mri brain without contrast",
        cpt_code="70551",
        price=1150.00,
        category="imaging",
        description="MRI brain without gadolinium contrast",
        complexity="complex",
        expected_price_range=(900.0, 1500.0)
    ),
    TestCostEntry(
        test_name="mri brain with contrast",
        cpt_code="70553",
        price=1350.00,
        category="imaging",
        description="MRI brain with gadolinium contrast",
        complexity="complex",
        expected_price_range=(1000.0, 1800.0)
    ),
    TestCostEntry(
        test_name="mammography bilateral",
        cpt_code="77067",
        price="185.50",
        category="imaging",
        description="Bilateral screening mammography",
        complexity="moderate",
        expected_price_range=(120.0, 250.0)
    ),
    TestCostEntry(
        test_name="ultrasound abdomen complete",
        cpt_code="76700",
        price=245.75,
        category="imaging",
        description="Complete abdominal ultrasound",
        complexity="moderate", 
        expected_price_range=(180.0, 350.0)
    ),
]

# Cardiology Tests
CARDIOLOGY_TESTS = [
    TestCostEntry(
        test_name="electrocardiogram",
        cpt_code="93000",
        price=45.25,
        category="cardiology",
        description="12-lead ECG with interpretation",
        complexity="simple",
        expected_price_range=(30.0, 65.0)
    ),
    TestCostEntry(
        test_name="echocardiogram transthoracic",
        cpt_code="93307",
        price=275.50,
        category="cardiology",
        description="Complete transthoracic echocardiogram",
        complexity="moderate",
        expected_price_range=(200.0, 400.0)
    ),
    TestCostEntry(
        test_name="stress test treadmill",
        cpt_code="93017",
        price=350.00,
        category="cardiology",
        description="Treadmill stress test with ECG monitoring",
        complexity="moderate",
        expected_price_range=(250.0, 500.0)
    ),
    TestCostEntry(
        test_name="holter monitor 24 hour",
        cpt_code="93224",
        price="195.25",
        category="cardiology",
        description="24-hour Holter monitor recording",
        complexity="moderate",
        expected_price_range=(150.0, 300.0)
    ),
]

# Procedure Tests
PROCEDURE_TESTS = [
    TestCostEntry(
        test_name="colonoscopy screening",
        cpt_code="45378",
        price=850.00,
        category="procedure",
        description="Screening colonoscopy with biopsy if needed",
        complexity="complex",
        expected_price_range=(600.0, 1200.0)
    ),
    TestCostEntry(
        test_name="upper endoscopy",
        cpt_code="43235",
        price=485.75,
        category="procedure",
        description="Upper endoscopy (EGD) diagnostic",
        complexity="moderate",
        expected_price_range=(350.0, 700.0)
    ),
    TestCostEntry(
        test_name="bone marrow biopsy",
        cpt_code="38221",
        price=725.50,
        category="procedure",
        description="Bone marrow biopsy and aspiration",
        complexity="complex",
        expected_price_range=(500.0, 1000.0)
    ),
    TestCostEntry(
        test_name="skin biopsy punch",
        cpt_code="11104",
        price=125.25,
        category="procedure",
        description="Punch biopsy of skin lesion",
        complexity="simple",
        expected_price_range=(80.0, 180.0)
    ),
]

# Office Visit Tests
OFFICE_VISIT_TESTS = [
    TestCostEntry(
        test_name="office visit new patient level 3",
        cpt_code="99203",
        price=150.00,
        category="office_visit",
        description="New patient visit, moderate complexity",
        complexity="moderate",
        expected_price_range=(120.0, 200.0)
    ),
    TestCostEntry(
        test_name="office visit established patient level 4",
        cpt_code="99214",
        price=125.50,
        category="office_visit",
        description="Established patient visit, high complexity",
        complexity="moderate",
        expected_price_range=(100.0, 170.0)
    ),
]

# Combine all test data
ALL_TEST_ENTRIES = (
    LABORATORY_TESTS + 
    IMAGING_TESTS + 
    CARDIOLOGY_TESTS + 
    PROCEDURE_TESTS + 
    OFFICE_VISIT_TESTS
)


# ============================================================================
# CPT CODE VALIDATION TEST CASES
# ============================================================================

CPT_VALIDATION_CASES = [
    # Valid Standard CPT Codes (Category I)
    CPTValidationCase("85027", True, "standard", "CBC with differential"),
    CPTValidationCase("99213", True, "standard", "Office visit established patient"),
    CPTValidationCase("71046", True, "standard", "Chest X-ray 2 views"),
    CPTValidationCase("70553", True, "standard", "MRI brain with contrast"),
    CPTValidationCase("93307", True, "standard", "Echocardiogram complete"),
    
    # Valid HCPCS Level II Codes
    CPTValidationCase("G0463", True, "hcpcs", "Hospital outpatient clinic visit"),
    CPTValidationCase("A4217", True, "hcpcs", "Sterile water for injection"),
    CPTValidationCase("J1745", True, "hcpcs", "Infliximab injection"),
    CPTValidationCase("S0265", True, "hcpcs", "Genetic counseling session"),
    
    # Valid Category III CPT Codes (Emerging Technology)
    CPTValidationCase("0001T", True, "category_iii", "Emerging technology code"),
    CPTValidationCase("0584T", True, "category_iii", "Fractional exhaled nitric oxide"),
    CPTValidationCase("0639T", True, "category_iii", "Wireless capsule endoscopy"),
    
    # Valid Modifier Codes
    CPTValidationCase("85027-26", True, "modifier", "CBC professional component"),
    CPTValidationCase("71046-TC", True, "modifier", "Chest X-ray technical component"),
    
    # Invalid Codes - Format Issues
    CPTValidationCase("1234", False, "invalid", "Too short"),
    CPTValidationCase("123456", False, "invalid", "Too long"),
    CPTValidationCase("ABCDE", False, "invalid", "All letters, no valid pattern"),
    CPTValidationCase("85O27", False, "invalid", "Contains letter O instead of 0"),
    CPTValidationCase("", False, "invalid", "Empty code"),
    
    # Invalid Codes - Non-existent
    CPTValidationCase("00000", False, "invalid", "All zeros"),
    CPTValidationCase("99999", False, "invalid", "Likely non-existent high number"),
    CPTValidationCase("Z9999", False, "invalid", "Invalid HCPCS prefix"),
]


# ============================================================================
# PRICING SCENARIOS FOR COMPREHENSIVE TESTING
# ============================================================================

PRICING_SCENARIOS = [
    PricingScenario(
        scenario_name="basic_lab_panel",
        test_entries=LABORATORY_TESTS[:4],  # CBC, BMP, CMP, Lipid
        expected_coverage=1.0,
        description="Basic laboratory panel for annual physical"
    ),
    PricingScenario(
        scenario_name="cardiac_workup",
        test_entries=CARDIOLOGY_TESTS + [LABORATORY_TESTS[4]],  # All cardiac + A1C
        expected_coverage=1.0,
        description="Comprehensive cardiac evaluation"
    ),
    PricingScenario(
        scenario_name="imaging_battery",
        test_entries=IMAGING_TESTS[:3],  # Chest X-ray, CT head w/o and w/ contrast
        expected_coverage=1.0,
        description="Emergency department imaging workup"
    ),
    PricingScenario(
        scenario_name="preventive_screening",
        test_entries=[
            IMAGING_TESTS[5],  # Mammography
            PROCEDURE_TESTS[0],  # Colonoscopy screening
            LABORATORY_TESTS[7],  # PSA
        ],
        expected_coverage=1.0,
        description="Age-appropriate cancer screening tests"
    ),
    PricingScenario(
        scenario_name="mixed_complexity",
        test_entries=[
            entry for entry in ALL_TEST_ENTRIES 
            if entry.complexity in ["simple", "moderate", "complex"]
        ][:10],
        expected_coverage=1.0,
        description="Mixed complexity tests across all categories"
    ),
]


# ============================================================================
# PERFORMANCE TEST DATA GENERATORS
# ============================================================================

def generate_large_test_dataset(size: int = 1000) -> List[TestCostEntry]:
    """Generate a large dataset for performance testing."""
    test_entries = []
    categories = ["laboratory", "imaging", "cardiology", "procedure", "office_visit"]
    complexities = ["simple", "moderate", "complex"]
    
    for i in range(size):
        entry = TestCostEntry(
            test_name=f"test_{i:04d}",
            cpt_code=f"{85000 + i}",
            price=round(10.0 + (i % 500) + (i * 0.01), 2),
            category=categories[i % len(categories)],
            description=f"Generated test case {i}",
            complexity=complexities[i % len(complexities)],
            expected_price_range=(
                round(5.0 + (i % 400), 2),
                round(50.0 + (i % 1000), 2)
            )
        )
        test_entries.append(entry)
    
    return test_entries


def generate_cpt_code_variations() -> List[str]:
    """Generate various CPT code formats for validation testing."""
    codes = []
    
    # Standard 5-digit CPT codes
    for i in range(10000, 99999, 1000):
        codes.append(str(i))
    
    # HCPCS Level II codes (letter + 4 digits)
    prefixes = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'J', 'K', 'L', 'M', 'P', 'Q', 'R', 'S', 'T', 'V']
    for prefix in prefixes[:5]:  # Limit for testing
        for i in range(1000, 9999, 1000):
            codes.append(f"{prefix}{i}")
    
    # Category III codes (4 digits + T)
    for i in range(1, 999, 100):
        codes.append(f"{i:04d}T")
    
    return codes


# ============================================================================
# REAL-WORLD PRICING DATA VALIDATION
# ============================================================================

# CMS-based pricing validation data (approximate 2024 rates)
CMS_REFERENCE_PRICES = {
    "85027": {"min": 8.50, "max": 28.00, "national_avg": 18.25},   # CBC
    "80048": {"min": 7.25, "max": 22.50, "national_avg": 14.75},   # BMP
    "71046": {"min": 35.00, "max": 125.00, "national_avg": 68.50}, # Chest X-ray
    "70450": {"min": 180.00, "max": 550.00, "national_avg": 365.00}, # CT Head
    "93307": {"min": 125.00, "max": 400.00, "national_avg": 262.50}, # Echo
    "45378": {"min": 285.00, "max": 1200.00, "national_avg": 742.50}, # Colonoscopy
}


@dataclass
class CMSValidationEntry:
    """CMS pricing validation entry."""
    cpt_code: str
    test_name: str
    min_price: float
    max_price: float
    national_avg: float
    geographic_adjustment: float = 1.0


CMS_VALIDATION_DATA = [
    CMSValidationEntry("85027", "complete blood count", 8.50, 28.00, 18.25),
    CMSValidationEntry("80048", "basic metabolic panel", 7.25, 22.50, 14.75),
    CMSValidationEntry("71046", "chest x-ray 2 views", 35.00, 125.00, 68.50),
    CMSValidationEntry("70450", "ct head without contrast", 180.00, 550.00, 365.00),
    CMSValidationEntry("93307", "echocardiogram transthoracic", 125.00, 400.00, 262.50),
    CMSValidationEntry("45378", "colonoscopy screening", 285.00, 1200.00, 742.50),
]


# ============================================================================
# EDGE CASES AND ERROR CONDITIONS
# ============================================================================

EDGE_CASE_TEST_DATA = [
    # Price edge cases
    TestCostEntry("zero_price_test", "99999", 0.0, "test", "Zero price test", "simple", (0.0, 0.0)),
    TestCostEntry("high_price_test", "99998", 50000.0, "test", "Very expensive test", "complex", (40000.0, 60000.0)),
    TestCostEntry("decimal_precision", "99997", 123.456789, "test", "High precision price", "simple", (120.0, 130.0)),
    
    # Text edge cases
    TestCostEntry("unicode_test_naïve", "99996", 25.50, "test", "Unicode in name", "simple", (20.0, 30.0)),
    TestCostEntry("very long test name that exceeds normal expectations", "99995", 35.75, "test", "Long name test", "simple", (30.0, 40.0)),
    TestCostEntry("", "99994", 15.25, "test", "Empty name test", "simple", (10.0, 20.0)),
    
    # Category edge cases
    TestCostEntry("mixed_case_test", "99993", 45.00, "Laboratory", "Mixed case category", "simple", (40.0, 50.0)),
    TestCostEntry("special_chars_test", "99992", 55.25, "test-category", "Special chars in category", "simple", (50.0, 60.0)),
]


# ============================================================================
# UTILITY FUNCTIONS FOR TEST DATA MANAGEMENT
# ============================================================================

def create_csv_from_test_entries(entries: List[TestCostEntry], 
                                include_category: bool = True,
                                include_expected_range: bool = False) -> str:
    """Create CSV content from test entries."""
    fieldnames = ["test_name", "cpt_code", "price"]
    if include_category:
        fieldnames.append("category")
    if include_expected_range:
        fieldnames.extend(["min_price", "max_price"])
    
    output = []
    output.append(",".join(fieldnames))
    
    for entry in entries:
        row = [entry.test_name, entry.cpt_code, str(entry.price)]
        if include_category:
            row.append(entry.category)
        if include_expected_range:
            row.extend([str(entry.expected_price_range[0]), str(entry.expected_price_range[1])])
        output.append(",".join(row))
    
    return "\n".join(output)


def create_temp_csv_file(entries: List[TestCostEntry], **kwargs) -> str:
    """Create a temporary CSV file with test entries."""
    csv_content = create_csv_from_test_entries(entries, **kwargs)
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_file.write(csv_content)
    temp_file.close()
    
    return temp_file.name


def create_cms_pricing_csv(validation_data: List[CMSValidationEntry]) -> str:
    """Create CMS pricing CSV for coverage validation."""
    output = ["cpt_code,national_avg,min_price,max_price"]
    
    for entry in validation_data:
        row = f"{entry.cpt_code},{entry.national_avg},{entry.min_price},{entry.max_price}"
        output.append(row)
    
    return "\n".join(output)


def get_test_entries_by_category(category: str) -> List[TestCostEntry]:
    """Get test entries for a specific category."""
    return [entry for entry in ALL_TEST_ENTRIES if entry.category == category]


def get_test_entries_by_complexity(complexity: str) -> List[TestCostEntry]:
    """Get test entries for a specific complexity level."""
    return [entry for entry in ALL_TEST_ENTRIES if entry.complexity == complexity]


def get_test_entries_by_price_range(min_price: float, 
                                   max_price: float) -> List[TestCostEntry]:
    """Get test entries within a specific price range."""
    return [
        entry for entry in ALL_TEST_ENTRIES 
        if min_price <= entry.price <= max_price
    ]


def validate_test_entry_prices() -> Dict[str, bool]:
    """Validate that test entry prices fall within expected ranges."""
    results = {}
    
    for entry in ALL_TEST_ENTRIES:
        min_expected, max_expected = entry.expected_price_range
        price_valid = min_expected <= entry.price <= max_expected
        results[entry.test_name] = price_valid
    
    return results


def get_benchmark_expectations() -> Dict[str, Any]:
    """Get performance benchmark expectations for testing."""
    return {
        "lookup_performance_threshold": 0.001,  # 1ms per lookup
        "bulk_load_threshold": 5.0,  # 5 seconds for 1000 entries
        "memory_usage_per_entry": 1024,  # ~1KB per entry estimate
        "concurrent_success_rate": 1.0,  # 100% success rate
        "cms_coverage_threshold": 0.95,  # 95% coverage minimum
        "price_accuracy_tolerance": 0.01,  # $0.01 tolerance
    }


# ============================================================================
# MOCK DATA FOR TESTING EXTERNAL INTEGRATIONS
# ============================================================================

MOCK_CMS_RESPONSE_DATA = {
    "valid_response": {
        "85027": 18.25,
        "80048": 14.75,
        "71046": 68.50,
        "70450": 365.00,
        "93307": 262.50,
    },
    "partial_response": {
        "85027": 18.25,
        "80048": 14.75,
        # Missing other codes to test coverage
    },
    "error_response": {
        "error": "Service unavailable",
        "code": 503
    },
    "malformed_response": {
        "85027": "invalid_price",
        "80048": None,
        "71046": "",
    }
}


MOCK_LLM_RESPONSES = {
    "successful_lookups": {
        "blood work": "85027",
        "cbc": "85027",
        "metabolic panel": "80048",
        "chest xray": "71046",
        "brain scan": "70450",
    },
    "failed_lookups": {
        "unknown test": None,
        "invalid request": None,
        "gibberish": None,
    },
    "ambiguous_responses": {
        "cardiac test": "93307",  # Could be ECG, Echo, etc.
        "blood test": "85027",   # Could be many different tests
    }
}


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_all_test_data_json(filepath: str) -> None:
    """Export all test data to JSON for external use."""
    data = {
        "laboratory_tests": [asdict(entry) for entry in LABORATORY_TESTS],
        "imaging_tests": [asdict(entry) for entry in IMAGING_TESTS],
        "cardiology_tests": [asdict(entry) for entry in CARDIOLOGY_TESTS],
        "procedure_tests": [asdict(entry) for entry in PROCEDURE_TESTS],
        "office_visit_tests": [asdict(entry) for entry in OFFICE_VISIT_TESTS],
        "cpt_validation_cases": [asdict(case) for case in CPT_VALIDATION_CASES],
        "pricing_scenarios": [asdict(scenario) for scenario in PRICING_SCENARIOS],
        "cms_validation_data": [asdict(entry) for entry in CMS_VALIDATION_DATA],
        "edge_cases": [asdict(entry) for entry in EDGE_CASE_TEST_DATA],
        "benchmark_expectations": get_benchmark_expectations(),
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_test_data_csv(filepath: str, entries: List[TestCostEntry]) -> None:
    """Export test entries to CSV format."""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["test_name", "cpt_code", "price", "category", "description", "complexity"])
        
        for entry in entries:
            writer.writerow([
                entry.test_name,
                entry.cpt_code,
                entry.price,
                entry.category,
                entry.description,
                entry.complexity
            ])


if __name__ == "__main__":
    # Example usage - export all test data
    export_all_test_data_json("cost_estimator_test_data.json")
    export_test_data_csv("cost_estimator_test_entries.csv", ALL_TEST_ENTRIES)
    
    # Validate test entry prices
    validation_results = validate_test_entry_prices()
    invalid_entries = [name for name, valid in validation_results.items() if not valid]
    
    if invalid_entries:
        print(f"WARNING: {len(invalid_entries)} test entries have prices outside expected ranges:")
        for name in invalid_entries:
            print(f"  - {name}")
    else:
        print("All test entry prices are within expected ranges.")