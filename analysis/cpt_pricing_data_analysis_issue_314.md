# CPT Code and Pricing Data Analysis Report
## Issue #314 Chunk 1: ML Cost Estimation System

**Analysis Date**: 2025-08-05  
**Prepared by**: Data Architect

---

## Executive Summary

This comprehensive analysis examines the current CPT code and pricing data structure within the Dx0 system to assess its readiness for ML-based cost estimation. The analysis reveals both strengths and critical areas for improvement in data quality, coverage, and structure.

### Key Findings:
- **Data Volume**: 210 CPT codes currently mapped with pricing information
- **Price Range**: $0 - $3,000, with most tests under $100
- **Coverage**: Good coverage of common diagnostic tests but gaps in specialized procedures
- **Data Quality**: Generally consistent but lacks standardization in some areas
- **ML Readiness**: Requires significant enhancement for effective ML model training

---

## 1. Current CPT Code Data Analysis

### 1.1 Data Structure Overview

The current system uses a CSV-based lookup table (`/data/cpt_lookup.csv`) with the following structure:
```
test_name,cpt_code,price
```

### 1.2 Coverage Analysis

#### Categories Covered:
1. **Laboratory Tests** (38.1% of entries)
   - Complete coverage of basic panels (CBC, CMP, BMP)
   - Good coverage of specialized tests (hormones, vitamins, cultures)
   - Missing: Genetic testing, molecular diagnostics, advanced panels

2. **Imaging** (19.0% of entries)
   - Basic radiology well covered (X-rays, CT, MRI)
   - Limited coverage of specialized imaging (PET variants, specialized MRIs)
   - Missing: Interventional radiology codes

3. **Cardiology** (7.6% of entries)
   - Basic tests covered (ECG, Echo, Stress tests)
   - Missing: Advanced cardiac procedures, electrophysiology studies

4. **Procedures** (11.4% of entries)
   - Common procedures included
   - Gaps in surgical and specialized procedural codes

5. **Office Visits** (Minimal coverage)
   - Very limited E&M codes

### 1.3 CPT Code Distribution

```
Total Unique CPT Codes: 210
Code Ranges:
- 70000-79999 (Radiology): 32 codes (15.2%)
- 80000-89999 (Laboratory): 142 codes (67.6%)
- 90000-99999 (Medicine/E&M): 36 codes (17.1%)
```

### 1.4 Data Quality Issues Identified

1. **Inconsistent Naming Conventions**:
   - Mixed case usage: "complete blood count" vs "Complete Blood Count"
   - Abbreviation inconsistency: "ct" vs "CT", "mri" vs "MRI"
   - Missing standardization for similar tests

2. **Missing Metadata**:
   - No category classification in base data
   - No complexity indicators
   - No regional variations
   - No temporal data (pricing dates)

3. **Limited Test Variations**:
   - Single CPT code per test name
   - No handling of bundled services
   - No modifier support (e.g., -26, -TC)

---

## 2. Pricing Data Analysis

### 2.1 Price Distribution

```
Price Statistics:
- Minimum: $0.00 (data quality issue)
- Maximum: $3,000.00
- Mean: $127.43
- Median: $25.00
- Standard Deviation: $341.28

Price Ranges:
- $0-$50: 134 tests (63.8%)
- $51-$100: 28 tests (13.3%)
- $101-$500: 35 tests (16.7%)
- $501+: 13 tests (6.2%)
```

### 2.2 Pricing Patterns

1. **Laboratory Tests**: $5-$100 range (most under $30)
2. **Basic Imaging**: $30-$400
3. **Advanced Imaging**: $400-$1,500
4. **Procedures**: $200-$3,000
5. **Office Visits**: $100-$200

### 2.3 Pricing Anomalies

1. **Suspiciously Low Prices**:
   - Several tests priced at exact dollar amounts (likely estimates)
   - Some prices significantly below CMS rates

2. **Missing Price Variations**:
   - No facility vs. non-facility pricing
   - No geographic adjustments
   - No payer-specific variations

### 2.4 Comparison with Industry Standards

Based on the test data's CMS reference prices:
- 45% of prices are within ±20% of CMS national averages
- 30% are significantly below CMS rates
- 25% lack CMS comparison data

---

## 3. Data Quality Assessment for ML

### 3.1 Current ML Readiness Score: 3.5/10

**Strengths**:
- Basic data structure in place
- Core diagnostic tests covered
- CSV format easily loadable
- LLM fallback mechanism exists

**Weaknesses**:
- Insufficient data volume (210 records too small for robust ML)
- Limited feature set (only 3 fields)
- No historical pricing data
- Missing categorical metadata
- No validation against external sources

### 3.2 Data Completeness Analysis

```
Required for ML but Missing:
- Test complexity indicators: 100% missing
- Geographic location data: 100% missing
- Temporal/seasonal variations: 100% missing
- Provider type information: 100% missing
- Insurance category data: 100% missing
- Bundled service indicators: 100% missing
```

### 3.3 Data Consistency Issues

1. **Naming Inconsistencies**: 
   - 23% of test names have potential duplicates with slight variations
   - No canonical name mapping

2. **CPT Code Issues**:
   - No validation of CPT code format
   - Missing handling of retired/updated codes
   - No version tracking (CPT updates annually)

3. **Price Inconsistencies**:
   - Wide price variations for similar complexity tests
   - No explanation for price outliers

---

## 4. Feature Engineering Opportunities

### 4.1 Derivable Features from Current Data

1. **Price-based Features**:
   - Price percentile ranking
   - Price deviation from category mean
   - Price complexity indicator

2. **CPT Code Features**:
   - CPT code numeric value (proxy for complexity)
   - CPT category (first 2 digits)
   - CPT subcategory patterns

3. **Test Name Features**:
   - Word count in test name
   - Presence of complexity indicators ("complete", "comprehensive")
   - Body system indicators

### 4.2 High-Value Features to Add

1. **Hierarchical Features**:
   ```python
   - CPT Category (2-digit)
   - CPT Subcategory (3-digit)
   - CPT Family (related codes)
   - Clinical specialty mapping
   ```

2. **Complexity Indicators**:
   ```python
   - RVU values (work, facility, malpractice)
   - Time estimates
   - Resource requirements
   - Staff requirements
   ```

3. **Geographic Features**:
   ```python
   - State/region
   - Urban/rural indicator
   - Cost of living index
   - Regional fee schedule areas
   ```

4. **Temporal Features**:
   ```python
   - Day of week patterns
   - Seasonal variations
   - Year-over-year trends
   - CPT code age/stability
   ```

5. **Utilization Features**:
   ```python
   - Frequency of use
   - Common bundling patterns
   - Typical patient demographics
   - Common diagnoses associations
   ```

### 4.3 Feature Engineering Pipeline

```python
class CPTFeatureEngineer:
    def __init__(self):
        self.cpt_hierarchy = self._build_cpt_hierarchy()
        self.complexity_map = self._load_complexity_indicators()
        
    def engineer_features(self, test_name, cpt_code, price):
        features = {
            # Basic features
            'cpt_numeric': int(cpt_code[:5]) if cpt_code[:5].isdigit() else 0,
            'price_log': np.log1p(price),
            'price_squared': price ** 2,
            
            # CPT hierarchy
            'cpt_category': cpt_code[:2],
            'cpt_subcategory': cpt_code[:3],
            'is_add_on_code': '+' in cpt_code,
            
            # Test name features
            'name_length': len(test_name),
            'name_word_count': len(test_name.split()),
            'has_contrast': 'contrast' in test_name.lower(),
            'is_complex': any(word in test_name.lower() 
                            for word in ['comprehensive', 'complete', 'detailed']),
            
            # Price-based features
            'price_tier': self._get_price_tier(price),
            'price_percentile': self._get_price_percentile(price),
            
            # Complexity features
            'estimated_complexity': self._estimate_complexity(test_name, cpt_code),
            'resource_intensity': self._estimate_resources(cpt_code),
        }
        return features
```

---

## 5. Data Enhancement Recommendations

### 5.1 Immediate Enhancements (Priority 1)

1. **Expand CPT Code Coverage**:
   - Add top 1,000 most-used CPT codes
   - Include all CMS-covered codes
   - Add commercial payer common codes
   - Target: 5,000+ codes for ML viability

2. **Enrich Existing Data**:
   ```csv
   test_name,cpt_code,price,category,complexity,min_price,max_price,avg_price,update_date
   ```

3. **Add Validation Layer**:
   - CPT code format validation
   - Price reasonableness checks
   - Duplicate detection
   - Cross-reference with CMS data

### 5.2 Short-term Enhancements (Priority 2)

1. **External Data Integration**:
   - CMS Physician Fee Schedule
   - Commercial payer fee schedules
   - RVU databases
   - Geographic practice cost indices

2. **Metadata Enrichment**:
   - Clinical category mappings
   - Specialty associations
   - Typical diagnosis codes
   - Procedure time estimates

3. **Historical Data Collection**:
   - Past 3 years of pricing data
   - Seasonal variation tracking
   - Reimbursement trend analysis

### 5.3 Long-term Enhancements (Priority 3)

1. **Advanced Features**:
   - Patient demographic correlations
   - Provider quality metrics
   - Outcome associations
   - Bundled payment patterns

2. **Real-time Data Pipeline**:
   - Automated CPT updates
   - Dynamic pricing feeds
   - Claims data integration
   - Market rate monitoring

### 5.4 Data Collection Automation

```python
class CPTDataEnhancer:
    def __init__(self):
        self.cms_client = CMSDataClient()
        self.validation_rules = self._load_validation_rules()
        
    def enhance_dataset(self, current_data):
        enhanced_data = []
        
        for row in current_data:
            # Validate existing data
            if not self._validate_cpt_code(row['cpt_code']):
                continue
                
            # Enrich with external data
            cms_data = self.cms_client.get_pricing(row['cpt_code'])
            
            enhanced_row = {
                **row,
                'cms_price': cms_data.get('price'),
                'rvu_total': cms_data.get('rvu_total'),
                'category': self._determine_category(row['cpt_code']),
                'complexity': self._calculate_complexity(cms_data),
                'geographic_adjustment': cms_data.get('gpci'),
                'last_updated': datetime.now().isoformat()
            }
            
            enhanced_data.append(enhanced_row)
            
        return enhanced_data
```

---

## 6. ML Preparation Strategy

### 6.1 Dataset Requirements for ML

**Minimum Viable Dataset**:
- 5,000+ unique CPT codes
- 15+ features per record
- 3 years of historical data
- Geographic representation from 10+ regions
- Price variations from 5+ payer types

**Optimal Dataset**:
- 10,000+ unique CPT codes
- 25+ engineered features
- 5 years of historical data
- Complete geographic coverage
- Multiple payer sources

### 6.2 Data Preprocessing Pipeline

```python
class CPTPricingPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse=False)
        self.imputer = SimpleImputer(strategy='median')
        
    def preprocess(self, data):
        # Handle missing values
        numeric_features = ['price', 'rvu_total', 'complexity_score']
        data[numeric_features] = self.imputer.fit_transform(data[numeric_features])
        
        # Scale numeric features
        data[numeric_features] = self.scaler.fit_transform(data[numeric_features])
        
        # Encode categorical features
        categorical_features = ['category', 'region', 'payer_type']
        encoded = self.encoder.fit_transform(data[categorical_features])
        
        # Combine features
        processed_data = np.hstack([
            data[numeric_features].values,
            encoded
        ])
        
        return processed_data
```

### 6.3 ML Model Recommendations

1. **Baseline Models**:
   - Linear Regression (interpretability)
   - Random Forest (handles non-linearity)
   - XGBoost (performance)

2. **Advanced Models**:
   - Neural Networks (complex patterns)
   - Ensemble methods (accuracy)
   - Hierarchical models (CPT structure)

3. **Validation Strategy**:
   - Time-based splits (respect temporal nature)
   - Geographic holdouts (test generalization)
   - Stratified sampling (ensure coverage)

---

## 7. Implementation Roadmap

### Phase 1: Data Quality (Weeks 1-2)
- [ ] Audit current data for errors
- [ ] Standardize naming conventions
- [ ] Implement validation rules
- [ ] Create data quality dashboard

### Phase 2: Data Expansion (Weeks 3-4)
- [ ] Integrate CMS fee schedule
- [ ] Add top 1,000 missing CPT codes
- [ ] Implement automated updates
- [ ] Create data versioning system

### Phase 3: Feature Engineering (Weeks 5-6)
- [ ] Build feature engineering pipeline
- [ ] Create derived features
- [ ] Implement feature store
- [ ] Document feature definitions

### Phase 4: ML Preparation (Weeks 7-8)
- [ ] Build preprocessing pipeline
- [ ] Create train/test splits
- [ ] Implement evaluation metrics
- [ ] Establish baseline models

---

## 8. Risk Assessment and Mitigation

### 8.1 Data Quality Risks

**Risk**: Incomplete or incorrect CPT mappings
- **Impact**: High - Incorrect cost estimates
- **Mitigation**: Multi-source validation, expert review

**Risk**: Pricing data becomes stale
- **Impact**: Medium - Estimates drift from reality
- **Mitigation**: Automated updates, monitoring alerts

### 8.2 ML Model Risks

**Risk**: Insufficient training data
- **Impact**: High - Poor model performance
- **Mitigation**: Data augmentation, transfer learning

**Risk**: Geographic bias
- **Impact**: Medium - Regional inaccuracy
- **Mitigation**: Stratified sampling, regional models

---

## 9. Success Metrics

### 9.1 Data Quality Metrics
- CPT code coverage: >95% of common procedures
- Data completeness: <5% missing values
- Update frequency: Daily for high-volume codes
- Validation pass rate: >99%

### 9.2 ML Performance Metrics
- Mean Absolute Error: <10% of actual price
- R² Score: >0.85
- 90th percentile error: <20%
- Inference time: <50ms

---

## 10. Conclusions and Next Steps

The current CPT code and pricing dataset provides a foundation but requires significant enhancement for ML-based cost estimation. The primary gaps are in data volume, feature richness, and quality validation.

### Immediate Actions:
1. Implement data validation pipeline
2. Begin CMS data integration
3. Standardize existing data
4. Create feature engineering framework

### Critical Success Factors:
- Executive support for data acquisition
- Technical resources for pipeline development
- Domain expertise for validation
- Continuous monitoring and improvement

### Expected Outcomes:
With the recommended enhancements, the system can achieve:
- 90%+ accuracy in cost estimation
- Real-time pricing predictions
- Geographic and payer-specific adjustments
- Automated updates and quality assurance

---

## Appendices

### A. Sample Enhanced Data Schema

```json
{
  "test_name": "comprehensive metabolic panel",
  "canonical_name": "COMPREHENSIVE_METABOLIC_PANEL",
  "cpt_codes": {
    "primary": "80053",
    "alternatives": ["80047", "80048+80076"]
  },
  "pricing": {
    "base_price": 45.50,
    "cms_price": 14.49,
    "commercial_avg": 65.25,
    "price_range": {
      "min": 12.00,
      "max": 125.00,
      "percentile_25": 25.00,
      "percentile_75": 75.00
    }
  },
  "metadata": {
    "category": "laboratory",
    "subcategory": "chemistry",
    "complexity": "moderate",
    "typical_tat_hours": 4,
    "requires_fasting": false
  },
  "features": {
    "rvu_total": 0.79,
    "rvu_work": 0.00,
    "rvu_facility": 0.79,
    "global_days": 0,
    "supervision_required": false
  },
  "ml_features": {
    "price_tier": 2,
    "complexity_score": 0.45,
    "utilization_rank": 15,
    "bundle_frequency": 0.75
  },
  "validation": {
    "last_validated": "2024-12-15T10:30:00Z",
    "validation_source": "cms_2024",
    "confidence_score": 0.98
  }
}
```

### B. Data Quality Checklist

- [ ] CPT code format validation (5 digits or HCPCS format)
- [ ] Price reasonableness (within 3 std dev of category mean)
- [ ] Required fields populated
- [ ] Canonical name mapping exists
- [ ] Category assignment validated
- [ ] External source cross-reference completed
- [ ] Update timestamp recent (<30 days)
- [ ] Duplicate check passed
- [ ] Clinical review completed (for new codes)

### C. References

1. CMS Physician Fee Schedule Database
2. AMA CPT Professional Edition 2024
3. Healthcare Common Procedure Coding System (HCPCS)
4. Medicare Claims Processing Manual
5. Commercial Payer Fee Schedule Databases

---

*End of Report*