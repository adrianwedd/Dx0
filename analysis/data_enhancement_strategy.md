# CPT Data Enhancement Strategy for ML Cost Estimation
## Strategic Roadmap for Issue #314

---

## Executive Summary

This document outlines a comprehensive strategy to enhance the CPT code and pricing dataset from its current state (209 records, 3 features) to an ML-ready dataset (5,000+ records, 25+ features) capable of supporting accurate cost estimation models.

---

## Current State Assessment

### Data Volume
- **Current**: 209 CPT codes
- **ML Minimum**: 5,000 codes
- **Gap**: 4,791 codes (95.8%)

### Feature Set
- **Current**: 3 features (test_name, cpt_code, price)
- **ML Minimum**: 25 features
- **Gap**: 22 features (88%)

### Data Quality Score
- **Overall**: 62.5/100 (Good)
- **Breakdown**:
  - Completeness: 100%
  - Consistency: 94.3%
  - Volume: 4.2%
  - Features: 20%

---

## Enhancement Phases

### Phase 1: Immediate Data Expansion (Weeks 1-2)

#### 1.1 CMS Integration
```python
# Priority CPT codes to add from CMS
TOP_PRIORITY_CODES = {
    # E&M Codes (Office Visits)
    "99202-99205": "New Patient Office Visits",
    "99211-99215": "Established Patient Office Visits",
    "99281-99285": "Emergency Department Visits",
    
    # Common Procedures
    "36415": "Venipuncture",
    "90792": "Psychiatric Diagnostic Evaluation",
    "97110": "Physical Therapy",
    
    # High-Volume Lab Tests
    "87635": "COVID-19 Detection",
    "83036": "Hemoglobin A1c",
    "84443": "TSH",
    
    # Imaging Studies
    "74177": "CT Abdomen/Pelvis",
    "72148": "MRI Lumbar Spine",
    "76700": "Abdominal Ultrasound"
}
```

#### 1.2 Data Sources
1. **CMS Physician Fee Schedule** (Primary)
   - URL: https://www.cms.gov/medicare/payment/fee-schedules/physician
   - Contains: ~10,000 CPT codes with RVU values
   - Update frequency: Annual

2. **Commercial Payer Data** (Secondary)
   - Fair Health Consumer: https://www.fairhealthconsumer.org
   - Healthcare Bluebook: https://www.healthcarebluebook.com
   - Regional variations included

3. **RVU Database**
   - AMA RVU Data Manager
   - Work, Facility, Malpractice components
   - Geographic adjustments (GPCI)

#### 1.3 Automated Collection Script
```python
class DataCollectionPipeline:
    def __init__(self):
        self.sources = {
            'cms': CMSDataCollector(),
            'fair_health': FairHealthCollector(),
            'rvudata': RVUDataCollector()
        }
        
    def collect_cpt_data(self, cpt_code: str) -> Dict:
        """Collect comprehensive data for a CPT code."""
        data = {
            'cpt_code': cpt_code,
            'sources': {}
        }
        
        for source_name, collector in self.sources.items():
            try:
                source_data = collector.get_cpt_info(cpt_code)
                data['sources'][source_name] = source_data
            except Exception as e:
                logger.error(f"Failed to collect from {source_name}: {e}")
                
        # Aggregate data
        data['aggregated'] = self._aggregate_sources(data['sources'])
        data['confidence_score'] = self._calculate_confidence(data['sources'])
        
        return data
```

### Phase 2: Feature Engineering Implementation (Weeks 3-4)

#### 2.1 Core Feature Categories

1. **CPT Hierarchical Features**
   ```python
   features = {
       'cpt_section': extract_section(cpt_code),  # 00-99
       'cpt_subsection': extract_subsection(cpt_code),  # 000-999
       'cpt_category': map_to_category(cpt_code),  # Radiology, Lab, etc.
       'cpt_subcategory': map_to_subcategory(cpt_code),  # Detailed classification
   }
   ```

2. **Complexity Indicators**
   ```python
   features = {
       'rvu_work': rvu_data.get('work_rvu', 0),
       'rvu_practice': rvu_data.get('practice_expense_rvu', 0),
       'rvu_malpractice': rvu_data.get('malpractice_rvu', 0),
       'rvu_total': sum([work, practice, malpractice]),
       'global_period': cpt_data.get('global_days', 0),
       'is_major_procedure': global_period > 10,
       'complexity_score': calculate_complexity(rvu_total, global_period)
   }
   ```

3. **Pricing Features**
   ```python
   features = {
       'cms_national_price': cms_data.get('national_payment', 0),
       'commercial_avg_price': commercial_data.get('average', 0),
       'price_variance': calculate_variance(all_prices),
       'price_percentile_25': np.percentile(all_prices, 25),
       'price_percentile_75': np.percentile(all_prices, 75),
       'price_ratio_to_cms': price / cms_price if cms_price > 0 else 1
   }
   ```

4. **Temporal Features**
   ```python
   features = {
       'cpt_effective_year': get_cpt_introduction_year(cpt_code),
       'years_since_introduction': current_year - effective_year,
       'last_price_update': price_history.get('last_update'),
       'price_trend_1yr': calculate_trend(price_history, 1),
       'seasonal_factor': get_seasonal_factor(procedure_type, month)
   }
   ```

5. **Geographic Features**
   ```python
   features = {
       'gpci_work': geographic_data.get('work_gpci', 1.0),
       'gpci_practice': geographic_data.get('pe_gpci', 1.0),
       'gpci_malpractice': geographic_data.get('mp_gpci', 1.0),
       'locality_adjustment': calculate_locality_adjustment(gpcis),
       'urban_rural_indicator': location_data.get('rurality', 'urban'),
       'state_avg_adjustment': state_factors.get(state, 1.0)
   }
   ```

#### 2.2 Feature Store Architecture
```python
class CPTFeatureStore:
    def __init__(self):
        self.feature_definitions = self._load_feature_definitions()
        self.feature_cache = {}
        
    def compute_features(self, cpt_code: str, context: Dict) -> Dict:
        """Compute all features for a CPT code."""
        features = {}
        
        for feature_name, definition in self.feature_definitions.items():
            try:
                value = definition.compute(cpt_code, context)
                features[feature_name] = value
                
                # Cache computed features
                cache_key = f"{cpt_code}:{feature_name}"
                self.feature_cache[cache_key] = {
                    'value': value,
                    'timestamp': datetime.now(),
                    'version': definition.version
                }
            except Exception as e:
                logger.error(f"Failed to compute {feature_name}: {e}")
                features[feature_name] = None
                
        return features
```

### Phase 3: Data Quality Framework (Weeks 5-6)

#### 3.1 Validation Pipeline
```python
class DataValidationPipeline:
    def __init__(self):
        self.validators = [
            CPTFormatValidator(),
            PriceReasonablenessValidator(),
            RVUConsistencyValidator(),
            TemporalConsistencyValidator(),
            CrossSourceValidator()
        ]
        
    def validate_record(self, record: Dict) -> ValidationResult:
        """Run all validators on a record."""
        results = []
        
        for validator in self.validators:
            result = validator.validate(record)
            results.append(result)
            
        return ValidationResult(
            is_valid=all(r.is_valid for r in results),
            errors=[e for r in results for e in r.errors],
            warnings=[w for r in results for w in r.warnings],
            confidence_score=self._calculate_confidence(results)
        )
```

#### 3.2 Anomaly Detection
```python
class AnomalyDetector:
    def __init__(self):
        self.models = {
            'isolation_forest': IsolationForest(contamination=0.05),
            'local_outlier': LocalOutlierFactor(novelty=True),
            'statistical': StatisticalOutlierDetector()
        }
        
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in pricing data."""
        anomalies = pd.DataFrame(index=data.index)
        
        for name, model in self.models.items():
            # Prepare features for anomaly detection
            features = self._prepare_features(data)
            
            # Detect anomalies
            predictions = model.fit_predict(features)
            anomalies[f'{name}_anomaly'] = predictions == -1
            
        # Combine results
        anomalies['is_anomaly'] = anomalies.any(axis=1)
        anomalies['anomaly_score'] = anomalies.sum(axis=1) / len(self.models)
        
        return anomalies
```

### Phase 4: ML Pipeline Development (Weeks 7-8)

#### 4.1 Feature Processing Pipeline
```python
class MLFeaturePipeline:
    def __init__(self):
        self.numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ])
        
        self.categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', TargetEncoder(smooth=20))
        ])
        
        self.text_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=100)),
            ('svd', TruncatedSVD(n_components=10))
        ])
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Process features for ML training."""
        # Separate feature types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        text_features = ['test_name']  # Special handling for text
        
        # Process each type
        transformers = [
            ('numeric', self.numeric_pipeline, numeric_features),
            ('categorical', self.categorical_pipeline, categorical_features),
            ('text', self.text_pipeline, 'test_name')
        ]
        
        # Combine all transformers
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        return preprocessor.fit_transform(X, y)
```

#### 4.2 Model Architecture
```python
class CPTPricingModel:
    def __init__(self):
        self.models = {
            'baseline': LinearRegression(),
            'tree': XGBRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=8,
                subsample=0.8
            ),
            'neural': self._build_neural_network(),
            'ensemble': VotingRegressor([
                ('xgb', XGBRegressor()),
                ('rf', RandomForestRegressor()),
                ('lgb', LGBMRegressor())
            ])
        }
        
    def _build_neural_network(self):
        """Build neural network for price prediction."""
        model = Sequential([
            Dense(256, activation='relu', input_shape=(n_features,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mape']
        )
        
        return model
```

---

## Implementation Timeline

### Month 1: Foundation
- Week 1-2: CMS data integration
- Week 3-4: Feature engineering framework

### Month 2: Enhancement
- Week 5-6: Data quality systems
- Week 7-8: ML pipeline setup

### Month 3: Optimization
- Week 9-10: Model training and evaluation
- Week 11-12: Production deployment

---

## Success Metrics

### Data Quality Metrics
- CPT Code Coverage: >5,000 unique codes
- Feature Completeness: >95% non-null values
- Data Freshness: <30 days for high-volume codes
- Cross-source Agreement: >85% within 20% variance

### ML Performance Metrics
- Mean Absolute Error: <$50 or 10% of actual price
- R² Score: >0.85
- 90th Percentile Error: <20%
- Inference Latency: <50ms

### Business Impact Metrics
- Cost Estimation Accuracy: 90%+ within range
- User Satisfaction: >4.5/5 rating
- API Response Time: <100ms p95
- System Uptime: 99.9%

---

## Risk Mitigation

### Data Quality Risks
1. **Stale Pricing Data**
   - Mitigation: Automated daily updates
   - Monitoring: Freshness alerts

2. **Incomplete CPT Coverage**
   - Mitigation: Fallback to similar codes
   - Monitoring: Coverage reports

### Technical Risks
1. **Model Drift**
   - Mitigation: Continuous retraining
   - Monitoring: Performance tracking

2. **Scalability Issues**
   - Mitigation: Distributed processing
   - Monitoring: Load testing

---

## Budget Estimation

### Data Acquisition
- CMS Data: Free (public)
- Commercial Data: $5,000/year
- RVU Database: $2,000/year

### Infrastructure
- Cloud Compute: $500/month
- Storage: $200/month
- Monitoring: $300/month

### Development
- Engineering: 3 FTE × 3 months
- Data Science: 2 FTE × 3 months
- QA: 1 FTE × 3 months

### Total Estimated Cost
- One-time: $15,000 (data + setup)
- Recurring: $1,000/month
- Development: ~$150,000 (assuming $100k/year FTE)

---

## Conclusion

This comprehensive enhancement strategy will transform the current limited dataset into a robust, ML-ready foundation for accurate cost estimation. The phased approach ensures continuous value delivery while building toward the complete solution.

Key success factors:
1. Executive commitment to data acquisition
2. Dedicated engineering resources
3. Continuous monitoring and improvement
4. Strong validation framework

With proper execution, this strategy will enable 90%+ accuracy in cost estimation within 3 months.