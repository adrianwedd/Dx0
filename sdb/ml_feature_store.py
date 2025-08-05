#!/usr/bin/env python3
"""
Feature Store and Caching System for ML Cost Estimation
Efficient storage, retrieval, and caching of engineered features
"""

import os
import json
import pickle
import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata for stored features."""
    
    feature_name: str
    feature_type: str  # 'numerical', 'categorical', 'text', 'binary'
    description: str
    data_type: str
    created_at: datetime
    last_updated: datetime
    version: str = "1.0"
    source_columns: List[str] = None
    transformation_steps: List[str] = None
    quality_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.source_columns is None:
            self.source_columns = []
        if self.transformation_steps is None:
            self.transformation_steps = []
        if self.quality_metrics is None:
            self.quality_metrics = {}


@dataclass
class FeatureSet:
    """Container for a set of related features."""
    
    name: str
    features: Dict[str, np.ndarray]
    metadata: Dict[str, FeatureMetadata]
    target_column: Optional[str] = None
    sample_ids: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
            
    def to_dataframe(self) -> pd.DataFrame:
        """Convert feature set to pandas DataFrame."""
        df = pd.DataFrame(self.features)
        if self.sample_ids:
            df.index = self.sample_ids
        return df
        
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str, 
                      metadata: Optional[Dict[str, FeatureMetadata]] = None) -> 'FeatureSet':
        """Create FeatureSet from DataFrame."""
        features = {col: df[col].values for col in df.columns}
        sample_ids = df.index.tolist() if df.index.name else None
        
        if metadata is None:
            metadata = {}
            for col in df.columns:
                metadata[col] = FeatureMetadata(
                    feature_name=col,
                    feature_type='numerical' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical',
                    description=f"Feature: {col}",
                    data_type=str(df[col].dtype),
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
                
        return cls(
            name=name,
            features=features,
            metadata=metadata,
            sample_ids=sample_ids
        )


class FeatureCache:
    """In-memory cache for frequently accessed features."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._access_times = {}
        self._lock = threading.RLock()
        
    def _generate_key(self, data_hash: str, transformation_params: Dict) -> str:
        """Generate cache key from data and parameters."""
        params_str = json.dumps(transformation_params, sort_keys=True)
        combined = f"{data_hash}_{params_str}"
        return hashlib.md5(combined.encode()).hexdigest()
        
    def get(self, data_hash: str, transformation_params: Dict) -> Optional[Any]:
        """Retrieve cached features."""
        key = self._generate_key(data_hash, transformation_params)
        
        with self._lock:
            if key in self._cache:
                # Check TTL
                access_time = self._access_times.get(key, datetime.now())
                if datetime.now() - access_time < timedelta(seconds=self.ttl_seconds):
                    self._access_times[key] = datetime.now()
                    logger.debug(f"Cache hit for key: {key[:8]}...")
                    return self._cache[key]
                else:
                    # Expired
                    del self._cache[key]
                    del self._access_times[key]
                    
        return None
        
    def put(self, data_hash: str, transformation_params: Dict, features: Any):
        """Store features in cache."""
        key = self._generate_key(data_hash, transformation_params)
        
        with self._lock:
            # Implement LRU eviction if cache is full
            if len(self._cache) >= self.max_size:
                # Remove oldest accessed item
                oldest_key = min(self._access_times.keys(), 
                               key=lambda k: self._access_times[k])
                del self._cache[oldest_key]
                del self._access_times[oldest_key]
                
            self._cache[key] = features
            self._access_times[key] = datetime.now()
            logger.debug(f"Cached features for key: {key[:8]}...")
            
    def clear(self):
        """Clear all cached features."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class SQLiteFeatureStore:
    """SQLite-based feature store for persistent storage."""
    
    def __init__(self, db_path: str = "feature_store.db"):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Feature sets table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_sets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    sample_count INTEGER,
                    feature_count INTEGER
                )
            ''')
            
            # Individual features table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_set_id INTEGER,
                    feature_name TEXT NOT NULL,
                    feature_type TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    description TEXT,
                    version TEXT DEFAULT '1.0',
                    source_columns TEXT,
                    transformation_steps TEXT,
                    quality_metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (feature_set_id) REFERENCES feature_sets (id)
                )
            ''')
            
            # Feature data table (stores actual feature values)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_set_id INTEGER,
                    sample_id TEXT,
                    feature_data BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (feature_set_id) REFERENCES feature_sets (id)
                )
            ''')
            
            # Create indices for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feature_sets_name ON feature_sets (name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_features_set_name ON features (feature_set_id, feature_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feature_data_set_sample ON feature_data (feature_set_id, sample_id)')
            
            conn.commit()
            
    def store_feature_set(self, feature_set: FeatureSet) -> int:
        """Store a complete feature set."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert or update feature set
            cursor.execute('''
                INSERT OR REPLACE INTO feature_sets 
                (name, metadata, sample_count, feature_count, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                feature_set.name,
                json.dumps({"created_at": feature_set.created_at.isoformat()}),
                len(feature_set.sample_ids) if feature_set.sample_ids else len(next(iter(feature_set.features.values()))),
                len(feature_set.features)
            ))
            
            feature_set_id = cursor.lastrowid
            
            # Clear existing features for this set
            cursor.execute('DELETE FROM features WHERE feature_set_id = ?', (feature_set_id,))
            cursor.execute('DELETE FROM feature_data WHERE feature_set_id = ?', (feature_set_id,))
            
            # Store feature metadata
            for feature_name, metadata in feature_set.metadata.items():
                cursor.execute('''
                    INSERT INTO features 
                    (feature_set_id, feature_name, feature_type, data_type, description,
                     version, source_columns, transformation_steps, quality_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    feature_set_id,
                    metadata.feature_name,
                    metadata.feature_type,
                    metadata.data_type,
                    metadata.description,
                    metadata.version,
                    json.dumps(metadata.source_columns),
                    json.dumps(metadata.transformation_steps),
                    json.dumps(metadata.quality_metrics)
                ))
                
            # Store feature data
            feature_data_blob = pickle.dumps(feature_set.features)
            sample_ids_str = json.dumps(feature_set.sample_ids) if feature_set.sample_ids else None
            
            cursor.execute('''
                INSERT INTO feature_data (feature_set_id, sample_id, feature_data)
                VALUES (?, ?, ?)
            ''', (feature_set_id, sample_ids_str, feature_data_blob))
            
            conn.commit()
            logger.info(f"Stored feature set '{feature_set.name}' with {len(feature_set.features)} features")
            
            return feature_set_id
            
    def load_feature_set(self, name: str) -> Optional[FeatureSet]:
        """Load a feature set by name."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get feature set info
            cursor.execute('SELECT * FROM feature_sets WHERE name = ?', (name,))
            set_row = cursor.fetchone()
            
            if not set_row:
                return None
                
            feature_set_id = set_row[0]
            
            # Get feature metadata
            cursor.execute('SELECT * FROM features WHERE feature_set_id = ?', (feature_set_id,))
            feature_rows = cursor.fetchall()
            
            metadata = {}
            for row in feature_rows:
                feature_name = row[2]
                metadata[feature_name] = FeatureMetadata(
                    feature_name=feature_name,
                    feature_type=row[3],
                    data_type=row[4],
                    description=row[5],
                    version=row[6],
                    source_columns=json.loads(row[7]) if row[7] else [],
                    transformation_steps=json.loads(row[8]) if row[8] else [],
                    quality_metrics=json.loads(row[9]) if row[9] else {},
                    created_at=datetime.fromisoformat(row[10]),
                    last_updated=datetime.fromisoformat(row[10])
                )
                
            # Get feature data
            cursor.execute('SELECT sample_id, feature_data FROM feature_data WHERE feature_set_id = ?', 
                         (feature_set_id,))
            data_row = cursor.fetchone()
            
            if not data_row:
                return None
                
            sample_ids = json.loads(data_row[0]) if data_row[0] else None
            features = pickle.loads(data_row[1])
            
            feature_set = FeatureSet(
                name=name,
                features=features,
                metadata=metadata,
                sample_ids=sample_ids,
                created_at=datetime.fromisoformat(set_row[2])
            )
            
            logger.info(f"Loaded feature set '{name}' with {len(features)} features")
            return feature_set
            
    def list_feature_sets(self) -> List[Dict[str, Any]]:
        """List all stored feature sets."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT name, created_at, last_updated, sample_count, feature_count 
                FROM feature_sets ORDER BY last_updated DESC
            ''')
            
            return [
                {
                    'name': row[0],
                    'created_at': row[1],
                    'last_updated': row[2],
                    'sample_count': row[3],
                    'feature_count': row[4]
                }
                for row in cursor.fetchall()
            ]
            
    def delete_feature_set(self, name: str) -> bool:
        """Delete a feature set and all associated data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get feature set ID
            cursor.execute('SELECT id FROM feature_sets WHERE name = ?', (name,))
            result = cursor.fetchone()
            
            if not result:
                return False
                
            feature_set_id = result[0]
            
            # Delete all associated data
            cursor.execute('DELETE FROM feature_data WHERE feature_set_id = ?', (feature_set_id,))
            cursor.execute('DELETE FROM features WHERE feature_set_id = ?', (feature_set_id,))
            cursor.execute('DELETE FROM feature_sets WHERE id = ?', (feature_set_id,))
            
            conn.commit()
            logger.info(f"Deleted feature set '{name}'")
            return True


class CachedFeatureTransformer(BaseEstimator, TransformerMixin):
    """Feature transformer with automatic caching and storage."""
    
    def __init__(self, 
                 transformer: BaseEstimator,
                 feature_store: Optional[SQLiteFeatureStore] = None,
                 cache: Optional[FeatureCache] = None,
                 feature_set_name: Optional[str] = None,
                 enable_caching: bool = True,
                 enable_storage: bool = True):
        
        self.transformer = transformer
        self.feature_store = feature_store or SQLiteFeatureStore()
        self.cache = cache or FeatureCache()
        self.feature_set_name = feature_set_name
        self.enable_caching = enable_caching
        self.enable_storage = enable_storage
        
        self.fitted_ = False
        self.feature_metadata_ = {}
        
    def _compute_data_hash(self, X) -> str:
        """Compute hash of input data for caching."""
        if isinstance(X, pd.DataFrame):
            data_str = X.to_csv()
        else:
            data_str = str(X)
        return hashlib.md5(data_str.encode()).hexdigest()
        
    def _get_transformation_params(self) -> Dict:
        """Get parameters used for transformation."""
        params = {}
        if hasattr(self.transformer, 'get_params'):
            params = self.transformer.get_params()
        return params
        
    def fit(self, X, y=None):
        """Fit the transformer."""
        # Try to load from feature store if name provided
        if self.feature_set_name and self.enable_storage:
            stored_features = self.feature_store.load_feature_set(self.feature_set_name)
            if stored_features:
                logger.info(f"Loaded pre-fitted transformer for '{self.feature_set_name}'")
                self.fitted_ = True
                self.feature_metadata_ = stored_features.metadata
                return self
                
        # Fit transformer normally
        self.transformer.fit(X, y)
        self.fitted_ = True
        
        return self
        
    def transform(self, X):
        """Transform with caching."""
        if not self.fitted_:
            raise ValueError("Transformer must be fitted before transform")
            
        # Check cache first
        if self.enable_caching:
            data_hash = self._compute_data_hash(X)
            params = self._get_transformation_params()
            cached_result = self.cache.get(data_hash, params)
            
            if cached_result is not None:
                logger.debug("Using cached transformation result")
                return cached_result
                
        # Perform transformation
        result = self.transformer.transform(X)
        
        # Cache result
        if self.enable_caching:
            data_hash = self._compute_data_hash(X)
            params = self._get_transformation_params()
            self.cache.put(data_hash, params, result)
            
        return result
        
    def fit_transform(self, X, y=None):
        """Fit and transform with storage."""
        result = self.fit(X, y).transform(X)
        
        # Store result if enabled and name provided
        if self.enable_storage and self.feature_set_name:
            self._store_features(X, result)
            
        return result
        
    def _store_features(self, X, features):
        """Store features in feature store."""
        try:
            # Create sample IDs
            sample_ids = [f"sample_{i}" for i in range(len(features))]
            
            # Convert features to dictionary
            if isinstance(features, np.ndarray):
                feature_dict = {}
                for i in range(features.shape[1]):
                    feature_dict[f"feature_{i}"] = features[:, i]
            else:
                feature_dict = {'features': features}
                
            # Create metadata
            metadata = {}
            for name in feature_dict.keys():
                metadata[name] = FeatureMetadata(
                    feature_name=name,
                    feature_type='numerical',
                    description=f"Transformed feature: {name}",
                    data_type='float64',
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    transformation_steps=[str(type(self.transformer).__name__)]
                )
                
            # Create and store feature set
            feature_set = FeatureSet(
                name=self.feature_set_name,
                features=feature_dict,
                metadata=metadata,
                sample_ids=sample_ids
            )
            
            self.feature_store.store_feature_set(feature_set)
            
        except Exception as e:
            logger.warning(f"Failed to store features: {e}")


class FeatureRegistry:
    """Registry for managing feature definitions and transformations."""
    
    def __init__(self, registry_path: str = "feature_registry.json"):
        self.registry_path = registry_path
        self.registry = self._load_registry()
        
    def _load_registry(self) -> Dict:
        """Load feature registry from file."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {
            'feature_definitions': {},
            'transformation_pipelines': {},
            'version': '1.0',
            'created_at': datetime.now().isoformat()
        }
        
    def _save_registry(self):
        """Save registry to file."""
        self.registry['last_updated'] = datetime.now().isoformat()
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
            
    def register_feature(self, feature_name: str, definition: Dict):
        """Register a feature definition."""
        self.registry['feature_definitions'][feature_name] = {
            **definition,
            'registered_at': datetime.now().isoformat()
        }
        self._save_registry()
        logger.info(f"Registered feature: {feature_name}")
        
    def register_pipeline(self, pipeline_name: str, pipeline_config: Dict):
        """Register a transformation pipeline."""
        self.registry['transformation_pipelines'][pipeline_name] = {
            **pipeline_config,
            'registered_at': datetime.now().isoformat()
        }
        self._save_registry()
        logger.info(f"Registered pipeline: {pipeline_name}")
        
    def get_feature(self, feature_name: str) -> Optional[Dict]:
        """Get feature definition."""
        return self.registry['feature_definitions'].get(feature_name)
        
    def get_pipeline(self, pipeline_name: str) -> Optional[Dict]:
        """Get pipeline configuration."""
        return self.registry['transformation_pipelines'].get(pipeline_name)
        
    def list_features(self) -> List[str]:
        """List all registered features."""
        return list(self.registry['feature_definitions'].keys())
        
    def list_pipelines(self) -> List[str]:
        """List all registered pipelines."""
        return list(self.registry['transformation_pipelines'].keys())


def demonstrate_feature_store():
    """Demonstrate the feature store system."""
    
    print("=== Feature Store Demonstration ===\n")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.randint(0, 5, 100),
        'target': np.random.randn(100)
    })
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Initialize feature store
    feature_store = SQLiteFeatureStore("demo_feature_store.db")
    
    # Create feature set
    features = {col: sample_data[col].values for col in sample_data.columns if col != 'target'}
    metadata = {}
    
    for col in features.keys():
        metadata[col] = FeatureMetadata(
            feature_name=col,
            feature_type='numerical' if col.startswith('feature') else 'categorical',
            description=f"Demo feature: {col}",
            data_type=str(sample_data[col].dtype),
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
    feature_set = FeatureSet(
        name="demo_features",
        features=features,
        metadata=metadata,
        target_column="target",
        sample_ids=[f"sample_{i}" for i in range(100)]
    )
    
    # Store feature set
    print(f"\nStoring feature set...")
    feature_store.store_feature_set(feature_set)
    
    # List stored feature sets
    print(f"\nStored feature sets:")
    feature_sets = feature_store.list_feature_sets()
    for fs in feature_sets:
        print(f"- {fs['name']}: {fs['feature_count']} features, {fs['sample_count']} samples")
        
    # Load feature set
    print(f"\nLoading feature set...")
    loaded_features = feature_store.load_feature_set("demo_features")
    
    if loaded_features:
        print(f"Loaded feature set: {loaded_features.name}")
        print(f"Features: {list(loaded_features.features.keys())}")
        print(f"Sample count: {len(loaded_features.sample_ids)}")
        
        # Convert to DataFrame
        df = loaded_features.to_dataframe()
        print(f"DataFrame shape: {df.shape}")
        print(f"First few rows:")
        print(df.head())
        
    # Demonstrate caching
    print(f"\n=== Caching Demonstration ===")
    cache = FeatureCache(max_size=10, ttl_seconds=60)
    
    # Test cache operations
    test_features = np.random.randn(50, 5)
    data_hash = "test_hash"
    params = {"param1": "value1"}
    
    print(f"Cache size before: {cache.size()}")
    cache.put(data_hash, params, test_features)
    print(f"Cache size after storing: {cache.size()}")
    
    cached_result = cache.get(data_hash, params)
    print(f"Cache retrieval successful: {cached_result is not None}")
    
    # Cleanup
    if os.path.exists("demo_feature_store.db"):
        os.remove("demo_feature_store.db")
        print(f"\nCleaned up demo database")
        
    print(f"\n=== Feature Store System Ready ===")


if __name__ == "__main__":
    demonstrate_feature_store()