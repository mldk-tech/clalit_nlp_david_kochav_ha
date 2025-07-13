"""
Disease Clustering System for Medical Appointment Summaries
===========================================================

This module provides comprehensive disease clustering capabilities based on:
- Symptom patterns
- Treatment approaches
- Anatomical systems
- Outcome patterns
- Clinical presentation similarities
"""

import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.decomposition import PCA
import scipy.sparse
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import skfuzzy
from skfuzzy import cmeans
import warnings
import os
import hashlib
import pickle
from functools import wraps
from joblib import Parallel, delayed, Memory, dump, load
import plotly.express as px
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import plotly.graph_objects as go
from sqlalchemy.orm import Session
from backend.app.db.session import SessionLocal
from backend.app.models.disease_clustering import DiseaseClusteringResult
warnings.filterwarnings('ignore')

class DiseaseClusteringEngine:
    """Advanced disease clustering engine with caching and optimization"""
    
    def __init__(self, data_file='../doctor_appointment_summaries.json', cache_dir='./cache'):
        """Initialize the clustering engine with caching support"""
        self.data_file = data_file
        self.data = None
        self.features = []
        self.feature_matrix = None
        self.feature_names = []
        self.clusters = {}
        self.summaries = []
        
        # Initialize caching
        self.cache_dir = cache_dir
        self.memory = Memory(cache_dir, verbose=0)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Define medical patterns
        self.disease_patterns = self._define_disease_patterns()
        self.symptom_patterns = self._define_symptom_patterns()
        self.treatment_patterns = self._define_treatment_patterns()
        self.outcome_patterns = self._define_outcome_patterns()
        self.normalization_dict = self._define_normalization_dict()
    
    def _get_cache_key(self, func_name, *args, **kwargs):
        """Generate a cache key for function calls"""
        # Create a hash of the function name and arguments
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else None,
            'data_hash': self._get_data_hash() if hasattr(self, 'data') and self.data else None
        }
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_data_hash(self):
        """Generate a hash of the current data for cache invalidation"""
        if not self.data:
            return None
        data_str = json.dumps(self.data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _cache_result(self, cache_key, result):
        """Cache a result with the given key"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"Warning: Could not cache result: {e}")
    
    def _load_cached_result(self, cache_key):
        """Load a cached result if it exists"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load cached result: {e}")
        return None
    
    def clear_cache(self):
        """Clear all cached results"""
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, file))
        print("Cache cleared.")
    
    def get_cache_info(self):
        """Get information about cached results"""
        if not os.path.exists(self.cache_dir):
            return {"cached_items": 0, "cache_size": "0 MB"}
        
        files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in files)
        
        return {
            "cached_items": len(files),
            "cache_size": f"{total_size / (1024*1024):.2f} MB",
            "cache_files": files
        }
    
    def _memory_property(self):
        """Property for memory access"""
        return self._memory
    
    def _memory_setter(self, value):
        """Setter for memory property"""
        self._memory = value
    
    memory = property(_memory_property, _memory_setter)

    def _define_disease_patterns(self):
        """Define regex patterns for disease identification"""
        return {
            'diabetes': r'\bdiabetes\b',
            'hypertension': r'\bhypertension\b',
            'asthma': r'\basthma\b',
            'eczema': r'\beczema\b',
            'migraine': r'\bmigraine\b',
            'anemia': r'\banemia\b',
            'bronchitis': r'\bbronchitis\b',
            'sinusitis': r'\bsinusitis\b',
            'uti': r'\buti\b',
            'pneumonia': r'\bpneumonia\b',
            'arthritis': r'\barthritis\b',
            'depression': r'\bdepression\b',
            'anxiety': r'\banxiety\b',
            'obesity': r'\bobesity\b',
            'thyroid': r'\bthyroid\b'
        }
    
    def _define_symptom_patterns(self):
        """Define regex patterns for symptom identification"""
        return {
            'pain_symptoms': {
                'headache': r'\bheadache\b',
                'back_pain': r'\bback pain\b',
                'abdominal_pain': r'\babdominal pain\b',
                'chest_pain': r'\bchest pain\b'
            },
            'respiratory_symptoms': {
                'shortness_of_breath': r'\bshortness of breath\b',
                'cough': r'\bcough\b',
                'wheezing': r'\bwheezing\b'
            },
            'systemic_symptoms': {
                'fatigue': r'\bfatigue\b',
                'dizziness': r'\bdizziness\b',
                'fever': r'\bfever\b',
                'nausea': r'\bnausea\b'
            },
            'skin_symptoms': {
                'rash': r'\brash\b',
                'itching': r'\bitching\b',
                'swelling': r'\bswelling\b'
            }
        }
    
    def _define_treatment_patterns(self):
        """Define regex patterns for treatment identification"""
        return {
            'medications': {
                'antibiotics': r'\b(amoxicillin|penicillin|azithromycin)\b',
                'pain_relievers': r'\b(ibuprofen|paracetamol|acetaminophen|aspirin)\b',
                'inhalers': r'\b(ventolin|albuterol|inhaler)\b',
                'diabetes_meds': r'\b(metformin|insulin)\b',
                'hypertension_meds': r'\b(lisinopril|amlodipine|atenolol)\b'
            },
            'procedures': {
                'imaging': r'\b(x-ray|xray|ct scan|mri|ultrasound)\b',
                'blood_tests': r'\b(blood test|blood work|lab work)\b',
                'vaccinations': r'\b(vaccination|vaccine|shot)\b'
            },
            'referrals': {
                'cardiology': r'\b(cardiology|cardiologist)\b',
                'neurology': r'\b(neurology|neurologist)\b',
                'orthopedics': r'\b(orthopedics|orthopedic)\b',
                'dermatology': r'\b(dermatology|dermatologist)\b'
            }
        }
    
    def _define_outcome_patterns(self):
        """Define regex patterns for outcome identification"""
        return {
            'positive_outcomes': {
                'improvement': r'\b(improved|better|recovered|healed)\b',
                'stable': r'\b(stable|unchanged|maintained)\b',
                'resolved': r'\b(resolved|cleared|gone)\b'
            },
            'negative_outcomes': {
                'deterioration': r'\b(deteriorated|worsened|declined)\b',
                'critical': r'\b(critical|severe|emergency)\b',
                'deceased': r'\b(deceased|passed away|died)\b'
            }
        }
    
    def _define_normalization_dict(self):
        """Define mapping for medical terminology normalization"""
        return {
            # Diseases
            'high blood pressure': 'hypertension',
            'dm': 'diabetes',
            'htn': 'hypertension',
            'mi': 'myocardial infarction',
            'heart attack': 'myocardial infarction',
            # Medications
            'paracetamol': 'acetaminophen',
            'tylenol': 'acetaminophen',
            'advil': 'ibuprofen',
            # Symptoms
            'sob': 'shortness of breath',
            'tiredness': 'fatigue',
            'tired': 'fatigue',
            'bp': 'blood pressure',
            # Treatments
            'xray': 'x-ray',
            'ct': 'ct scan',
            'mri scan': 'mri',
            # @TODO: Add more as needed
        }

    def _normalize_text(self, text):
        """Normalize medical terms in the input text using the normalization dictionary"""
        text = text.lower()
        for variant, canonical in self.normalization_dict.items():
            text = re.sub(rf'\b{re.escape(variant)}\b', canonical, text)
        return text

    def _outcome_severity_score(self, outcome):
        """Assign a numerical severity score to the outcome text"""
        if 'deceased' in outcome or 'passed away' in outcome or 'died' in outcome:
            return 3
        if 'critical' in outcome or 'emergency' in outcome or 'intensive care' in outcome:
            return 2
        if 'deteriorated' in outcome or 'worsened' in outcome or 'declined' in outcome:
            return 1
        if 'improved' in outcome or 'better' in outcome or 'recovered' in outcome or 'healed' in outcome or 'resolved' in outcome or 'cleared' in outcome or 'gone' in outcome:
            return -1
        if 'stable' in outcome or 'unchanged' in outcome or 'maintained' in outcome or 'no complaints' in outcome:
            return 0
        return 0

    def _treatment_complexity(self, treatments):
        """Quantify treatment complexity by counting unique treatments and procedures"""
        count = 0
        for category in treatments.values():
            count += len(category)
        return count

    def load_data(self):
        """Load and preprocess the appointment data"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"File size: {len(content)} characters")
                self.data = json.loads(content)
            
            print(f"Loaded {len(self.data)} appointment records")
            
            # Verify data integrity
            if len(self.data) < 2000:  # Should be 2402
                print(f"WARNING: Expected ~2402 records, but loaded only {len(self.data)}")
                print("This might indicate a truncated file or parsing issue")
            
            return self
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def extract_medical_features(self, n_jobs=-1):
        """Extract medical features from appointment summaries with caching and parallel processing"""
        cache_key = self._get_cache_key('extract_medical_features', n_jobs)
        cached_result = self._load_cached_result(cache_key)
        if cached_result is not None:
            print("Loading cached medical features...")
            self.features = list(cached_result) if not isinstance(cached_result, list) else cached_result
            if self.data is not None:
                self.summaries = [record['summary'] for record in self.data if isinstance(record, dict) and 'summary' in record]
            else:
                self.summaries = []
            return self
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() before extracting features.")
        print("Extracting medical features (parallel)...")
        def process_record(record):
            summary = record.get('summary', '').lower() if isinstance(record, dict) else ''
            future_outcome = record.get('future_outcome', '').lower() if isinstance(record, dict) else ''
            diseases = []
            for category, patterns in self.disease_patterns.items():
                if isinstance(patterns, dict):
                    for disease, pattern in patterns.items():
                        if re.search(pattern, summary):
                            diseases.append(disease)
                else:
                    if re.search(patterns, summary):
                        diseases.append(category)
            symptoms = []
            for category, patterns in self.symptom_patterns.items():
                if isinstance(patterns, dict):
                    for symptom, pattern in patterns.items():
                        if re.search(pattern, summary):
                            symptoms.append(symptom)
            treatments = []
            for category, patterns in self.treatment_patterns.items():
                if isinstance(patterns, dict):
                    for treatment, pattern in patterns.items():
                        if re.search(pattern, summary):
                            treatments.append(treatment)
            outcomes = []
            for category, patterns in self.outcome_patterns.items():
                if isinstance(patterns, dict):
                    for outcome, pattern in patterns.items():
                        if re.search(pattern, future_outcome):
                            outcomes.append(outcome)
            severity_score = self._outcome_severity_score(future_outcome)
            complexity_score = self._treatment_complexity({'dummy': treatments}) if isinstance(treatments, list) else 0
            return {
                'id': record.get('id', None),
                'doctor_id': record.get('doctor_id', None),
                'summary': summary,
                'future_outcome': future_outcome,
                'diseases': diseases,
                'symptoms': symptoms,
                'treatments': treatments,
                'outcomes': outcomes,
                'severity_score': severity_score,
                'complexity_score': complexity_score
            }
        features = list(Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(process_record)(record) for record in self.data if isinstance(record, dict)
        ))
        self.features = features
        self.summaries = [record['summary'] for record in self.data if isinstance(record, dict) and 'summary' in record]
        self._cache_result(cache_key, self.features)
        print(f"Extracted features from {len(features)} records (parallel)")
        return self
    
    def create_feature_matrix(self, use_sparse=True, add_tfidf=True, n_jobs=-1):
        """Create numerical feature matrix for clustering with caching and parallel processing"""
        cache_key = self._get_cache_key('create_feature_matrix', use_sparse, add_tfidf, n_jobs)
        cached_result = self._load_cached_result(cache_key)
        if cached_result is not None:
            print("Loading cached feature matrix...")
            self.feature_matrix, self.feature_names = cached_result
            return self
        print("Creating feature matrix (parallel)...")
        all_diseases = set()
        all_symptoms = set()
        all_treatments = set()
        all_outcomes = set()
        if self.features is None:
            raise ValueError("Features not extracted. Call extract_medical_features() first.")
        for feature in self.features:
            if not isinstance(feature, dict):
                continue
            all_diseases.update(feature.get('diseases', []))
            all_symptoms.update(feature.get('symptoms', []))
            all_treatments.update(feature.get('treatments', []))
            all_outcomes.update(feature.get('outcomes', []))
        disease_list = list(sorted(all_diseases))
        symptom_list = list(sorted(all_symptoms))
        treatment_list = list(sorted(all_treatments))
        outcome_list = list(sorted(all_outcomes))
        def make_vector(feature):
            vector = []
            for disease in disease_list:
                vector.append(1 if disease in feature.get('diseases', []) else 0)
            for symptom in symptom_list:
                vector.append(1 if symptom in feature.get('symptoms', []) else 0)
            for treatment in treatment_list:
                vector.append(1 if treatment in feature.get('treatments', []) else 0)
            for outcome in outcome_list:
                vector.append(1 if outcome in feature.get('outcomes', []) else 0)
            vector.append(feature.get('severity_score', 0))
            vector.append(feature.get('complexity_score', 0))
            return vector
        feature_vectors = list(Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(make_vector)(feature) for feature in self.features if isinstance(feature, dict)
        ))
        X = scipy.sparse.csr_matrix(feature_vectors) if use_sparse else np.array(feature_vectors)
        feature_names = (
            disease_list + symptom_list + treatment_list + outcome_list +
            ['outcome_severity', 'treatment_complexity']
        )
        if add_tfidf:
            tfidf = TfidfVectorizer(max_features=100)
            tfidf_matrix = tfidf.fit_transform(self.summaries)
            X = scipy.sparse.hstack([X, tfidf_matrix], format='csr')
            feature_names += [f'tfidf_{w}' for w in tfidf.get_feature_names_out()]
        if scipy.sparse.issparse(X) and hasattr(X, 'tocsr') and not isinstance(X, scipy.sparse.csr_matrix):
            X = X.tocsr()
        self.feature_matrix = X
        self.feature_names = feature_names
        self._cache_result(cache_key, (self.feature_matrix, self.feature_names))
        print(f"Created feature matrix with {self.feature_matrix.shape[0]} samples and {self.feature_matrix.shape[1]} features (parallel, sparse={use_sparse})")
        return self
    
    def optimize_clustering_params(self, method, param_grid, cv_splits=3, n_jobs=-1):
        """Optimize clustering parameters using grid search with cross-validation and caching"""
        cache_key = self._get_cache_key('optimize_clustering_params', method, str(param_grid), cv_splits, n_jobs)
        cached_result = self._load_cached_result(cache_key)
        
        if cached_result is not None:
            print(f"Loading cached optimization results for {method}...")
            return cached_result
        
        print(f"Optimizing parameters for {method} clustering...")
        
        # Extract appropriate features based on method
        if method == 'symptom_based':
            symptom_start = len([f for f in self.feature_names if f in self.disease_patterns])
            mat = self.feature_matrix
            if scipy.sparse.issparse(mat):
                mat = mat.tocsr()
            features = mat[:, symptom_start:symptom_start+len([s for cat in self.symptom_patterns.values() for s in cat.keys()])]
            if scipy.sparse.issparse(features):
                features = features.toarray()
            algorithm = KMeans(random_state=42, n_init='auto')
            
        elif method == 'treatment_based':
            treatment_start = len([f for f in self.feature_names if f in self.disease_patterns or 
                                  any(f in cat for cat in self.symptom_patterns.values())])
            mat = self.feature_matrix
            if scipy.sparse.issparse(mat):
                mat = mat.tocsr()
            features = mat[:, treatment_start:]
            if scipy.sparse.issparse(features):
                features = features.toarray()
            algorithm = AgglomerativeClustering()
            
        elif method == 'outcome_based':
            outcome_start = len([f for f in self.feature_names if f not in 
                                [o for cat in self.outcome_patterns.values() for o in cat.keys()]])
            mat = self.feature_matrix
            if scipy.sparse.issparse(mat):
                mat = mat.tocsr()
            features = mat[:, outcome_start:]
            if scipy.sparse.issparse(features):
                features = features.toarray()
            algorithm = DBSCAN()
            
        elif method == 'comprehensive':
            features = self.feature_matrix
            if scipy.sparse.issparse(features):
                features = features.toarray()
            scaler = StandardScaler(with_mean=False)
            features = scaler.fit_transform(features)
            algorithm = KMeans(random_state=42, n_init='auto')
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Custom scoring function for clustering
        def silhouette_scorer(estimator, X):
            labels = estimator.fit_predict(X)
            if len(set(labels)) < 2:
                return -1.0  # Penalize single cluster
            try:
                return silhouette_score(X, labels)
            except:
                return -1.0
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=algorithm,
            param_grid=param_grid,
            scoring=silhouette_scorer,
            cv=cv_splits,
            n_jobs=n_jobs,
            verbose=0
        )
        
        grid_search.fit(features)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Best parameters for {method}: {best_params}")
        print(f"Best Silhouette score: {best_score:.4f}")
        
        result = (best_params, best_score)
        self._cache_result(cache_key, result)
        
        return result
    
    def perform_symptom_based_clustering(self, n_clusters=6, optimize_params=True, n_jobs=-1):
        """Cluster based on symptom patterns with optional parameter optimization"""
        print("Performing symptom-based clustering...")
        
        if optimize_params:
            # Parameter grid for KMeans optimization
            param_grid = {
                'n_clusters': [4, 5, 6, 7, 8],
                'init': ['k-means++', 'random'],
                'max_iter': [200, 300, 500]
            }
            
            best_params, best_score = self.optimize_clustering_params('symptom_based', param_grid, n_jobs=n_jobs)
            n_clusters = best_params['n_clusters']
            init_method = best_params['init']
            max_iter = best_params['max_iter']
        else:
            init_method = 'k-means++'
            max_iter = 300
        
        symptom_start = len([f for f in self.feature_names if f in self.disease_patterns])
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not created. Call create_feature_matrix() first.")
        
        mat = self.feature_matrix
        if scipy.sparse.issparse(mat):
            mat = mat.tocsr()
        symptom_features = mat[:, symptom_start:symptom_start+len([s for cat in self.symptom_patterns.values() for s in cat.keys()])]
        if scipy.sparse.issparse(symptom_features):
            symptom_features = symptom_features.toarray()
        
        # Apply K-means clustering with optimized parameters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto', 
                       init=init_method, max_iter=max_iter)
        symptom_clusters = kmeans.fit_predict(symptom_features)
        
        # Analyze clusters
        symptom_cluster_analysis = self._analyze_clusters(symptom_clusters, 'symptom')
        
        self.clusters['symptom_based'] = {
            'labels': symptom_clusters,
            'analysis': symptom_cluster_analysis,
            'algorithm': 'K-means',
            'n_clusters': n_clusters,
            'optimized_params': best_params if optimize_params else None,
            'best_score': best_score if optimize_params else None
        }
        
        return self
    
    def perform_treatment_based_clustering(self, n_clusters=5, optimize_params=True, n_jobs=-1):
        """Cluster based on treatment patterns with optional parameter optimization"""
        print("Performing treatment-based clustering...")
        
        if optimize_params:
            # Parameter grid for AgglomerativeClustering optimization
            param_grid = {
                'n_clusters': [3, 4, 5, 6, 7],
                'linkage': ['ward', 'complete', 'average', 'single']
            }
            
            best_params, best_score = self.optimize_clustering_params('treatment_based', param_grid, n_jobs=n_jobs)
            n_clusters = best_params['n_clusters']
            linkage_method = best_params['linkage']
        else:
            linkage_method = 'ward'
        
        treatment_start = len([f for f in self.feature_names if f in self.disease_patterns or 
                              any(f in cat for cat in self.symptom_patterns.values())])
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not created. Call create_feature_matrix() first.")

        mat = self.feature_matrix
        if scipy.sparse.issparse(mat):
            mat = mat.tocsr()
        treatment_features = mat[:, treatment_start:]
        if scipy.sparse.issparse(treatment_features):
            treatment_features = treatment_features.toarray()
        
        # Apply hierarchical clustering with optimized parameters
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        treatment_clusters = hierarchical.fit_predict(treatment_features)
        
        # Analyze clusters
        treatment_cluster_analysis = self._analyze_clusters(treatment_clusters, 'treatment')
        
        self.clusters['treatment_based'] = {
            'labels': treatment_clusters,
            'analysis': treatment_cluster_analysis,
            'algorithm': 'Hierarchical',
            'n_clusters': n_clusters,
            'optimized_params': best_params if optimize_params else None,
            'best_score': best_score if optimize_params else None
        }
        
        return self
    
    def perform_outcome_based_clustering(self, n_clusters=4, optimize_params=True, n_jobs=-1):
        """Cluster based on outcome patterns with optional parameter optimization"""
        print("Performing outcome-based clustering...")
        
        if optimize_params:
            # Parameter grid for DBSCAN optimization
            param_grid = {
                'eps': [0.1, 0.2, 0.3, 0.4, 0.5],
                'min_samples': [3, 5, 7, 10]
            }
            
            best_params, best_score = self.optimize_clustering_params('outcome_based', param_grid, n_jobs=n_jobs)
            eps = best_params['eps']
            min_samples = best_params['min_samples']
        else:
            eps = 0.3
            min_samples = 5
        
        outcome_start = len([f for f in self.feature_names if f not in 
                            [o for cat in self.outcome_patterns.values() for o in cat.keys()]])
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not created. Call create_feature_matrix() first.")
            
        mat = self.feature_matrix
        if scipy.sparse.issparse(mat):
            mat = mat.tocsr()
        outcome_features = mat[:, outcome_start:]
        if scipy.sparse.issparse(outcome_features):
            outcome_features = outcome_features.toarray()
        
        # Apply DBSCAN clustering with optimized parameters
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        outcome_clusters = dbscan.fit_predict(outcome_features)
        
        # Analyze clusters
        outcome_cluster_analysis = self._analyze_clusters(outcome_clusters, 'outcome')
        
        self.clusters['outcome_based'] = {
            'labels': outcome_clusters,
            'analysis': outcome_cluster_analysis,
            'algorithm': 'DBSCAN',
            'n_clusters': len(set(outcome_clusters)) - (1 if -1 in outcome_clusters else 0),
            'optimized_params': best_params if optimize_params else None,
            'best_score': best_score if optimize_params else None
        }
        
        return self
    
    def perform_comprehensive_clustering(self, n_clusters=8, optimize_params=True, n_jobs=-1):
        """Perform comprehensive clustering using all features with optional parameter optimization"""
        print("Performing comprehensive clustering...")
        
        if optimize_params:
            # Parameter grid for comprehensive KMeans optimization
            param_grid = {
                'n_clusters': [6, 7, 8, 9, 10],
                'init': ['k-means++', 'random'],
                'max_iter': [200, 300, 500]
            }
            
            best_params, best_score = self.optimize_clustering_params('comprehensive', param_grid, n_jobs=n_jobs)
            n_clusters = best_params['n_clusters']
            init_method = best_params['init']
            max_iter = best_params['max_iter']
        else:
            init_method = 'k-means++'
            max_iter = 300
        
        # Standardize features
        scaler = StandardScaler(with_mean=False)
        scaled_features = scaler.fit_transform(self.feature_matrix)
        
        # Apply K-means clustering with optimized parameters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto',
                       init=init_method, max_iter=max_iter)
        comprehensive_clusters = kmeans.fit_predict(scaled_features)
        
        # Analyze clusters
        comprehensive_analysis = self._analyze_clusters(comprehensive_clusters, 'comprehensive')
        self.clusters['comprehensive'] = {
            'labels': comprehensive_clusters,
            'analysis': comprehensive_analysis,
            'algorithm': 'K-means',
            'n_clusters': n_clusters,
            'optimized_params': best_params if optimize_params else None,
            'best_score': best_score if optimize_params else None
        }
        return self
    
    def perform_fuzzy_cmeans_clustering(self, n_clusters=5, m=2.0, error=0.005, maxiter=1000, optimize_params=True, n_jobs=-1):
        """Cluster using fuzzy C-means for overlapping conditions with optional parameter optimization"""
        print("Performing fuzzy C-means clustering...")
        
        if optimize_params:
            # Parameter grid for fuzzy C-means optimization
            param_grid = {
                'c': [3, 4, 5, 6, 7],  # number of clusters
                'm': [1.5, 2.0, 2.5, 3.0],  # fuzziness parameter
                'error': [0.001, 0.005, 0.01],
                'maxiter': [500, 1000, 1500]
            }
            
            best_params, best_score = self.optimize_fuzzy_cmeans_params(param_grid, n_jobs=n_jobs)
            n_clusters = best_params['c']
            m = best_params['m']
            error = best_params['error']
            maxiter = best_params['maxiter']
        
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not created. Call create_feature_matrix() first.")
        
        # Standardize features
        scaler = StandardScaler(with_mean=False)
        scaled_features = scaler.fit_transform(self.feature_matrix)
        if scipy.sparse.issparse(scaled_features):
            scaled_features = scaled_features.toarray()
        scaled_features = scaled_features.T  # Transpose for skfuzzy
        
        # Fuzzy C-means
        cntr, u, u0, d, jm, p, fpc = cmeans(
            scaled_features, c=n_clusters, m=m, error=error, maxiter=maxiter, init=None, seed=42
        )
        
        # Assign each sample to the cluster with highest membership
        labels = u.argmax(axis=0)
        
        # Analyze clusters
        fuzzy_cluster_analysis = self._analyze_clusters(labels, 'fuzzy_cmeans')
        self.clusters['fuzzy_cmeans'] = {
            'labels': labels,
            'analysis': fuzzy_cluster_analysis,
            'algorithm': 'Fuzzy C-means',
            'n_clusters': n_clusters,
            'fpc': fpc,
            'membership': u.tolist(),
            'optimized_params': best_params if optimize_params else None,
            'best_score': best_score if optimize_params else None
        }
        print(f"Fuzzy partition coefficient (FPC): {fpc:.4f}")
        return self
    
    def optimize_fuzzy_cmeans_params(self, param_grid, n_jobs=-1):
        """Optimize fuzzy C-means parameters using grid search with caching"""
        cache_key = self._get_cache_key('optimize_fuzzy_cmeans_params', str(param_grid), n_jobs)
        cached_result = self._load_cached_result(cache_key)
        
        if cached_result is not None:
            print("Loading cached fuzzy C-means optimization results...")
            return cached_result
        
        print("Optimizing fuzzy C-means parameters...")
        
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not created. Call create_feature_matrix() first.")
        
        # Standardize features
        scaler = StandardScaler(with_mean=False)
        scaled_features = scaler.fit_transform(self.feature_matrix)
        if scipy.sparse.issparse(scaled_features):
            scaled_features = scaled_features.toarray()
        scaled_features = scaled_features.T
        
        best_score = -1
        best_params = None
        
        # Grid search for fuzzy C-means
        for c in param_grid['c']:
            for m in param_grid['m']:
                for error in param_grid['error']:
                    for maxiter in param_grid['maxiter']:
                        try:
                            cntr, u, u0, d, jm, p, fpc = cmeans(
                                scaled_features, c=c, m=m, error=error, maxiter=maxiter, init=None, seed=42
                            )
                            labels = u.argmax(axis=0)
                            
                            if len(set(labels)) > 1:
                                # Use FPC as score (higher is better)
                                if fpc > best_score:
                                    best_score = fpc
                                    best_params = {'c': c, 'm': m, 'error': error, 'maxiter': maxiter}
                        except:
                            continue
        
        if best_params:
            print(f"Best fuzzy C-means parameters: {best_params}")
            print(f"Best FPC score: {best_score:.4f}")
        else:
            print("No valid parameters found for fuzzy C-means")
            best_params = {'c': 5, 'm': 2.0, 'error': 0.005, 'maxiter': 1000}
            best_score = 0.0
        
        result = (best_params, best_score)
        self._cache_result(cache_key, result)
        
        return result
    
    def _analyze_clusters(self, cluster_labels, cluster_type):
        """Analyze cluster characteristics"""
        analysis = {}
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
                
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if self.feature_matrix is None:
                raise ValueError("Feature matrix not created. Call create_feature_matrix() first.")
            
            mat = self.feature_matrix
            if scipy.sparse.issparse(mat) and hasattr(mat, 'tocsr'):
                mat = mat.tocsr()
            if hasattr(mat, '__getitem__'):
                cluster_features = mat[cluster_indices]
            else:
                continue
            # Find most common features in this cluster
            if scipy.sparse.issparse(cluster_features):
                feature_counts = np.array(cluster_features.sum(axis=0)).flatten()
            elif hasattr(cluster_features, 'sum'):
                feature_counts = np.sum(cluster_features, axis=0)
            else:
                continue
            
            top_features = []
            
            for i, count in enumerate(feature_counts):
                if count > len(cluster_indices) * 0.3:  # Feature present in >30% of cases
                    top_features.append((self.feature_names[i], int(count)))
            
            top_features.sort(key=lambda x: x[1], reverse=True)
            
            # Analyze cluster characteristics
            cluster_summaries = [self.features[i]['summary'] for i in cluster_indices]
            cluster_outcomes = [self.features[i]['future_outcome'] for i in cluster_indices]
            
            analysis[str(cluster_id)] = {
                'size': int(len(cluster_indices)),
                'top_features': top_features[:10],
                'sample_summaries': cluster_summaries[:3],
                'sample_outcomes': cluster_outcomes[:3],
                'indices': cluster_indices.tolist()
            }
        
        return analysis
    
    def generate_cluster_report(self):
        """Generate comprehensive cluster analysis report"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() before generating report.")
        
        report = {
            'summary': {
                'total_records': len(self.data),
                'clustering_methods': list(self.clusters.keys()),
                'feature_count': len(self.feature_names)
            },
            'clusters': {}
        }
        
        for method, cluster_data in self.clusters.items():
            report['clusters'][method] = {
                'algorithm': cluster_data['algorithm'],
                'n_clusters': cluster_data['n_clusters'],
                'cluster_analysis': cluster_data['analysis']
            }
        
        return report
    
    def visualize_clusters(self, method='comprehensive'):
        """Create visualizations for cluster analysis"""
        if method not in self.clusters:
            print(f"Method {method} not found. Available methods: {list(self.clusters.keys())}")
            return
        
        cluster_data = self.clusters[method]
        labels = cluster_data['labels']
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(self.feature_matrix)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        unique_labels = set(labels)
        # Use simple colors instead of colormap
        base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        colors = base_colors * (len(unique_labels) // len(base_colors) + 1)
        
        for i, label in enumerate(unique_labels):
            color = colors[i]
            if label == -1:  # Noise points
                mask = labels == label
                plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1], 
                           c='black', marker='x', s=50, alpha=0.5, label='Noise')
            else:
                mask = labels == label
                plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1], 
                           c=color, s=50, alpha=0.7, label=f'Cluster {label}')
        
        plt.title(f'{method.replace("_", " ").title()} Clustering Results')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{method}_clustering.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self
    
    def visualize_clusters_interactive(self, method='comprehensive', filename=None):
        """Create interactive Plotly visualization for cluster analysis and save as HTML"""
        if method not in self.clusters:
            print(f"Method {method} not found. Available methods: {list(self.clusters.keys())}")
            return
        cluster_data = self.clusters[method]
        labels = cluster_data['labels']
        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(self.feature_matrix.toarray() if hasattr(self.feature_matrix, 'toarray') else self.feature_matrix)
        # Prepare hover text
        hover_text = []
        for i, feature in enumerate(self.features):
            summary = feature.get('summary', '') if isinstance(feature, dict) else ''
            outcome = feature.get('future_outcome', '') if isinstance(feature, dict) else ''
            hover_text.append(f"Summary: {summary}<br>Outcome: {outcome}")
        df = pd.DataFrame({
            'PC1': reduced_features[:, 0],
            'PC2': reduced_features[:, 1],
            'Cluster': labels,
            'Hover': hover_text
        })
        fig = px.scatter(
            df, x='PC1', y='PC2', color=df['Cluster'].astype(str),
            hover_name='Hover',
            title=f'Interactive {method.replace("_", " ").title()} Clustering Results',
            labels={'color': 'Cluster'}
        )
        fig.update_traces(marker=dict(size=8, opacity=0.7), selector=dict(mode='markers'))
        fig.update_layout(legend_title_text='Cluster', legend=dict(itemsizing='constant'))
        if filename is None:
            filename = f'{method}_clustering_interactive.html'
        fig.write_html(filename)
        print(f"Interactive cluster plot saved to {filename}")
        return self
    
    def get_similar_diseases(self, disease_name, method='comprehensive', top_k=5):
        """Find diseases similar to a given disease"""
        if method not in self.clusters:
            print(f"Method {method} not found")
            return []
        
        # Find the cluster containing the disease
        disease_indices = []
        for i, feature in enumerate(self.features):
            if disease_name in feature['diseases']:
                disease_indices.append(i)
        
        if not disease_indices:
            print(f"Disease '{disease_name}' not found in the dataset")
            return []
        
        cluster_labels = self.clusters[method]['labels']
        disease_cluster = cluster_labels[disease_indices[0]]
        
        # Find all diseases in the same cluster
        cluster_indices = np.where(cluster_labels == disease_cluster)[0]
        similar_diseases = []
        
        for idx in cluster_indices:
            diseases = self.features[idx]['diseases']
            for disease in diseases:
                if disease != disease_name and disease not in [d[0] for d in similar_diseases]:
                    similar_diseases.append((disease, 1))
        
        # Sort by frequency
        similar_diseases.sort(key=lambda x: x[1], reverse=True)
        return similar_diseases[:top_k]

    def compute_cluster_quality_metrics(self, method):
        """Compute Silhouette, Calinski-Harabasz, and Davies-Bouldin scores for a given clustering method"""
        if method not in self.clusters:
            print(f"Method {method} not found in clusters.")
            return None
        labels = self.clusters[method]['labels']
        if self.feature_matrix is None:
            print("Feature matrix not created.")
            return None
        # Fuzzy C-means labels may not be suitable for these metrics
        if method == 'fuzzy_cmeans':
            print("Cluster quality metrics are not computed for fuzzy C-means.")
            return None
        # Some metrics require at least 2 clusters and no noise (-1)
        if len(set(labels)) < 2 or (set(labels) == {-1}):
            print(f"Not enough clusters for quality metrics in method {method}.")
            return None
        try:
            sil = silhouette_score(self.feature_matrix, labels) if len(set(labels)) > 1 and -1 not in set(labels) else None
        except Exception:
            sil = None
        try:
            ch = calinski_harabasz_score(self.feature_matrix, labels)
        except Exception:
            ch = None
        try:
            db = davies_bouldin_score(self.feature_matrix, labels)
        except Exception:
            db = None
        metrics = {
            'silhouette_score': sil,
            'calinski_harabasz_score': ch,
            'davies_bouldin_score': db
        }
        self.clusters[method]['quality_metrics'] = metrics
        print(f"Cluster quality metrics for {method}: {metrics}")
        return metrics

    def evaluate_all_clusters(self):
        """Automated quality assessment for all clustering methods (except fuzzy C-means)"""
        results = {}
        best_method = None
        best_score = float('-inf')
        for method in ['symptom_based', 'treatment_based', 'outcome_based', 'comprehensive']:
            metrics = self.compute_cluster_quality_metrics(method)
            if metrics and metrics['silhouette_score'] is not None:
                results[method] = metrics
                if metrics['silhouette_score'] > best_score:
                    best_score = metrics['silhouette_score']
                    best_method = method
        print("\nAutomated Cluster Quality Assessment Summary:")
        for method, metrics in results.items():
            print(f"  {method}: Silhouette={metrics['silhouette_score']}, Calinski-Harabasz={metrics['calinski_harabasz_score']}, Davies-Bouldin={metrics['davies_bouldin_score']}")
        if best_method:
            print(f"\nBest clustering method by Silhouette score: {best_method} (score={best_score})")
        else:
            print("No valid clustering method found for automated assessment.")
        return {'results': results, 'best_method': best_method, 'best_score': best_score}

    def cross_validate_clustering(self, method, n_splits=5):
        """Cross-validation for clustering results using Silhouette score stability"""
        if self.feature_matrix is None:
            print("Feature matrix not created.")
            return None
        if method not in ['symptom_based', 'treatment_based', 'outcome_based', 'comprehensive']:
            print(f"Cross-validation not supported for method: {method}")
            return None
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        sil_scores = []
        for train_idx, _ in kf.split(self.feature_matrix):
            mat = self.feature_matrix
            if scipy.sparse.issparse(mat):
                mat = mat.tocsr()
            X_train = mat[train_idx]
            if scipy.sparse.issparse(X_train):
                X_train_dense = X_train.toarray()
            else:
                X_train_dense = X_train
            # Fit clustering on train split
            if method == 'symptom_based':
                kmeans = KMeans(n_clusters=self.clusters[method]['n_clusters'], random_state=42, n_init='auto')
                labels = kmeans.fit_predict(X_train_dense)
            elif method == 'treatment_based':
                hierarchical = AgglomerativeClustering(n_clusters=self.clusters[method]['n_clusters'])
                labels = hierarchical.fit_predict(X_train_dense)
            elif method == 'outcome_based':
                dbscan = DBSCAN(eps=0.3, min_samples=5)
                labels = dbscan.fit_predict(X_train_dense)
            elif method == 'comprehensive':
                scaler = StandardScaler(with_mean=False)
                scaled = scaler.fit_transform(X_train_dense)
                kmeans = KMeans(n_clusters=self.clusters[method]['n_clusters'], random_state=42, n_init='auto')
                labels = kmeans.fit_predict(scaled)
            else:
                continue
            # Only compute Silhouette if more than 1 cluster and no noise
            if len(set(labels)) > 1 and -1 not in set(labels):
                try:
                    sil = silhouette_score(X_train_dense, labels)
                    sil_scores.append(sil)
                except Exception:
                    continue
        if sil_scores:
            mean_sil = float(np.mean(sil_scores))
            std_sil = float(np.std(sil_scores))
            print(f"Cross-validated Silhouette score for {method}: mean={mean_sil:.4f}, std={std_sil:.4f}")
            self.clusters[method]['crossval_silhouette_mean'] = mean_sil
            self.clusters[method]['crossval_silhouette_std'] = std_sil
            return {'mean': mean_sil, 'std': std_sil, 'scores': sil_scores}
        else:
            print(f"No valid Silhouette scores computed for {method} in cross-validation.")
            return None

    def compare_clusters(self, method1, method2, filename=None):
        """Compare two clustering results: overlap matrix, best matches, ARI/NMI, and interactive heatmap"""
        if method1 not in self.clusters or method2 not in self.clusters:
            print(f"Both methods must be in self.clusters. Found: {list(self.clusters.keys())}")
            return
        labels1 = np.array(self.clusters[method1]['labels'])
        labels2 = np.array(self.clusters[method2]['labels'])
        # Compute overlap/contingency matrix
        unique1 = np.unique(labels1)
        unique2 = np.unique(labels2)
        overlap = np.zeros((len(unique1), len(unique2)), dtype=int)
        for i, c1 in enumerate(unique1):
            for j, c2 in enumerate(unique2):
                overlap[i, j] = np.sum((labels1 == c1) & (labels2 == c2))
        # Best match for each cluster in both directions
        best_match_1to2 = overlap.argmax(axis=1)
        best_match_2to1 = overlap.argmax(axis=0)
        # Agreement metrics
        ari = adjusted_rand_score(labels1, labels2)
        nmi = normalized_mutual_info_score(labels1, labels2)
        print(f"Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
        # Print best match table
        print("\nBest match for each cluster in", method1, "(row) to", method2, "(col):")
        for i, j in enumerate(best_match_1to2):
            print(f"  {method1} cluster {unique1[i]} best matches {method2} cluster {unique2[j]} (overlap: {overlap[i, j]})")
        print("\nBest match for each cluster in", method2, "(col) to", method1, "(row):")
        for j, i in enumerate(best_match_2to1):
            print(f"  {method2} cluster {unique2[j]} best matches {method1} cluster {unique1[i]} (overlap: {overlap[i, j]})")
        # Plotly heatmap with best matches highlighted
        z = overlap
        hovertext = [[f"{method1} {unique1[i]} & {method2} {unique2[j]}: {z[i, j]}" for j in range(len(unique2))] for i in range(len(unique1))]
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=[f"{method2} {c}" for c in unique2],
            y=[f"{method1} {c}" for c in unique1],
            text=hovertext,
            hoverinfo='text',
            colorscale='Blues',
            showscale=True
        ))
        # Highlight best matches
        for i, j in enumerate(best_match_1to2):
            fig.add_shape(type="rect",
                x0=j-0.5, x1=j+0.5, y0=i-0.5, y1=i+0.5,
                line=dict(color="red", width=2), fillcolor="rgba(0,0,0,0)")
        fig.update_layout(
            title=f"Cluster Overlap: {method1} vs {method2}<br>ARI={ari:.3f}, NMI={nmi:.3f}",
            xaxis_title=method2,
            yaxis_title=method1,
            autosize=False,
            width=700,
            height=600
        )
        if filename is None:
            filename = f"compare_{method1}_vs_{method2}_heatmap.html"
        fig.write_html(filename)
        print(f"Interactive overlap heatmap saved to {filename}")
        return {'overlap_matrix': overlap, 'best_match_1to2': best_match_1to2, 'best_match_2to1': best_match_2to1, 'ari': ari, 'nmi': nmi}

    def validate_success_criteria(self, method='comprehensive', time_taken=None):
        """Validate cluster coherence (>0.7) and processing time (<30s) for a clustering method."""
        print(f"\n=== Success Criteria Validation for '{method}' ===")
        # Cluster coherence (Silhouette)
        metrics = self.compute_cluster_quality_metrics(method)
        sil = metrics['silhouette_score'] if metrics else None
        if sil is not None:
            print(f"Silhouette score: {sil:.3f} ({'PASS' if sil > 0.7 else 'FAIL'})")
        else:
            print("Silhouette score: N/A (FAIL)")
        # Processing time
        if time_taken is not None:
            print(f"Processing time: {time_taken:.2f} seconds ({'PASS' if time_taken < 30 else 'FAIL'})")
        else:
            print("Processing time: N/A")
        return {'silhouette_score': sil, 'silhouette_pass': sil is not None and sil > 0.7, 'processing_time': time_taken, 'processing_time_pass': time_taken is not None and time_taken < 30}

    def save_clustering_results_to_db(self, db: Session = None):
        """Save clustering results to the database instead of JSON file"""
        if not self.clusters:
            print("No clustering results to save.")
            return
        
        # Use provided session or create new one
        if db is None:
            db = SessionLocal()
            should_close = True
        else:
            should_close = False
        
        try:
            # Clear existing results for this run
            db.query(DiseaseClusteringResult).delete()
            db.commit()
            
            saved_count = 0
            
            for method, cluster_data in self.clusters.items():
                labels = cluster_data['labels']
                algorithm = cluster_data['algorithm']
                n_clusters = cluster_data['n_clusters']
                parameters = {
                    'n_clusters': n_clusters,
                    'algorithm': algorithm
                }
                
                # Add quality metrics if available
                if 'quality_metrics' in cluster_data:
                    parameters['quality_metrics'] = cluster_data['quality_metrics']
                
                # Process each record
                for i, (feature, label) in enumerate(zip(self.features, labels)):
                    # Extract data from feature
                    summary = feature.get('summary', '')
                    diseases = feature.get('diseases', [])
                    symptoms = feature.get('symptoms', [])
                    treatments = feature.get('treatments', [])
                    outcomes = feature.get('outcomes', [])
                    severity_score = feature.get('severity_score', 0)
                    complexity_score = feature.get('complexity_score', 0)
                    doctor_id = feature.get('doctor_id', None)
                    
                    # Create database record
                    clustering_result = DiseaseClusteringResult(
                        clustering_method=method,
                        cluster_label=int(label),
                        appointment_id=None,  # Could be linked if appointment_id is available in features
                        doctor_id=doctor_id,
                        summary=summary,
                        diseases=diseases,
                        symptoms=symptoms,
                        treatments=treatments,
                        outcomes=outcomes,
                        severity_score=severity_score,
                        complexity_score=complexity_score,
                        algorithm=algorithm,
                        parameters=parameters
                    )
                    
                    db.add(clustering_result)
                    saved_count += 1
            
            db.commit()
            print(f"Successfully saved {saved_count} clustering results to database for {len(self.clusters)} methods.")
            
        except Exception as e:
            db.rollback()
            print(f"Error saving clustering results to database: {e}")
            raise
        finally:
            if should_close:
                db.close()
    
    def get_clustering_results_from_db(self, method: str = None, db: Session = None):
        """Retrieve clustering results from database"""
        if db is None:
            db = SessionLocal()
            should_close = True
        else:
            should_close = False
        
        try:
            query = db.query(DiseaseClusteringResult)
            if method:
                query = query.filter(DiseaseClusteringResult.clustering_method == method)
            
            results = query.all()
            
            # Group by method and cluster
            grouped_results = {}
            for result in results:
                if result.clustering_method not in grouped_results:
                    grouped_results[result.clustering_method] = {}
                
                if result.cluster_label not in grouped_results[result.clustering_method]:
                    grouped_results[result.clustering_method][result.cluster_label] = []
                
                grouped_results[result.clustering_method][result.cluster_label].append({
                    'id': str(result.id),
                    'summary': result.summary,
                    'diseases': result.diseases,
                    'symptoms': result.symptoms,
                    'treatments': result.treatments,
                    'outcomes': result.outcomes,
                    'severity_score': result.severity_score,
                    'complexity_score': result.complexity_score,
                    'doctor_id': result.doctor_id,
                    'created_at': result.created_at.isoformat() if result.created_at else None
                })
            
            return grouped_results
            
        except Exception as e:
            print(f"Error retrieving clustering results from database: {e}")
            raise
        finally:
            if should_close:
                db.close()

def run_disease_clustering_pipeline():
    """
    Run the disease clustering pipeline and save results to database.
    """
    import time
    summary = {}
    engine = DiseaseClusteringEngine(cache_dir='./cache')
    cache_info = engine.get_cache_info()
    summary['cache_info_start'] = cache_info
    start_time = time.time()
    engine.load_data()
    engine.extract_medical_features(n_jobs=-1)
    engine.create_feature_matrix(use_sparse=True, add_tfidf=True, n_jobs=-1)
    engine.perform_symptom_based_clustering(optimize_params=True, n_jobs=-1)
    engine.perform_treatment_based_clustering(optimize_params=True, n_jobs=-1)
    engine.perform_outcome_based_clustering(optimize_params=True, n_jobs=-1)
    engine.perform_comprehensive_clustering(optimize_params=True, n_jobs=-1)
    engine.perform_fuzzy_cmeans_clustering(optimize_params=True, n_jobs=-1)
    report = engine.generate_cluster_report()
    summary['cluster_report'] = report
    metrics = engine.evaluate_all_clusters()
    summary['quality_metrics'] = metrics
    crossval = {}
    for method in ['symptom_based', 'treatment_based', 'outcome_based', 'comprehensive']:
        crossval[method] = engine.cross_validate_clustering(method)
    summary['cross_validation'] = crossval
    engine.visualize_clusters(method='comprehensive')
    
    # Save results to database instead of JSON file
    try:
        engine.save_clustering_results_to_db()
        summary['database_save'] = 'success'
    except Exception as e:
        summary['database_save'] = f'error: {str(e)}'
        print(f"Warning: Failed to save to database: {e}")
    
    final_cache_info = engine.get_cache_info()
    summary['cache_info_end'] = final_cache_info
    end_time = time.time()
    total_time = end_time - start_time
    summary['processing_time_sec'] = total_time
    summary['success_criteria'] = engine.validate_success_criteria(method='comprehensive', time_taken=total_time)
    return summary

if __name__ == "__main__":
    run_disease_clustering_pipeline() 