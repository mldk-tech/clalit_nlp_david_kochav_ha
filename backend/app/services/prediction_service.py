import joblib
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from gensim.models import Word2Vec
import scipy.sparse

logger = logging.getLogger(__name__)

class ClinicalFeatureExtractor:
    """Extracts medical and clinical features from appointment summaries."""
    
    def __init__(self):
        self.chronic_conditions = {
            'diabetes', 'hypertension', 'asthma', 'eczema', 'migraine', 'migraines', 'anemia'
        }
        self.acute_symptoms = {
            'headache', 'abdominal pain', 'back pain', 'fatigue', 'dizziness', 
            'shortness of breath', 'cough', 'rash'
        }
        self.severe_terms = {
            'critical', 'severe', 'emergency', 'intensive care', 'admitted', 
            'deteriorated', 'declined', 'passed away', 'deceased'
        }
        self.moderate_terms = {
            'persistent', 'unchanged', 'stable', 'monitoring', 'observation', 
            'manageable', 'minor', 'no significant progress'
        }
        self.mild_terms = {
            'improvement', 'improved', 'recovery', 'no complaints', 'doing well', 
            'subsided', 'complete', 'good health', 'no further issues'
        }

    def extract_features(self, text: str) -> Dict[str, int]:
        """Extract all clinical features from text."""
        text_lower = text.lower()
        
        features = {}
        
        # Chronic conditions
        for condition in self.chronic_conditions:
            features[f'has_{condition}'] = int(condition in text_lower)
        
        # Acute symptoms
        for symptom in self.acute_symptoms:
            features[f'has_{symptom.replace(" ", "_")}'] = int(symptom in text_lower)
        
        # Severity indicators
        features['has_severe_term'] = int(any(term in text_lower for term in self.severe_terms))
        features['has_moderate_term'] = int(any(term in text_lower for term in self.moderate_terms))
        features['has_mild_term'] = int(any(term in text_lower for term in self.mild_terms))
        
        # Diagnostics
        diagnostics = {
            'x-ray', 'xray', 'ct scan', 'mri', 'blood test', 'bloodtest', 'ecg', 'imaging'
        }
        for diagnostic in diagnostics:
            features[f'has_{diagnostic.replace(" ", "_").replace("-", "_")}'] = int(diagnostic in text_lower)
        
        # Test results
        features['test_result_normal'] = int('normal' in text_lower)
        features['test_result_abnormal'] = int(any(term in text_lower for term in ['abnormal', 'elevated', 'inflammation']))
        features['test_result_pending'] = int(any(term in text_lower for term in ['pending', 'awaiting']))
        
        # Treatments
        treatments = {
            'amoxicillin', 'ibuprofen', 'paracetamol', 'lisinopril', 'metformin', 'ventolin'
        }
        for treatment in treatments:
            features[f'has_{treatment}'] = int(treatment in text_lower)
        
        # Treatment types
        features['has_prescription'] = int('prescribed' in text_lower)
        features['has_referral'] = int('referral' in text_lower)
        features['has_lifestyle'] = int('lifestyle' in text_lower)
        features['has_dietary'] = int('dietary' in text_lower)
        features['has_exercise'] = int('exercise' in text_lower)
        
        # Specialist referrals
        specialists = ['cardiology', 'neurology', 'orthopedics', 'dermatology']
        for specialist in specialists:
            features[f'referral_{specialist}'] = int(specialist in text_lower)
        
        # Clinical patterns
        features['is_initial_assessment'] = int(any(term in text_lower for term in ['initial assessment', 'consultation']))
        features['is_follow_up'] = int(any(term in text_lower for term in ['follow-up', 'follow up']))
        features['is_test_discussion'] = int(any(term in text_lower for term in ['test results', 'lab results']))
        
        # Text characteristics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['has_medication'] = int('medication' in text_lower or 'prescribed' in text_lower)
        features['has_symptoms'] = int(any(symptom in text_lower for symptom in self.acute_symptoms))
        
        return features

class TextFeatureEngineer:
    """Text feature engineering for prediction."""
    
    def __init__(self, tfidf_vectorizer_path: Optional[str] = None, 
                 word2vec_model_path: Optional[str] = None):
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        
        # Load pre-trained models if available
        if tfidf_vectorizer_path and os.path.exists(tfidf_vectorizer_path):
            try:
                self.tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
                logger.info(f"Loaded TF-IDF vectorizer from {tfidf_vectorizer_path}")
            except Exception as e:
                logger.warning(f"Failed to load TF-IDF vectorizer: {e}")
        
        if word2vec_model_path and os.path.exists(word2vec_model_path):
            try:
                self.word2vec_model = Word2Vec.load(word2vec_model_path)
                logger.info(f"Loaded Word2Vec model from {word2vec_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load Word2Vec model: {e}")
    
    def extract_tfidf_features(self, text: str) -> Dict[str, float]:
        """Extract TF-IDF features from text."""
        if self.tfidf_vectorizer is None:
            return {}
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.transform([text])
            if scipy.sparse.issparse(tfidf_matrix):
                tfidf_array = tfidf_matrix.toarray()
            else:
                tfidf_array = np.array(tfidf_matrix)
            
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            features = {}
            for i, name in enumerate(feature_names):
                features[f"tfidf_{name}"] = float(tfidf_array[0, i])
            
            return features
        except Exception as e:
            logger.error(f"Error extracting TF-IDF features: {e}")
            return {}
    
    def extract_word2vec_features(self, text: str, vector_size: int = 50) -> Dict[str, float]:
        """Extract Word2Vec embedding features from text."""
        if self.word2vec_model is None:
            return {}
        
        try:
            tokens = text.split()
            vectors = [self.word2vec_model.wv[word] for word in tokens if word in self.word2vec_model.wv]
            
            if vectors:
                avg_vector = np.mean(vectors, axis=0)
            else:
                avg_vector = np.zeros(vector_size)
            
            features = {}
            for i in range(vector_size):
                features[f"w2v_{i}"] = float(avg_vector[i])
            
            return features
        except Exception as e:
            logger.error(f"Error extracting Word2Vec features: {e}")
            return {}

class PredictionService:
    """Main prediction service that integrates models and feature extraction."""
    
    def __init__(self, models_dir: str = "backend/app/services/training_model/results"):
        self.models_dir = models_dir
        self.models = {}
        self.feature_extractor = ClinicalFeatureExtractor()
        self.text_engineer = TextFeatureEngineer()
        self.feature_names = None
        self.load_models()
    
    def load_models(self):
        """Load trained models from disk."""
        model_paths = {
            'random_forest': os.path.join(self.models_dir, 'random_forest_model.pkl'),
            'xgboost': os.path.join(self.models_dir, 'xgboost_model.pkl')
        }
        
        for model_name, model_path in model_paths.items():
            if os.path.exists(model_path):
                try:
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} model from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load {model_name} model: {e}")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        # Load feature names if available
        feature_names_path = os.path.join(self.models_dir, 'feature_names.pkl')
        if os.path.exists(feature_names_path):
            try:
                self.feature_names = joblib.load(feature_names_path)
                logger.info(f"Loaded feature names: {len(self.feature_names)} features")
            except Exception as e:
                logger.error(f"Failed to load feature names: {e}")
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract all features from text using the same pipeline as training."""
        # Clinical features
        clinical_features = self.feature_extractor.extract_features(text)
        
        # Text features (TF-IDF and Word2Vec)
        tfidf_features = self.text_engineer.extract_tfidf_features(text)
        word2vec_features = self.text_engineer.extract_word2vec_features(text)
        
        # Combine all features
        all_features = {**clinical_features, **tfidf_features, **word2vec_features}
        
        return all_features
    
    def prepare_features_for_model(self, features: Dict[str, Any], model_name: str) -> np.ndarray:
        """Prepare features for model prediction."""
        # Convert features to DataFrame
        df = pd.DataFrame([features])
        
        # Ensure we have the expected feature names for the model
        if self.feature_names is not None:
            # Add missing features with 0 values
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Select only the expected features in the correct order
            df = df[self.feature_names]
        
        # Convert to numpy array
        X = df.values
        
        # Handle XGBoost specific requirements
        if model_name == 'xgboost':
            # Ensure only numeric features
            X = X.astype(np.float32)
        
        return X
    
    def predict_with_fallback(self, text: str) -> Dict[str, Any]:
        """Make prediction using clinical features only (fallback when models not available)."""
        try:
            # Extract clinical features
            features = self.feature_extractor.extract_features(text)
            
            # Simple rule-based prediction based on clinical features
            score = 0
            
            # Positive indicators
            if features.get('has_severe_term', 0):
                score += 3
            if features.get('has_abnormal', 0):
                score += 2
            if features.get('has_referral', 0):
                score += 1
            if features.get('has_prescription', 0):
                score += 1
            if features.get('text_length', 0) > 100:
                score += 1
            
            # Negative indicators
            if features.get('has_mild_term', 0):
                score -= 2
            if features.get('test_result_normal', 0):
                score -= 1
            if features.get('is_follow_up', 0):
                score -= 1
            
            # Determine prediction
            prediction = "positive" if score > 0 else "negative"
            confidence = min(0.95, max(0.05, abs(score) / 5.0))  # Normalize confidence
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": {
                    "negative": 1 - confidence if prediction == "positive" else confidence,
                    "positive": confidence if prediction == "positive" else 1 - confidence
                },
                "model_name": "clinical_rules",
                "features_used": features,
                "feature_importance": {k: 1.0 for k in features.keys()},
                "text_length": len(text),
                "word_count": len(text.split()),
                "method": "clinical_rules_fallback"
            }
            
        except Exception as e:
            logger.error(f"Fallback prediction failed: {e}")
            raise ValueError(f"Fallback prediction failed: {str(e)}")

    def predict(self, text: str, model_name: str = 'random_forest') -> Dict[str, Any]:
        """Make prediction using the specified model."""
        if model_name not in self.models:
            if not self.models:
                # If no models are available, use fallback
                logger.warning("No trained models available, using clinical rules fallback")
                return self.predict_with_fallback(text)
            else:
                raise ValueError(f"Model '{model_name}' not available. Available models: {list(self.models.keys())}")
        
        try:
            # Extract features
            features = self.extract_features(text)
            
            # Prepare features for model
            X = self.prepare_features_for_model(features, model_name)
            
            # Make prediction
            model = self.models[model_name]
            
            # Get prediction and probability
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            # Determine outcome label
            outcome_label = "positive" if prediction == 1 else "negative"
            confidence = float(max(probabilities))
            
            # Get feature importance if available
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                if self.feature_names is not None:
                    for i, importance in enumerate(model.feature_importances_):
                        if i < len(self.feature_names):
                            feature_importance[self.feature_names[i]] = float(importance)
                else:
                    feature_importance = {f"feature_{i}": float(importance) 
                                       for i, importance in enumerate(model.feature_importances_)}
            
            return {
                "prediction": outcome_label,
                "confidence": confidence,
                "probabilities": {
                    "negative": float(probabilities[0]),
                    "positive": float(probabilities[1])
                },
                "model_name": model_name,
                "features_used": features,
                "feature_importance": feature_importance,
                "text_length": len(text),
                "word_count": len(text.split())
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def predict_with_ensemble(self, text: str) -> Dict[str, Any]:
        """Make prediction using ensemble of available models."""
        predictions = {}
        confidences = {}
        
        for model_name in self.models.keys():
            try:
                result = self.predict(text, model_name)
                predictions[model_name] = result["prediction"]
                confidences[model_name] = result["confidence"]
            except Exception as e:
                logger.error(f"Prediction failed for {model_name}: {e}")
                predictions[model_name] = "error"
                confidences[model_name] = 0.0
        
        # Ensemble decision (majority vote)
        positive_votes = sum(1 for pred in predictions.values() if pred == "positive")
        total_votes = len(predictions)
        
        ensemble_prediction = "positive" if positive_votes > total_votes / 2 else "negative"
        ensemble_confidence = sum(confidences.values()) / len(confidences) if confidences else 0.0
        
        return {
            "prediction": ensemble_prediction,
            "confidence": ensemble_confidence,
            "individual_predictions": predictions,
            "individual_confidences": confidences,
            "ensemble_method": "majority_vote"
        }
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with metadata."""
        models_info = []
        
        for model_name, model in self.models.items():
            model_info = {
                "name": model_name,
                "type": type(model).__name__,
                "available": True,
                "feature_count": len(self.feature_names) if self.feature_names else "unknown"
            }
            
            # Add model-specific information
            if hasattr(model, 'n_estimators'):
                model_info["n_estimators"] = model.n_estimators
            if hasattr(model, 'max_depth'):
                model_info["max_depth"] = model.max_depth
            
            models_info.append(model_info)
        
        return models_info
    
    def get_model_health(self) -> Dict[str, Any]:
        """Get health status of prediction models."""
        return {
            "total_models": len(self.models),
            "available_models": list(self.models.keys()),
            "feature_names_loaded": self.feature_names is not None,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "models_dir": self.models_dir,
            "status": "healthy" if self.models else "unhealthy"
        } 