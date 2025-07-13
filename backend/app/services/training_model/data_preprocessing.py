import pandas as pd
import numpy as np
import re
import string
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import logging
from data_loader import AppointmentDataLoader
from outcome_labeling import OutcomeLabeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for doctor appointment summaries.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the preprocessor.
        
        Args:
            df: DataFrame with appointment data
        """
        self.df = df.copy()
        self.label_encoder = LabelEncoder()
        self.processed_df = None
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation except for medical abbreviations
        # Keep periods, commas, and hyphens for medical terms
        text = re.sub(r'[^\w\s\.\,\-]', '', text)
        
        # Normalize medical abbreviations
        text = re.sub(r'\bct\b', 'ct scan', text)
        text = re.sub(r'\bmri\b', 'mri scan', text)
        text = re.sub(r'\bx-ray\b', 'xray', text)
        text = re.sub(r'\bblood test\b', 'bloodtest', text)
        
        # Remove extra spaces
        text = text.strip()
        
        return text
    
    def extract_text_features(self, text: str) -> Dict[str, Any]:
        """
        Extract basic text features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of text features
        """
        if not text:
            return {
                'length': 0,
                'word_count': 0,
                'avg_word_length': 0,
                'sentence_count': 0
            }
        
        words = text.split()
        sentences = text.split('.')
        
        features = {
            'length': len(text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'sentence_count': len([s for s in sentences if s.strip()])
        }
        
        return features
    
    def create_medical_features(self, text: str) -> Dict[str, int]:
        """
        Create binary features for medical terms.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of binary medical features
        """
        text_lower = text.lower()
        
        # Chronic conditions
        chronic_conditions = {
            'has_diabetes': int('diabetes' in text_lower),
            'has_hypertension': int('hypertension' in text_lower),
            'has_asthma': int('asthma' in text_lower),
            'has_eczema': int('eczema' in text_lower),
            'has_migraines': int('migraine' in text_lower),
            'has_anemia': int('anemia' in text_lower)
        }
        
        # Acute symptoms
        acute_symptoms = {
            'has_headache': int('headache' in text_lower),
            'has_abdominal_pain': int('abdominal pain' in text_lower),
            'has_back_pain': int('back pain' in text_lower),
            'has_fatigue': int('fatigue' in text_lower),
            'has_dizziness': int('dizziness' in text_lower),
            'has_shortness_of_breath': int('shortness of breath' in text_lower),
            'has_cough': int('cough' in text_lower),
            'has_rash': int('rash' in text_lower)
        }
        
        # Diagnostic procedures
        diagnostic_procedures = {
            'has_xray': int(('x-ray' in text_lower) or ('xray' in text_lower)),
            'has_ct_scan': int('ct scan' in text_lower),
            'has_mri': int('mri' in text_lower),
            'has_blood_test': int('blood test' in text_lower),
            'has_ecg': int('ecg' in text_lower)
        }
        
        # Medications
        medications = {
            'has_amoxicillin': int('amoxicillin' in text_lower),
            'has_ibuprofen': int('ibuprofen' in text_lower),
            'has_paracetamol': int('paracetamol' in text_lower),
            'has_lisinopril': int('lisinopril' in text_lower),
            'has_metformin': int('metformin' in text_lower),
            'has_ventolin': int('ventolin' in text_lower)
        }
        
        # Treatment types
        treatment_types = {
            'has_prescription': int('prescribed' in text_lower),
            'has_referral': int('referral' in text_lower),
            'has_lifestyle': int('lifestyle' in text_lower),
            'has_dietary': int('dietary' in text_lower),
            'has_exercise': int('exercise' in text_lower)
        }
        
        # Combine all features
        all_features = {}
        all_features.update(chronic_conditions)
        all_features.update(acute_symptoms)
        all_features.update(diagnostic_procedures)
        all_features.update(medications)
        all_features.update(treatment_types)
        
        return all_features
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Apply complete preprocessing pipeline.
        
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing...")
        
        # Apply outcome labeling
        labeler = OutcomeLabeler(self.df)
        self.df = labeler.apply_labeling()
        
        # Clean summary text
        self.df['cleaned_summary'] = self.df['summary'].apply(self.clean_text)
        
        # Extract text features
        text_features = self.df['cleaned_summary'].apply(self.extract_text_features)
        text_features_df = pd.DataFrame(text_features.tolist())
        
        # Create medical features
        medical_features = self.df['cleaned_summary'].apply(self.create_medical_features)
        medical_features_df = pd.DataFrame(medical_features.tolist())
        
        # Encode outcome labels
        self.df['outcome_encoded'] = self.label_encoder.fit_transform(self.df['outcome_label'])
        
        # Combine all features
        self.processed_df = pd.concat([
            self.df[['id', 'doctor_id', 'summary', 'cleaned_summary', 'outcome_label', 'outcome_encoded']],
            text_features_df,
            medical_features_df
        ], axis=1)
        
        logger.info(f"Preprocessing completed. Final shape: {self.processed_df.shape}")
        return self.processed_df
    
    def create_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/test split.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.processed_df is None:
            self.processed_df = self.preprocess_data()
        
        # Stratified split based on outcome
        train_df, test_df = train_test_split(
            self.processed_df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.processed_df['outcome_encoded']
        )
        train_df = pd.DataFrame(train_df)
        test_df = pd.DataFrame(test_df)
        logger.info(f"Train/test split created. Train: {len(train_df)}, Test: {len(test_df)}")
        return train_df, test_df
    
    def create_cross_validation_folds(self, n_splits: int = 5, random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation folds.
        
        Args:
            n_splits: Number of CV folds
            random_state: Random seed for reproducibility
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        if self.processed_df is None:
            self.processed_df = self.preprocess_data()
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = []
        
        for train_idx, test_idx in skf.split(
            self.processed_df, 
            self.processed_df['outcome_encoded']
        ):
            folds.append((train_idx, test_idx))
        
        logger.info(f"Created {n_splits} cross-validation folds")
        return folds
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature columns for model training.
        
        Returns:
            List of feature column names
        """
        if self.processed_df is None:
            self.processed_df = self.preprocess_data()
        
        # Exclude non-feature columns
        exclude_cols = ['id', 'doctor_id', 'summary', 'cleaned_summary', 'outcome_label', 'outcome_encoded']
        feature_cols = [col for col in self.processed_df.columns if col not in exclude_cols]
        
        return feature_cols
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing steps and results.
        
        Returns:
            Dictionary with preprocessing summary
        """
        if self.processed_df is None:
            self.processed_df = self.preprocess_data()
        
        summary = {
            'original_shape': self.df.shape,
            'processed_shape': self.processed_df.shape,
            'feature_columns': len(self.get_feature_columns()),
            'outcome_distribution': self.processed_df['outcome_label'].value_counts().to_dict(),
            'text_features': ['length', 'word_count', 'avg_word_length', 'sentence_count'],
            'medical_features': len([col for col in self.processed_df.columns if col.startswith('has_')]),
            'missing_values': self.processed_df.isnull().sum().sum()
        }
        
        return summary

def main():
    """
    Main function to demonstrate preprocessing pipeline.
    """
    # Load data
    loader = AppointmentDataLoader()
    df = loader.to_dataframe()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(df)
    
    # Apply preprocessing
    processed_df = preprocessor.preprocess_data()
    
    # Create train/test split
    train_df, test_df = preprocessor.create_train_test_split()
    
    # Create CV folds
    cv_folds = preprocessor.create_cross_validation_folds()
    
    # Get summary
    summary = preprocessor.get_preprocessing_summary()
    
    # Print results
    print("=== Data Preprocessing Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print(f"\n=== Train/Test Split ===")
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Train outcome distribution: {train_df['outcome_label'].value_counts().to_dict()}")
    print(f"Test outcome distribution: {test_df['outcome_label'].value_counts().to_dict()}")
    
    print(f"\n=== Feature Information ===")
    feature_cols = preprocessor.get_feature_columns()
    print(f"Total features: {len(feature_cols)}")
    print(f"Text features: {len([col for col in feature_cols if col in ['length', 'word_count', 'avg_word_length', 'sentence_count']])}")
    print(f"Medical features: {len([col for col in feature_cols if col.startswith('has_')])}")
    
    print(f"\n=== Sample Processed Data ===")
    print(processed_df[['cleaned_summary', 'outcome_label', 'length', 'word_count']].head())
    
    return preprocessor, processed_df, train_df, test_df

if __name__ == "__main__":
    main() 