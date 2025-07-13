import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalFeatureExtractor:
    """
    Extracts medical and clinical features from appointment summaries.
    """
    def __init__(self, df: pd.DataFrame, text_column: str = 'cleaned_summary'):
        self.df = df.copy()
        self.text_column = text_column

    def extract_chronic_conditions(self, text: str) -> Dict[str, int]:
        chronic = {
            'has_diabetes': int('diabetes' in text),
            'has_hypertension': int('hypertension' in text),
            'has_asthma': int('asthma' in text),
            'has_eczema': int('eczema' in text),
            'has_migraines': int('migraine' in text or 'migraines' in text),
            'has_anemia': int('anemia' in text)
        }
        return chronic

    def extract_acute_symptoms(self, text: str) -> Dict[str, int]:
        acute = {
            'has_headache': int('headache' in text),
            'has_abdominal_pain': int('abdominal pain' in text),
            'has_back_pain': int('back pain' in text),
            'has_fatigue': int('fatigue' in text),
            'has_dizziness': int('dizziness' in text),
            'has_shortness_of_breath': int('shortness of breath' in text),
            'has_cough': int('cough' in text),
            'has_rash': int('rash' in text)
        }
        return acute

    def extract_condition_severity(self, text: str) -> Dict[str, int]:
        # Severe terms
        severe_terms = [
            'critical', 'severe', 'emergency', 'intensive care', 'admitted', 'deteriorated', 'declined', 'passed away', 'deceased'
        ]
        moderate_terms = [
            'persistent', 'unchanged', 'stable', 'monitoring', 'observation', 'manageable', 'minor', 'no significant progress'
        ]
        mild_terms = [
            'improvement', 'improved', 'recovery', 'no complaints', 'doing well', 'subsided', 'complete', 'good health', 'no further issues'
        ]
        text_lower = text.lower()
        severity = {
            'has_severe_term': int(any(term in text_lower for term in severe_terms)),
            'has_moderate_term': int(any(term in text_lower for term in moderate_terms)),
            'has_mild_term': int(any(term in text_lower for term in mild_terms))
        }
        return severity

    def extract_diagnostics(self, text: str) -> Dict[str, int]:
        diagnostics = {
            'has_xray': int('x-ray' in text or 'xray' in text),
            'has_ct_scan': int('ct scan' in text),
            'has_mri': int('mri' in text),
            'has_blood_test': int('blood test' in text or 'bloodtest' in text),
            'has_ecg': int('ecg' in text),
            'has_imaging': int('imaging' in text)
        }
        # Test result indicators
        diagnostics['test_result_normal'] = int('normal' in text)
        diagnostics['test_result_abnormal'] = int('abnormal' in text or 'elevated' in text or 'inflammation' in text)
        diagnostics['test_result_pending'] = int('pending' in text or 'awaiting' in text)
        return diagnostics

    def extract_treatments(self, text: str) -> Dict[str, int]:
        treatments = {
            'has_amoxicillin': int('amoxicillin' in text),
            'has_ibuprofen': int('ibuprofen' in text),
            'has_paracetamol': int('paracetamol' in text),
            'has_lisinopril': int('lisinopril' in text),
            'has_metformin': int('metformin' in text),
            'has_ventolin': int('ventolin' in text),
            'has_prescription': int('prescribed' in text),
            'has_referral': int('referral' in text),
            'has_lifestyle': int('lifestyle' in text),
            'has_dietary': int('dietary' in text),
            'has_exercise': int('exercise' in text)
        }
        # Specialist referrals
        treatments['referral_cardiology'] = int('cardiology' in text)
        treatments['referral_neurology'] = int('neurology' in text)
        treatments['referral_orthopedics'] = int('orthopedics' in text)
        treatments['referral_dermatology'] = int('dermatology' in text)
        return treatments

    def extract_clinical_patterns(self, text: str) -> Dict[str, Any]:
        patterns = {
            'is_initial_assessment': int('initial assessment' in text or 'consultation' in text),
            'is_follow_up': int('follow-up' in text or 'follow up' in text),
            'is_test_result_discussion': int('test result' in text or 'discussed test results' in text),
            'has_time_reference': int(bool(re.search(r'\b(week|month|day|year|reassess|follow-up|follow up|return)\b', text))),
            'has_followup_scheduling': int('reassess' in text or 'follow-up' in text or 'follow up' in text or 'schedule' in text)
        }
        # Clinical language complexity
        word_count = len(text.split())
        med_terms = [
            'diabetes', 'hypertension', 'asthma', 'eczema', 'migraines', 'anemia',
            'headache', 'abdominal pain', 'back pain', 'fatigue', 'dizziness', 'shortness of breath', 'cough', 'rash',
            'x-ray', 'ct scan', 'mri', 'blood test', 'ecg', 'imaging',
            'amoxicillin', 'ibuprofen', 'paracetamol', 'lisinopril', 'metformin', 'ventolin',
            'prescribed', 'referral', 'lifestyle', 'dietary', 'exercise',
            'cardiology', 'neurology', 'orthopedics', 'dermatology'
        ]
        med_term_count = sum(1 for term in med_terms if term in text)
        patterns['medical_term_density'] = int(med_term_count / word_count) if word_count else 0
        patterns['text_complexity'] = word_count
        return patterns

    def extract_all_features(self) -> pd.DataFrame:
        logger.info("Extracting all clinical and medical features...")
        features = []
        for text in self.df[self.text_column]:
            text = text.lower() if isinstance(text, str) else ''
            row_features = {}
            row_features.update(self.extract_chronic_conditions(text))
            row_features.update(self.extract_acute_symptoms(text))
            row_features.update(self.extract_condition_severity(text))
            row_features.update(self.extract_diagnostics(text))
            row_features.update(self.extract_treatments(text))
            row_features.update(self.extract_clinical_patterns(text))
            features.append(row_features)
        features_df = pd.DataFrame(features)
        logger.info(f"Extracted features shape: {features_df.shape}")
        return pd.concat([self.df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

def main():
    from data_loader import AppointmentDataLoader
    from data_preprocessing import DataPreprocessor
    # Load and preprocess data
    loader = AppointmentDataLoader()
    df = loader.to_dataframe()
    preprocessor = DataPreprocessor(df)
    processed_df = preprocessor.preprocess_data()
    # Extract clinical features
    extractor = ClinicalFeatureExtractor(processed_df)
    features_df = extractor.extract_all_features()
    print(features_df.head())
    print(f"\nExtracted feature columns: {features_df.columns.tolist()[-20:]}")
    # Save to CSV
    features_df.to_csv('clinical_feature_extraction.csv', index=False)

if __name__ == "__main__":
    main() 