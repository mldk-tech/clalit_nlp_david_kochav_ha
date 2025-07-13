import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from typing import Dict, List, Tuple, Any
import logging
from data_loader import AppointmentDataLoader
import numpy as np
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExploratoryDataAnalysis:
    """
    Exploratory Data Analysis for doctor appointment summaries dataset.
    """
    
    def __init__(self, data_loader: AppointmentDataLoader):
        """
        Initialize the EDA module.
        
        Args:
            data_loader: Initialized AppointmentDataLoader instance
        """
        self.data_loader = data_loader
        self.df = data_loader.to_dataframe()
        
    def analyze_doctor_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of appointments across doctors.
        
        Returns:
            Dictionary with doctor distribution analysis
        """
        doctor_counts = self.df['doctor_id'].value_counts()
        
        analysis = {
            'total_doctors': len(doctor_counts),
            'appointments_per_doctor': doctor_counts.to_dict(),
            'mean_appointments': doctor_counts.mean(),
            'std_appointments': doctor_counts.std(),
            'min_appointments': doctor_counts.min(),
            'max_appointments': doctor_counts.max()
        }
        
        logger.info(f"Doctor distribution analysis completed. Found {analysis['total_doctors']} doctors.")
        return analysis
    
    def analyze_summary_characteristics(self) -> Dict[str, Any]:
        """
        Analyze the characteristics of appointment summaries.
        
        Returns:
            Dictionary with summary analysis
        """
        summary_lengths = self.df['summary'].str.len()
        word_counts = self.df['summary'].str.split().str.len()
        
        analysis = {
            'length_stats': {
                'mean': summary_lengths.mean(),
                'median': summary_lengths.median(),
                'min': summary_lengths.min(),
                'max': summary_lengths.max(),
                'std': summary_lengths.std()
            },
            'word_count_stats': {
                'mean': word_counts.mean(),
                'median': word_counts.median(),
                'min': word_counts.min(),
                'max': word_counts.max(),
                'std': word_counts.std()
            },
            'unique_summaries': self.df['summary'].nunique(),
            'duplicate_summaries': len(self.df) - self.df['summary'].nunique()
        }
        
        logger.info(f"Summary analysis completed. Average length: {analysis['length_stats']['mean']:.1f} characters")
        return analysis
    
    def analyze_outcome_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in future outcomes.
        
        Returns:
            Dictionary with outcome analysis
        """
        outcome_counts = self.df['future_outcome'].value_counts()
        
        # Categorize outcomes
        positive_keywords = [
            'responded positively', 'returned to baseline', 'feeling much better',
            'recovery is on track', 'no complaints', 'doing well', 'subsided',
            'discharged in good condition', 'excellent recovery'
        ]
        
        negative_keywords = [
            'deteriorated', 'deceased', 'passed away', 'critical condition',
            'emergency intervention', 'worsened', 'declined', 'admitted'
        ]
        
        neutral_keywords = [
            'unchanged', 'persistent but manageable', 'stable', 'no significant progress'
        ]
        
        def categorize_outcome(outcome):
            outcome_lower = outcome.lower()
            if any(keyword in outcome_lower for keyword in positive_keywords):
                return 'positive'
            elif any(keyword in outcome_lower for keyword in negative_keywords):
                return 'negative'
            elif any(keyword in outcome_lower for keyword in neutral_keywords):
                return 'neutral'
            else:
                return 'uncategorized'
        
        self.df['outcome_category'] = self.df['future_outcome'].apply(categorize_outcome)
        category_counts = self.df['outcome_category'].value_counts()
        
        analysis = {
            'total_unique_outcomes': len(outcome_counts),
            'most_common_outcomes': outcome_counts.head(10).to_dict(),
            'outcome_categories': category_counts.to_dict(),
            'category_distribution': (category_counts / len(self.df) * 100).to_dict()
        }
        
        logger.info(f"Outcome analysis completed. Found {analysis['total_unique_outcomes']} unique outcomes")
        return analysis
    
    def identify_medical_terms(self) -> Dict[str, Dict[str, int]]:
        """
        Identify common medical terms in the summaries.
        
        Returns:
            Dictionary with medical terms by category
        """
        # Medical conditions
        chronic_conditions = ['diabetes', 'hypertension', 'asthma', 'eczema', 'migraines', 'anemia']
        acute_symptoms = ['headache', 'abdominal pain', 'back pain', 'fatigue', 'dizziness', 
                         'shortness of breath', 'cough', 'rash']
        
        # Diagnostic procedures
        imaging_tests = ['x-ray', 'ct scan', 'mri', 'imaging']
        lab_tests = ['blood test', 'ecg', 'laboratory']
        
        # Medications
        medications = ['amoxicillin', 'ibuprofen', 'paracetamol', 'lisinopril', 'metformin', 'ventolin']
        
        # Treatment types
        treatments = ['prescribed', 'referral', 'lifestyle', 'dietary', 'exercise']
        
        def find_terms_in_text(text, terms):
            found_terms = []
            text_lower = text.lower()
            for term in terms:
                if term in text_lower:
                    found_terms.append(term)
            return found_terms
        
        # Count occurrences
        term_counts = {
            'chronic_conditions': Counter(),
            'acute_symptoms': Counter(),
            'imaging_tests': Counter(),
            'lab_tests': Counter(),
            'medications': Counter(),
            'treatments': Counter()
        }
        
        for summary in self.df['summary']:
            for term in find_terms_in_text(summary, chronic_conditions):
                term_counts['chronic_conditions'][term] += 1
            for term in find_terms_in_text(summary, acute_symptoms):
                term_counts['acute_symptoms'][term] += 1
            for term in find_terms_in_text(summary, imaging_tests):
                term_counts['imaging_tests'][term] += 1
            for term in find_terms_in_text(summary, lab_tests):
                term_counts['lab_tests'][term] += 1
            for term in find_terms_in_text(summary, medications):
                term_counts['medications'][term] += 1
            for term in find_terms_in_text(summary, treatments):
                term_counts['treatments'][term] += 1
        
        analysis = {
            'chronic_conditions': dict(term_counts['chronic_conditions']),
            'acute_symptoms': dict(term_counts['acute_symptoms']),
            'imaging_tests': dict(term_counts['imaging_tests']),
            'lab_tests': dict(term_counts['lab_tests']),
            'medications': dict(term_counts['medications']),
            'treatments': dict(term_counts['treatments'])
        }
        
        logger.info("Medical terms analysis completed")
        return analysis
    
    def create_visualizations(self, save_path: str = "eda_plots"):
        """
        Create and save visualizations for the dataset.
        
        Args:
            save_path: Directory to save plots
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Doctor distribution
        plt.figure(figsize=(12, 6))
        doctor_counts = self.df['doctor_id'].value_counts()
        plt.bar(np.asarray(doctor_counts.index), np.asarray(doctor_counts.values))
        plt.title('Distribution of Appointments Across Doctors')
        plt.xlabel('Doctor ID')
        plt.ylabel('Number of Appointments')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_path}/doctor_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Summary length distribution
        plt.figure(figsize=(10, 6))
        summary_lengths = self.df['summary'].str.len()
        plt.hist(summary_lengths, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Summary Lengths')
        plt.xlabel('Number of Characters')
        plt.ylabel('Frequency')
        plt.axvline(summary_lengths.mean(), color='red', linestyle='--', 
                   label=f'Mean: {summary_lengths.mean():.1f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}/summary_length_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Outcome category distribution
        plt.figure(figsize=(10, 6))
        category_counts = self.df['outcome_category'].value_counts()
        plt.pie(np.asarray(category_counts.values), labels=[str(x) for x in category_counts.index], autopct='%1.1f%%')
        plt.title('Distribution of Outcome Categories')
        plt.tight_layout()
        plt.savefig(f"{save_path}/outcome_category_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Medical terms heatmap
        medical_terms = self.identify_medical_terms()
        
        # Create a matrix for heatmap
        term_categories = ['chronic_conditions', 'acute_symptoms', 'imaging_tests', 
                          'lab_tests', 'medications', 'treatments']
        
        # Count total occurrences per category
        category_totals = {}
        for category in term_categories:
            category_totals[category] = sum(medical_terms[category].values())
        
        plt.figure(figsize=(12, 6))
        categories = list(category_totals.keys())
        counts = list(category_totals.values())
        
        plt.bar(categories, counts)
        plt.title('Medical Terms by Category')
        plt.xlabel('Medical Term Category')
        plt.ylabel('Total Occurrences')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_path}/medical_terms_by_category.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {save_path}")
    
    def generate_eda_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive EDA report.
        
        Returns:
            Dictionary with complete EDA analysis
        """
        logger.info("Starting comprehensive EDA analysis...")
        
        report = {
            'dataset_overview': {
                'total_records': len(self.df),
                'total_doctors': self.df['doctor_id'].nunique(),
                'unique_outcomes': self.df['future_outcome'].nunique(),
                'date_range': 'Not available (no date field)'
            },
            'doctor_analysis': self.analyze_doctor_distribution(),
            'summary_analysis': self.analyze_summary_characteristics(),
            'outcome_analysis': self.analyze_outcome_patterns(),
            'medical_terms_analysis': self.identify_medical_terms(),
            'data_quality': {
                'missing_values': self.df.isnull().sum().to_dict(),
                'duplicate_records': self.df.duplicated().sum(),
                'unique_ids': self.df['id'].nunique()
            }
        }
        
        logger.info("EDA report generated successfully")
        return report

def main():
    """
    Main function to run comprehensive EDA.
    """
    # Initialize data loader
    loader = AppointmentDataLoader()
    
    # Initialize EDA
    eda = ExploratoryDataAnalysis(loader)
    
    # Generate comprehensive report
    report = eda.generate_eda_report()
    
    # Create visualizations
    eda.create_visualizations()
    
    # Print summary
    print("=== EDA Summary ===")
    print(f"Total records: {report['dataset_overview']['total_records']}")
    print(f"Total doctors: {report['dataset_overview']['total_doctors']}")
    print(f"Unique outcomes: {report['dataset_overview']['unique_outcomes']}")
    
    print(f"\n=== Outcome Categories ===")
    for category, count in report['outcome_analysis']['outcome_categories'].items():
        percentage = report['outcome_analysis']['category_distribution'][category]
        print(f"{category}: {count} ({percentage:.1f}%)")
    
    print(f"\n=== Most Common Medical Terms ===")
    for category, terms in report['medical_terms_analysis'].items():
        if terms:
            most_common = max(terms.items(), key=lambda x: x[1])
            print(f"{category}: {most_common[0]} ({most_common[1]} occurrences)")
    
    return eda, report

if __name__ == "__main__":
    main() 