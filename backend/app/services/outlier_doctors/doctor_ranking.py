"""
Doctor Ranking and Outlier Detection System

This module provides functionality to:
1. Load and analyze doctor appointment data
2. Score doctors based on patient outcomes
3. Rank doctors by performance
4. Detect outlier doctors (both good and bad performers)
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import statistics
from enum import Enum


class OutcomeCategory(Enum):
    """Enumeration for categorizing patient outcomes"""
    EXCELLENT = "excellent"
    GOOD = "good"
    NEUTRAL = "neutral"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class DoctorScore:
    """Data class to hold doctor performance metrics"""
    doctor_id: str
    total_cases: int
    outcome_scores: Dict[OutcomeCategory, int]
    average_score: float
    weighted_score: float
    rank: Optional[int] = None
    is_outlier: bool = False
    outlier_type: Optional[str] = None


class OutcomeScorer:
    """Handles the scoring of different patient outcomes"""
    
    def __init__(self):
        self.outcome_mapping = {
            # Excellent outcomes (score: 5)
            "Excellent recovery progress, no ongoing concerns.": OutcomeCategory.EXCELLENT,
            "Symptoms resolved, patient discharged in good condition.": OutcomeCategory.EXCELLENT,
            
            # Good outcomes (score: 4)
            "Patient is feeling much better and has resumed daily activities.": OutcomeCategory.GOOD,
            "Health has returned to baseline with no complications.": OutcomeCategory.GOOD,
            "Symptoms have subsided and recovery is complete.": OutcomeCategory.GOOD,
            "Recovery is on track, no concerns noted.": OutcomeCategory.GOOD,
            "Patient responded positively to treatment.": OutcomeCategory.GOOD,
            "No complaints at follow-up, patient in good health.": OutcomeCategory.GOOD,
            "No further issues reported, patient doing well.": OutcomeCategory.GOOD,
            "Stable condition with signs of improvement.": OutcomeCategory.GOOD,
            "Patient reports minor improvements but overall stable.": OutcomeCategory.GOOD,
            
            # Neutral outcomes (score: 3)
            "Condition remains unchanged, monitoring continues.": OutcomeCategory.NEUTRAL,
            "Symptoms are persistent but manageable.": OutcomeCategory.NEUTRAL,
            "No significant progress, further observation required.": OutcomeCategory.NEUTRAL,
            
            # Poor outcomes (score: 2)
            "Condition has deteriorated since last visit.": OutcomeCategory.POOR,
            "Symptoms worsened despite intervention.": OutcomeCategory.POOR,
            "No improvement noted; escalation of care necessary.": OutcomeCategory.POOR,
            "Patient's state declined; alternative treatment considered.": OutcomeCategory.POOR,
            
            # Critical outcomes (score: 1)
            "Patient admitted for further evaluation due to worsening condition.": OutcomeCategory.CRITICAL,
            "Patient is in critical condition and under intensive care.": OutcomeCategory.CRITICAL,
            "Emergency intervention required due to severe decline.": OutcomeCategory.CRITICAL,
            "Health continues to decline under current treatment.": OutcomeCategory.CRITICAL,
            "Patient deceased following complications.": OutcomeCategory.CRITICAL,
            "Despite efforts, the patient passed away.": OutcomeCategory.CRITICAL,
        }
        
        self.score_values = {
            OutcomeCategory.EXCELLENT: 5,
            OutcomeCategory.GOOD: 4,
            OutcomeCategory.NEUTRAL: 3,
            OutcomeCategory.POOR: 2,
            OutcomeCategory.CRITICAL: 1
        }
    
    def get_outcome_category(self, outcome: str) -> OutcomeCategory:
        """Map outcome text to category"""
        return self.outcome_mapping.get(outcome, OutcomeCategory.NEUTRAL)
    
    def get_score(self, outcome: str) -> int:
        """Get numerical score for an outcome"""
        category = self.get_outcome_category(outcome)
        return self.score_values[category]


class DoctorAnalyzer:
    """Main class for analyzing doctor performance"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.scorer = OutcomeScorer()
        self.data = None
        self.doctor_scores = {}
        
    def load_data(self) -> None:
        """Load appointment data from JSON file"""
        try:
            with open(self.data_file, 'r') as f:
                self.data = json.load(f)
            print(f"Loaded {len(self.data)} appointment records")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file {self.data_file} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {self.data_file}")
    
    def calculate_doctor_scores(self) -> Dict[str, DoctorScore]:
        """Calculate performance scores for all doctors"""
        if not self.data:
            return {}
            
        doctor_data: Dict[str, Dict] = {}
        
        # Initialize doctor data structure
        for record in self.data:
            doctor_id = record['doctor_id']
            if doctor_id not in doctor_data:
                doctor_data[doctor_id] = {
                    'cases': [],
                    'outcomes': defaultdict(int),
                    'scores': []
                }
        
        # Group data by doctor
        for record in self.data:
            doctor_id = record['doctor_id']
            outcome = record['future_outcome']
            score = self.scorer.get_score(outcome)
            category = self.scorer.get_outcome_category(outcome)
            
            doctor_data[doctor_id]['cases'].append(record)
            doctor_data[doctor_id]['outcomes'][category] += 1
            doctor_data[doctor_id]['scores'].append(score)
        
        # Calculate metrics for each doctor
        for doctor_id, data in doctor_data.items():
            total_cases = len(data['cases'])
            scores = data['scores']
            
            # Calculate average score
            average_score = np.mean(scores) if scores else 0
            
            # Calculate weighted score (considering case volume)
            weighted_score = average_score * np.log(total_cases + 1)  # Log to reduce impact of very high case counts
            
            doctor_score = DoctorScore(
                doctor_id=doctor_id,
                total_cases=total_cases,
                outcome_scores=dict(data['outcomes']),
                average_score=round(average_score, 3),
                weighted_score=round(weighted_score, 3)
            )
            
            self.doctor_scores[doctor_id] = doctor_score
        
        return self.doctor_scores
    
    def rank_doctors(self) -> List[DoctorScore]:
        """Rank doctors by weighted score"""
        if not self.doctor_scores:
            self.calculate_doctor_scores()
        
        # Sort by weighted score (descending)
        ranked_doctors = sorted(
            self.doctor_scores.values(),
            key=lambda x: x.weighted_score,
            reverse=True
        )
        
        # Assign ranks
        for i, doctor in enumerate(ranked_doctors):
            doctor.rank = i + 1
        
        return ranked_doctors
    
    def detect_outliers(self, method: str = 'iqr') -> List[DoctorScore]:
        """Detect outlier doctors using specified method"""
        if not self.doctor_scores:
            self.calculate_doctor_scores()
        
        scores = [doc.weighted_score for doc in self.doctor_scores.values()]
        
        if method == 'iqr':
            outliers = self._detect_outliers_iqr(scores)
        elif method == 'zscore':
            outliers = self._detect_outliers_zscore(scores)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        # Mark outliers in doctor scores
        for doctor in self.doctor_scores.values():
            if doctor.weighted_score in outliers['high']:
                doctor.is_outlier = True
                doctor.outlier_type = "good"
            elif doctor.weighted_score in outliers['low']:
                doctor.is_outlier = True
                doctor.outlier_type = "bad"
        
        return [doc for doc in self.doctor_scores.values() if doc.is_outlier]
    
    def _detect_outliers_iqr(self, scores: List[float]) -> Dict[str, List[float]]:
        """Detect outliers using IQR method"""
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = {
            'low': [s for s in scores if s < lower_bound],
            'high': [s for s in scores if s > upper_bound]
        }
        
        return outliers
    
    def _detect_outliers_zscore(self, scores: List[float], threshold: float = 2.0) -> Dict[str, List[float]]:
        """Detect outliers using Z-score method"""
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        outliers = {
            'low': [s for s in scores if (s - mean_score) / std_score < -threshold],
            'high': [s for s in scores if (s - mean_score) / std_score > threshold]
        }
        
        return outliers
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of doctor rankings and outliers"""
        ranked_doctors = self.rank_doctors()
        outliers = self.detect_outliers()
        
        report = []
        report.append("=" * 80)
        report.append("DOCTOR PERFORMANCE RANKING REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS:")
        report.append(f"Total doctors analyzed: {len(ranked_doctors)}")
        report.append(f"Total cases analyzed: {sum(doc.total_cases for doc in ranked_doctors)}")
        report.append(f"Average cases per doctor: {np.mean([doc.total_cases for doc in ranked_doctors]):.1f}")
        report.append(f"Average weighted score: {np.mean([doc.weighted_score for doc in ranked_doctors]):.3f}")
        report.append("")
        
        # Doctor rankings
        report.append("DOCTOR RANKINGS (by weighted score):")
        report.append("-" * 80)
        report.append(f"{'Rank':<4} {'Doctor ID':<10} {'Cases':<6} {'Avg Score':<10} {'Weighted Score':<15} {'Outlier':<8}")
        report.append("-" * 80)
        
        for doctor in ranked_doctors:
            outlier_mark = f"({doctor.outlier_type})" if doctor.is_outlier else ""
            report.append(
                f"{doctor.rank:<4} {doctor.doctor_id:<10} {doctor.total_cases:<6} "
                f"{doctor.average_score:<10.3f} {doctor.weighted_score:<15.3f} {outlier_mark:<8}"
            )
        
        report.append("")
        
        # Outlier analysis
        if outliers:
            report.append("OUTLIER DOCTORS DETECTED:")
            report.append("-" * 40)
            
            good_outliers = [doc for doc in outliers if doc.outlier_type == "good"]
            bad_outliers = [doc for doc in outliers if doc.outlier_type == "bad"]
            
            if good_outliers:
                report.append("EXCEPTIONAL PERFORMERS:")
                for doctor in good_outliers:
                    report.append(f"  {doctor.doctor_id}: Weighted score {doctor.weighted_score:.3f} "
                                f"(Rank {doctor.rank}, {doctor.total_cases} cases)")
            
            if bad_outliers:
                report.append("UNDERPERFORMING DOCTORS:")
                for doctor in bad_outliers:
                    report.append(f"  {doctor.doctor_id}: Weighted score {doctor.weighted_score:.3f} "
                                f"(Rank {doctor.rank}, {doctor.total_cases} cases)")
        else:
            report.append("No outliers detected.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def get_detailed_doctor_analysis(self, doctor_id: str) -> str:
        """Get detailed analysis for a specific doctor"""
        if doctor_id not in self.doctor_scores:
            return f"Doctor {doctor_id} not found in dataset"
        
        doctor = self.doctor_scores[doctor_id]
        
        analysis = []
        analysis.append(f"DETAILED ANALYSIS FOR {doctor_id}")
        analysis.append("=" * 50)
        analysis.append(f"Total cases: {doctor.total_cases}")
        analysis.append(f"Average score: {doctor.average_score}")
        analysis.append(f"Weighted score: {doctor.weighted_score}")
        analysis.append(f"Rank: {doctor.rank}")
        analysis.append(f"Outlier status: {'Yes' if doctor.is_outlier else 'No'}")
        if doctor.is_outlier:
            analysis.append(f"Outlier type: {doctor.outlier_type}")
        
        analysis.append("")
        analysis.append("Outcome Distribution:")
        for category in OutcomeCategory:
            count = doctor.outcome_scores.get(category, 0)
            percentage = (count / doctor.total_cases * 100) if doctor.total_cases > 0 else 0
            analysis.append(f"  {category.value.title()}: {count} ({percentage:.1f}%)")
        
        return "\n".join(analysis)

    def save_ranking_to_csv(self, csv_path: str = 'outlier_doctors/doctor_ranking_table.csv') -> None:
        """Save the doctor ranking table to a CSV file"""
        ranked_doctors = self.rank_doctors()
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Rank', 'Doctor ID', 'Cases', 'Avg Score', 'Weighted Score', 'Outlier', 'Outlier Type'
            ])
            for doctor in ranked_doctors:
                writer.writerow([
                    doctor.rank,
                    doctor.doctor_id,
                    doctor.total_cases,
                    doctor.average_score,
                    doctor.weighted_score,
                    'Yes' if doctor.is_outlier else 'No',
                    doctor.outlier_type if doctor.is_outlier else ''
                ])


def main():
    """Main function to run the doctor ranking analysis"""
    # Initialize analyzer
    analyzer = DoctorAnalyzer('doctor_appointment_summaries.json')
    
    try:
        # Load data
        analyzer.load_data()
        
        # Generate and print report
        report = analyzer.generate_report()
        print(report)
        
        # Save report to file
        with open('outlier_doctors/doctor_ranking_report.txt', 'w') as f:
            f.write(report)
        
        # Save ranking table to CSV
        analyzer.save_ranking_to_csv('outlier_doctors/doctor_ranking_table.csv')
        print("\nRanking table saved to 'outlier_doctors/doctor_ranking_table.csv'")
        
        print("\nReport saved to 'outlier_doctors/doctor_ranking_report.txt'")
        
        # Example detailed analysis
        print("\n" + "="*50)
        print("EXAMPLE DETAILED ANALYSIS")
        print("="*50)
        
        # Get top and bottom performers for detailed analysis
        ranked_doctors = analyzer.rank_doctors()
        if ranked_doctors:
            top_doctor = ranked_doctors[0]
            bottom_doctor = ranked_doctors[-1]
            
            print(f"\nTop performer ({top_doctor.doctor_id}):")
            print(analyzer.get_detailed_doctor_analysis(top_doctor.doctor_id))
            
            print(f"\nBottom performer ({bottom_doctor.doctor_id}):")
            print(analyzer.get_detailed_doctor_analysis(bottom_doctor.doctor_id))
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 