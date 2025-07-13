"""
Test script for doctor ranking functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from doctor_ranking import DoctorAnalyzer, OutcomeScorer


def test_outcome_scorer():
    """Test the outcome scoring functionality"""
    print("Testing OutcomeScorer...")
    
    scorer = OutcomeScorer()
    
    # Test some known outcomes
    test_cases = [
        ("Excellent recovery progress, no ongoing concerns.", 5),
        ("Patient is feeling much better and has resumed daily activities.", 4),
        ("Condition remains unchanged, monitoring continues.", 3),
        ("Condition has deteriorated since last visit.", 2),
        ("Patient deceased following complications.", 1),
        ("Unknown outcome", 3),  # Should default to neutral
    ]
    
    for outcome, expected_score in test_cases:
        score = scorer.get_score(outcome)
        category = scorer.get_outcome_category(outcome)
        print(f"  {outcome[:50]}... -> Score: {score}, Category: {category.value}")
        assert score == expected_score, f"Expected {expected_score}, got {score}"
    
    print("✓ OutcomeScorer tests passed!")


def test_doctor_analyzer():
    """Test the doctor analyzer functionality"""
    print("\nTesting DoctorAnalyzer...")
    
    # Initialize analyzer
    analyzer = DoctorAnalyzer('../doctor_appointment_summaries.json')
    
    # Load data
    analyzer.load_data()
    if analyzer.data:
        print(f"  Loaded {len(analyzer.data)} records")
    else:
        print("  No data loaded")
    
    # Calculate scores
    doctor_scores = analyzer.calculate_doctor_scores()
    print(f"  Calculated scores for {len(doctor_scores)} doctors")
    
    # Test ranking
    ranked_doctors = analyzer.rank_doctors()
    print(f"  Ranked {len(ranked_doctors)} doctors")
    
    # Verify ranking order
    for i in range(len(ranked_doctors) - 1):
        assert ranked_doctors[i].weighted_score >= ranked_doctors[i + 1].weighted_score, \
            "Ranking order is incorrect"
    
    # Test outlier detection
    outliers = analyzer.detect_outliers()
    print(f"  Detected {len(outliers)} outliers")
    
    # Print top 3 and bottom 3
    print("\n  Top 3 doctors:")
    for i, doctor in enumerate(ranked_doctors[:3]):
        print(f"    {i+1}. {doctor.doctor_id}: {doctor.weighted_score:.3f} (Rank {doctor.rank})")
    
    print("\n  Bottom 3 doctors:")
    for i, doctor in enumerate(ranked_doctors[-3:]):
        print(f"    {i+1}. {doctor.doctor_id}: {doctor.weighted_score:.3f} (Rank {doctor.rank})")
    
    print("✓ DoctorAnalyzer tests passed!")


def test_detailed_analysis():
    """Test detailed analysis functionality"""
    print("\nTesting detailed analysis...")
    
    analyzer = DoctorAnalyzer('../doctor_appointment_summaries.json')
    analyzer.load_data()
    analyzer.calculate_doctor_scores()
    
    # Test detailed analysis for first doctor
    ranked_doctors = analyzer.rank_doctors()
    if ranked_doctors:
        first_doctor = ranked_doctors[0]
        detailed_analysis = analyzer.get_detailed_doctor_analysis(first_doctor.doctor_id)
        print(f"  Detailed analysis for {first_doctor.doctor_id}:")
        print("  " + "\n  ".join(detailed_analysis.split('\n')[:10]) + "...")
    
    print("✓ Detailed analysis tests passed!")


def main():
    """Run all tests"""
    print("Running doctor ranking tests...\n")
    
    try:
        test_outcome_scorer()
        test_doctor_analyzer()
        test_detailed_analysis()
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 