#!/usr/bin/env python3
"""
Test script for doctor analytics functionality
"""

import json
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.services.outlier_doctors.doctor_ranking import DoctorAnalyzer

def test_doctor_analytics():
    """Test the doctor analytics functionality"""
    print("Testing Doctor Analytics Pipeline...")
    
    # Check if data file exists
    data_file = "doctor_appointment_summaries.json"
    if not os.path.exists(data_file):
        print(f"❌ Data file {data_file} not found")
        return False
    
    try:
        # Initialize analyzer
        analyzer = DoctorAnalyzer(data_file)
        
        # Load data
        print("📊 Loading appointment data...")
        analyzer.load_data()
        
        # Calculate scores
        print("🔢 Calculating doctor scores...")
        doctor_scores = analyzer.calculate_doctor_scores()
        print(f"✅ Calculated scores for {len(doctor_scores)} doctors")
        
        # Rank doctors
        print("🏆 Ranking doctors...")
        ranked_doctors = analyzer.rank_doctors()
        print(f"✅ Ranked {len(ranked_doctors)} doctors")
        
        # Detect outliers
        print("🎯 Detecting outliers...")
        outliers = analyzer.detect_outliers()
        print(f"✅ Found {len(outliers)} outliers")
        
        # Show top 5 doctors
        print("\n🏅 Top 5 Doctors:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Doctor ID':<10} {'Cases':<6} {'Avg Score':<10} {'Weighted Score':<15} {'Outlier':<8}")
        print("-" * 80)
        
        for doctor in ranked_doctors[:5]:
            outlier_mark = f"({doctor.outlier_type})" if doctor.is_outlier else ""
            print(f"{doctor.rank:<4} {doctor.doctor_id:<10} {doctor.total_cases:<6} "
                  f"{doctor.average_score:<10.3f} {doctor.weighted_score:<15.3f} {outlier_mark:<8}")
        
        # Show outliers
        if outliers:
            print(f"\n🎯 Outlier Doctors ({len(outliers)} found):")
            print("-" * 50)
            
            good_outliers = [doc for doc in outliers if doc.outlier_type == "good"]
            bad_outliers = [doc for doc in outliers if doc.outlier_type == "bad"]
            
            if good_outliers:
                print("✅ Exceptional Performers:")
                for doctor in good_outliers:
                    print(f"  {doctor.doctor_id}: Score {doctor.weighted_score:.3f} (Rank {doctor.rank})")
            
            if bad_outliers:
                print("❌ Underperforming Doctors:")
                for doctor in bad_outliers:
                    print(f"  {doctor.doctor_id}: Score {doctor.weighted_score:.3f} (Rank {doctor.rank})")
        
        # Generate report
        print("\n📋 Generating report...")
        report = analyzer.generate_report()
        
        # Save report
        with open('doctor_analytics_test_report.txt', 'w') as f:
            f.write(report)
        print("✅ Report saved to 'doctor_analytics_test_report.txt'")
        
        # Save CSV
        analyzer.save_ranking_to_csv('doctor_analytics_test_ranking.csv')
        print("✅ Ranking table saved to 'doctor_analytics_test_ranking.csv'")
        
        print("\n🎉 Doctor analytics test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_doctor_analytics()
    sys.exit(0 if success else 1) 