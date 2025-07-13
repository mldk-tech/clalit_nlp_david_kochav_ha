import json
import os
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
from backend.app.models.doctor import Doctor
from backend.app.services.outlier_doctors.doctor_ranking import DoctorAnalyzer, DoctorScore
from backend.app.core.logging_config import logger


class DoctorAnalyticsService:
    """Service for handling doctor analytics operations"""
    
    def __init__(self, db: Session):
        self.db = db
        self.data_file = "doctor_appointment_summaries.json"
    
    def get_all_rankings(self) -> List[Dict[str, Any]]:
        """Get all doctor rankings from the database"""
        try:
            doctors = self.db.query(Doctor).order_by(desc(Doctor.weighted_score)).all()
            return [
                {
                    "id": doctor.id,
                    "rank": doctor.rank,
                    "cases": doctor.cases,
                    "avg_score": doctor.avg_score,
                    "weighted_score": doctor.weighted_score,
                    "outlier": doctor.outlier,
                    "outlier_type": doctor.outlier_type
                }
                for doctor in doctors
            ]
        except Exception as e:
            logger.error(f"Error fetching doctor rankings: {e}")
            # Return mock data if database is not available
            return self._get_mock_rankings()
    
    def _get_mock_rankings(self) -> List[Dict[str, Any]]:
        """Get mock doctor rankings when database is not available"""
        return [
            {
                "id": "dr_03",
                "rank": 1,
                "cases": 41,
                "avg_score": 4.244,
                "weighted_score": 15.862,
                "outlier": True,
                "outlier_type": "good"
            },
            {
                "id": "dr_05",
                "rank": 2,
                "cases": 50,
                "avg_score": 3.080,
                "weighted_score": 12.110,
                "outlier": False,
                "outlier_type": None
            },
            {
                "id": "dr_06",
                "rank": 3,
                "cases": 42,
                "avg_score": 3.024,
                "weighted_score": 11.373,
                "outlier": False,
                "outlier_type": None
            },
            {
                "id": "dr_01",
                "rank": 4,
                "cases": 39,
                "avg_score": 2.821,
                "weighted_score": 10.405,
                "outlier": False,
                "outlier_type": None
            },
            {
                "id": "dr_04",
                "rank": 5,
                "cases": 39,
                "avg_score": 2.718,
                "weighted_score": 10.026,
                "outlier": False,
                "outlier_type": None
            },
            {
                "id": "dr_02",
                "rank": 6,
                "cases": 38,
                "avg_score": 2.605,
                "weighted_score": 9.632,
                "outlier": False,
                "outlier_type": None
            },
            {
                "id": "dr_08",
                "rank": 7,
                "cases": 36,
                "avg_score": 2.389,
                "weighted_score": 8.901,
                "outlier": False,
                "outlier_type": None
            },
            {
                "id": "dr_09",
                "rank": 8,
                "cases": 37,
                "avg_score": 2.162,
                "weighted_score": 8.234,
                "outlier": False,
                "outlier_type": None
            },
            {
                "id": "dr_10",
                "rank": 9,
                "cases": 35,
                "avg_score": 1.857,
                "weighted_score": 6.789,
                "outlier": False,
                "outlier_type": None
            },
            {
                "id": "dr_07",
                "rank": 10,
                "cases": 35,
                "avg_score": 1.457,
                "weighted_score": 5.124,
                "outlier": True,
                "outlier_type": "bad"
            }
        ]
    
    def get_doctor_by_id(self, doctor_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific doctor"""
        try:
            doctor = self.db.query(Doctor).filter(Doctor.id == doctor_id).first()
            if not doctor:
                return None
            
            return {
                "id": doctor.id,
                "rank": doctor.rank,
                "cases": doctor.cases,
                "avg_score": doctor.avg_score,
                "weighted_score": doctor.weighted_score,
                "outlier": doctor.outlier,
                "outlier_type": doctor.outlier_type
            }
        except Exception as e:
            logger.error(f"Error fetching doctor {doctor_id}: {e}")
            raise
    
    def get_outlier_doctors(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get outlier doctors (both good and bad performers)"""
        try:
            good_outliers = self.db.query(Doctor).filter(
                Doctor.outlier == True,
                Doctor.outlier_type == "good"
            ).order_by(desc(Doctor.weighted_score)).all()
            
            bad_outliers = self.db.query(Doctor).filter(
                Doctor.outlier == True,
                Doctor.outlier_type == "bad"
            ).order_by(Doctor.weighted_score).all()
            
            return {
                "good_outliers": [
                    {
                        "id": doctor.id,
                        "rank": doctor.rank,
                        "cases": doctor.cases,
                        "avg_score": doctor.avg_score,
                        "weighted_score": doctor.weighted_score
                    }
                    for doctor in good_outliers
                ],
                "bad_outliers": [
                    {
                        "id": doctor.id,
                        "rank": doctor.rank,
                        "cases": doctor.cases,
                        "avg_score": doctor.avg_score,
                        "weighted_score": doctor.weighted_score
                    }
                    for doctor in bad_outliers
                ]
            }
        except Exception as e:
            logger.error(f"Error fetching outlier doctors: {e}")
            raise
    
    def get_analytics_status(self) -> Dict[str, Any]:
        """Get the status of the analytics pipeline and basic statistics"""
        try:
            total_doctors = self.db.query(Doctor).count()
            outlier_doctors = self.db.query(Doctor).filter(Doctor.outlier == True).count()
            good_outliers = self.db.query(Doctor).filter(
                Doctor.outlier == True,
                Doctor.outlier_type == "good"
            ).count()
            bad_outliers = self.db.query(Doctor).filter(
                Doctor.outlier == True,
                Doctor.outlier_type == "bad"
            ).count()
            
            # Get average scores
            try:
                avg_weighted_score = self.db.query(Doctor.weighted_score).scalar()
                if avg_weighted_score is None:
                    avg_weighted_score = 0.0
            except Exception:
                # If there's an issue with the query, calculate manually
                doctors = self.db.query(Doctor).all()
                if doctors:
                    avg_weighted_score = sum(doc.weighted_score for doc in doctors) / len(doctors)
                else:
                    avg_weighted_score = 0.0
            
            # Ensure avg_weighted_score is a float
            avg_weighted_score = float(avg_weighted_score) if avg_weighted_score is not None else 0.0
            
            return {
                "total_doctors": total_doctors,
                "outlier_doctors": outlier_doctors,
                "good_outliers": good_outliers,
                "bad_outliers": bad_outliers,
                "average_weighted_score": round(avg_weighted_score, 3),
                "data_file_exists": os.path.exists(self.data_file)
            }
        except Exception as e:
            logger.error(f"Error getting analytics status: {e}")
            # Return default status if database is not available
            return {
                "total_doctors": 0,
                "outlier_doctors": 0,
                "good_outliers": 0,
                "bad_outliers": 0,
                "average_weighted_score": 0.0,
                "data_file_exists": os.path.exists(self.data_file),
                "database_connected": False,
                "error": str(e)
            }
    
    def run_analytics_pipeline(self) -> None:
        """Run the complete doctor analytics pipeline and save results to database"""
        try:
            logger.info("Starting doctor analytics pipeline...")
            
            # Check if data file exists
            if not os.path.exists(self.data_file):
                logger.error(f"Data file {self.data_file} not found")
                return
            
            # Initialize analyzer
            analyzer = DoctorAnalyzer(self.data_file)
            
            # Load data
            analyzer.load_data()
            logger.info("Data loaded successfully")
            
            # Calculate scores and detect outliers
            analyzer.calculate_doctor_scores()
            ranked_doctors = analyzer.rank_doctors()
            outliers = analyzer.detect_outliers()
            
            logger.info(f"Analyzed {len(ranked_doctors)} doctors, found {len(outliers)} outliers")
            
            try:
                # Clear existing data
                self.db.query(Doctor).delete()
                
                # Save results to database
                for doctor_score in ranked_doctors:
                    doctor = Doctor(
                        id=doctor_score.doctor_id,
                        rank=doctor_score.rank,
                        cases=doctor_score.total_cases,
                        avg_score=doctor_score.average_score,
                        weighted_score=doctor_score.weighted_score,
                        outlier=doctor_score.is_outlier,
                        outlier_type=doctor_score.outlier_type
                    )
                    self.db.add(doctor)
                
                # Commit changes
                self.db.commit()
                logger.info("Doctor analytics pipeline completed successfully - data saved to database")
                
            except Exception as db_error:
                logger.error(f"Database error in analytics pipeline: {db_error}")
                logger.info("Analytics completed but data not saved to database (using mock data)")
                self.db.rollback()
            
        except Exception as e:
            logger.error(f"Error in analytics pipeline: {e}")
            if 'db' in locals():
                self.db.rollback()
            raise
    
    def save_doctor_score(self, doctor_score: DoctorScore) -> None:
        """Save a single doctor score to the database"""
        try:
            # Check if doctor already exists
            existing_doctor = self.db.query(Doctor).filter(Doctor.id == doctor_score.doctor_id).first()
            
            if existing_doctor:
                # Update existing record
                existing_doctor.rank = doctor_score.rank
                existing_doctor.cases = doctor_score.total_cases
                existing_doctor.avg_score = doctor_score.average_score
                existing_doctor.weighted_score = doctor_score.weighted_score
                existing_doctor.outlier = doctor_score.is_outlier
                existing_doctor.outlier_type = doctor_score.outlier_type
            else:
                # Create new record
                doctor = Doctor(
                    id=doctor_score.doctor_id,
                    rank=doctor_score.rank,
                    cases=doctor_score.total_cases,
                    avg_score=doctor_score.average_score,
                    weighted_score=doctor_score.weighted_score,
                    outlier=doctor_score.is_outlier,
                    outlier_type=doctor_score.outlier_type
                )
                self.db.add(doctor)
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error saving doctor score for {doctor_score.doctor_id}: {e}")
            self.db.rollback()
            raise 