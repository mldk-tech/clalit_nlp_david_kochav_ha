from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta
import os
from backend.app.services.outlier_doctors.doctor_ranking import DoctorAnalyzer
from backend.app.services.model_metrics_service import ModelMetricsService
from backend.app.db.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

class PerformanceMetrics(BaseModel):
    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    timestamp: str
    data_version: str

class SystemMetrics(BaseModel):
    total_predictions: int
    average_response_time: float
    error_rate: float
    active_models: int
    last_updated: str

@router.get("/metrics/performance", response_model=List[PerformanceMetrics])
async def get_performance_metrics(
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    days: int = Query(30, description="Number of days to look back")
):
    """
    Get model performance metrics over time (from database)
    """
    try:
        db = next(get_db())
        metrics_service = ModelMetricsService(db)
        
        # Get all model metrics from database
        metrics_data = metrics_service.get_all_model_metrics()
        
        # Convert to PerformanceMetrics objects
        metrics = []
        for data in metrics_data:
            metrics.append(PerformanceMetrics(
                model_id=data["model_id"],
                accuracy=data["accuracy"],
                precision=data["precision"],
                recall=data["recall"],
                f1_score=data["f1_score"],
                auc_roc=data["auc_roc"],
                timestamp=data["timestamp"],
                data_version=data["data_version"]
            ))
        
        # Filter by model_id if provided
        if model_id:
            metrics = [m for m in metrics if m.model_id == model_id]
            
        return metrics
        
    except Exception as e:
        logger.error(f"Error retrieving performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")

@router.get("/metrics/system", response_model=SystemMetrics)
async def get_system_metrics():
    """
    Get system-wide performance metrics
    """
    try:
        # TODO: Implement actual system metrics calculation
        return SystemMetrics(
            total_predictions=1250,
            average_response_time=0.85,
            error_rate=0.02,
            active_models=2,
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error retrieving system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system metrics: {str(e)}")

@router.get("/metrics/doctors")
async def get_doctor_metrics():
    """
    Get doctor performance metrics (real data)
    """
    try:
        # Path to the real data file
        data_file = os.path.join(os.path.dirname(__file__), "../../../..", "doctor_appointment_summaries.json")
        analyzer = DoctorAnalyzer(data_file)
        analyzer.load_data()
        analyzer.calculate_doctor_scores()
        ranked = analyzer.rank_doctors()
        outliers = analyzer.detect_outliers()
        # Prepare top performers (top 5 by weighted score)
        top_performers = [
            {
                "doctor_id": doc.doctor_id,
                "avg_score": doc.average_score,
                "cases": doc.total_cases
            }
            for doc in ranked[:5]
        ]
        # Prepare outliers (good/bad)
        outlier_list = [
            {
                "doctor_id": doc.doctor_id,
                "type": doc.outlier_type,
                "score": doc.average_score
            }
            for doc in outliers if doc.is_outlier
        ]
        total_doctors = len(ranked)
        average_score = round(sum(doc.average_score for doc in ranked) / total_doctors, 2) if total_doctors > 0 else 0
        return {
            "top_performers": top_performers,
            "outliers": outlier_list,
            "total_doctors": total_doctors,
            "average_score": average_score
        }
    except Exception as e:
        logger.error(f"Error retrieving doctor metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve doctor metrics: {str(e)}")

@router.get("/metrics/diseases")
async def get_disease_clustering_metrics():
    """
    Get disease clustering metrics
    """
    try:
        # TODO: Implement actual clustering metrics
        clustering_metrics = {
            "total_clusters": 6,
            "cluster_sizes": {
                "symptom_based": 6,
                "treatment_based": 5,
                "outcome_based": 4
            },
            "silhouette_score": 0.72,
            "total_records": 400
        }
        return clustering_metrics
        
    except Exception as e:
        logger.error(f"Error retrieving disease metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve disease metrics: {str(e)}")

@router.post("/metrics/calculate")
async def calculate_metrics():
    """
    Trigger metrics calculation for all models
    """
    try:
        logger.info("Triggering metrics calculation")
        
        # TODO: Implement actual metrics calculation
        
        return {
            "status": "calculation_started",
            "job_id": "metrics_12345",
            "estimated_completion": (datetime.now() + timedelta(minutes=5)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics calculation failed: {str(e)}") 