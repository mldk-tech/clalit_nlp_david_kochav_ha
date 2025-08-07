from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta
import os
from backend.app.services.outlier_doctors.doctor_ranking import DoctorAnalyzer
from backend.app.services.model_metrics_service import ModelMetricsService
from backend.app.db.session import get_db
import numpy as np

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
    Get system-wide performance metrics (real data)
    """
    try:
        db = next(get_db())
        # Total predictions
        from backend.app.models.prediction import Prediction
        from backend.app.models.model_version import ModelVersion
        total_predictions = db.query(Prediction).count()
        # Active models
        active_models = db.query(ModelVersion).count()
        # Last updated: latest prediction or model version
        last_pred = db.query(Prediction).order_by(Prediction.created_at.desc()).first()
        last_model = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).first()
        last_pred_time = last_pred.created_at if last_pred else None
        last_model_time = last_model.created_at if last_model else None
        if last_pred_time and last_model_time:
            last_updated = max(last_pred_time, last_model_time).isoformat()
        elif last_pred_time:
            last_updated = last_pred_time.isoformat()
        elif last_model_time:
            last_updated = last_model_time.isoformat()
        else:
            last_updated = datetime.now().isoformat()
        # Average response time and error rate (not tracked, return 0.0 for now)
        average_response_time = 0.0
        error_rate = 0.0
        return SystemMetrics(
            total_predictions=total_predictions,
            average_response_time=average_response_time,
            error_rate=error_rate,
            active_models=active_models,
            last_updated=last_updated
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
async def get_disease_clustering_metrics(method: Optional[str] = None):
    """
    Get disease clustering metrics (real data from database)
    """
    try:
        db = next(get_db())
        from backend.app.models.disease_clustering import DiseaseClusteringResult
        # Optionally filter by clustering method
        query = db.query(DiseaseClusteringResult)
        if method:
            query = query.filter(DiseaseClusteringResult.clustering_method == method)
        results = query.all()
        if not results:
            return {
                "total_clusters": 0,
                "cluster_sizes": {},
                "silhouette_score": None,
                "total_records": 0
            }
        # Calculate cluster sizes
        from collections import Counter
        cluster_labels = [r.cluster_label for r in results]
        cluster_sizes = dict(Counter(cluster_labels))
        total_clusters = len(cluster_sizes)
        total_records = len(results)
        # Try to get silhouette score from parameters (if available)
        silhouette_scores = []
        for r in results:
            if r.parameters and isinstance(r.parameters, dict):
                score = r.parameters.get("silhouette_score")
                if score is not None:
                    silhouette_scores.append(score)
        silhouette_score = float(np.mean(silhouette_scores)) if silhouette_scores else None
        return {
            "total_clusters": total_clusters,
            "cluster_sizes": cluster_sizes,
            "silhouette_score": silhouette_score,
            "total_records": total_records
        }
    except Exception as e:
        logger.error(f"Error retrieving disease metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve disease metrics: {str(e)}")

@router.post("/metrics/calculate")
async def calculate_metrics():
    """
    Trigger metrics calculation for all models, clusters, and doctors (real pipeline)
    """
    try:
        db = next(get_db())
        from backend.app.services.model_metrics_service import ModelMetricsService
        from backend.app.services.alike_diseases_clusters.disease_clustering import DiseaseClusteringEngine
        from backend.app.services.outlier_doctors.doctor_ranking import DoctorAnalyzer
        import os
        import traceback
        summary = {"models": [], "clustering": None, "doctors": None, "errors": []}
        # --- Model metrics ---
        try:
            metrics_service = ModelMetricsService(db)
            # For each model version, recalculate metrics if possible
            from backend.app.models.model_version import ModelVersion
            model_versions = db.query(ModelVersion).all()
            for model in model_versions:
                # Placeholder: In a real system, load model and test data, recalculate metrics
                # Here, just log the model id
                summary["models"].append({"model_id": str(model.id), "version": model.version, "status": "metrics recalculation not implemented in this demo"})
        except Exception as e:
            summary["errors"].append(f"Model metrics error: {str(e)}\n{traceback.format_exc()}")
        # --- Clustering metrics ---
        try:
            clustering_engine = DiseaseClusteringEngine()
            # Example: Recompute quality metrics for all methods
            clustering_methods = ["symptom_based", "treatment_based", "outcome_based", "comprehensive"]
            clustering_results = {}
            for method in clustering_methods:
                metrics = clustering_engine.compute_cluster_quality_metrics(method)
                clustering_results[method] = metrics
            summary["clustering"] = clustering_results
        except Exception as e:
            summary["errors"].append(f"Clustering metrics error: {str(e)}\n{traceback.format_exc()}")
        # --- Doctor analytics ---
        try:
            data_file = os.path.join(os.path.dirname(__file__), "../../../..", "doctor_appointment_summaries.json")
            analyzer = DoctorAnalyzer(data_file)
            analyzer.load_data()
            analyzer.calculate_doctor_scores()
            ranked = analyzer.rank_doctors()
            summary["doctors"] = {"total": len(ranked), "top": [d.doctor_id for d in ranked[:5]]}
        except Exception as e:
            summary["errors"].append(f"Doctor analytics error: {str(e)}\n{traceback.format_exc()}")
        return {
            "status": "calculation_completed",
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        import traceback
        raise HTTPException(status_code=500, detail={"status": "error", "error": str(e), "trace": traceback.format_exc()}) 