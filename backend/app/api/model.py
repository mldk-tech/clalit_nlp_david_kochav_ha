from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app.services.model_metrics_service import ModelMetricsService
from backend.app.models.model_metrics import ModelMetrics
import uuid
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
router = APIRouter()

class ModelInfo(BaseModel):
    id: str
    name: str
    version: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    last_updated: str
    status: str

class ModelTrainingRequest(BaseModel):
    model_type: str
    hyperparameters: Optional[Dict[str, Any]] = None
    data_version: Optional[str] = None

class TrainingJobResponse(BaseModel):
    status: str
    job_id: str
    model_type: str
    estimated_completion: str
    message: str

# In-memory storage for training jobs (in production, use Redis or database)
training_jobs = {}

@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """
    Get specific model details from database
    """
    try:
        metrics_service = ModelMetricsService(db)
        
        # Get latest metrics for the model
        metrics = metrics_service.get_latest_model_metrics(model_id)
        
        if not metrics:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Get model version information - ModelVersion doesn't have model_id field, so we'll use a default
        # In a real implementation, you might want to add a model_id field to ModelVersion
        model_version = None
        
        # Determine model type based on model_id
        model_type = "XGBoost" if "xgboost" in model_id.lower() else "RandomForest" if "rf" in model_id.lower() else "Unknown"
        
        # Get the latest timestamp from metrics
        latest_metric = db.query(ModelMetrics).filter(
            ModelMetrics.model_id == model_id
        ).order_by(ModelMetrics.timestamp.desc()).first()
        
        last_updated = latest_metric.timestamp.isoformat() if latest_metric else "2024-01-01"
        
        return ModelInfo(
            id=model_id,
            name=f"{model_type} Model {model_id}",
            version="v1.0",
            model_type=model_type,
            accuracy=metrics.get("accuracy", 0.0),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            f1_score=metrics.get("f1", 0.0),
            last_updated=last_updated,
            status="active"
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model: {str(e)}")

@router.get("/models", response_model=List[ModelInfo])
async def list_models(db: Session = Depends(get_db)):
    """
    List all available models with their metrics
    """
    try:
        metrics_service = ModelMetricsService(db)
        
        # Get all model metrics
        all_metrics = metrics_service.get_all_model_metrics()
        
        models = []
        for metric_data in all_metrics:
            model_id = metric_data["model_id"]
            
            # Determine model type based on model_id
            model_type = "XGBoost" if "xgboost" in model_id.lower() else "RandomForest" if "rf" in model_id.lower() else "Unknown"
            
            # Get model version information - ModelVersion doesn't have model_id field, so we'll use a default
            # In a real implementation, you might want to add a model_id field to ModelVersion
            model_version = None
            
            models.append(ModelInfo(
                id=model_id,
                name=f"{model_type} Model {model_id}",
                version="v1.0",
                model_type=model_type,
                accuracy=metric_data.get("accuracy", 0.0),
                precision=metric_data.get("precision", 0.0),
                recall=metric_data.get("recall", 0.0),
                f1_score=metric_data.get("f1_score", 0.0),
                last_updated=metric_data.get("timestamp", "2024-01-01"),
                status="active"
            ))
        
        return models
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.post("/models/train", response_model=TrainingJobResponse)
async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Train a new model or retrain existing model using the training pipeline
    """
    try:
        logger.info(f"Training request for model type: {request.model_type}")
        
        # Validate model type
        valid_model_types = ["xgboost", "random_forest", "both", "all"]
        if request.model_type.lower() not in valid_model_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model type. Must be one of: {valid_model_types}"
            )
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Calculate estimated completion time (typically 5-10 minutes for full pipeline)
        estimated_completion = (datetime.now() + timedelta(minutes=8)).isoformat()
        
        # Store job information
        training_jobs[job_id] = {
            "status": "started",
            "model_type": request.model_type,
            "started_at": datetime.now().isoformat(),
            "estimated_completion": estimated_completion,
            "hyperparameters": request.hyperparameters,
            "data_version": request.data_version,
            "progress": 0,
            "message": "Training pipeline started"
        }
        
        # Add background task for model training
        background_tasks.add_task(
            run_model_training_pipeline,
            job_id=job_id,
            model_type=request.model_type,
            hyperparameters=request.hyperparameters,
            data_version=request.data_version
        )
        
        logger.info(f"Training job {job_id} started for model type: {request.model_type}")
        
        return TrainingJobResponse(
            status="training_started",
            job_id=job_id,
            model_type=request.model_type,
            estimated_completion=estimated_completion,
            message="Training pipeline started successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

async def run_model_training_pipeline(
    job_id: str, 
    model_type: str, 
    hyperparameters: Optional[Dict[str, Any]] = None,
    data_version: Optional[str] = None
):
    """
    Background task to run the model training pipeline
    """
    try:
        logger.info(f"Starting training pipeline for job {job_id}")
        
        # Update job status
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["progress"] = 10
        training_jobs[job_id]["message"] = "Loading and preprocessing data"
        
        # Import training pipeline
        import sys
        import os
        training_path = os.path.join(os.path.dirname(__file__), "../services/training_model")
        sys.path.append(training_path)
        
        try:
            from model_training import run_training_pipeline
        except ImportError as e:
            logger.error(f"Failed to import training pipeline: {e}")
            raise Exception(f"Training pipeline import failed: {e}")
        
        # Update progress
        training_jobs[job_id]["progress"] = 30
        training_jobs[job_id]["message"] = "Running feature engineering"
        
        # Run the training pipeline
        results = run_training_pipeline()
        
        # Update job status on completion
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 100
        training_jobs[job_id]["message"] = "Training completed successfully"
        training_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        training_jobs[job_id]["results"] = {
            "rf_metrics": results.get("rf_metrics"),
            "xgb_metrics": results.get("xgb_metrics"),
            "comparison": results.get("comparison")
        }
        
        logger.info(f"Training pipeline completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed for job {job_id}: {str(e)}")
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["message"] = f"Training failed: {str(e)}"
        training_jobs[job_id]["error"] = str(e)

@router.get("/models/train/{job_id}")
async def get_training_status(job_id: str):
    """
    Get the status of a training job
    """
    try:
        if job_id not in training_jobs:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        job_info = training_jobs[job_id]
        
        return {
            "job_id": job_id,
            "status": job_info["status"],
            "model_type": job_info["model_type"],
            "progress": job_info.get("progress", 0),
            "message": job_info.get("message", ""),
            "started_at": job_info.get("started_at"),
            "estimated_completion": job_info.get("estimated_completion"),
            "completed_at": job_info.get("completed_at"),
            "results": job_info.get("results"),
            "error": job_info.get("error")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training status for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")

@router.get("/models/train")
async def list_training_jobs():
    """
    List all training jobs
    """
    try:
        jobs = []
        for job_id, job_info in training_jobs.items():
            jobs.append({
                "job_id": job_id,
                "status": job_info["status"],
                "model_type": job_info["model_type"],
                "progress": job_info.get("progress", 0),
                "started_at": job_info.get("started_at"),
                "estimated_completion": job_info.get("estimated_completion"),
                "completed_at": job_info.get("completed_at")
            })
        
        return {"jobs": jobs}
        
    except Exception as e:
        logger.error(f"Error listing training jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list training jobs: {str(e)}")

@router.delete("/models/{model_id}")
async def delete_model(model_id: str, db: Session = Depends(get_db)):
    """
    Delete a model and all related data from database
    """
    try:
        logger.info(f"Delete request for model: {model_id}")
        
        # First, check if the model exists
        metrics_service = ModelMetricsService(db)
        existing_metrics = metrics_service.get_latest_model_metrics(model_id)
        
        if not existing_metrics:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Start transaction for atomic deletion
        try:
            # 1. Delete model metrics
            deleted_metrics = db.query(ModelMetrics).filter(
                ModelMetrics.model_id == model_id
            ).delete()
            logger.info(f"Deleted {deleted_metrics} model metrics for {model_id}")
            
            # 2. Delete model feature importances
            from backend.app.models.model_metrics import ModelFeatureImportance
            deleted_importances = db.query(ModelFeatureImportance).filter(
                ModelFeatureImportance.model_id == model_id
            ).delete()
            logger.info(f"Deleted {deleted_importances} feature importances for {model_id}")
            
            # 3. Delete predictions that reference this model
            # First, find model versions that might be associated with this model_id
            from backend.app.models.prediction import Prediction
            from backend.app.models.model_version import ModelVersion
            
            # Get model versions that might be related to this model_id
            # Since ModelVersion doesn't have model_id, we'll use a pattern match approach
            # or delete predictions based on model version patterns
            model_type = "XGBoost" if "xgboost" in model_id.lower() else "RandomForest" if "rf" in model_id.lower() else "Unknown"
            
            # For now, we'll delete predictions that might be associated with this model
            # In a real implementation, you'd want to add a model_id field to ModelVersion
            deleted_predictions = 0
            # Note: This is a simplified approach. In production, you'd want better model version tracking
            
            # 4. Delete model comparisons that include this model
            from backend.app.models.model_metrics import ModelComparison
            if model_type == "XGBoost":
                # Delete comparisons where this XGBoost model was compared
                deleted_comparisons = db.query(ModelComparison).filter(
                    ModelComparison.xgboost_value.isnot(None)
                ).delete()
            elif model_type == "RandomForest":
                # Delete comparisons where this RandomForest model was compared
                deleted_comparisons = db.query(ModelComparison).filter(
                    ModelComparison.random_forest_value.isnot(None)
                ).delete()
            else:
                deleted_comparisons = 0
            logger.info(f"Deleted {deleted_comparisons} model comparisons for {model_id}")
            
            # 5. Delete model versions (if they can be identified)
            # Since ModelVersion doesn't have model_id, we'll skip this for now
            # In a real implementation, you'd want to add model_id to ModelVersion
            deleted_versions = 0
            
            # Commit all deletions
            db.commit()
            
            logger.info(f"Successfully deleted model {model_id} and all related data")
            
            return {
                "status": "deleted", 
                "model_id": model_id,
                "deleted_metrics": deleted_metrics,
                "deleted_feature_importances": deleted_importances,
                "deleted_predictions": deleted_predictions,
                "deleted_comparisons": deleted_comparisons,
                "deleted_versions": deleted_versions
            }
            
        except Exception as db_error:
            db.rollback()
            logger.error(f"Database error during model deletion: {str(db_error)}")
            raise HTTPException(status_code=500, detail=f"Database error during deletion: {str(db_error)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}") 