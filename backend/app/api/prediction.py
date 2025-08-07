from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
from backend.app.services.prediction_service import PredictionService
from backend.app.db.session import get_db
from sqlalchemy.orm import Session
from backend.app.models.prediction import Prediction
from backend.app.models.model_version import ModelVersion
import uuid

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize prediction service
prediction_service = PredictionService()

class PredictionRequest(BaseModel):
    doctor_id: str
    summary: str
    model_version: Optional[str] = "latest"
    use_ensemble: Optional[bool] = False

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_version: str
    features_used: Dict[str, Any]
    probabilities: Dict[str, float]
    model_name: str
    text_length: int
    word_count: int
    feature_importance: Dict[str, float]
    timestamp: str

class ModelInfo(BaseModel):
    id: str
    name: str
    version: str
    model_type: str
    accuracy: float
    last_updated: str
    status: str
    feature_count: int

@router.post("/predict", response_model=PredictionResponse)
async def predict_outcome(request: PredictionRequest, db: Session = get_db()):
    """
    Predict patient outcome based on appointment summary
    """
    try:
        logger.info(f"Prediction request for doctor {request.doctor_id}")
        
        # Validate input
        if not request.summary.strip():
            raise HTTPException(status_code=400, detail="Summary cannot be empty")
        
        if len(request.summary) > 10000:  # Limit summary length
            raise HTTPException(status_code=400, detail="Summary too long (max 10000 characters)")
        
        # Make prediction
        if request.use_ensemble:
            result = prediction_service.predict_with_ensemble(request.summary)
            model_name = "ensemble"
            prediction = result["prediction"]
            confidence = result["confidence"]
            probabilities = {"negative": 0.5, "positive": 0.5}  # Placeholder for ensemble
            feature_importance = {}
        else:
            # Use specified model or default to random_forest
            model_name = request.model_version if request.model_version != "latest" else "random_forest"
            result = prediction_service.predict(request.summary, model_name)
            prediction = result["prediction"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]
            feature_importance = result["feature_importance"]
        
        # Create prediction response
        prediction_response = PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_version=request.model_version,
            features_used=result.get("features_used", {}),
            probabilities=probabilities,
            model_name=model_name,
            text_length=result.get("text_length", len(request.summary)),
            word_count=result.get("word_count", len(request.summary.split())),
            feature_importance=feature_importance,
            timestamp=datetime.now().isoformat()
        )
        
        # Save prediction to database
        try:
            db_prediction = Prediction(
                id=uuid.uuid4(),
                doctor_id=request.doctor_id,
                summary=request.summary,
                predicted_outcome=prediction,
                confidence_score=confidence,
                model_version=request.model_version,
                model_name=model_name,
                features_used=result.get("features_used", {}),
                created_at=datetime.now()
            )
            db.add(db_prediction)
            db.commit()
            logger.info(f"Saved prediction to database with ID: {db_prediction.id}")
        except Exception as db_error:
            logger.error(f"Failed to save prediction to database: {db_error}")
            # Don't fail the request if database save fails
            db.rollback()
        
        return prediction_response
        
    except ValueError as e:
        logger.error(f"Prediction validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/models", response_model=List[ModelInfo])
async def list_models(db: Session = get_db()):
    """
    List available prediction models with their metadata
    """
    try:
        # Get models from prediction service
        available_models = prediction_service.get_available_models()
        
        # Get model metrics from database
        from backend.app.services.model_metrics_service import ModelMetricsService
        metrics_service = ModelMetricsService(db)
        
        models_info = []
        for model in available_models:
            # Get latest metrics for this model
            metrics = metrics_service.get_latest_model_metrics(model["name"])
            
            model_info = ModelInfo(
                id=model["name"],
                name=f"{model['type']} Model {model['name']}",
                version="1.0.0",
                model_type=model["type"],
                accuracy=metrics.get("accuracy", 0.0) if metrics else 0.0,
                last_updated=datetime.now().isoformat(),
                status="active" if model["available"] else "inactive",
                feature_count=model.get("feature_count", 0)
            )
            models_info.append(model_info)
        
        return models_info
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.get("/models/{model_name}/health")
async def get_model_health(model_name: str):
    """
    Get health status of a specific model
    """
    try:
        health_info = prediction_service.get_model_health()
        
        # Add specific model information
        if model_name in health_info["available_models"]:
            model_health = {
                "model_name": model_name,
                "status": "healthy",
                "available": True,
                "feature_count": health_info["feature_count"],
                "loaded_at": datetime.now().isoformat()
            }
        else:
            model_health = {
                "model_name": model_name,
                "status": "unhealthy",
                "available": False,
                "error": f"Model '{model_name}' not found"
            }
        
        return model_health
        
    except Exception as e:
        logger.error(f"Error getting model health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model health: {str(e)}")

@router.get("/predictions/history")
async def get_prediction_history(
    doctor_id: Optional[str] = None,
    limit: int = 10,
    db: Session = get_db()
):
    """
    Get prediction history
    """
    try:
        query = db.query(Prediction)
        
        if doctor_id:
            query = query.filter(Prediction.doctor_id == doctor_id)
        
        predictions = query.order_by(Prediction.created_at.desc()).limit(limit).all()
        
        return {
            "predictions": [
                {
                    "id": str(pred.id),
                    "doctor_id": pred.doctor_id,
                    "predicted_outcome": pred.predicted_outcome,
                    "confidence_score": pred.confidence_score,
                    "model_name": pred.model_name,
                    "model_version": pred.model_version,
                    "created_at": pred.created_at.isoformat() if pred.created_at else None
                }
                for pred in predictions
            ],
            "total": len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Error getting prediction history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get prediction history: {str(e)}")

@router.post("/predict/batch")
async def predict_batch(request: List[PredictionRequest]):
    """
    Make predictions for multiple summaries
    """
    try:
        if len(request) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
        
        results = []
        for i, pred_request in enumerate(request):
            try:
                result = prediction_service.predict(pred_request.summary, "random_forest")
                results.append({
                    "index": i,
                    "doctor_id": pred_request.doctor_id,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "model_name": result["model_name"],
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "doctor_id": pred_request.doctor_id,
                    "error": str(e),
                    "status": "failed"
                })
        
        return {
            "batch_results": results,
            "total_processed": len(request),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "failed"])
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}") 