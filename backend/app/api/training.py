from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from backend.app.db.session import get_db
from sqlalchemy.orm import Session
from backend.app.models.model_version import ModelVersion
import uuid
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory job status store (for demo; replace with persistent job tracking in production)
TRAINING_JOBS = {}

class TrainRequest(BaseModel):
    model_type: str = "random_forest"  # or "xgboost"
    description: Optional[str] = None
    parameters: Optional[dict] = None

class TrainResponse(BaseModel):
    job_id: str
    status: str
    started_at: str

class TrainingStatusResponse(BaseModel):
    job_id: str
    status: str
    started_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None

class TrainedModelInfo(BaseModel):
    id: str
    version: str
    description: Optional[str]
    created_at: str
    status: str

@router.post("/train", response_model=TrainResponse)
async def start_training(request: TrainRequest):
    """
    Trigger a new model training job (demo: synchronous, extend for async/background jobs)
    """
    try:
        job_id = str(uuid.uuid4())
        started_at = datetime.now().isoformat()
        TRAINING_JOBS[job_id] = {"status": "running", "started_at": started_at, "finished_at": None, "error": None}
        # TODO: Replace with real training pipeline (call your training service/module)
        # For demo, simulate training and create a new model version
        import time
        time.sleep(1)  # Simulate training time
        db = next(get_db())
        new_model = ModelVersion(
            version=f"{request.model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            description=request.description or f"Trained {request.model_type} model",
            created_at=datetime.now()
        )
        db.add(new_model)
        db.commit()
        TRAINING_JOBS[job_id]["status"] = "completed"
        TRAINING_JOBS[job_id]["finished_at"] = datetime.now().isoformat()
        return TrainResponse(job_id=job_id, status="completed", started_at=started_at)
    except Exception as e:
        logger.error(f"Training job failed: {str(e)}")
        TRAINING_JOBS[job_id]["status"] = "failed"
        TRAINING_JOBS[job_id]["error"] = str(e)
        return TrainResponse(job_id=job_id, status="failed", started_at=started_at)

@router.get("/training-status/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str):
    """
    Get the status of a training job
    """
    job = TRAINING_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return TrainingStatusResponse(
        job_id=job_id,
        status=job["status"],
        started_at=job["started_at"],
        finished_at=job.get("finished_at"),
        error=job.get("error")
    )

@router.get("/trained-models", response_model=List[TrainedModelInfo])
async def list_trained_models():
    """
    List all trained models
    """
    db = next(get_db())
    models = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    return [
        TrainedModelInfo(
            id=str(m.id),
            version=m.version,
            description=m.description,
            created_at=m.created_at.isoformat() if m.created_at else None,
            status="active"
        ) for m in models
    ]

@router.delete("/trained-models/{model_id}")
async def delete_trained_model(model_id: str):
    """
    Delete a trained model version
    """
    db = next(get_db())
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    db.delete(model)
    db.commit()
    return {"status": "deleted", "model_id": model_id}
