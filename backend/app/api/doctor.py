from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any
from backend.app.services.doctor_analytics import DoctorAnalyticsService
from backend.app.models.doctor import Doctor
from backend.app.db.session import get_db
from sqlalchemy.orm import Session
from fastapi import Depends

router = APIRouter()

@router.get("/doctors/rankings", response_model=List[Dict[str, Any]])
async def get_doctor_rankings(db: Session = Depends(get_db)):
    """Get all doctor rankings from the database"""
    try:
        service = DoctorAnalyticsService(db)
        rankings = service.get_all_rankings()
        return rankings
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch doctor rankings: {str(e)}")

@router.get("/doctors/outliers", response_model=Dict[str, List[Dict[str, Any]]])
async def get_outlier_doctors(db: Session = Depends(get_db)):
    """Get outlier doctors (both good and bad performers)"""
    try:
        service = DoctorAnalyticsService(db)
        outliers = service.get_outlier_doctors()
        return outliers
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch outlier doctors: {str(e)}")

@router.get("/doctors/analytics/status")
async def get_analytics_status(db: Session = Depends(get_db)):
    """Get the status of the analytics pipeline and basic statistics"""
    try:
        service = DoctorAnalyticsService(db)
        status = service.get_analytics_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics status: {str(e)}")

@router.get("/doctors/{doctor_id}", response_model=Dict[str, Any])
async def get_doctor_details(doctor_id: str, db: Session = Depends(get_db)):
    """Get detailed information for a specific doctor"""
    try:
        service = DoctorAnalyticsService(db)
        doctor = service.get_doctor_by_id(doctor_id)
        if not doctor:
            raise HTTPException(status_code=404, detail=f"Doctor {doctor_id} not found")
        return doctor
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch doctor details: {str(e)}")

@router.post("/doctors/run-analytics")
async def run_doctor_analytics(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Trigger the doctor analytics pipeline in the background"""
    try:
        service = DoctorAnalyticsService(db)
        background_tasks.add_task(service.run_analytics_pipeline)
        return {"message": "Doctor analytics pipeline started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start analytics pipeline: {str(e)}")
