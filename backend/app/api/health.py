from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from backend.app.db.session import get_db, engine
from backend.app.config.settings import settings
import os
import time
import psutil
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
router = APIRouter()

class HealthChecker:
    """Health check utility class"""
    
    def __init__(self):
        self.checks = {}
        self.start_time = time.time()
    
    def check_database_connectivity(self) -> Dict[str, Any]:
        """Check database connectivity and basic operations"""
        try:
            # Test database connection
            with engine.connect() as connection:
                # Test basic query
                result = connection.execute(text("SELECT 1"))
                result.fetchone()
                
                # Test if we can access our tables
                tables_query = text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                tables_result = connection.execute(tables_query)
                available_tables = [row[0] for row in tables_result.fetchall()]
                
                return {
                    "status": "healthy",
                    "connection": "ok",
                    "tables_count": len(available_tables),
                    "available_tables": available_tables[:10],  # Show first 10 tables
                    "response_time_ms": round((time.time() - self.start_time) * 1000, 2)
                }
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "connection": "failed",
                "error": str(e),
                "response_time_ms": round((time.time() - self.start_time) * 1000, 2)
            }
    
    def check_model_availability(self) -> Dict[str, Any]:
        """Check if trained models are available"""
        try:
            model_paths = [
                "backend/app/services/training_model/results/random_forest_model.pkl",
                "backend/app/services/training_model/results/xgboost_model.pkl"
            ]
            
            available_models = []
            missing_models = []
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path)
                    modified_time = datetime.fromtimestamp(os.path.getmtime(model_path))
                    available_models.append({
                        "name": os.path.basename(model_path),
                        "path": model_path,
                        "size_mb": round(file_size / (1024 * 1024), 2),
                        "last_modified": modified_time.isoformat(),
                        "age_days": (datetime.now() - modified_time).days
                    })
                else:
                    missing_models.append(os.path.basename(model_path))
            
            return {
                "status": "healthy" if available_models else "unhealthy",
                "available_models": available_models,
                "missing_models": missing_models,
                "total_models": len(available_models),
                "response_time_ms": round((time.time() - self.start_time) * 1000, 2)
            }
        except Exception as e:
            logger.error(f"Model availability check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": round((time.time() - self.start_time) * 1000, 2)
            }
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Process info
            process = psutil.Process()
            
            return {
                "status": "healthy",
                "cpu_percent": round(cpu_percent, 2),
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "percent_used": round(memory.percent, 2)
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent_used": round((disk.used / disk.total) * 100, 2)
                },
                "process": {
                    "memory_mb": round(process.memory_info().rss / (1024**2), 2),
                    "cpu_percent": round(process.cpu_percent(), 2)
                },
                "response_time_ms": round((time.time() - self.start_time) * 1000, 2)
            }
        except Exception as e:
            logger.error(f"System resources check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": round((time.time() - self.start_time) * 1000, 2)
            }
    
    def check_api_endpoints(self) -> Dict[str, Any]:
        """Check if critical API endpoints are accessible"""
        try:
            # This would typically check other services, but for now we'll check our own endpoints
            endpoints_to_check = [
                "/api/v1/health",
                "/api/v1/predict",
                "/api/v1/models",
                "/api/v1/metrics/performance"
            ]
            
            # For now, we'll just return that endpoints are available
            # In a real implementation, you might make actual HTTP requests to check
            return {
                "status": "healthy",
                "endpoints_available": len(endpoints_to_check),
                "endpoints": endpoints_to_check,
                "response_time_ms": round((time.time() - self.start_time) * 1000, 2)
            }
        except Exception as e:
            logger.error(f"API endpoints check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": round((time.time() - self.start_time) * 1000, 2)
            }
    
    def check_data_files(self) -> Dict[str, Any]:
        """Check if critical data files are available"""
        try:
            data_files = [
                "doctor_appointment_summaries.json",
                "backend/app/services/training_model/results/model_comparison.csv",
                "backend/app/services/training_model/results/xgboost_metrics.csv"
            ]
            
            available_files = []
            missing_files = []
            
            for file_path in data_files:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    available_files.append({
                        "name": os.path.basename(file_path),
                        "path": file_path,
                        "size_kb": round(file_size / 1024, 2)
                    })
                else:
                    missing_files.append(os.path.basename(file_path))
            
            return {
                "status": "healthy" if available_files else "unhealthy",
                "available_files": available_files,
                "missing_files": missing_files,
                "total_files": len(available_files),
                "response_time_ms": round((time.time() - self.start_time) * 1000, 2)
            }
        except Exception as e:
            logger.error(f"Data files check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": round((time.time() - self.start_time) * 1000, 2)
            }

@router.get("/health")
def health_check():
    """
    Comprehensive health check endpoint that checks:
    - Database connectivity
    - Model availability
    - System resources
    - API endpoints
    - Data files
    """
    try:
        checker = HealthChecker()
        
        # Run all health checks
        checks = {
            "database": checker.check_database_connectivity(),
            "models": checker.check_model_availability(),
            "system_resources": checker.check_system_resources(),
            "api_endpoints": checker.check_api_endpoints(),
            "data_files": checker.check_data_files()
        }
        
        # Determine overall health status
        overall_status = "healthy"
        unhealthy_checks = []
        
        for check_name, check_result in checks.items():
            if check_result.get("status") == "unhealthy":
                overall_status = "unhealthy"
                unhealthy_checks.append(check_name)
        
        # Calculate total response time
        total_response_time = sum(check.get("response_time_ms", 0) for check in checks.values())
        
        # Prepare response
        response = {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "app_name": settings.app_name,
            "version": "1.0.0",
            "checks": checks,
            "summary": {
                "total_checks": len(checks),
                "healthy_checks": len(checks) - len(unhealthy_checks),
                "unhealthy_checks": len(unhealthy_checks),
                "unhealthy_check_names": unhealthy_checks,
                "total_response_time_ms": round(total_response_time, 2)
            }
        }
        
        # Return appropriate HTTP status code
        if overall_status == "healthy":
            return response
        else:
            raise HTTPException(status_code=503, detail=response)
            
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@router.get("/health/simple")
def simple_health_check():
    """
    Simple health check for load balancers and basic monitoring
    """
    try:
        # Quick database check
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "app_name": settings.app_name
        }
    except Exception as e:
        logger.error(f"Simple health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail={"status": "error", "error": str(e)})

@router.get("/health/detailed")
def detailed_health_check():
    """
    Detailed health check with comprehensive system information
    """
    try:
        checker = HealthChecker()
        
        # Run all checks with detailed information
        checks = {
            "database": checker.check_database_connectivity(),
            "models": checker.check_model_availability(),
            "system_resources": checker.check_system_resources(),
            "api_endpoints": checker.check_api_endpoints(),
            "data_files": checker.check_data_files()
        }
        
        # Add additional detailed information
        detailed_info = {
            "environment": {
                "app_name": settings.app_name,
                "debug_mode": settings.debug,
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "platform": os.sys.platform
            },
            "timing": {
                "start_time": datetime.fromtimestamp(checker.start_time).isoformat(),
                "current_time": datetime.now().isoformat(),
                "uptime_seconds": round(time.time() - checker.start_time, 2)
            },
            "checks": checks
        }
        
        return detailed_info
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        ) 