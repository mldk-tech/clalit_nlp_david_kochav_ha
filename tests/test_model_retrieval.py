#!/usr/bin/env python3
"""
Test script for model retrieval functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.orm import Session
from backend.app.db.session import SessionLocal
from backend.app.services.model_metrics_service import ModelMetricsService
from backend.app.models.model_metrics import ModelMetrics
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_retrieval():
    """Test the model retrieval functionality"""
    try:
        # Get database session
        db = SessionLocal()
        
        # Check if we have any model metrics in the database
        metrics_count = db.query(ModelMetrics).count()
        logger.info(f"Found {metrics_count} model metrics in database")
        
        if metrics_count == 0:
            logger.info("No model metrics found in database. Adding test data...")
            
            # Add some test model metrics
            metrics_service = ModelMetricsService(db)
            
            # Test data for XGBoost model
            xgboost_metrics = {
                "accuracy": 0.85,
                "precision": 0.87,
                "recall": 0.83,
                "f1": 0.85,
                "roc_auc": 0.89
            }
            metrics_service.save_model_metrics("xgboost_v1", xgboost_metrics, "v1.0")
            
            # Test data for Random Forest model
            rf_metrics = {
                "accuracy": 0.82,
                "precision": 0.84,
                "recall": 0.80,
                "f1": 0.82,
                "roc_auc": 0.86
            }
            metrics_service.save_model_metrics("random_forest_v1", rf_metrics, "v1.0")
            
            logger.info("Test data added successfully")
        
        # Test the model retrieval service
        metrics_service = ModelMetricsService(db)
        
        # Test getting metrics for a specific model
        xgboost_metrics = metrics_service.get_latest_model_metrics("xgboost_v1")
        logger.info(f"XGBoost metrics: {xgboost_metrics}")
        
        # Test getting all model metrics
        all_metrics = metrics_service.get_all_model_metrics()
        logger.info(f"All model metrics: {all_metrics}")
        
        # Test the API endpoints (simulate)
        logger.info("Testing API endpoint simulation...")
        
        # Simulate GET /models/{model_id}
        if xgboost_metrics:
            model_info = {
                "id": "xgboost_v1",
                "name": "XGBoost Model xgboost_v1",
                "version": "v1.0",
                "model_type": "XGBoost",
                "accuracy": xgboost_metrics.get("accuracy", 0.0),
                "precision": xgboost_metrics.get("precision", 0.0),
                "recall": xgboost_metrics.get("recall", 0.0),
                "f1_score": xgboost_metrics.get("f1", 0.0),
                "last_updated": "2024-01-01",
                "status": "active"
            }
            logger.info(f"Model info: {model_info}")
        
        # Simulate GET /models
        models_list = []
        for metric_data in all_metrics:
            model_id = metric_data["model_id"]
            model_type = "XGBoost" if "xgboost" in model_id.lower() else "RandomForest" if "rf" in model_id.lower() else "Unknown"
            
            models_list.append({
                "id": model_id,
                "name": f"{model_type} Model {model_id}",
                "version": "v1.0",
                "model_type": model_type,
                "accuracy": metric_data.get("accuracy", 0.0),
                "precision": metric_data.get("precision", 0.0),
                "recall": metric_data.get("recall", 0.0),
                "f1_score": metric_data.get("f1_score", 0.0),
                "last_updated": metric_data.get("timestamp", "2024-01-01"),
                "status": "active"
            })
        
        logger.info(f"Models list: {models_list}")
        
        logger.info("✅ Model retrieval functionality test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    test_model_retrieval() 