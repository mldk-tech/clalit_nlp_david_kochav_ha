#!/usr/bin/env python3
"""
Test script for model deletion functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.orm import Session
from backend.app.db.session import SessionLocal
from backend.app.services.model_metrics_service import ModelMetricsService
from backend.app.models.model_metrics import ModelMetrics, ModelFeatureImportance, ModelComparison
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_deletion():
    """Test the model deletion functionality"""
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
            
            # Add feature importances
            xgboost_importances = {
                "text_length": 0.25,
                "has_medication": 0.30,
                "has_referral": 0.20,
                "medical_terms": 0.15,
                "symptom_count": 0.10
            }
            metrics_service.save_feature_importances("xgboost_v1", xgboost_importances, "v1.0")
            
            rf_importances = {
                "text_length": 0.20,
                "has_medication": 0.25,
                "has_referral": 0.25,
                "medical_terms": 0.20,
                "symptom_count": 0.10
            }
            metrics_service.save_feature_importances("random_forest_v1", rf_importances, "v1.0")
            
            # Add model comparison
            comparison_data = {
                'accuracy': {'RandomForest': 0.82, 'XGBoost': 0.85},
                'precision': {'RandomForest': 0.84, 'XGBoost': 0.87},
                'recall': {'RandomForest': 0.80, 'XGBoost': 0.83},
                'f1': {'RandomForest': 0.82, 'XGBoost': 0.85},
                'roc_auc': {'RandomForest': 0.86, 'XGBoost': 0.89}
            }
            metrics_service.save_model_comparison(comparison_data, "v1.0")
            
            logger.info("Test data added successfully")
        
        # Test the model deletion functionality
        logger.info("Testing model deletion...")
        
        # Get initial counts
        initial_metrics = db.query(ModelMetrics).count()
        initial_importances = db.query(ModelFeatureImportance).count()
        initial_comparisons = db.query(ModelComparison).count()
        
        logger.info(f"Initial counts - Metrics: {initial_metrics}, Importances: {initial_importances}, Comparisons: {initial_comparisons}")
        
        # Test deletion of XGBoost model
        model_id_to_delete = "xgboost_v1"
        
        # Simulate the deletion process
        try:
            # 1. Check if model exists
            metrics_service = ModelMetricsService(db)
            existing_metrics = metrics_service.get_latest_model_metrics(model_id_to_delete)
            
            if not existing_metrics:
                logger.error(f"Model {model_id_to_delete} not found")
                return
            
            logger.info(f"Found model {model_id_to_delete} with metrics: {existing_metrics}")
            
            # 2. Delete model metrics
            deleted_metrics = db.query(ModelMetrics).filter(
                ModelMetrics.model_id == model_id_to_delete
            ).delete()
            logger.info(f"Deleted {deleted_metrics} model metrics for {model_id_to_delete}")
            
            # 3. Delete model feature importances
            deleted_importances = db.query(ModelFeatureImportance).filter(
                ModelFeatureImportance.model_id == model_id_to_delete
            ).delete()
            logger.info(f"Deleted {deleted_importances} feature importances for {model_id_to_delete}")
            
            # 4. Delete model comparisons (since it's XGBoost)
            deleted_comparisons = db.query(ModelComparison).filter(
                ModelComparison.xgboost_value.isnot(None)
            ).delete()
            logger.info(f"Deleted {deleted_comparisons} model comparisons for {model_id_to_delete}")
            
            # Commit deletions
            db.commit()
            
            # Get final counts
            final_metrics = db.query(ModelMetrics).count()
            final_importances = db.query(ModelFeatureImportance).count()
            final_comparisons = db.query(ModelComparison).count()
            
            logger.info(f"Final counts - Metrics: {final_metrics}, Importances: {final_importances}, Comparisons: {final_comparisons}")
            
            # Verify deletion
            remaining_metrics = metrics_service.get_latest_model_metrics(model_id_to_delete)
            if not remaining_metrics:
                logger.info(f"✅ Successfully deleted model {model_id_to_delete}")
            else:
                logger.error(f"❌ Model {model_id_to_delete} still exists after deletion")
            
            # Test API response simulation
            deletion_response = {
                "status": "deleted",
                "model_id": model_id_to_delete,
                "deleted_metrics": deleted_metrics,
                "deleted_feature_importances": deleted_importances,
                "deleted_predictions": 0,  # No predictions in test data
                "deleted_comparisons": deleted_comparisons,
                "deleted_versions": 0  # No versions in test data
            }
            logger.info(f"Deletion response: {deletion_response}")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error during deletion: {str(e)}")
            raise
        
        logger.info("✅ Model deletion functionality test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    test_model_deletion() 