from sqlalchemy.orm import Session
from backend.app.models.model_metrics import ModelMetrics, ModelFeatureImportance, ModelComparison
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ModelMetricsService:
    """Service for handling model metrics database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def save_model_metrics(self, model_id: str, metrics: Dict[str, float], model_version: str = "v1.0"):
        """Save model metrics to database"""
        try:
            for metric_name, metric_value in metrics.items():
                db_metric = ModelMetrics(
                    model_id=model_id,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    model_version=model_version
                )
                self.db.add(db_metric)
            self.db.commit()
            logger.info(f"Saved metrics for model {model_id}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error saving metrics for model {model_id}: {str(e)}")
            raise
    
    def save_feature_importances(self, model_id: str, feature_importances: Dict[str, float], model_version: str = "v1.0"):
        """Save feature importances to database"""
        try:
            for feature_name, importance_value in feature_importances.items():
                db_importance = ModelFeatureImportance(
                    model_id=model_id,
                    feature_name=feature_name,
                    importance_value=importance_value,
                    model_version=model_version
                )
                self.db.add(db_importance)
            self.db.commit()
            logger.info(f"Saved feature importances for model {model_id}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error saving feature importances for model {model_id}: {str(e)}")
            raise
    
    def save_model_comparison(self, comparison_data: Dict[str, Dict[str, float]], comparison_version: str = "v1.0"):
        """Save model comparison to database"""
        try:
            for metric_name, model_values in comparison_data.items():
                db_comparison = ModelComparison(
                    metric_name=metric_name,
                    random_forest_value=model_values.get("RandomForest", 0.0),
                    xgboost_value=model_values.get("XGBoost", 0.0),
                    comparison_version=comparison_version
                )
                self.db.add(db_comparison)
            self.db.commit()
            logger.info("Saved model comparison")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error saving model comparison: {str(e)}")
            raise
    
    def get_latest_model_metrics(self, model_id: str) -> Dict[str, float]:
        """Get latest metrics for a specific model"""
        try:
            metrics = self.db.query(ModelMetrics).filter(
                ModelMetrics.model_id == model_id
            ).order_by(ModelMetrics.timestamp.desc()).all()
            
            return {str(metric.metric_name): float(metric.metric_value) for metric in metrics}
        except Exception as e:
            logger.error(f"Error retrieving metrics for model {model_id}: {str(e)}")
            return {}
    
    def get_all_model_metrics(self) -> List[Dict[str, Any]]:
        """Get all model metrics for API response"""
        try:
            # Get the latest metrics for each model
            latest_metrics = {}
            all_metrics = self.db.query(ModelMetrics).order_by(
                ModelMetrics.model_id, ModelMetrics.timestamp.desc()
            ).all()
            
            for metric in all_metrics:
                if metric.model_id not in latest_metrics:
                    latest_metrics[metric.model_id] = {
                        "model_id": metric.model_id,
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1_score": 0.0,
                        "auc_roc": 0.0,
                        "timestamp": metric.timestamp.isoformat(),
                        "data_version": metric.model_version
                    }
                latest_metrics[metric.model_id][metric.metric_name] = metric.metric_value
            
            return list(latest_metrics.values())
        except Exception as e:
            logger.error(f"Error retrieving all model metrics: {str(e)}")
            return [] 