from sqlalchemy import Column, String, Float, DateTime, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
import uuid
import datetime
from backend.app.models.base import Base

class ModelMetrics(Base):
    __tablename__ = 'model_metrics'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)  # accuracy, precision, recall, f1, roc_auc
    metric_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    model_version = Column(String, default="v1.0")

class ModelFeatureImportance(Base):
    __tablename__ = 'model_feature_importances'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String, nullable=False)
    feature_name = Column(String, nullable=False)
    importance_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    model_version = Column(String, default="v1.0")

class ModelComparison(Base):
    __tablename__ = 'model_comparisons'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String, nullable=False)  # accuracy, precision, recall, f1, roc_auc
    random_forest_value = Column(Float, nullable=False)
    xgboost_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    comparison_version = Column(String, default="v1.0") 