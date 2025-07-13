from sqlalchemy import Column, String, DateTime
import uuid
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import datetime

from backend.app.models.base import Base

class ModelVersion(Base):
    __tablename__ = 'model_versions'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    description = Column(String)
    predictions = relationship('Prediction', back_populates='model_version') 