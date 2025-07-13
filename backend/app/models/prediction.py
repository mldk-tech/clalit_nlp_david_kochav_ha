from sqlalchemy import Column, String, Float, DateTime, ForeignKey
import uuid
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import datetime

from backend.app.models.base import Base

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    appointment_id = Column(UUID(as_uuid=True), ForeignKey('appointments.id'))
    model_version_id = Column(UUID(as_uuid=True), ForeignKey('model_versions.id'))
    prediction_label = Column(String)
    prediction_score = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    appointment = relationship('Appointment', back_populates='predictions')
    model_version = relationship('ModelVersion', back_populates='predictions') 