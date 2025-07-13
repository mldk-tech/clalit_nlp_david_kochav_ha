from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON
import uuid
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import datetime

from backend.app.models.base import Base

class DiseaseClusteringResult(Base):
    __tablename__ = 'disease_clustering_results'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clustering_method = Column(String, nullable=False)  # e.g., 'symptom_based', 'treatment_based', etc.
    cluster_label = Column(Integer, nullable=False)
    appointment_id = Column(UUID(as_uuid=True), ForeignKey('appointments.id'), nullable=True)  # Optional: link to appointment
    doctor_id = Column(String, nullable=True)
    summary = Column(String, nullable=True)
    diseases = Column(JSON, nullable=True)  # List of diseases
    symptoms = Column(JSON, nullable=True)  # List of symptoms
    treatments = Column(JSON, nullable=True)  # List of treatments
    outcomes = Column(JSON, nullable=True)  # List of outcomes
    severity_score = Column(Integer, nullable=True)
    complexity_score = Column(Integer, nullable=True)
    algorithm = Column(String, nullable=True)  # e.g., 'KMeans', 'DBSCAN', etc.
    parameters = Column(JSON, nullable=True)  # Store clustering parameters
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    # Optionally, add relationships
    appointment = relationship('Appointment') 