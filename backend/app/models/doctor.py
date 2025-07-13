from sqlalchemy import Column, String, Integer, Boolean, Float
from backend.app.models.base import Base

class Doctor(Base):
    __tablename__ = 'doctors'
    id = Column(String, primary_key=True)
    rank = Column(Integer)
    cases = Column(Integer)
    avg_score = Column(Float)
    weighted_score = Column(Float)
    outlier = Column(Boolean)
    outlier_type = Column(String) 