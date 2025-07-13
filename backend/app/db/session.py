from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import os
from typing import Generator

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://postgres:postgres@localhost:5433/clalit')

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 