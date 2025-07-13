from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.logging_config import logger
from backend.app.config.settings import settings
from backend.app.api import (
    health_router,
    alike_diseases_router,
    prediction_router,
    model_router,
    metrics_router,
    doctor_router
)

app = FastAPI(title=settings.app_name, debug=settings.debug)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routers
app.include_router(health_router, prefix="/api/v1", tags=["health"])
app.include_router(alike_diseases_router, prefix="/api/v1", tags=["diseases"])
app.include_router(prediction_router, prefix="/api/v1", tags=["predictions"])
app.include_router(model_router, prefix="/api/v1", tags=["models"])
app.include_router(metrics_router, prefix="/api/v1", tags=["metrics"])
app.include_router(doctor_router, prefix="/api/v1", tags=["doctors"])

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": f"Welcome to the {settings.app_name}!"} 