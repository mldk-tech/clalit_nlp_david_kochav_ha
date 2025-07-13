from backend.app.api.health import router as health_router
from backend.app.api.alike_diseases import router as alike_diseases_router
from backend.app.api.prediction import router as prediction_router
from backend.app.api.model import router as model_router
from backend.app.api.metrics import router as metrics_router
from backend.app.api.doctor import router as doctor_router

# Note: training.py is currently empty, so we'll import it when implemented
# from backend.app.api.training import router as training_router 