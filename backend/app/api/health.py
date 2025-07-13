from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    # TODO: Implement logic to check system status and model availability
    return {"status": "ok", "model_available": True} 