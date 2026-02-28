from fastapi import APIRouter, Depends

from app.api.deps import get_app_settings
from app.core.config import Settings
from app.schemas.common import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
def health(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app_name)
