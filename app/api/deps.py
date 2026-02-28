from app.core.config import Settings, get_settings
from app.services.ocr_service import OCRService


def get_app_settings() -> Settings:
    return get_settings()


def get_ocr_service() -> OCRService:
    return OCRService()
