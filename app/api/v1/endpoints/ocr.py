from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from starlette import status

from app.api.deps import get_ocr_service
from app.core.config import get_settings
from app.schemas.ocr import OCRExtractRequest, OCRExtractResponse, OCRScanResponse
from app.services.ocr_service import OCRService

router = APIRouter(prefix="/ocr", tags=["OCR"])


@router.post("/extract", response_model=OCRExtractResponse)
def extract_text(
    payload: OCRExtractRequest,
    service: OCRService = Depends(get_ocr_service),
) -> OCRExtractResponse:
    text = service.extract_text(payload.content)
    return OCRExtractResponse(text=text)


@router.post("/scan", response_model=OCRScanResponse)
async def scan_image(
    file: UploadFile = File(...),
    service: OCRService = Depends(get_ocr_service),
) -> OCRScanResponse:
    settings = get_settings()

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing file name",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ocr_scan_allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension: {ext}",
        )

    image_bytes = await file.read()
    if len(image_bytes) > settings.ocr_scan_max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                "File too large. "
                f"Limit: {settings.ocr_scan_max_file_size} bytes"
            ),
        )

    try:
        text = service.extract_text_from_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return OCRScanResponse(text=text)

