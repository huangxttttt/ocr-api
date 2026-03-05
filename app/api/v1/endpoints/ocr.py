import asyncio
import logging
from pathlib import Path
import time

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from starlette import status

from app.api.deps import get_ocr_service
from app.core.config import get_settings
from app.schemas.ocr import OCRExtractRequest, OCRExtractResponse, OCRScanResponse
from app.services.ocr_service import OCRService

router = APIRouter(prefix="/ocr", tags=["OCR"])
LOGGER = logging.getLogger(__name__)
_SCAN_SEMAPHORE: asyncio.Semaphore | None = None
_SCAN_SEMAPHORE_LIMIT: int | None = None


def _get_scan_semaphore(limit: int) -> asyncio.Semaphore:
    global _SCAN_SEMAPHORE, _SCAN_SEMAPHORE_LIMIT

    normalized_limit = max(1, limit)
    if _SCAN_SEMAPHORE is None or _SCAN_SEMAPHORE_LIMIT != normalized_limit:
        _SCAN_SEMAPHORE = asyncio.Semaphore(normalized_limit)
        _SCAN_SEMAPHORE_LIMIT = normalized_limit

    return _SCAN_SEMAPHORE


@router.post("/extract", response_model=OCRExtractResponse)
def extract_text(
    payload: OCRExtractRequest,
    service: OCRService = Depends(get_ocr_service),
) -> OCRExtractResponse:
    text = service.extract_text(payload.content)
    return OCRExtractResponse(text=text)


@router.post("/scan", response_model=OCRScanResponse)
async def scan_image(
    file: UploadFile | None = File(default=None),
    image: UploadFile | None = File(default=None),
    service: OCRService = Depends(get_ocr_service),
) -> OCRScanResponse:
    started_at = time.perf_counter()
    settings = get_settings()
    upload = file or image

    if upload is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing upload file. Use form field 'file' or 'image'.",
        )

    if not upload.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing file name",
        )

    ext = Path(upload.filename).suffix.lower()
    if ext not in settings.ocr_scan_allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension: {ext}",
        )

    image_bytes = await upload.read()
    if len(image_bytes) > settings.ocr_scan_max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                "File too large. "
                f"Limit: {settings.ocr_scan_max_file_size} bytes"
            ),
        )

    semaphore = _get_scan_semaphore(settings.ocr_scan_max_concurrency)
    queue_timeout = max(0.1, settings.ocr_scan_queue_timeout_seconds)
    waiting_started_at = time.perf_counter()
    acquired = False

    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=queue_timeout)
        acquired = True
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "OCR scan busy. Please retry later or reduce request concurrency."
            ),
        ) from exc

    inference_started_at = time.perf_counter()
    try:
        text = await run_in_threadpool(service.extract_text_from_image, image_bytes)
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
    finally:
        if acquired:
            semaphore.release()
        LOGGER.info(
            "ocr.scan timing wait=%.3fs infer=%.3fs total=%.3fs bytes=%d limit=%d",
            inference_started_at - waiting_started_at,
            time.perf_counter() - inference_started_at,
            time.perf_counter() - started_at,
            len(image_bytes),
            max(1, settings.ocr_scan_max_concurrency),
        )

    return OCRScanResponse(text=text)
