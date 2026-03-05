import logging
import time

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.services.ocr_service import OCRService


LOGGER = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(debug=settings.debug)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        debug=settings.debug,
    )

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.on_event("startup")
    async def warmup_ocr_runtime() -> None:
        if not settings.ocr_scan_warmup_on_startup:
            return

        started_at = time.perf_counter()
        try:
            await run_in_threadpool(OCRService.warmup_runtime)
        except Exception:
            LOGGER.exception("OCR runtime warmup failed")
            return

        LOGGER.info("OCR runtime warmup completed in %.3fs", time.perf_counter() - started_at)

    @app.get("/", tags=["Meta"])
    def root() -> dict[str, str]:
        return {"message": f"{settings.app_name} is running"}

    return app


app = create_app()
