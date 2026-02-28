from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(debug=settings.debug)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        debug=settings.debug,
    )

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/", tags=["Meta"])
    def root() -> dict[str, str]:
        return {"message": f"{settings.app_name} is running"}

    return app


app = create_app()
