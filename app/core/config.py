from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "OCR API"
    app_env: str = "dev"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    ocr_scan_max_file_size: int = 10 * 1024 * 1024
    ocr_scan_allowed_extensions: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    )
    deepseek_model_path: str = "models/DeepSeek-OCR"
    deepseek_prompt: str = "<image>\n<|grounding|>Convert the document to markdown. "
    deepseek_base_size: int = 1024
    deepseek_image_size: int = 1024
    deepseek_crop_mode: str = "none"
    deepseek_device: str = "cuda"
    deepseek_use_bfloat16: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
