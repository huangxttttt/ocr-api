from io import BytesIO
import logging
import re
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
import time
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.core.config import get_settings


LOGGER = logging.getLogger(__name__)


class OCRService:
    _MAX_TESTED_TRANSFORMERS_VERSION = (4, 46, 3)
    _TRUE_VALUES = {"1", "true", "yes", "y", "on"}
    _FALSE_VALUES = {"0", "false", "no", "n", "off", "none", ""}

    _deepseek_model: Any | None = None
    _deepseek_tokenizer: Any | None = None
    _deepseek_load_lock = Lock()

    def extract_text(self, content: str) -> str:
        return content.strip()

    @staticmethod
    def _resolve_local_model_path(raw_path: str) -> Path:
        model_path = Path(raw_path)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        return model_path

    @classmethod
    def _warn_on_transformers_version(cls) -> None:
        try:
            raw_version = metadata.version("transformers")
        except metadata.PackageNotFoundError:
            return

        parsed = cls._parse_semver(raw_version)
        if parsed > cls._MAX_TESTED_TRANSFORMERS_VERSION:
            LOGGER.warning(
                "DeepSeek OCR upstream was validated with transformers==4.46.3; "
                "current version is %s and may cause runtime issues.",
                raw_version,
            )

    @staticmethod
    def _parse_semver(raw: str) -> tuple[int, int, int]:
        parts = raw.split(".")
        values: list[int] = []

        for part in parts[:3]:
            match = re.match(r"^(\d+)", part)
            values.append(int(match.group(1)) if match else 0)

        while len(values) < 3:
            values.append(0)

        return values[0], values[1], values[2]

    @classmethod
    def _normalize_crop_mode(cls, raw_mode: bool | str) -> bool:
        if isinstance(raw_mode, bool):
            return raw_mode

        value = raw_mode.strip().lower()
        if value in cls._TRUE_VALUES:
            return True
        if value in cls._FALSE_VALUES:
            return False

        raise RuntimeError(
            "Invalid DEEPSEEK_CROP_MODE value. Use true/false."
        )

    @staticmethod
    def _ensure_padding_tokens(model: Any, tokenizer: Any) -> None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(getattr(model, "config", None), "eos_token_id", None)
        if eos_token_id is None:
            return

        tokenizer.pad_token_id = eos_token_id
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None and getattr(generation_config, "pad_token_id", None) is None:
            generation_config.pad_token_id = eos_token_id

    @classmethod
    def _build_inference_error_message(cls, exc: Exception) -> str:
        detail = str(exc).strip()
        hints: list[str] = []

        if (
            "masked_scatter_size_check" in detail
            or "IndexKernel.cu" in detail
            or "device-side assert triggered" in detail
        ):
            hints.append(
                "Try DEEPSEEK_TEST_COMPRESS=false and DEEPSEEK_CROP_MODE=false."
            )

        try:
            raw_version = metadata.version("transformers")
        except metadata.PackageNotFoundError:
            raw_version = ""

        if raw_version and cls._parse_semver(raw_version) > cls._MAX_TESTED_TRANSFORMERS_VERSION:
            hints.append(
                f"Current transformers=={raw_version}; "
                "DeepSeek OCR upstream validates against transformers==4.46.3."
            )

        if not hints:
            return f"DeepSeek inference failed: {detail}"

        return f"DeepSeek inference failed: {detail} {' '.join(hints)}"

    @classmethod
    def _ensure_deepseek_runtime(cls) -> tuple[Any, Any]:
        if cls._deepseek_model is not None and cls._deepseek_tokenizer is not None:
            return cls._deepseek_model, cls._deepseek_tokenizer

        with cls._deepseek_load_lock:
            if cls._deepseek_model is not None and cls._deepseek_tokenizer is not None:
                return cls._deepseek_model, cls._deepseek_tokenizer

            settings = get_settings()
            model_path = cls._resolve_local_model_path(settings.deepseek_model_path)
            if not model_path.exists():
                raise RuntimeError(
                    "DeepSeek local model path not found: "
                    f"{model_path}. Please mount/copy your pre-downloaded model."
                )

            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise RuntimeError(
                    "DeepSeek backend requires torch and transformers. "
                    "Install with: pip install -e \".[ocr-deepseek]\""
                ) from exc

            cls._warn_on_transformers_version()

            load_started_at = time.perf_counter()
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    trust_remote_code=True,
                    local_files_only=True,
                )

                model = AutoModel.from_pretrained(
                    str(model_path),
                    trust_remote_code=True,
                    use_safetensors=True,
                    local_files_only=True,
                )

                if settings.deepseek_device == "cuda" and torch.cuda.is_available():
                    model = model.eval().cuda()
                    if settings.deepseek_use_bfloat16:
                        model = model.to(torch.bfloat16)
                else:
                    model = model.eval()

                cls._ensure_padding_tokens(model, tokenizer)
            except Exception as exc:  # pragma: no cover - remote/model specific
                raise RuntimeError(f"Failed to initialize DeepSeek model: {exc}") from exc

            cls._deepseek_model = model
            cls._deepseek_tokenizer = tokenizer
            LOGGER.info(
                "DeepSeek runtime initialized in %.3fs (device=%s, bfloat16=%s, model=%s)",
                time.perf_counter() - load_started_at,
                settings.deepseek_device,
                settings.deepseek_use_bfloat16,
                model_path,
            )
            return model, tokenizer

    @classmethod
    def warmup_runtime(cls) -> None:
        cls._ensure_deepseek_runtime()

    def extract_text_from_image(self, image_bytes: bytes) -> str:
        if not image_bytes:
            raise ValueError("Empty image payload")

        settings = get_settings()
        model, tokenizer = self._ensure_deepseek_runtime()

        try:
            with Image.open(BytesIO(image_bytes)) as source_image:
                image = source_image.convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError("Invalid image file") from exc

        with TemporaryDirectory(prefix="ocr_scan_") as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.png"
            output_dir = temp_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            image.save(input_path, format="PNG")

            crop_mode = self._normalize_crop_mode(settings.deepseek_crop_mode)
            infer_kwargs = {
                "prompt": settings.deepseek_prompt,
                "image_file": str(input_path),
                "output_path": str(output_dir),
            }
            deepseek_kwargs = {
                "base_size": settings.deepseek_base_size,
                "image_size": settings.deepseek_image_size,
                "crop_mode": crop_mode,
                "save_results": True,
                "test_compress": settings.deepseek_test_compress,
            }

            try:
                model.infer(
                    tokenizer,
                    **deepseek_kwargs,
                    **infer_kwargs,
                )
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise RuntimeError(self._build_inference_error_message(exc)) from exc
                # Compatibility fallback for alternative remote-code infer signatures.
                try:
                    model.infer(tokenizer, **infer_kwargs)
                except Exception as inner_exc:  # pragma: no cover - model runtime specific
                    raise RuntimeError(
                        self._build_inference_error_message(inner_exc)
                    ) from inner_exc
            except Exception as exc:  # pragma: no cover - model runtime specific
                raise RuntimeError(self._build_inference_error_message(exc)) from exc

            result_file = output_dir / "result.mmd"
            if not result_file.exists():
                raise RuntimeError("DeepSeek did not produce result.mmd")

            text = result_file.read_text(encoding="utf-8")
            return text.strip()
