from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.core.config import get_settings

class OCRService:
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
            except Exception as exc:  # pragma: no cover - remote/model specific
                raise RuntimeError(f"Failed to initialize DeepSeek model: {exc}") from exc

            cls._deepseek_model = model
            cls._deepseek_tokenizer = tokenizer
            return model, tokenizer

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

            infer_kwargs = {
                "prompt": settings.deepseek_prompt,
                "image_file": str(input_path),
                "output_path": str(output_dir),
            }

            try:
                model.infer(
                    tokenizer,
                    base_size=settings.deepseek_base_size,
                    image_size=settings.deepseek_image_size,
                    crop_mode=settings.deepseek_crop_mode,
                    save_results=True,
                    test_compress=True,
                    **infer_kwargs,
                )
            except TypeError:
                # Compatibility fallback for alternative remote-code infer signatures.
                model.infer(tokenizer, **infer_kwargs)
            except Exception as exc:  # pragma: no cover - model runtime specific
                raise RuntimeError(f"DeepSeek inference failed: {exc}") from exc

            result_file = output_dir / "result.mmd"
            if not result_file.exists():
                raise RuntimeError("DeepSeek did not produce result.mmd")

            text = result_file.read_text(encoding="utf-8")
            return text.strip()
