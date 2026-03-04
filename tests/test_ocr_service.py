import pytest

from app.services.ocr_service import OCRService


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("false", False),
        ("none", False),
        ("0", False),
    ],
)
def test_normalize_crop_mode(raw: bool | str, expected: bool) -> None:
    assert OCRService._normalize_crop_mode(raw) is expected


def test_normalize_crop_mode_rejects_unknown_value() -> None:
    with pytest.raises(RuntimeError, match="Invalid DEEPSEEK_CROP_MODE"):
        OCRService._normalize_crop_mode("auto")


def test_build_inference_error_message_adds_hints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.ocr_service.metadata.version", lambda _: "4.57.6")
    exc = RuntimeError("IndexKernel.cu:400: masked_scatter_size_check failed")

    message = OCRService._build_inference_error_message(exc)

    assert "DEEPSEEK_TEST_COMPRESS=false" in message
    assert "DEEPSEEK_CROP_MODE=false" in message
    assert "transformers==4.57.6" in message
