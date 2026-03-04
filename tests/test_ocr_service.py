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


def test_build_scan_prompt_adds_strict_rules() -> None:
    prompt = OCRService._build_scan_prompt(
        base_prompt="<image>\n<|grounding|>Convert the document to markdown.",
        enforce_verbatim=True,
        unreadable_placeholder="[UNREADABLE]",
    )

    assert "Only transcribe content that is visible" in prompt
    assert "Do not infer, guess, or autocomplete" in prompt
    assert "For tables, preserve row/column structure" in prompt
    assert "[UNREADABLE]" in prompt


def test_build_scan_prompt_returns_original_when_verbatim_disabled() -> None:
    base_prompt = "<image>\n<|grounding|>Convert the document to markdown."
    prompt = OCRService._build_scan_prompt(
        base_prompt=base_prompt,
        enforce_verbatim=False,
        unreadable_placeholder="[UNREADABLE]",
    )

    assert prompt == base_prompt
