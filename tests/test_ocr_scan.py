from fastapi.testclient import TestClient

from app.api.deps import get_ocr_service
from app.main import app

client = TestClient(app)


class _FakeOCRService:
    def extract_text_from_image(self, image_bytes: bytes) -> str:
        return f"mocked-{len(image_bytes)}"


def test_ocr_scan_success() -> None:
    app.dependency_overrides[get_ocr_service] = lambda: _FakeOCRService()
    try:
        response = client.post(
            "/api/v1/ocr/scan",
            files={"file": ("invoice.png", b"fake-image-bytes", "image/png")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["text"] == "mocked-16"


def test_ocr_scan_rejects_unsupported_extension() -> None:
    app.dependency_overrides[get_ocr_service] = lambda: _FakeOCRService()
    try:
        response = client.post(
            "/api/v1/ocr/scan",
            files={"file": ("invoice.txt", b"fake-image-bytes", "text/plain")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
