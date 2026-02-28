from pydantic import BaseModel, Field


class OCRExtractRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Raw text or OCR source content")


class OCRExtractResponse(BaseModel):
    text: str


class OCRScanResponse(BaseModel):
    text: str = Field(..., description="OCR extracted text")
