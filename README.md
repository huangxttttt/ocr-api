## OCR API (FastAPI Scaffold)

### Project Structure

```text
.
|- app/
|  |- api/
|  |  |- deps.py
|  |  |- v1/
|  |     |- endpoints/
|  |     |  |- health.py
|  |     |  |- ocr.py
|  |     |- router.py
|  |- core/
|  |  |- config.py
|  |  |- logging.py
|  |- schemas/
|  |  |- common.py
|  |  |- ocr.py
|  |- services/
|  |  |- ocr_service.py
|  |- main.py
|- tests/
|  |- test_health.py
|- main.py
|- pyproject.toml
```

### Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,ocr-deepseek]"
uvicorn app.main:app --reload
```

Production run (no reload):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

OCR scanning uses DeepSeek OCR model only.
Model is loaded from local path only (no online download).
Default local path: `./models/DeepSeek-OCR` (config: `DEEPSEEK_MODEL_PATH`).
Recommended DeepSeek runtime knobs:
- `OCR_SCAN_MAX_CONCURRENCY=1` (limit concurrent `/scan` inference in each process)
- `OCR_SCAN_QUEUE_TIMEOUT_SECONDS=30` (fail fast with `429` when queue wait is too long)
- `OCR_SCAN_WARMUP_ON_STARTUP=true` (optional model warmup to reduce first-request latency)
- `DEEPSEEK_CROP_MODE=true|false` (boolean only)
- `DEEPSEEK_TEST_COMPRESS=false` (disable compression test path in production)
- `transformers==4.46.3` for best compatibility with upstream DeepSeek OCR code

### Docker

Build image:

```bash
docker build -t ocr-api:latest .
```

Run (CPU):

```bash
docker run --rm -p 8000:8000 \
  -e DEEPSEEK_DEVICE=cpu \
  -e DEEPSEEK_MODEL_PATH=/app/models/DeepSeek-OCR \
  -v ./models:/app/models:ro \
  ocr-api:latest
```

Run (GPU, NVIDIA runtime):

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e DEEPSEEK_DEVICE=cuda \
  -e DEEPSEEK_MODEL_PATH=/app/models/DeepSeek-OCR \
  -v ./models:/app/models:ro \
  ocr-api:latest
```

Docker Compose (GPU):

```bash
docker compose up -d --build
```

Before compose startup, place your pre-downloaded model files under:

```text
./models/DeepSeek-OCR
```

### API Endpoints

- `GET /`
- `GET /api/v1/health`
- `POST /api/v1/ocr/extract`
- `POST /api/v1/ocr/scan`

Text extract example:

```json
{
  "content": "  sample text  "
}
```

Example response:

```json
{
  "text": "sample text"
}
```

Image OCR scan example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/ocr/scan" \
  -F "file=@./sample.png"
```

Response:

```json
{
  "text": "recognized text from image"
}
```

### Tests

```bash
pytest
```
