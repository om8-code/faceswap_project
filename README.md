# Face Swap API (Two-API) using Gemini Image Editing (No Replicate)

Implements:
- POST /api/v1/face-swap/jobs
- GET  /api/v1/face-swap/jobs/{reference_id}

## Setup (Local)
1) Get a Gemini API key (Google AI Studio / Gemini API docs).
2) Run:

```bash
export GEMINI_API_KEY="YOUR_KEY"
export GEMINI_IMAGE_MODEL="gemini-2.5-flash-image"
export BASE_URL="http://localhost:8080"

docker build -t face-swap-gemini .
docker run --rm -p 8080:8080 \
  -e GEMINI_API_KEY \
  -e GEMINI_IMAGE_MODEL \
  -e BASE_URL \
  -v $(pwd)/data:/app/data \
  face-swap-gemini
