FROM python:3.11-slim

# Install FFmpeg and system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /agatsya

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ app/
COPY samples/ samples/

# Create storage directories
RUN mkdir -p app/storage/episodes

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
