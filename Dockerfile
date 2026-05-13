FROM python:3.10-slim

WORKDIR /workspace

# System deps for OpenCV / MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 git \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (better Docker layer caching)
COPY app/requirements.txt /tmp/app-reqs.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/app-reqs.txt

# Copy source
COPY src /workspace/src
COPY app /workspace/app

ENV PYTHONPATH=/workspace
ENV PORT=7860

EXPOSE 7860

CMD ["python", "-m", "app.app"]
