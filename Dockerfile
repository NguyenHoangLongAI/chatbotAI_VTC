# ===== BASE IMAGE: Ubuntu 22.04 with Python 3.11 + CUDA =====
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ===== INSTALL PYTHON 3.11 FROM DEADSNAKES PPA =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && rm -rf /var/lib/apt/lists/*

# ===== SYSTEM DEPENDENCIES =====
# - Python 3.11 + tesseract + poppler + OpenCV + ffmpeg
# - Additional dependencies for Docling
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip \
    tesseract-ocr tesseract-ocr-vie \
    poppler-utils \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg wget curl ca-certificates git dos2unix \
    # Docling dependencies
    libgomp1 \
    libmagic1 \
    libpoppler-cpp-dev \
  && rm -rf /var/lib/apt/lists/*

# Configure Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.11 1

WORKDIR /app

# ===== INSTALL PYTHON DEPENDENCIES =====
COPY requirements.txt /app/requirements.txt

# Clean requirements file and install
RUN dos2unix /app/requirements.txt || true && \
    sed -i '1s/^\xEF\xBB\xBF//' /app/requirements.txt && \
    grep -q "^# -*- coding: utf-8 -*-" /app/requirements.txt || sed -i '1i # -*- coding: utf-8 -*-' /app/requirements.txt && \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements.txt

# ===== DOWNLOAD EASYOCR MODELS (for Docling) =====
# Pre-download Vietnamese and English models to avoid runtime download
RUN python -c "import easyocr; reader = easyocr.Reader(['vi', 'en'], gpu=True, download_enabled=True)"

# Copy project
COPY . /app

# Runtime dirs
RUN mkdir -p /app/uploads /app/models

# Expose app ports (document-api:8000, rag-api:8501)
EXPOSE 8000 8501

# Default self-test (compose will override)
CMD ["bash", "-lc", "python - <<'PY'\nimport torch,sys\nprint('torch', torch.__version__, 'cuda_build', getattr(torch.version,'cuda',None), 'cuda_available', torch.cuda.is_available())\nPY"]