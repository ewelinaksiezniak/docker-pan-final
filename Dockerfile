FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# ===== 1. Zainstaluj systemowe zależności + git (do Hugging Face) =====
RUN apt-get update && apt-get install -y python3-pip \
    curl \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# ===== 2. Utwórz katalog roboczy i venv =====
WORKDIR /app

# ===== 3. Skopiuj kod i requirements =====
COPY requirements.txt /app/

# ===== 4. Zainstaluj wymagane biblioteki =====
RUN pip3 install --no-cache --break-system-packages -r requirements.txt

COPY MySoft.py /app/

# ===== 6. ENTRYPOINT =====
ENTRYPOINT ["python", "MySoft.py"]
