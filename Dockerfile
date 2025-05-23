FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Użyj systemowego Pythona 3.12 (który jest już dostępny)
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    curl \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Ustaw katalog roboczy
WORKDIR /app

# Stwórz venv z Pythona 3.12
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Skopiuj kod
COPY requirements.txt MySoft.py /app/
COPY models/ /app/models/

# Zainstaluj zależności
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Wstępne pobranie modelu transformers (opcjonalne)
RUN python -c 'from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base"); AutoModel.from_pretrained("FacebookAI/xlm-roberta-base")'

ENTRYPOINT ["python", "MySoft.py"]
