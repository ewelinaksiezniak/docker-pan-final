FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# ===== 1. Zainstaluj systemowe zależności + git (do Hugging Face) =====
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    curl \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# ===== 2. Utwórz katalog roboczy i venv =====
WORKDIR /app
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ===== 3. Skopiuj kod i requirements =====
COPY requirements.txt MySoft.py /app/

# ===== 4. Zainstaluj wymagane biblioteki =====
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ===== 5. Pobierz modele do lokalnego cache =====
# (Modele publiczne — bez tokena)
RUN python -c 'from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained("Ewel/FacebookAI_xlm_roberta_base", cache_dir="models/FacebookAI_xlm_roberta_base"); \
AutoModel.from_pretrained("Ewel/FacebookAI_xlm_roberta_base", cache_dir="models/FacebookAI_xlm_roberta_base")'

RUN python -c 'from transformers import AutoModel; \
AutoModel.from_pretrained("Ewel/modeltrained_on_contrastive_encoder_10_epoch_quote_easy_freeze_0", cache_dir="models/modeltrained_on_contrastive_encoder_10_epoch_quote_easy_freeze_0")'

RUN python -c 'from transformers import AutoModel; \
AutoModel.from_pretrained("Ewel/model_trained_on_contrastive_encoder_10_epoch_question_freeze_0", cache_dir="models/model_trained_on_contrastive_encoder_10_epoch_question_freeze_0")'

RUN python -c 'from transformers import AutoModel; \
AutoModel.from_pretrained("Ewel/model_trained_on_contrastive_encoder_10_epoch_question_medium_freeze_2", cache_dir="models/model_trained_on_contrastive_encoder_10_epoch_question_medium_freeze_2")'

# ===== 6. ENTRYPOINT =====
ENTRYPOINT ["python", "MySoft.py"]
