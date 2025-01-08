# Usa una imagen ligera de Python 3.10
FROM python:3.10.12-slim

# -------------------------
# 1. Instalar dependencias del sistema
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    libsndfile1 \
    libjpeg-dev \
    libpng-dev \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# 2. Instalar Poetry
# -------------------------
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# -------------------------
# 3. Copiar archivos de dependencias
# -------------------------
COPY pyproject.toml poetry.lock ./

# -------------------------
# 4. Configurar Poetry para NO crear venvs
# -------------------------
RUN poetry config virtualenvs.create false

# -------------------------
# 5. Instalar av antes de instalar el resto
# -------------------------
RUN pip install av --no-cache-dir

# -------------------------
# 6. Instalar dependencias definidas en pyproject.toml 
#    (excluyendo las dev si tienes configurado "main" en Poetry)
# -------------------------
RUN poetry install --no-root --only main --no-interaction --no-ansi

# -------------------------
# 7. Instalar PyTorch con soporte CUDA (cu121)
# -------------------------
RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# -------------------------
# 8. Copiar la carpeta src a /app/src (con todos sus subdirectorios)
# -------------------------
COPY src /app/src

# -------------------------
# 9. Agregar la carpeta src al PYTHONPATH
# -------------------------
ENV PYTHONPATH="/app/src"

# -------------------------
# 10. Exponer el puerto que usará Flask
# -------------------------
EXPOSE 5000

# -------------------------
# 11. Ajustar Gunicorn con un timeout mayor
#     para que los modelos de TTS no causen Worker Timeout
# -------------------------

CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "--timeout", "600", "f5_tts.infer.infer_gradio:app"]

