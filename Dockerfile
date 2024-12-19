# Usa una imagen ligera de Python 3.10
FROM python:3.10.12-slim

# Instalar dependencias necesarias para compilar algunas librerías
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

# Instalar Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos de dependencias al contenedor
COPY pyproject.toml poetry.lock ./

# Configurar Poetry para no usar virtualenvs
RUN poetry config virtualenvs.create false

# Instalar av primero para usar precompilados si están disponibles
RUN pip install av --no-cache-dir

# Instalar las dependencias definidas en pyproject.toml (solo las main, sin dev)
RUN poetry install --no-root --only main --no-interaction --no-ansi

# Instalar PyTorch con soporte CUDA (cu121)
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Copiar la carpeta src al contenedor, incluyendo todas sus subcarpetas como model
COPY src /app/src

# Agregar la carpeta src al PYTHONPATH
ENV PYTHONPATH="/app/src"

# Exponer el puerto que usará Flask
EXPOSE 5000

# Comando para ejecutar la aplicación Flask utilizando Gunicorn
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "f5_tts.infer.infer_gradio:app"]
