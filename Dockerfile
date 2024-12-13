# Usa una imagen ligera de Python 3.10
FROM python:3.10.12-slim

# Instalar dependencias necesarias para compilar algunas librerías
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libjpeg-dev \
    libpng-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos de dependencias al contenedor
COPY pyproject.toml ./

# Instalar setuptools, wheel y pip para asegurar compatibilidad
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instalar Poetry para manejar dependencias de pyproject.toml
RUN pip install --no-cache-dir poetry

# Instalar las dependencias definidas en pyproject.toml
RUN poetry install --no-root --no-dev

# Copiar todos los archivos de la carpeta actual al contenedor
COPY . .

# Exponer el puerto que usará Flask
EXPOSE 5000

# Comando para ejecutar la aplicación Flask utilizando Gunicorn
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "f5_tts.infer.infer_gradio:app"]
