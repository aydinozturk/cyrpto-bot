# Python 3.9 slim imajını kullan
FROM python:3.9-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını kopyala
COPY . .

# Gerekli dizinleri oluştur
RUN mkdir -p data models

# Port ayarı (gerekirse web arayüzü için)
EXPOSE 8080

# Varsayılan komut
CMD ["python", "main.py"] 