#!/bin/bash

# Kripto Para Sinyal Sistemi Docker Çalıştırma Scripti

echo "🚀 Kripto Para Alım-Satım Sinyali Sistemi Docker Başlatılıyor..."
echo "=================================================="

# Docker'ın yüklü olup olmadığını kontrol et
if ! command -v docker &> /dev/null; then
    echo "❌ Docker yüklü değil. Lütfen Docker'ı yükleyin."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose yüklü değil. Lütfen Docker Compose'u yükleyin."
    exit 1
fi

# Gerekli dizinleri oluştur
echo "📁 Gerekli dizinler oluşturuluyor..."
mkdir -p data models notebooks

# Environment dosyasını kontrol et
if [ ! -f .env ]; then
    echo "⚠️  .env dosyası bulunamadı. env.example dosyasından kopyalanıyor..."
    cp env.example .env
    echo "📝 Lütfen .env dosyasını düzenleyerek API anahtarlarınızı ekleyin."
fi

# Docker imajını oluştur ve çalıştır
echo "🐳 Docker container oluşturuluyor..."
docker-compose build

if [ $? -eq 0 ]; then
    echo "✅ Docker imajı başarıyla oluşturuldu."
    
    echo "🚀 Sistem başlatılıyor..."
    docker-compose up crypto-signal-system
    
else
    echo "❌ Docker imajı oluşturulurken hata oluştu."
    exit 1
fi 