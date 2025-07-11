#!/bin/bash

# Docker Temizleme Scripti

echo "🧹 Docker Temizleme İşlemi Başlatılıyor..."
echo "========================================"

# Container'ları durdur ve sil
echo "🛑 Container'lar durduruluyor..."
docker-compose down

# Kullanılmayan imajları sil
echo "🗑️  Kullanılmayan Docker imajları siliniyor..."
docker image prune -f

# Kullanılmayan volume'ları sil
echo "🗑️  Kullanılmayan volume'lar siliniyor..."
docker volume prune -f

# Kullanılmayan network'leri sil
echo "🗑️  Kullanılmayan network'ler siliniyor..."
docker network prune -f

# Tüm kullanılmayan Docker kaynaklarını temizle
echo "🧹 Tüm kullanılmayan Docker kaynakları temizleniyor..."
docker system prune -f

echo "✅ Docker temizleme işlemi tamamlandı!"
echo ""
echo "📋 Temizlenen kaynaklar:"
echo "   - Durdurulan container'lar"
echo "   - Kullanılmayan imajlar"
echo "   - Kullanılmayan volume'lar"
echo "   - Kullanılmayan network'ler"
echo "   - Sistem cache'i" 