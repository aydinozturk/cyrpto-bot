#!/bin/bash

# Crypto Bot Production Deployment Script
# Kullanım: ./deploy.sh [production|development]

set -e

# Renkler
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log fonksiyonu
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Environment kontrolü
check_environment() {
    log "Environment dosyası kontrol ediliyor..."
    
    if [ ! -f ".env" ]; then
        error ".env dosyası bulunamadı! Lütfen env.example dosyasını kopyalayıp düzenleyin."
    fi
    
    # Gerekli environment değişkenlerini kontrol et
    source .env
    
    if [ -z "$BINANCE_API_KEY" ] || [ "$BINANCE_API_KEY" = "your_binance_api_key_here" ]; then
        error "BINANCE_API_KEY ayarlanmamış!"
    fi
    
    if [ -z "$BINANCE_SECRET_KEY" ] || [ "$BINANCE_SECRET_KEY" = "your_binance_secret_key_here" ]; then
        error "BINANCE_SECRET_KEY ayarlanmamış!"
    fi
    
    log "Environment dosyası doğru yapılandırılmış ✓"
}

# Docker imajını güncelle
update_image() {
    log "Docker Hub'dan en son imajı çekiliyor..."
    docker pull aydinozturk/cyrpto-bot:latest || warn "İmaj çekilemedi, yerel imaj kullanılacak"
}

# Gerekli dizinleri oluştur
create_directories() {
    log "Gerekli dizinler oluşturuluyor..."
    mkdir -p data models notebooks ssl
    log "Dizinler oluşturuldu ✓"
}

# Backup oluştur
create_backup() {
    if [ -d "data" ] && [ "$(ls -A data)" ]; then
        log "Mevcut veriler yedekleniyor..."
        backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$backup_dir"
        cp -r data "$backup_dir/"
        cp -r models "$backup_dir/" 2>/dev/null || true
        log "Yedek oluşturuldu: $backup_dir ✓"
    fi
}

# Servisleri başlat
start_services() {
    local profile=${1:-production}
    
    log "Servisler başlatılıyor (profile: $profile)..."
    
    if [ "$profile" = "production" ]; then
        docker compose --profile production up -d
    elif [ "$profile" = "development" ]; then
        docker compose --profile development up -d
    else
        docker compose up -d
    fi
    
    log "Servisler başlatıldı ✓"
}

# Health check
health_check() {
    log "Health check yapılıyor..."
    
    # Container'ların çalışıp çalışmadığını kontrol et
    sleep 10
    
    if docker compose ps | grep -q "Up"; then
        log "Tüm servisler çalışıyor ✓"
    else
        error "Bazı servisler çalışmıyor!"
    fi
    
    # Logları göster
    log "Son loglar:"
    docker compose logs --tail=20
}

# Ana fonksiyon
main() {
    local profile=${1:-production}
    
    log "🚀 Crypto Bot Deployment Başlatılıyor..."
    log "Profile: $profile"
    
    # Kontroller
    check_environment
    create_directories
    create_backup
    
    # Eski servisleri durdur
    log "Eski servisler durduruluyor..."
    docker compose down 2>/dev/null || true
    
    # İmajı güncelle
    update_image
    
    # Servisleri başlat
    start_services "$profile"
    
    # Health check
    health_check
    
    log "✅ Deployment tamamlandı!"
    log "📊 Servisler:"
    docker compose ps
    
    log "📝 Logları izlemek için: docker compose logs -f"
    log "🌐 Web arayüzü: http://localhost:8080"
    
    if [ "$profile" = "development" ]; then
        log "📓 Jupyter Notebook: http://localhost:8888"
    fi
}

# Script'i çalıştır
main "$@" 