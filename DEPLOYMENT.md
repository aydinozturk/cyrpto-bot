# Crypto Bot Production Deployment

Bu dokümantasyon, Crypto Bot uygulamasını production ortamında nasıl deploy edeceğinizi açıklar.

## 🚀 Hızlı Başlangıç

### 1. Gereksinimler

- Docker ve Docker Compose yüklü
- Binance API anahtarları
- Telegram Bot token'ı (opsiyonel)

### 2. Environment Dosyası

`.env` dosyanızın doğru yapılandırıldığından emin olun:

```bash
# Binance API Anahtarları
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

# Telegram Bot Ayarları
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Sistem Ayarları
INITIAL_CAPITAL=10000
MAX_RISK_PER_TRADE=0.02
TIMEFRAME=1h
DATA_LIMIT=500
```

### 3. Deployment

#### Otomatik Deployment (Önerilen)

```bash
# Production deployment
./deploy.sh production

# Development deployment (Jupyter Notebook ile)
./deploy.sh development
```

#### Manuel Deployment

```bash
# Servisleri başlat
docker compose up -d

# Logları izle
docker compose logs -f

# Servisleri durdur
docker compose down
```

## 📊 Servisler

### Production Profile

- **crypto-signal-system**: Ana uygulama (port 8080)
- **nginx**: Reverse proxy (port 80, 443)

### Development Profile

- **crypto-signal-system**: Ana uygulama (port 8080)
- **jupyter**: Jupyter Notebook (port 8888)

## 🔧 Konfigürasyon

### Docker Compose Profiles

#### Production

```bash
docker compose --profile production up -d
```

#### Development

```bash
docker compose --profile development up -d
```

### Environment Değişkenleri

| Değişken             | Açıklama                   | Varsayılan |
| -------------------- | -------------------------- | ---------- |
| `BINANCE_API_KEY`    | Binance API anahtarı       | Gerekli    |
| `BINANCE_SECRET_KEY` | Binance gizli anahtarı     | Gerekli    |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token'ı       | Opsiyonel  |
| `TELEGRAM_CHAT_ID`   | Telegram chat ID'si        | Opsiyonel  |
| `INITIAL_CAPITAL`    | Başlangıç sermayesi        | 10000      |
| `MAX_RISK_PER_TRADE` | İşlem başına maksimum risk | 0.02       |
| `TIMEFRAME`          | Zaman dilimi               | 1h         |
| `DATA_LIMIT`         | Veri limiti                | 500        |

## 📁 Dizin Yapısı

```
cyrpto-bot/
├── .env                    # Environment değişkenleri
├── docker-compose.yml      # Production deployment
├── docker-compose.dev.yml  # Development deployment
├── deploy.sh              # Deployment script
├── nginx.conf             # Nginx konfigürasyonu
├── data/                  # Veri dosyaları
├── models/                # Eğitilmiş modeller
├── notebooks/             # Jupyter notebook'ları
└── ssl/                   # SSL sertifikaları
```

## 🔍 Monitoring

### Logları İzleme

```bash
# Tüm servislerin logları
docker compose logs -f

# Belirli bir servisin logları
docker compose logs -f crypto-signal-system

# Son 100 satır log
docker compose logs --tail=100
```

### Health Check

```bash
# Servis durumları
docker compose ps

# Health check endpoint
curl http://localhost:8080/health
```

### Sistem Kaynakları

```bash
# Container kaynak kullanımı
docker stats

# Disk kullanımı
docker system df
```

## 🔒 Güvenlik

### Nginx Güvenlik Başlıkları

- X-Frame-Options
- X-XSS-Protection
- X-Content-Type-Options
- Content-Security-Policy

### Rate Limiting

- API endpoint'leri için 10 request/saniye
- Burst limit: 20 request

### SSL/HTTPS

SSL sertifikalarınızı `ssl/` dizinine yerleştirin ve `nginx.conf` dosyasındaki HTTPS bölümünü aktif edin.

## 🚨 Troubleshooting

### Yaygın Sorunlar

#### 1. Environment Dosyası Bulunamadı

```bash
Error: .env dosyası bulunamadı!
```

**Çözüm**: `env.example` dosyasını `.env` olarak kopyalayın ve düzenleyin.

#### 2. API Anahtarları Eksik

```bash
Error: BINANCE_API_KEY ayarlanmamış!
```

**Çözüm**: `.env` dosyasında API anahtarlarınızı doğru şekilde ayarlayın.

#### 3. Port Çakışması

```bash
Error: Port 8080 is already in use
```

**Çözüm**:

```bash
# Port'u kullanan servisi bulun
lsof -i :8080

# Servisi durdurun veya port'u değiştirin
```

#### 4. Docker İmajı Bulunamadı

```bash
Error: manifest for aydinozturk/cyrpto-bot:latest not found
```

**Çözüm**: İmajı Docker Hub'a push edin veya yerel imajı kullanın.

### Log Analizi

```bash
# Hata loglarını filtrele
docker compose logs | grep -i error

# Son 1 saatteki loglar
docker compose logs --since=1h

# Belirli tarihten sonraki loglar
docker compose logs --since="2024-01-01T00:00:00"
```

## 🔄 Güncelleme

### Otomatik Güncelleme

```bash
# En son imajı çek ve yeniden başlat
./deploy.sh production
```

### Manuel Güncelleme

```bash
# İmajı güncelle
docker pull aydinozturk/cyrpto-bot:latest

# Servisleri yeniden başlat
docker compose down
docker compose up -d
```

## 📞 Destek

Sorun yaşarsanız:

1. Logları kontrol edin: `docker compose logs`
2. Environment dosyasını kontrol edin
3. Docker servislerinin çalıştığından emin olun
4. Port'ların açık olduğunu kontrol edin

## 📝 Notlar

- Production ortamında SSL sertifikası kullanmanız önerilir
- Düzenli backup alın
- Monitoring ve alerting sistemi kurun
- Log rotasyonu yapılandırın
