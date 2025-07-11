# Kripto Para Alım-Satım Sinyali Sistemi

Bu proje, kripto para piyasalarında alım-satım sinyalleri üretmek için yapay öğrenme modelleri ve teknik analiz göstergelerini kullanan, Telegram entegrasyonlu kapsamlı bir sistemdir.

## Özellikler

- **Teknik Analiz Göstergeleri**: RSI, MACD, Bollinger Bantları, Hareketli Ortalamalar
- **Yapay Öğrenme Modelleri**: Bagged Tree, Random Forest, Karar Ağacı, KNN, Neural Network
- **Risk Yönetimi**: Stop-loss, hedef fiyat, portföy çeşitlendirme
- **Gerçek Zamanlı Veri**: Binance API entegrasyonu (veya test verisi)
- **Görselleştirme**: Sinyal grafikleri ve performans analizi
- **Telegram Bot**: Sinyaller ve özetler Telegram'a otomatik gönderilir
- **Docker Desteği**: Kolay kurulum ve dağıtım

---

## 🐳 Docker ile Hızlı Başlangıç

### Gereksinimler

- Docker
- Docker Compose

### Kurulum ve Çalıştırma

1. **Projeyi klonlayın:**

   ```bash
   git clone <repository-url>
   cd cyrpto-bot
   ```

2. **Environment dosyasını hazırlayın:**

   ```bash
   cp env.example .env
   # .env dosyasını düzenleyerek API ve Telegram anahtarlarınızı ekleyin
   ```

3. **Docker ile çalıştırın:**

   ```bash
   # Hızlı başlatma
   ./scripts/docker-run.sh

   # Veya manuel olarak
   docker compose build
   docker compose up
   ```

### Geliştirme Ortamı

Jupyter Notebook ile geliştirme yapmak için:

```bash
./scripts/docker-dev.sh
```

Jupyter Notebook'a `http://localhost:8888` adresinden erişebilirsiniz.

### Docker Komutları

```bash
# Sistemi başlat
docker compose up

# Arka planda çalıştır
docker compose up -d

# Logları görüntüle
docker compose logs -f

# Sistemi durdur
docker compose down

# Temizlik yap
./scripts/docker-clean.sh
```

---

## 📲 Telegram Bot Entegrasyonu

1. **@BotFather ile yeni bir bot oluşturun** ve token alın.
2. Bot ile Telegram'da konuşmaya başlayın.
3. `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates` adresine giderek chat_id'nizi öğrenin.
4. `.env` dosyasına aşağıdaki satırları ekleyin:
   ```
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   ```
5. Test için:
   ```bash
   python test_telegram.py
   ```
   Başarılıysa Telegram'da test mesajı göreceksiniz.

---

## 📊 Sonuçlar

Sistem çalıştıktan sonra sonuçlar `data/` klasöründe bulunabilir:

- `data_*.csv`: Ham veriler
- `*_signals.csv`: Sinyal verileri
- `analysis_*.html`: İnteraktif grafikler

---

## 🔧 Manuel Kurulum

### Gereksinimler

- Python 3.9+
- pip

### Kurulum

```bash
pip install -r requirements.txt
```

### Kullanım

```bash
python main.py
```

---

## .env Örneği

```ini
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

# Model Ayarları
TARGET_TYPE=buy_signal
CONFIDENCE_THRESHOLD=0.6
```

---

## Proje Yapısı

```
├── data/               # Veri dosyaları
├── models/             # Eğitilmiş modeller
├── scripts/            # Docker scriptleri
│   ├── docker-run.sh
│   ├── docker-dev.sh
│   └── docker-clean.sh
├── src/                # Kaynak kod
│   ├── data_collector.py
│   ├── feature_engineering.py
│   ├── ml_models.py
│   ├── signal_generator.py
│   ├── risk_management.py
│   ├── visualization.py
│   └── telegram_bot.py
├── main.py             # Ana uygulama
├── test_telegram.py    # Telegram bot test dosyası
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── env.example
```

---

# Crypto Trading Signal System (English)

This project is a comprehensive system that generates buy/sell signals for cryptocurrencies using machine learning models and technical analysis indicators, with full Telegram integration.

## Features

- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Machine Learning Models**: Bagged Tree, Random Forest, Decision Tree, KNN, Neural Network
- **Risk Management**: Stop-loss, target price, portfolio diversification
- **Real-Time Data**: Binance API integration (or test data)
- **Visualization**: Signal charts and performance analysis
- **Telegram Bot**: Signals and summaries are automatically sent to Telegram
- **Docker Support**: Easy setup and deployment

---

## 🐳 Quick Start with Docker

### Requirements

- Docker
- Docker Compose

### Setup & Run

1. **Clone the project:**

   ```bash
   git clone <repository-url>
   cd cyrpto-bot
   ```

2. **Prepare the environment file:**

   ```bash
   cp env.example .env
   # Edit .env and add your API and Telegram keys
   ```

3. **Run with Docker:**

   ```bash
   # Quick start
   ./scripts/docker-run.sh

   # Or manually
   docker compose build
   docker compose up
   ```

### Development Environment

To use Jupyter Notebook:

```bash
./scripts/docker-dev.sh
```

Access Jupyter at `http://localhost:8888`.

### Docker Commands

```bash
# Start the system
docker compose up

# Run in background
docker compose up -d

# View logs
docker compose logs -f

# Stop the system
docker compose down

# Clean up
./scripts/docker-clean.sh
```

---

## 📲 Telegram Bot Integration

1. **Create a new bot with @BotFather** and get your token.
2. Start a chat with your bot on Telegram.
3. Go to `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates` to find your chat_id.
4. Add the following to your `.env` file:
   ```
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   ```
5. To test:
   ```bash
   python test_telegram.py
   ```
   If successful, you will receive a test message on Telegram.

---

## 📊 Results

After running, results are saved in the `data/` folder:

- `data_*.csv`: Raw data
- `*_signals.csv`: Signal data
- `analysis_*.html`: Interactive charts

---

## 🔧 Manual Setup

### Requirements

- Python 3.9+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
python main.py
```

---

## .env Example

```ini
# Binance API Keys
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

# Telegram Bot Settings
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# System Settings
INITIAL_CAPITAL=10000
MAX_RISK_PER_TRADE=0.02
TIMEFRAME=1h
DATA_LIMIT=500

# Model Settings
TARGET_TYPE=buy_signal
CONFIDENCE_THRESHOLD=0.6
```

---

## Project Structure

```
├── data/               # Data files
├── models/             # Trained models
├── scripts/            # Docker scripts
│   ├── docker-run.sh
│   ├── docker-dev.sh
│   └── docker-clean.sh
├── src/                # Source code
│   ├── data_collector.py
│   ├── feature_engineering.py
│   ├── ml_models.py
│   ├── signal_generator.py
│   ├── risk_management.py
│   ├── visualization.py
│   └── telegram_bot.py
├── main.py             # Main application
├── test_telegram.py    # Telegram bot test file
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── env.example
```

---

For any questions or support, feel free to contact.
