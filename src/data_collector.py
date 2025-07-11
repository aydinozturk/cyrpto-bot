"""
Kripto para verilerini toplamak için veri toplama modülü.
Binance API kullanarak gerçek zamanlı veri toplar.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

load_dotenv()

class CryptoDataCollector:
    def __init__(self, exchange_name='binance'):
        """
        Kripto para veri toplayıcısını başlatır.
        
        Args:
            exchange_name (str): Borsa adı (varsayılan: 'binance')
        """
        self.exchange_name = exchange_name
        self.exchange = self._initialize_exchange()
        
    def _initialize_exchange(self):
        """Borsa bağlantısını başlatır."""
        try:
            # API anahtarlarını al
            api_key = os.getenv('BINANCE_API_KEY')
            secret_key = os.getenv('BINANCE_SECRET_KEY')
            
            # Borsa nesnesini oluştur
            exchange_class = getattr(ccxt, self.exchange_name)
            exchange = exchange_class({
                'apiKey': api_key,
                'secret': secret_key,
                'sandbox': False,  # Gerçek API kullan
                'enableRateLimit': True,
            })
            
            return exchange
            
        except Exception as e:
            print(f"Borsa başlatma hatası: {e}")
            return None
    
    def get_historical_data(self, symbol, timeframe='1h', limit=1000):
        """
        Geçmiş verileri toplar.
        
        Args:
            symbol (str): Kripto para çifti (örn: 'BTC/USDT')
            timeframe (str): Zaman dilimi (1m, 5m, 15m, 1h, 4h, 1d)
            limit (int): Toplanacak veri sayısı
            
        Returns:
            pd.DataFrame: OHLCV verileri
        """
        try:
            if not self.exchange:
                raise Exception("Borsa bağlantısı kurulamadı")
            
            # OHLCV verilerini al
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # DataFrame'e dönüştür
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Veri toplama hatası: {e}")
            return pd.DataFrame()
    
    def get_realtime_data(self, symbol, timeframe='1m'):
        """
        Gerçek zamanlı veri alır.
        
        Args:
            symbol (str): Kripto para çifti
            timeframe (str): Zaman dilimi
            
        Returns:
            dict: Güncel fiyat bilgileri
        """
        try:
            if not self.exchange:
                raise Exception("Borsa bağlantısı kurulamadı")
            
            # Ticker bilgilerini al
            ticker = self.exchange.fetch_ticker(symbol)
            
            return {
                'symbol': symbol,
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Gerçek zamanlı veri alma hatası: {e}")
            return {}
    
    def get_multiple_symbols_data(self, symbols, timeframe='1h', limit=500):
        """
        Birden fazla sembol için veri toplar.
        
        Args:
            symbols (list): Kripto para çiftleri listesi
            timeframe (str): Zaman dilimi
            limit (int): Her sembol için veri sayısı
            
        Returns:
            dict: Sembol bazında veriler
        """
        data = {}
        
        for symbol in symbols:
            print(f"{symbol} verisi toplanıyor...")
            df = self.get_historical_data(symbol, timeframe, limit)
            if not df.empty:
                data[symbol] = df
            time.sleep(0.5)  # Rate limiting için bekleme
            
        return data
    
    def save_data(self, data, filename):
        """
        Verileri dosyaya kaydeder.
        
        Args:
            data: Kaydedilecek veri
            filename (str): Dosya adı
        """
        try:
            if isinstance(data, pd.DataFrame):
                data.to_csv(f"data/{filename}.csv")
            elif isinstance(data, dict):
                for symbol, df in data.items():
                    df.to_csv(f"data/{filename}_{symbol.replace('/', '_')}.csv")
            
            print(f"Veriler {filename} dosyasına kaydedildi.")
            
        except Exception as e:
            print(f"Veri kaydetme hatası: {e}")
    
    def load_data(self, filename):
        """
        Kaydedilmiş verileri yükler.
        
        Args:
            filename (str): Dosya adı
            
        Returns:
            pd.DataFrame: Yüklenen veriler
        """
        try:
            filepath = f"data/{filename}.csv"
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                return df
            else:
                print(f"Dosya bulunamadı: {filepath}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Veri yükleme hatası: {e}")
            return pd.DataFrame()

# Test fonksiyonu
if __name__ == "__main__":
    collector = CryptoDataCollector()
    
    # Test verileri
    symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'BNB/USDT']
    
    # Geçmiş verileri topla
    data = collector.get_multiple_symbols_data(symbols, timeframe='1h', limit=500)
    
    # Verileri kaydet
    collector.save_data(data, "crypto_data")
    
    # Gerçek zamanlı veri al
    realtime_data = collector.get_realtime_data('BTC/USDT')
    print("Gerçek zamanlı BTC fiyatı:", realtime_data) 