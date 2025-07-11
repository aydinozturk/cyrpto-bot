"""
Kripto Para Alım-Satım Sinyali Sistemi - Ana Uygulama
Bu dosya, tüm modülleri birleştirerek kapsamlı bir kripto para alım-satım sinyali sistemi oluşturur.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Modülleri import et
from src.data_collector import CryptoDataCollector
from src.feature_engineering import FeatureEngineer
from src.ml_models import CryptoMLModels
from src.signal_generator import SignalGenerator
from src.risk_management import RiskManager
from src.visualization import CryptoVisualizer
from src.telegram_bot import TelegramSignalBot
import asyncio

class CryptoSignalSystem:
    def __init__(self, symbols: list = None, initial_capital: float = 10000):
        """
        Kripto para sinyal sistemi sınıfını başlatır.
        
        Args:
            symbols (list): İşlem yapılacak kripto para çiftleri
            initial_capital (float): Başlangıç sermayesi
        """
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'BNB/USDT']
        self.initial_capital = initial_capital
        
        # Modülleri başlat
        self.data_collector = CryptoDataCollector()
        self.feature_engineer = FeatureEngineer()
        self.ml_models = CryptoMLModels()
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager(initial_capital=initial_capital)
        self.visualizer = CryptoVisualizer()
        
        # Veri ve modeller
        self.data = {}
        self.features = {}
        self.signals = {}
        self.trained_models = {}
        
        # Telegram bot
        self.telegram_bot = TelegramSignalBot()
        
    def collect_data(self, timeframe: str = '1h', limit: int = 500):
        """
        Kripto para verilerini toplar.
        
        Args:
            timeframe (str): Zaman dilimi
            limit (int): Her sembol için veri sayısı
        """
        print("Veri toplama başlatılıyor...")
        
        try:
            # Verileri topla
            self.data = self.data_collector.get_multiple_symbols_data(
                self.symbols, timeframe=timeframe, limit=limit
            )
            
            # Verileri kaydet
            for symbol, df in self.data.items():
                if not df.empty:
                    self.data_collector.save_data(df, f"data_{symbol.replace('/', '_')}")
                    print(f"{symbol}: {len(df)} veri noktası toplandı")
            
            print("Veri toplama tamamlandı!")
            
        except Exception as e:
            print(f"Veri toplama hatası: {e}")
            # Test verisi oluştur
            self._create_test_data()
    
    def _create_test_data(self):
        """Test verisi oluşturur."""
        print("Test verisi oluşturuluyor...")
        
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=500, freq='H')
        
        for symbol in self.symbols:
            # Gerçekçi fiyat simülasyonu
            base_price = 45000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 1.0
            
            # Trend ve volatilite ile fiyat oluştur
            trend = np.cumsum(np.random.normal(0, 0.001, len(dates)))
            noise = np.random.normal(0, 0.02, len(dates))
            price_multiplier = np.exp(trend + noise)
            
            close_prices = base_price * price_multiplier
            
            # OHLCV verileri
            volatility = 0.02
            self.data[symbol] = pd.DataFrame({
                'open': close_prices * (1 + np.random.normal(0, volatility, len(dates))),
                'high': close_prices * (1 + abs(np.random.normal(0, volatility, len(dates)))),
                'low': close_prices * (1 - abs(np.random.normal(0, volatility, len(dates)))),
                'close': close_prices,
                'volume': np.random.uniform(1000, 10000, len(dates))
            }, index=dates)
            
            print(f"{symbol}: Test verisi oluşturuldu")
    
    def engineer_features(self):
        """Özellik mühendisliği yapar."""
        print("Özellik mühendisliği başlatılıyor...")
        
        for symbol, df in self.data.items():
            if not df.empty:
                # Teknik göstergeleri hesapla
                features = self.feature_engineer.calculate_all_features(df)
                
                # Sinyal özelliklerini oluştur
                signal_features = self.feature_engineer.generate_signal_features(features)
                
                self.features[symbol] = signal_features
                print(f"{symbol}: Özellikler hesaplandı")
        
        print("Özellik mühendisliği tamamlandı!")
    
    def train_models(self, target_type: str = 'buy_signal'):
        """
        Makine öğrenmesi modellerini eğitir.
        
        Args:
            target_type (str): Hedef değişken türü
        """
        print("Model eğitimi başlatılıyor...")
        
        for symbol, features in self.features.items():
            if not features.empty:
                print(f"\n{symbol} için modeller eğitiliyor...")
                
                # Veriyi hazırla
                X, y = self.ml_models.prepare_data(features, target_type)
                
                if len(X) > 0 and len(y) > 0:
                    # Modelleri eğit
                    results = self.ml_models.train_models(X, y)
                    
                    # En iyi modeli seç
                    best_model_name, best_model = self.ml_models.get_best_model()
                    self.trained_models[symbol] = {
                        'model': best_model,
                        'scaler': self.ml_models.scaler,
                        'results': results
                    }
                    
                    print(f"{symbol} - En iyi model: {best_model_name}")
                    
                    # Modeli kaydet
                    self.ml_models.save_model(best_model_name, f"model_{symbol.replace('/', '_')}")
        
        print("Model eğitimi tamamlandı!")
    
    async def generate_signals(self):
        """Alım-satım sinyallerini üretir."""
        print("Sinyal üretimi başlatılıyor...")
        
        for symbol, features in self.features.items():
            if not features.empty:
                print(f"\n{symbol} için sinyaller üretiliyor...")
                
                # ML modelini yükle
                ml_model = None
                scaler = None
                if symbol in self.trained_models:
                    ml_model = self.trained_models[symbol]['model']
                    scaler = self.trained_models[symbol]['scaler']
                
                # Sinyal üreticiyi başlat
                signal_gen = SignalGenerator(ml_model=ml_model, scaler=scaler)
                
                # Birleştirilmiş sinyaller
                combined_signals = signal_gen.generate_combined_signals(features)
                
                # Giriş/çıkış sinyalleri
                entry_exit_signals = signal_gen.generate_entry_exit_signals(combined_signals)
                
                # Risk yönetimi
                risk_signals = self.risk_manager.apply_position_sizing(
                    entry_exit_signals, entry_exit_signals['Entry_Signal']
                )
                
                self.signals[symbol] = risk_signals
                
                # Sinyal özeti
                signal_gen.print_signal_summary(risk_signals)
                
                # Sinyalleri kaydet
                signal_gen.save_signal_history(risk_signals, symbol.replace('/', '_'))
                
                # Telegram'a sinyal gönder
                await self.send_telegram_signal(symbol, risk_signals)
        
        print("Sinyal üretimi tamamlandı!")
    
    async def send_telegram_signal(self, symbol: str, signals: pd.DataFrame):
        """
        Telegram'a sinyal gönderir.
        
        Args:
            symbol (str): Kripto para sembolü
            signals (pd.DataFrame): Sinyal verileri
        """
        if not self.telegram_bot.is_configured():
            print("⚠️ Telegram bot yapılandırılmamış!")
            return
        
        try:
            # Son sinyal verilerini hazırla
            last_signals = signals.tail(3).to_dict('records')
            
            # Özet istatistikleri hesapla
            total_signals = len(signals)
            buy_signals = len(signals[signals['Signal_Direction'] == 'BUY'])
            sell_signals = len(signals[signals['Signal_Direction'] == 'SELL'])
            hold_signals = len(signals[signals['Signal_Direction'] == 'HOLD'])
            
            summary = {
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'hold_signals': hold_signals,
                'buy_percentage': (buy_signals / total_signals * 100) if total_signals > 0 else 0,
                'sell_percentage': (sell_signals / total_signals * 100) if total_signals > 0 else 0,
                'hold_percentage': (hold_signals / total_signals * 100) if total_signals > 0 else 0,
                'avg_strength': signals['Signal_Strength'].mean()
            }
            
            # Sinyal verilerini hazırla
            signal_data = {
                'last_signals': last_signals,
                'summary': summary
            }
            
            # Telegram'a gönder
            await self.telegram_bot.send_signal_alert(symbol, signal_data)
            
        except Exception as e:
            print(f"Telegram sinyal gönderme hatası: {e}")
    
    async def send_telegram_summary(self):
        """Tüm sinyallerin özetini Telegram'a gönderir."""
        if not self.telegram_bot.is_configured():
            print("⚠️ Telegram bot yapılandırılmamış!")
            return
        
        try:
            all_signals = {}
            
            for symbol, signals in self.signals.items():
                if not signals.empty:
                    # Son sinyal verilerini hazırla
                    last_signals = signals.tail(3).to_dict('records')
                    
                    # Özet istatistikleri hesapla
                    total_signals = len(signals)
                    buy_signals = len(signals[signals['Signal_Direction'] == 'BUY'])
                    sell_signals = len(signals[signals['Signal_Direction'] == 'SELL'])
                    hold_signals = len(signals[signals['Signal_Direction'] == 'HOLD'])
                    
                    summary = {
                        'total_signals': total_signals,
                        'buy_signals': buy_signals,
                        'sell_signals': sell_signals,
                        'hold_signals': hold_signals,
                        'buy_percentage': (buy_signals / total_signals * 100) if total_signals > 0 else 0,
                        'sell_percentage': (sell_signals / total_signals * 100) if total_signals > 0 else 0,
                        'hold_percentage': (hold_signals / total_signals * 100) if total_signals > 0 else 0,
                        'avg_strength': signals['Signal_Strength'].mean()
                    }
                    
                    all_signals[symbol] = {
                        'last_signals': last_signals,
                        'summary': summary
                    }
            
            # Telegram'a özet gönder
            await self.telegram_bot.send_daily_summary(all_signals)
            
        except Exception as e:
            print(f"Telegram özet gönderme hatası: {e}")
    
    def create_visualizations(self):
        """Görselleştirmeleri oluşturur."""
        print("Görselleştirmeler oluşturuluyor...")
        
        figures = {}
        
        for symbol, signals in self.signals.items():
            if not signals.empty:
                print(f"\n{symbol} için grafikler oluşturuluyor...")
                
                # Fiyat ve sinyaller
                price_fig = self.visualizer.plot_price_with_signals(
                    self.data[symbol], 
                    signals['Entry_Signal'],
                    f"{symbol} - Fiyat ve Sinyaller"
                )
                
                # Teknik göstergeler
                indicators_fig = self.visualizer.plot_technical_indicators(signals)
                
                # Sinyal analizi
                signal_analysis_fig = self.visualizer.plot_signal_analysis(signals)
                
                # Dashboard
                dashboard_fig = self.visualizer.create_dashboard(self.data[symbol], signals)
                
                # Performans metrikleri
                returns = self.data[symbol]['close'].pct_change().dropna()
                performance_fig = self.visualizer.plot_performance_metrics(returns)
                
                # Risk analizi
                risk_report = self.risk_manager.generate_risk_report(self.data[symbol])
                risk_fig = self.visualizer.plot_risk_analysis(self.data[symbol], risk_report)
                
                # Grafikleri kaydet
                symbol_figures = {
                    'price': price_fig,
                    'indicators': indicators_fig,
                    'signals': signal_analysis_fig,
                    'dashboard': dashboard_fig,
                    'performance': performance_fig,
                    'risk': risk_fig
                }
                
                self.visualizer.save_plots(symbol_figures, f"analysis_{symbol.replace('/', '_')}")
                figures[symbol] = symbol_figures
        
        print("Görselleştirmeler tamamlandı!")
    
    def generate_report(self):
        """Kapsamlı rapor oluşturur."""
        print("Rapor oluşturuluyor...")
        
        print("\n" + "="*80)
        print("KRİPTO PARA ALIM-SATIM SİNYALİ SİSTEMİ RAPORU")
        print("="*80)
        print(f"Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Analiz Edilen Semboller: {', '.join(self.symbols)}")
        print(f"Başlangıç Sermayesi: ${self.initial_capital:,.2f}")
        
        # Model performansları
        print("\nMODEL PERFORMANSLARI:")
        for symbol, model_data in self.trained_models.items():
            results = model_data['results']
            best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
            best_result = results[best_model_name]
            
            print(f"\n{symbol}:")
            print(f"  En İyi Model: {best_model_name}")
            print(f"  Doğruluk: {best_result['accuracy']:.4f}")
            print(f"  F1 Skoru: {best_result['f1_score']:.4f}")
            print(f"  Çapraz Doğrulama: {best_result['cv_mean']:.4f} (+/- {best_result['cv_std']*2:.4f})")
        
        # Sinyal özetleri
        print("\nSİNYAL ÖZETLERİ:")
        for symbol, signals in self.signals.items():
            if not signals.empty:
                total_signals = len(signals)
                buy_signals = len(signals[signals['Signal_Direction'] == 'BUY'])
                sell_signals = len(signals[signals['Signal_Direction'] == 'SELL'])
                
                print(f"\n{symbol}:")
                print(f"  Toplam Sinyal: {total_signals}")
                print(f"  Alım Sinyali: {buy_signals} ({buy_signals/total_signals*100:.1f}%)")
                print(f"  Satım Sinyali: {sell_signals} ({sell_signals/total_signals*100:.1f}%)")
        
        # Risk analizi
        print("\nRİSK ANALİZİ:")
        for symbol, signals in self.signals.items():
            if not signals.empty:
                risk_report = self.risk_manager.generate_risk_report(self.data[symbol])
                self.risk_manager.print_risk_summary(risk_report)
        
        print("\n" + "="*80)
        print("RAPOR TAMAMLANDI")
        print("="*80)
    
    async def run_complete_analysis(self):
        """Tam analiz sürecini çalıştırır."""
        print("Kripto Para Alım-Satım Sinyali Sistemi Başlatılıyor...")
        print("="*60)
        
        try:
            # Telegram başlangıç mesajı
            if self.telegram_bot.is_configured():
                await self.telegram_bot.send_startup_message()
            
            # 1. Veri toplama
            self.collect_data()
            
            # 2. Özellik mühendisliği
            self.engineer_features()
            
            # 3. Model eğitimi
            self.train_models()
            
            # 4. Sinyal üretimi
            await self.generate_signals()
            
            # 5. Telegram özeti gönder
            await self.send_telegram_summary()
            
            # 6. Görselleştirme
            self.create_visualizations()
            
            # 7. Rapor oluşturma
            self.generate_report()
            
            print("\nTüm analizler başarıyla tamamlandı!")
            print("Sonuçlar 'data/' klasöründe kaydedildi.")
            
        except Exception as e:
            print(f"Analiz sırasında hata oluştu: {e}")
            # Telegram'a hata mesajı gönder
            if self.telegram_bot.is_configured():
                await self.telegram_bot.send_error_alert(str(e))
            import traceback
            traceback.print_exc()

async def main():
    """Ana fonksiyon."""
    print("Kripto Para Alım-Satım Sinyali Sistemi")
    print("="*50)
    
    # Sistem parametreleri
    symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'BNB/USDT']
    initial_capital = 10000
    
    # Sistemi başlat
    system = CryptoSignalSystem(symbols=symbols, initial_capital=initial_capital)
    
    # Tam analizi çalıştır
    await system.run_complete_analysis()

if __name__ == "__main__":
    asyncio.run(main()) 