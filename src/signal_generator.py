"""
Kripto para alım-satım sinyallerini üreten modül.
Teknik analiz ve makine öğrenmesi modellerini kullanarak sinyaller üretir.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class SignalGenerator:
    def __init__(self, ml_model=None, scaler=None):
        """
        Sinyal üretici sınıfını başlatır.
        
        Args:
            ml_model: Eğitilmiş makine öğrenmesi modeli
            scaler: Veri ölçeklendirici
        """
        self.ml_model = ml_model
        self.scaler = scaler
        self.signal_history = []
        
    def generate_technical_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Teknik analiz göstergelerine dayalı sinyaller üretir.
        
        Args:
            df (pd.DataFrame): Teknik göstergelerle DataFrame
            
        Returns:
            pd.DataFrame: Sinyal göstergeleri
        """
        signals = df.copy()
        
        # RSI sinyalleri
        signals['RSI_Signal'] = 0
        signals.loc[df['RSI'] < 30, 'RSI_Signal'] = 1  # Aşırı satım - Alım
        signals.loc[df['RSI'] > 70, 'RSI_Signal'] = -1  # Aşırı alım - Satım
        signals.loc[(df['RSI'] > 60) & (df['RSI'] < 70), 'RSI_Signal'] = 0.5  # Momentum alım
        
        # MACD sinyalleri
        signals['MACD_Signal'] = 0
        signals.loc[df['MACD'] > df['MACD_Signal'], 'MACD_Signal'] = 1  # Boğa sinyali
        signals.loc[df['MACD'] < df['MACD_Signal'], 'MACD_Signal'] = -1  # Ayı sinyali
        
        # Bollinger Bantları sinyalleri
        signals['BB_Signal'] = 0
        signals.loc[df['close'] < df['BB_Lower'], 'BB_Signal'] = 1  # Alt banda dokunma - Alım
        signals.loc[df['close'] > df['BB_Upper'], 'BB_Signal'] = -1  # Üst banda dokunma - Satım
        
        # Hareketli ortalama sinyalleri
        signals['MA_Signal'] = 0
        if 'MA_10' in df.columns and 'MA_20' in df.columns:
            signals.loc[df['MA_10'] > df['MA_20'], 'MA_Signal'] = 1  # Kısa MA uzun MA'yı kesiyor
            signals.loc[df['MA_10'] < df['MA_20'], 'MA_Signal'] = -1
        
        # Stochastic sinyalleri
        signals['Stoch_Signal'] = 0
        if 'Stoch_K' in df.columns:
            signals.loc[df['Stoch_K'] < 20, 'Stoch_Signal'] = 1  # Aşırı satım
            signals.loc[df['Stoch_K'] > 80, 'Stoch_Signal'] = -1  # Aşırı alım
        
        # Hacim onayı
        signals['Volume_Signal'] = 0
        if 'volume' in df.columns:
            volume_ma = df['volume'].rolling(20).mean()
            signals.loc[df['volume'] > volume_ma * 1.5, 'Volume_Signal'] = 1  # Yüksek hacim
        
        return signals
    
    def generate_ml_signals(self, df: pd.DataFrame, confidence_threshold: float = 0.6) -> pd.DataFrame:
        """
        Makine öğrenmesi modeli ile sinyaller üretir.
        
        Args:
            df (pd.DataFrame): Özelliklerle DataFrame
            confidence_threshold (float): Güven eşiği
            
        Returns:
            pd.DataFrame: ML sinyalleri
        """
        if self.ml_model is None:
            print("ML modeli yüklenmemiş")
            return df
        
        signals = df.copy()
        
        # Özellik sütunlarını belirle
        feature_columns = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position',
            'MA_10', 'MA_20', 'MA_50', 'MA_200',
            'Stoch_K', 'Stoch_D', 'ATR', 'OBV', 'VWAP', 'MFI',
            'Price_Change', 'Volume_Change', 'High_Low_Ratio', 'Close_Open_Ratio',
            'RSI_Overbought', 'RSI_Oversold', 'RSI_Buy_Signal',
            'MACD_Buy_Signal', 'MACD_Bullish_Divergence',
            'BB_Buy_Signal', 'BB_Sell_Signal',
            'MA_Cross_Buy', 'MA_Cross_Sell',
            'Stoch_Overbought', 'Stoch_Oversold',
            'Volume_Confirmation'
        ]
        
        # Mevcut özellikleri al
        available_features = [col for col in feature_columns if col in df.columns]
        
        if len(available_features) == 0:
            print("Uygun özellik bulunamadı")
            return df
        
        # Tahmin yap
        try:
            X = df[available_features].fillna(0)
            
            if self.scaler:
                X_scaled = self.scaler.transform(X)
                predictions = self.ml_model.predict(X_scaled)
                probabilities = self.ml_model.predict_proba(X_scaled)
            else:
                predictions = self.ml_model.predict(X)
                probabilities = self.ml_model.predict_proba(X)
            
            # Sinyalleri ekle
            signals['ML_Signal'] = predictions
            signals['ML_Confidence'] = np.max(probabilities, axis=1)
            
            # Güven eşiğine göre filtrele
            signals['ML_Signal_Filtered'] = 0
            signals.loc[signals['ML_Confidence'] > confidence_threshold, 'ML_Signal_Filtered'] = signals['ML_Signal']
            
        except Exception as e:
            print(f"ML tahmin hatası: {e}")
            signals['ML_Signal'] = 0
            signals['ML_Confidence'] = 0
            signals['ML_Signal_Filtered'] = 0
        
        return signals
    
    def generate_combined_signals(self, df: pd.DataFrame, weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Teknik analiz ve ML sinyallerini birleştirir.
        
        Args:
            df (pd.DataFrame): Özelliklerle DataFrame
            weights (Dict[str, float]): Sinyal ağırlıkları
            
        Returns:
            pd.DataFrame: Birleştirilmiş sinyaller
        """
        # Varsayılan ağırlıklar
        if weights is None:
            weights = {
                'RSI': 0.2,
                'MACD': 0.2,
                'BB': 0.15,
                'MA': 0.15,
                'Stoch': 0.1,
                'Volume': 0.1,
                'ML': 0.1
            }
        
        signals = df.copy()
        
        # Teknik sinyalleri hesapla
        tech_signals = self.generate_technical_signals(df)
        
        # ML sinyallerini hesapla
        ml_signals = self.generate_ml_signals(df)
        
        # Sinyalleri birleştir
        combined_signal = (
            weights['RSI'] * tech_signals['RSI_Signal'] +
            weights['MACD'] * tech_signals['MACD_Signal'] +
            weights['BB'] * tech_signals['BB_Signal'] +
            weights['MA'] * tech_signals['MA_Signal'] +
            weights['Stoch'] * tech_signals['Stoch_Signal'] +
            weights['Volume'] * tech_signals['Volume_Signal'] +
            weights['ML'] * ml_signals['ML_Signal_Filtered']
        )
        
        signals['Combined_Signal'] = combined_signal
        
        # Sinyal gücünü hesapla
        signals['Signal_Strength'] = abs(combined_signal)
        
        # Sinyal yönünü belirle
        signals['Signal_Direction'] = np.where(combined_signal > 0.3, 'BUY', 
                                             np.where(combined_signal < -0.3, 'SELL', 'HOLD'))
        
        return signals
    
    def generate_entry_exit_signals(self, df: pd.DataFrame, 
                                  entry_threshold: float = 0.5,
                                  exit_threshold: float = -0.3) -> pd.DataFrame:
        """
        Giriş ve çıkış sinyallerini üretir.
        
        Args:
            df (pd.DataFrame): Sinyal göstergeleri
            entry_threshold (float): Giriş eşiği
            exit_threshold (float): Çıkış eşiği
            
        Returns:
            pd.DataFrame: Giriş/çıkış sinyalleri
        """
        signals = df.copy()
        
        # Giriş sinyalleri
        signals['Entry_Signal'] = 0
        signals.loc[signals['Combined_Signal'] > entry_threshold, 'Entry_Signal'] = 1
        
        # Çıkış sinyalleri
        signals['Exit_Signal'] = 0
        signals.loc[signals['Combined_Signal'] < exit_threshold, 'Exit_Signal'] = 1
        
        # Pozisyon durumu
        signals['Position'] = 0
        position = 0
        
        for i in range(len(signals)):
            if signals.iloc[i]['Entry_Signal'] == 1 and position == 0:
                position = 1
            elif signals.iloc[i]['Exit_Signal'] == 1 and position == 1:
                position = 0
            
            signals.iloc[i, signals.columns.get_loc('Position')] = position
        
        return signals
    
    def calculate_signal_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Sinyal metriklerini hesaplar.
        
        Args:
            df (pd.DataFrame): Sinyal göstergeleri
            
        Returns:
            Dict[str, Any]: Sinyal metrikleri
        """
        metrics = {}
        
        # Sinyal sayıları
        metrics['total_signals'] = len(df)
        metrics['buy_signals'] = len(df[df['Signal_Direction'] == 'BUY'])
        metrics['sell_signals'] = len(df[df['Signal_Direction'] == 'SELL'])
        metrics['hold_signals'] = len(df[df['Signal_Direction'] == 'HOLD'])
        
        # Sinyal oranları
        metrics['buy_ratio'] = metrics['buy_signals'] / metrics['total_signals']
        metrics['sell_ratio'] = metrics['sell_signals'] / metrics['total_signals']
        metrics['hold_ratio'] = metrics['hold_signals'] / metrics['total_signals']
        
        # Ortalama sinyal gücü
        metrics['avg_signal_strength'] = df['Signal_Strength'].mean()
        metrics['max_signal_strength'] = df['Signal_Strength'].max()
        
        # Sinyal değişim sıklığı
        signal_changes = (df['Signal_Direction'] != df['Signal_Direction'].shift(1)).sum()
        metrics['signal_change_frequency'] = signal_changes / metrics['total_signals']
        
        return metrics
    
    def get_latest_signals(self, df: pd.DataFrame, n_periods: int = 5) -> pd.DataFrame:
        """
        En son sinyalleri döndürür.
        
        Args:
            df (pd.DataFrame): Sinyal göstergeleri
            n_periods (int): Döndürülecek periyot sayısı
            
        Returns:
            pd.DataFrame: En son sinyaller
        """
        return df.tail(n_periods)
    
    def save_signal_history(self, df: pd.DataFrame, filename: str):
        """
        Sinyal geçmişini kaydeder.
        
        Args:
            df (pd.DataFrame): Sinyal göstergeleri
            filename (str): Dosya adı
        """
        try:
            df.to_csv(f"data/{filename}_signals.csv")
            print(f"Sinyal geçmişi {filename}_signals.csv olarak kaydedildi")
        except Exception as e:
            print(f"Sinyal kaydetme hatası: {e}")
    
    def print_signal_summary(self, df: pd.DataFrame):
        """
        Sinyal özetini yazdırır.
        
        Args:
            df (pd.DataFrame): Sinyal göstergeleri
        """
        metrics = self.calculate_signal_metrics(df)
        
        print("\n" + "="*60)
        print("SİNYAL ÖZETİ")
        print("="*60)
        print(f"Toplam Sinyal: {metrics['total_signals']}")
        print(f"Alım Sinyali: {metrics['buy_signals']} ({metrics['buy_ratio']:.2%})")
        print(f"Satım Sinyali: {metrics['sell_signals']} ({metrics['sell_ratio']:.2%})")
        print(f"Bekle Sinyali: {metrics['hold_signals']} ({metrics['hold_ratio']:.2%})")
        print(f"Ortalama Sinyal Gücü: {metrics['avg_signal_strength']:.4f}")
        print(f"Maksimum Sinyal Gücü: {metrics['max_signal_strength']:.4f}")
        print(f"Sinyal Değişim Sıklığı: {metrics['signal_change_frequency']:.4f}")
        print("="*60)
        
        # En son sinyaller
        latest_signals = self.get_latest_signals(df, 3)
        print("\nEN SON SİNYALLER:")
        print(latest_signals[['close', 'Combined_Signal', 'Signal_Direction', 'Signal_Strength']].to_string())

# Test fonksiyonu
if __name__ == "__main__":
    # Test verisi oluştur
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    
    test_data = pd.DataFrame({
        'open': np.random.uniform(40000, 50000, 100),
        'high': np.random.uniform(40000, 50000, 100),
        'low': np.random.uniform(40000, 50000, 100),
        'close': np.random.uniform(40000, 50000, 100),
        'volume': np.random.uniform(1000, 5000, 100),
        'RSI': np.random.uniform(0, 100, 100),
        'MACD': np.random.uniform(-2, 2, 100),
        'MACD_Signal': np.random.uniform(-2, 2, 100),
        'BB_Lower': np.random.uniform(39000, 49000, 100),
        'BB_Upper': np.random.uniform(41000, 51000, 100),
        'MA_10': np.random.uniform(40000, 50000, 100),
        'MA_20': np.random.uniform(40000, 50000, 100),
        'Stoch_K': np.random.uniform(0, 100, 100)
    }, index=dates)
    
    # Sinyal üretici
    signal_gen = SignalGenerator()
    
    # Teknik sinyaller
    tech_signals = signal_gen.generate_technical_signals(test_data)
    
    # Birleştirilmiş sinyaller
    combined_signals = signal_gen.generate_combined_signals(test_data)
    
    # Giriş/çıkış sinyalleri
    entry_exit_signals = signal_gen.generate_entry_exit_signals(combined_signals)
    
    # Sinyal özeti
    signal_gen.print_signal_summary(entry_exit_signals) 