"""
Teknik analiz göstergelerini hesaplayan özellik mühendisliği modülü.
RSI, MACD, Bollinger Bantları, Hareketli Ortalamalar ve diğer göstergeleri hesaplar.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Tuple

class FeatureEngineer:
    def __init__(self):
        """Özellik mühendisliği sınıfını başlatır."""
        pass
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Göreceli Güç Endeksi (RSI) hesaplar.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            period (int): RSI periyodu (varsayılan: 14)
            
        Returns:
            pd.Series: RSI değerleri
        """
        return ta.momentum.RSIIndicator(df['close'], window=period).rsi()
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence) hesaplar.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            fast (int): Hızlı MA periyodu
            slow (int): Yavaş MA periyodu
            signal (int): Sinyal çizgisi periyodu
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: MACD, Sinyal, Histogram
        """
        macd_indicator = ta.trend.MACD(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
        return macd_indicator.macd(), macd_indicator.macd_signal(), macd_indicator.macd_diff()
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bantları hesaplar.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            period (int): MA periyodu
            std (int): Standart sapma çarpanı
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: Üst bant, Orta bant, Alt bant
        """
        bb_indicator = ta.volatility.BollingerBands(df['close'], window=period, window_dev=std)
        return bb_indicator.bollinger_hband(), bb_indicator.bollinger_mavg(), bb_indicator.bollinger_lband()
    
    def calculate_moving_averages(self, df: pd.DataFrame, periods: List[int] = [10, 20, 50, 200]) -> Dict[str, pd.Series]:
        """
        Hareketli ortalamalar hesaplar.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            periods (List[int]): MA periyotları
            
        Returns:
            Dict[str, pd.Series]: Periyot bazında MA'lar
        """
        mas = {}
        for period in periods:
            mas[f'MA_{period}'] = ta.trend.SMAIndicator(df['close'], window=period).sma_indicator()
        return mas
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator hesaplar.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            k_period (int): %K periyodu
            d_period (int): %D periyodu
            
        Returns:
            Tuple[pd.Series, pd.Series]: %K, %D
        """
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 
                                                window=k_period, smooth_window=d_period)
        return stoch.stoch(), stoch.stoch_signal()
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR) hesaplar.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            period (int): ATR periyodu
            
        Returns:
            pd.Series: ATR değerleri
        """
        return ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Hacim göstergelerini hesaplar.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            
        Returns:
            Dict[str, pd.Series]: Hacim göstergeleri
        """
        indicators = {}
        
        # On Balance Volume (OBV)
        indicators['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Volume Weighted Average Price (VWAP)
        indicators['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
        
        # Money Flow Index (MFI)
        indicators['MFI'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        
        return indicators
    
    def calculate_price_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Fiyat kalıplarını hesaplar.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            
        Returns:
            Dict[str, pd.Series]: Fiyat kalıpları
        """
        patterns = {}
        
        # Basit fiyat kalıpları hesaplama (ta.candlestick modülü mevcut değil)
        # Doji pattern - açılış ve kapanış fiyatları neredeyse eşit
        body_size = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        patterns['DOJI'] = (body_size <= total_range * 0.1).astype(int)
        
        # Hammer pattern - alt gölge uzun, üst gölge kısa
        body_size = abs(df['close'] - df['open'])
        lower_shadow = np.minimum(df['open'], df['close']) - df['low']
        upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
        patterns['HAMMER'] = ((lower_shadow > body_size * 2) & (upper_shadow < body_size * 0.5)).astype(int)
        
        # Engulfing pattern - önceki mumu tamamen kaplayan mum
        bullish_engulfing = ((df['open'] < df['close'].shift(1)) & 
                            (df['close'] > df['open'].shift(1)) &
                            (df['open'] < df['close'].shift(1)) &
                            (df['close'] > df['open'].shift(1))).astype(int)
        patterns['ENGULFING'] = bullish_engulfing
        
        return patterns
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """
        Destek ve direnç seviyelerini hesaplar.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            window (int): Pencere boyutu
            
        Returns:
            Tuple[pd.Series, pd.Series]: Destek, Direnç
        """
        # Basit destek ve direnç hesaplama
        support = df['low'].rolling(window=window).min()
        resistance = df['high'].rolling(window=window).max()
        
        return support, resistance
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm teknik göstergeleri hesaplar.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            
        Returns:
            pd.DataFrame: Tüm özelliklerle birlikte DataFrame
        """
        # Orijinal veriyi kopyala
        features_df = df.copy()
        
        # RSI
        features_df['RSI'] = self.calculate_rsi(df)
        
        # MACD
        macd, signal, histogram = self.calculate_macd(df)
        features_df['MACD'] = macd
        features_df['MACD_Signal'] = signal
        features_df['MACD_Histogram'] = histogram
        
        # Bollinger Bantları
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df)
        features_df['BB_Upper'] = bb_upper
        features_df['BB_Middle'] = bb_middle
        features_df['BB_Lower'] = bb_lower
        features_df['BB_Width'] = bb_upper - bb_lower
        features_df['BB_Position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Hareketli Ortalamalar
        mas = self.calculate_moving_averages(df)
        for name, values in mas.items():
            features_df[name] = values
        
        # Stochastic
        stoch_k, stoch_d = self.calculate_stochastic(df)
        features_df['Stoch_K'] = stoch_k
        features_df['Stoch_D'] = stoch_d
        
        # ATR
        features_df['ATR'] = self.calculate_atr(df)
        
        # Hacim göstergeleri
        volume_indicators = self.calculate_volume_indicators(df)
        for name, values in volume_indicators.items():
            features_df[name] = values
        
        # Fiyat kalıpları
        patterns = self.calculate_price_patterns(df)
        for name, values in patterns.items():
            features_df[name] = values
        
        # Destek ve direnç
        support, resistance = self.calculate_support_resistance(df)
        features_df['Support'] = support
        features_df['Resistance'] = resistance
        
        # Ek özellikler
        features_df['Price_Change'] = df['close'].pct_change()
        features_df['Volume_Change'] = df['volume'].pct_change()
        features_df['High_Low_Ratio'] = df['high'] / df['low']
        features_df['Close_Open_Ratio'] = df['close'] / df['open']
        
        # NaN değerleri temizle
        features_df = features_df.dropna()
        
        return features_df
    
    def generate_signal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sinyal üretimi için özellikler oluşturur.
        
        Args:
            df (pd.DataFrame): Teknik göstergelerle DataFrame
            
        Returns:
            pd.DataFrame: Sinyal özellikleri
        """
        signal_df = df.copy()
        
        # RSI sinyalleri
        signal_df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
        signal_df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
        signal_df['RSI_Buy_Signal'] = (df['RSI'] > 60).astype(int)  # Momentum ticaret için
        
        # MACD sinyalleri
        signal_df['MACD_Buy_Signal'] = (df['MACD'] > df['MACD_Signal']).astype(int)
        signal_df['MACD_Bullish_Divergence'] = ((df['close'] < df['close'].shift(1)) & 
                                               (df['MACD'] > df['MACD'].shift(1))).astype(int)
        
        # Bollinger Bantları sinyalleri
        signal_df['BB_Buy_Signal'] = (df['close'] < df['BB_Lower']).astype(int)
        signal_df['BB_Sell_Signal'] = (df['close'] > df['BB_Upper']).astype(int)
        
        # Hareketli ortalama sinyalleri
        signal_df['MA_Cross_Buy'] = (df['MA_10'] > df['MA_20']).astype(int)
        signal_df['MA_Cross_Sell'] = (df['MA_10'] < df['MA_20']).astype(int)
        
        # Stochastic sinyalleri
        signal_df['Stoch_Overbought'] = (df['Stoch_K'] > 80).astype(int)
        signal_df['Stoch_Oversold'] = (df['Stoch_K'] < 20).astype(int)
        
        # Hacim onayı
        signal_df['Volume_Confirmation'] = (df['volume'] > df['volume'].rolling(20).mean()).astype(int)
        
        # Kombine sinyaller
        signal_df['Strong_Buy_Signal'] = (
            (df['RSI'] < 30) & 
            (df['close'] < df['BB_Lower']) & 
            (df['MACD'] > df['MACD_Signal']) &
            (df['volume'] > df['volume'].rolling(20).mean())
        ).astype(int)
        
        signal_df['Strong_Sell_Signal'] = (
            (df['RSI'] > 70) & 
            (df['close'] > df['BB_Upper']) & 
            (df['MACD'] < df['MACD_Signal']) &
            (df['volume'] > df['volume'].rolling(20).mean())
        ).astype(int)
        
        return signal_df

# Test fonksiyonu
if __name__ == "__main__":
    # Test verisi oluştur
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    test_data = pd.DataFrame({
        'open': np.random.uniform(40000, 50000, 100),
        'high': np.random.uniform(40000, 50000, 100),
        'low': np.random.uniform(40000, 50000, 100),
        'close': np.random.uniform(40000, 50000, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    }, index=dates)
    
    # Özellik mühendisliği
    fe = FeatureEngineer()
    features = fe.calculate_all_features(test_data)
    signals = fe.generate_signal_features(features)
    
    print("Özellikler hesaplandı:")
    print(features.columns.tolist())
    print("\nSinyal özellikleri:")
    print(signals.columns.tolist()) 