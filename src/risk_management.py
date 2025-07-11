"""
Risk yönetimi stratejilerini içeren modül.
Stop-loss, hedef fiyat, pozisyon boyutlandırma ve portföy çeşitlendirme.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RiskManager:
    def __init__(self, initial_capital: float = 10000, max_risk_per_trade: float = 0.02):
        """
        Risk yöneticisi sınıfını başlatır.
        
        Args:
            initial_capital (float): Başlangıç sermayesi
            max_risk_per_trade (float): İşlem başına maksimum risk oranı (0.02 = %2)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.positions = {}
        self.trade_history = []
        
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              risk_amount: float = None) -> float:
        """
        Pozisyon boyutunu hesaplar.
        
        Args:
            entry_price (float): Giriş fiyatı
            stop_loss (float): Stop-loss fiyatı
            risk_amount (float): Risk edilecek miktar (None ise max_risk_per_trade kullanılır)
            
        Returns:
            float: Pozisyon boyutu (miktar)
        """
        if risk_amount is None:
            risk_amount = self.current_capital * self.max_risk_per_trade
        
        # Risk per birim hesapla
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
        
        # Pozisyon boyutu = Risk miktarı / Birim başına risk
        position_size = risk_amount / risk_per_unit
        
        return position_size
    
    def calculate_stop_loss(self, df: pd.DataFrame, method: str = 'atr', 
                           multiplier: float = 2.0) -> pd.Series:
        """
        Stop-loss seviyelerini hesaplar.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            method (str): Stop-loss hesaplama yöntemi ('atr', 'percentage', 'support')
            multiplier (float): ATR çarpanı
            
        Returns:
            pd.Series: Stop-loss seviyeleri
        """
        stop_loss = pd.Series(index=df.index, dtype=float)
        
        if method == 'atr':
            # ATR tabanlı stop-loss
            if 'ATR' in df.columns:
                stop_loss = df['close'] - (df['ATR'] * multiplier)
            else:
                # ATR hesapla
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(14).mean()
                stop_loss = df['close'] - (atr * multiplier)
        
        elif method == 'percentage':
            # Yüzde tabanlı stop-loss (%3)
            stop_loss = df['close'] * 0.97
        
        elif method == 'support':
            # Destek seviyesi tabanlı stop-loss
            if 'Support' in df.columns:
                stop_loss = df['Support'] * 0.98  # Destek seviyesinin %2 altı
            else:
                # Basit destek hesaplama
                support = df['low'].rolling(20).min()
                stop_loss = support * 0.98
        
        return stop_loss
    
    def calculate_target_price(self, df: pd.DataFrame, method: str = 'risk_reward', 
                             risk_reward_ratio: float = 2.0) -> pd.Series:
        """
        Hedef fiyatları hesaplar.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            method (str): Hedef fiyat hesaplama yöntemi
            risk_reward_ratio (float): Risk/ödül oranı
            
        Returns:
            pd.Series: Hedef fiyatlar
        """
        target_price = pd.Series(index=df.index, dtype=float)
        
        if method == 'risk_reward':
            # Risk/ödül oranına göre hedef
            stop_loss = self.calculate_stop_loss(df, method='atr')
            risk = df['close'] - stop_loss
            target_price = df['close'] + (risk * risk_reward_ratio)
        
        elif method == 'resistance':
            # Direnç seviyesi tabanlı hedef
            if 'Resistance' in df.columns:
                target_price = df['Resistance'] * 0.98  # Direnç seviyesinin %2 altı
            else:
                # Basit direnç hesaplama
                resistance = df['high'].rolling(20).max()
                target_price = resistance * 0.98
        
        elif method == 'percentage':
            # Yüzde tabanlı hedef (%5)
            target_price = df['close'] * 1.05
        
        return target_price
    
    def calculate_portfolio_risk(self, positions: Dict[str, Dict]) -> Dict[str, float]:
        """
        Portföy riskini hesaplar.
        
        Args:
            positions (Dict[str, Dict]): Pozisyonlar sözlüğü
            
        Returns:
            Dict[str, float]: Risk metrikleri
        """
        total_value = 0
        total_risk = 0
        position_weights = {}
        
        for symbol, position in positions.items():
            current_value = position['quantity'] * position['current_price']
            total_value += current_value
            
            # Pozisyon riski
            if position['stop_loss'] > 0:
                position_risk = (position['current_price'] - position['stop_loss']) * position['quantity']
                total_risk += position_risk
        
        # Ağırlıkları hesapla
        for symbol, position in positions.items():
            current_value = position['quantity'] * position['current_price']
            position_weights[symbol] = current_value / total_value if total_value > 0 else 0
        
        # Risk metrikleri
        portfolio_risk = {
            'total_value': total_value,
            'total_risk': total_risk,
            'risk_percentage': (total_risk / total_value * 100) if total_value > 0 else 0,
            'position_weights': position_weights,
            'diversification_score': 1 - max(position_weights.values()) if position_weights else 0
        }
        
        return portfolio_risk
    
    def apply_position_sizing(self, df: pd.DataFrame, entry_signals: pd.Series, 
                            capital: float = None) -> pd.DataFrame:
        """
        Pozisyon boyutlandırmasını uygular.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            entry_signals (pd.Series): Giriş sinyalleri
            capital (float): Kullanılacak sermaye
            
        Returns:
            pd.DataFrame: Pozisyon boyutları ile DataFrame
        """
        if capital is None:
            capital = self.current_capital
        
        result_df = df.copy()
        result_df['Position_Size'] = 0.0
        result_df['Risk_Amount'] = 0.0
        result_df['Stop_Loss'] = self.calculate_stop_loss(df)
        result_df['Target_Price'] = self.calculate_target_price(df)
        
        # Giriş sinyali olan yerlerde pozisyon boyutunu hesapla
        entry_points = entry_signals[entry_signals == 1].index
        
        for idx in entry_points:
            entry_price = df.loc[idx, 'close']
            stop_loss = result_df.loc[idx, 'Stop_Loss']
            
            # Pozisyon boyutunu hesapla
            position_size = self.calculate_position_size(entry_price, stop_loss)
            risk_amount = abs(entry_price - stop_loss) * position_size
            
            result_df.loc[idx, 'Position_Size'] = position_size
            result_df.loc[idx, 'Risk_Amount'] = risk_amount
        
        return result_df
    
    def calculate_drawdown(self, equity_curve: pd.Series) -> Dict[str, float]:
        """
        Drawdown metriklerini hesaplar.
        
        Args:
            equity_curve (pd.Series): Sermaye eğrisi
            
        Returns:
            Dict[str, float]: Drawdown metrikleri
        """
        # Kümülatif maksimum
        running_max = equity_curve.expanding().max()
        
        # Drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Drawdown metrikleri
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]
        
        # Drawdown süresi
        underwater = drawdown < 0
        underwater_periods = underwater.sum()
        total_periods = len(drawdown)
        
        return {
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'underwater_periods': underwater_periods,
            'underwater_percentage': underwater_periods / total_periods if total_periods > 0 else 0
        }
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Value at Risk (VaR) hesaplar.
        
        Args:
            returns (pd.Series): Getiri serisi
            confidence_level (float): Güven seviyesi
            
        Returns:
            float: VaR değeri
        """
        # Parametrik VaR (normal dağılım varsayımı)
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Z-skor hesapla
        from scipy.stats import norm
        z_score = norm.ppf(1 - confidence_level)
        
        var = mean_return - (z_score * std_return)
        
        return var
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Sharpe oranını hesaplar.
        
        Args:
            returns (pd.Series): Getiri serisi
            risk_free_rate (float): Risksiz faiz oranı
            
        Returns:
            float: Sharpe oranı
        """
        excess_returns = returns - risk_free_rate
        sharpe_ratio = excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        
        return sharpe_ratio
    
    def generate_risk_report(self, df: pd.DataFrame, positions: Dict[str, Dict] = None) -> Dict[str, Any]:
        """
        Kapsamlı risk raporu oluşturur.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            positions (Dict[str, Dict]): Mevcut pozisyonlar
            
        Returns:
            Dict[str, Any]: Risk raporu
        """
        # Getiri hesapla
        returns = df['close'].pct_change().dropna()
        
        # Risk metrikleri
        volatility = returns.std() * np.sqrt(252)  # Yıllık volatilite
        var_95 = self.calculate_var(returns, 0.95)
        sharpe = self.calculate_sharpe_ratio(returns)
        
        # Drawdown hesapla
        cumulative_returns = (1 + returns).cumprod()
        drawdown_metrics = self.calculate_drawdown(cumulative_returns)
        
        # Portföy riski
        portfolio_risk = {}
        if positions:
            portfolio_risk = self.calculate_portfolio_risk(positions)
        
        risk_report = {
            'volatility': volatility,
            'var_95': var_95,
            'sharpe_ratio': sharpe,
            'max_drawdown': drawdown_metrics['max_drawdown'],
            'current_drawdown': drawdown_metrics['current_drawdown'],
            'underwater_percentage': drawdown_metrics['underwater_percentage'],
            'portfolio_risk': portfolio_risk,
            'total_return': (cumulative_returns.iloc[-1] - 1) if len(cumulative_returns) > 0 else 0
        }
        
        return risk_report
    
    def print_risk_summary(self, risk_report: Dict[str, Any]):
        """
        Risk özetini yazdırır.
        
        Args:
            risk_report (Dict[str, Any]): Risk raporu
        """
        print("\n" + "="*60)
        print("RİSK YÖNETİMİ ÖZETİ")
        print("="*60)
        print(f"Volatilite (Yıllık): {risk_report['volatility']:.2%}")
        print(f"VaR (95%): {risk_report['var_95']:.2%}")
        print(f"Sharpe Oranı: {risk_report['sharpe_ratio']:.4f}")
        print(f"Maksimum Drawdown: {risk_report['max_drawdown']:.2%}")
        print(f"Mevcut Drawdown: {risk_report['current_drawdown']:.2%}")
        print(f"Toplam Getiri: {risk_report['total_return']:.2%}")
        
        if risk_report['portfolio_risk']:
            portfolio = risk_report['portfolio_risk']
            print(f"\nPortföy Değeri: ${portfolio['total_value']:,.2f}")
            print(f"Toplam Risk: ${portfolio['total_risk']:,.2f}")
            print(f"Risk Yüzdesi: {portfolio['risk_percentage']:.2%}")
            print(f"Çeşitlendirme Skoru: {portfolio['diversification_score']:.4f}")
        
        print("="*60)
    
    def optimize_position_sizes(self, df: pd.DataFrame, entry_signals: pd.Series, 
                              target_volatility: float = 0.15) -> pd.DataFrame:
        """
        Volatilite hedefine göre pozisyon boyutlarını optimize eder.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            entry_signals (pd.Series): Giriş sinyalleri
            target_volatility (float): Hedef volatilite
            
        Returns:
            pd.DataFrame: Optimize edilmiş pozisyon boyutları
        """
        result_df = df.copy()
        result_df['Optimized_Position_Size'] = 0.0
        
        # Getiri hesapla
        returns = df['close'].pct_change().dropna()
        current_volatility = returns.std() * np.sqrt(252)
        
        # Volatilite oranı
        volatility_ratio = target_volatility / current_volatility if current_volatility > 0 else 1
        
        # Pozisyon boyutlarını ayarla
        entry_points = entry_signals[entry_signals == 1].index
        
        for idx in entry_points:
            entry_price = df.loc[idx, 'close']
            stop_loss = self.calculate_stop_loss(df.loc[idx:idx+1]).iloc[0]
            
            # Temel pozisyon boyutu
            base_position_size = self.calculate_position_size(entry_price, stop_loss)
            
            # Volatiliteye göre optimize et
            optimized_size = base_position_size * volatility_ratio
            
            result_df.loc[idx, 'Optimized_Position_Size'] = optimized_size
        
        return result_df

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
        'volume': np.random.uniform(1000, 5000, 100)
    }, index=dates)
    
    # Risk yöneticisi
    risk_manager = RiskManager(initial_capital=10000, max_risk_per_trade=0.02)
    
    # Stop-loss ve hedef fiyat hesapla
    test_data['Stop_Loss'] = risk_manager.calculate_stop_loss(test_data, method='atr')
    test_data['Target_Price'] = risk_manager.calculate_target_price(test_data, method='risk_reward')
    
    # Simüle edilmiş giriş sinyalleri
    entry_signals = pd.Series(0, index=test_data.index)
    entry_signals.iloc[::20] = 1  # Her 20 periyotta bir giriş
    
    # Pozisyon boyutlandırma
    result_df = risk_manager.apply_position_sizing(test_data, entry_signals)
    
    # Risk raporu
    risk_report = risk_manager.generate_risk_report(test_data)
    risk_manager.print_risk_summary(risk_report)
    
    print("\nPozisyon Boyutlandırma Örneği:")
    print(result_df[['close', 'Stop_Loss', 'Target_Price', 'Position_Size', 'Risk_Amount']].head(10)) 