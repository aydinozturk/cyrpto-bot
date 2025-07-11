"""
Kripto para sinyalleri ve performans görselleştirme modülü.
Teknik göstergeler, sinyaller ve performans grafikleri oluşturur.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter desteği için
plt.rcParams['font.family'] = 'DejaVu Sans'

class CryptoVisualizer:
    def __init__(self):
        """Görselleştirme sınıfını başlatır."""
        self.colors = {
            'buy': '#00ff88',
            'sell': '#ff4444',
            'hold': '#888888',
            'profit': '#00aa00',
            'loss': '#aa0000',
            'neutral': '#888888'
        }
    
    def plot_price_with_signals(self, df: pd.DataFrame, signals: pd.Series = None, 
                               title: str = "Fiyat ve Sinyaller") -> go.Figure:
        """
        Fiyat grafiği ile sinyalleri gösterir.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            signals (pd.Series): Sinyal serisi
            title (str): Grafik başlığı
            
        Returns:
            go.Figure: Plotly grafik nesnesi
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(title, 'Hacim'),
            row_width=[0.7, 0.3]
        )
        
        # Mum grafiği
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Fiyat",
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )
        
        # Sinyal noktaları
        if signals is not None:
            buy_signals = df[signals == 1]
            sell_signals = df[signals == -1]
            
            if len(buy_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['low'] * 0.99,
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=15, color='#00ff88'),
                        name='Alım Sinyali'
                    ),
                    row=1, col=1
                )
            
            if len(sell_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['high'] * 1.01,
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=15, color='#ff4444'),
                        name='Satım Sinyali'
                    ),
                    row=1, col=1
                )
        
        # Hacim grafiği
        colors = ['#00ff88' if close >= open else '#ff4444' 
                 for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Hacim',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        return fig
    
    def plot_technical_indicators(self, df: pd.DataFrame) -> go.Figure:
        """
        Teknik göstergeleri gösterir.
        
        Args:
            df (pd.DataFrame): Teknik göstergelerle DataFrame
            
        Returns:
            go.Figure: Plotly grafik nesnesi
        """
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Fiyat ve Bollinger Bantları', 'RSI', 'MACD', 'Stochastic'),
            row_width=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Fiyat ve Bollinger Bantları
        fig.add_trace(
            go.Scatter(x=df.index, y=df['close'], name='Fiyat', line=dict(color='#0000ff')),
            row=1, col=1
        )
        
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Üst', 
                          line=dict(color='#888888', dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Alt', 
                          line=dict(color='#888888', dash='dash'), fill='tonexty'),
                row=1, col=1
            )
        
        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#ff8800')),
                row=2, col=1
            )
            # RSI seviyeleri
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
        
        # MACD
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#0000ff')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD_Signal'], name='MACD Sinyal', 
                          line=dict(color='#ff0000')),
                row=3, col=1
            )
            if 'MACD_Histogram' in df.columns:
                fig.add_trace(
                    go.Bar(x=df.index, y=df['MACD_Histogram'], name='MACD Histogram',
                           marker_color='#888888'),
                    row=3, col=1
                )
        
        # Stochastic
        if 'Stoch_K' in df.columns and 'Stoch_D' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['Stoch_K'], name='%K', line=dict(color='#0000ff')),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['Stoch_D'], name='%D', line=dict(color='#ff0000')),
                row=4, col=1
            )
            # Stochastic seviyeleri
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)
        
        fig.update_layout(height=800, title_text="Teknik Göstergeler")
        
        return fig
    
    def plot_signal_analysis(self, df: pd.DataFrame) -> go.Figure:
        """
        Sinyal analizini gösterir.
        
        Args:
            df (pd.DataFrame): Sinyal göstergeleri
            
        Returns:
            go.Figure: Plotly grafik nesnesi
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Birleştirilmiş Sinyal', 'Sinyal Gücü', 'Sinyal Yönü'),
            row_width=[0.4, 0.3, 0.3]
        )
        
        # Birleştirilmiş sinyal
        if 'Combined_Signal' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['Combined_Signal'], name='Birleştirilmiş Sinyal',
                          line=dict(color='#0000ff')),
                row=1, col=1
            )
            # Sinyal seviyeleri
            fig.add_hline(y=0.3, line_dash="dash", line_color="green", row=1, col=1)
            fig.add_hline(y=-0.3, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        # Sinyal gücü
        if 'Signal_Strength' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['Signal_Strength'], name='Sinyal Gücü',
                          line=dict(color='#ff8800')),
                row=2, col=1
            )
        
        # Sinyal yönü
        if 'Signal_Direction' in df.columns:
            # Sinyal yönünü sayısal değerlere dönüştür
            direction_numeric = df['Signal_Direction'].map({'BUY': 1, 'SELL': -1, 'HOLD': 0})
            
            fig.add_trace(
                go.Scatter(x=df.index, y=direction_numeric, name='Sinyal Yönü',
                          line=dict(color='#00aa00'), mode='markers'),
                row=3, col=1
            )
        
        fig.update_layout(height=600, title_text="Sinyal Analizi")
        
        return fig
    
    def plot_performance_metrics(self, returns: pd.Series, equity_curve: pd.Series = None) -> go.Figure:
        """
        Performans metriklerini gösterir.
        
        Args:
            returns (pd.Series): Getiri serisi
            equity_curve (pd.Series): Sermaye eğrisi
            
        Returns:
            go.Figure: Plotly grafik nesnesi
        """
        if equity_curve is None:
            equity_curve = (1 + returns).cumprod()
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Sermaye Eğrisi', 'Getiri Dağılımı', 'Drawdown'),
            row_width=[0.5, 0.25, 0.25]
        )
        
        # Sermaye eğrisi
        fig.add_trace(
            go.Scatter(x=equity_curve.index, y=equity_curve, name='Sermaye Eğrisi',
                      line=dict(color='#0000ff')),
            row=1, col=1
        )
        
        # Getiri dağılımı
        fig.add_trace(
            go.Histogram(x=returns, name='Getiri Dağılımı', nbinsx=30,
                        marker_color='#888888'),
            row=2, col=1
        )
        
        # Drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown, name='Drawdown',
                      line=dict(color='#ff0000'), fill='tonexty'),
            row=3, col=1
        )
        
        fig.update_layout(height=800, title_text="Performans Metrikleri")
        
        return fig
    
    def plot_risk_analysis(self, df: pd.DataFrame, risk_metrics: Dict[str, Any]) -> go.Figure:
        """
        Risk analizini gösterir.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            risk_metrics (Dict[str, Any]): Risk metrikleri
            
        Returns:
            go.Figure: Plotly grafik nesnesi
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Volatilite', 'VaR Analizi', 'Pozisyon Boyutları', 'Risk Dağılımı'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # Volatilite (rolling)
        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        
        fig.add_trace(
            go.Scatter(x=volatility.index, y=volatility, name='Volatilite',
                      line=dict(color='#ff8800')),
            row=1, col=1
        )
        
        # VaR Analizi
        var_95 = []
        for i in range(20, len(returns)):
            window_returns = returns.iloc[i-20:i]
            var_95.append(np.percentile(window_returns, 5))
        
        var_index = returns.index[20:]
        fig.add_trace(
            go.Scatter(x=var_index, y=var_95, name='VaR (95%)',
                      line=dict(color='#ff0000')),
            row=1, col=2
        )
        
        # Pozisyon boyutları
        if 'Position_Size' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['Position_Size'], name='Pozisyon Boyutu',
                          line=dict(color='#00aa00')),
                row=2, col=1
            )
        
        # Risk dağılımı (pie chart)
        if 'portfolio_risk' in risk_metrics and 'position_weights' in risk_metrics['portfolio_risk']:
            weights = risk_metrics['portfolio_risk']['position_weights']
            if weights:
                fig.add_trace(
                    go.Pie(labels=list(weights.keys()), values=list(weights.values()),
                           name='Pozisyon Ağırlıkları'),
                    row=2, col=2
                )
        
        fig.update_layout(height=600, title_text="Risk Analizi")
        
        return fig
    
    def create_dashboard(self, df: pd.DataFrame, signals: pd.DataFrame) -> go.Figure:
        """
        Kapsamlı dashboard oluşturur.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            signals (pd.DataFrame): Sinyal göstergeleri
            
        Returns:
            go.Figure: Dashboard grafik
        """
        # Ana dashboard
        fig = make_subplots(
            rows=4, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.05,
            horizontal_spacing=0.05,
            subplot_titles=(
                'Fiyat ve Sinyaller', 'Teknik Göstergeler',
                'RSI', 'MACD',
                'Sinyal Analizi', 'Hacim',
                'Performans', 'Risk Metrikleri'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Fiyat ve sinyaller
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Fiyat"
            ),
            row=1, col=1
        )
        
        # RSI
        if 'RSI' in signals.columns:
            fig.add_trace(
                go.Scatter(x=signals.index, y=signals['RSI'], name='RSI',
                          line=dict(color='#ff8800')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'MACD' in signals.columns and 'MACD_Signal' in signals.columns:
            fig.add_trace(
                go.Scatter(x=signals.index, y=signals['MACD'], name='MACD',
                          line=dict(color='#0000ff')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=signals.index, y=signals['MACD_Signal'], name='MACD Sinyal',
                          line=dict(color='#ff0000')),
                row=2, col=2
            )
        
        # Sinyal analizi
        if 'Combined_Signal' in signals.columns:
            fig.add_trace(
                go.Scatter(x=signals.index, y=signals['Combined_Signal'], name='Birleştirilmiş Sinyal',
                          line=dict(color='#0000ff')),
                row=3, col=1
            )
        
        # Hacim
        colors = ['#00ff88' if close >= open else '#ff4444' 
                 for close, open in zip(df['close'], df['open'])]
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='Hacim',
                   marker_color=colors),
            row=3, col=2
        )
        
        # Performans
        returns = df['close'].pct_change().dropna()
        equity_curve = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(x=equity_curve.index, y=equity_curve, name='Sermaye Eğrisi',
                      line=dict(color='#00aa00')),
            row=4, col=1
        )
        
        # Risk metrikleri
        volatility = returns.rolling(20).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=volatility.index, y=volatility, name='Volatilite',
                      line=dict(color='#ff8800')),
            row=4, col=2
        )
        
        fig.update_layout(
            title="Kripto Para Alım-Satım Dashboard",
            height=1200,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def save_plots(self, figures: Dict[str, go.Figure], filename_prefix: str = "crypto_analysis"):
        """
        Grafikleri HTML dosyalarına kaydeder.
        
        Args:
            figures (Dict[str, go.Figure]): Grafik sözlüğü
            filename_prefix (str): Dosya adı öneki
        """
        for name, fig in figures.items():
            filename = f"data/{filename_prefix}_{name}.html"
            fig.write_html(filename)
            print(f"{name} grafiği {filename} olarak kaydedildi")
    
    def print_summary_stats(self, df: pd.DataFrame, signals: pd.DataFrame):
        """
        Özet istatistikleri yazdırır.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            signals (pd.DataFrame): Sinyal göstergeleri
        """
        print("\n" + "="*60)
        print("GÖRSELLEŞTİRME ÖZETİ")
        print("="*60)
        
        # Fiyat istatistikleri
        print(f"Veri Periyodu: {df.index[0]} - {df.index[-1]}")
        print(f"Toplam Veri Noktası: {len(df)}")
        print(f"Başlangıç Fiyatı: ${df['close'].iloc[0]:,.2f}")
        print(f"Bitiş Fiyatı: ${df['close'].iloc[-1]:,.2f}")
        print(f"Toplam Getiri: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
        
        # Sinyal istatistikleri
        if 'Signal_Direction' in signals.columns:
            signal_counts = signals['Signal_Direction'].value_counts()
            print(f"\nSinyal Dağılımı:")
            for direction, count in signal_counts.items():
                print(f"  {direction}: {count} ({count/len(signals)*100:.1f}%)")
        
        # Teknik gösterge istatistikleri
        if 'RSI' in signals.columns:
            print(f"\nRSI Ortalama: {signals['RSI'].mean():.2f}")
            print(f"RSI Std: {signals['RSI'].std():.2f}")
        
        if 'MACD' in signals.columns:
            print(f"MACD Ortalama: {signals['MACD'].mean():.4f}")
        
        print("="*60)

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
        'Combined_Signal': np.random.uniform(-1, 1, 100),
        'Signal_Direction': np.random.choice(['BUY', 'SELL', 'HOLD'], 100)
    }, index=dates)
    
    # Görselleştirici
    visualizer = CryptoVisualizer()
    
    # Grafikler oluştur
    price_fig = visualizer.plot_price_with_signals(test_data)
    indicators_fig = visualizer.plot_technical_indicators(test_data)
    signals_fig = visualizer.plot_signal_analysis(test_data)
    
    # Dashboard
    dashboard_fig = visualizer.create_dashboard(test_data, test_data)
    
    # Grafikleri kaydet
    figures = {
        'price': price_fig,
        'indicators': indicators_fig,
        'signals': signals_fig,
        'dashboard': dashboard_fig
    }
    
    visualizer.save_plots(figures)
    visualizer.print_summary_stats(test_data, test_data) 