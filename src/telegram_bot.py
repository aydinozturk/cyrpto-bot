"""
Telegram bot entegrasyonu modülü.
Kripto para sinyallerini Telegram üzerinden gönderir.
"""

import asyncio
import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# Telegram bot kütüphanesi
try:
    from telegram import Bot
    from telegram.constants import ParseMode
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("Telegram kütüphanesi yüklü değil. 'pip install python-telegram-bot' komutu ile yükleyin.")

load_dotenv()

class TelegramSignalBot:
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Telegram bot sınıfını başlatır.
        
        Args:
            bot_token (str): Telegram bot token'ı
            chat_id (str): Chat ID (kullanıcı veya grup ID'si)
        """
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.bot = None
        
        if TELEGRAM_AVAILABLE and self.bot_token:
            self.bot = Bot(token=self.bot_token)
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def is_configured(self) -> bool:
        """Bot'un yapılandırılıp yapılandırılmadığını kontrol eder."""
        return bool(self.bot_token and self.chat_id and TELEGRAM_AVAILABLE)
    
    async def send_message(self, message: str, parse_mode: str = ParseMode.HTML) -> bool:
        """
        Telegram'a mesaj gönderir.
        
        Args:
            message (str): Gönderilecek mesaj
            parse_mode (str): Mesaj formatı (HTML, Markdown)
            
        Returns:
            bool: Başarılı ise True
        """
        if not self.is_configured():
            self.logger.warning("Telegram bot yapılandırılmamış!")
            return False
            
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            self.logger.info("Telegram mesajı başarıyla gönderildi")
            return True
        except TelegramError as e:
            self.logger.error(f"Telegram mesajı gönderilemedi: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Beklenmeyen hata: {e}")
            return False
    
    def format_signal_message(self, symbol: str, signal_data: Dict) -> str:
        """
        Sinyal verilerini Telegram mesajı formatında düzenler.
        
        Args:
            symbol (str): Kripto para sembolü (örn: BTC/USDT)
            signal_data (Dict): Sinyal verileri
            
        Returns:
            str: Formatlanmış mesaj
        """
        # Son sinyal verilerini al
        last_signal = signal_data.get('last_signals', [])
        if not last_signal:
            return f"❌ {symbol} için sinyal verisi bulunamadı"
        
        latest = last_signal[-1] if isinstance(last_signal, list) else last_signal
        
        # Emoji seçimi
        direction_emoji = {
            'BUY': '🟢',
            'SELL': '🔴', 
            'HOLD': '🟡'
        }
        
        signal_emoji = direction_emoji.get(latest.get('Signal_Direction', 'HOLD'), '🟡')
        
        # Mesaj oluştur
        message = f"""
<b>📊 {symbol} Sinyal Raporu</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏰ <b>Zaman:</b> {latest.get('timestamp', 'N/A')}
💰 <b>Fiyat:</b> {latest.get('close', 'N/A'):,.2f} USDT
📈 <b>Sinyal:</b> {signal_emoji} {latest.get('Signal_Direction', 'HOLD')}
💪 <b>Güç:</b> {latest.get('Signal_Strength', 0):.2f}

<b>Teknik Göstergeler:</b>
• RSI: {latest.get('RSI', 0):.1f}
• MACD: {latest.get('MACD', 0):.1f}
• BB Pozisyon: {latest.get('BB_Position', 0):.2f}

<b>Son 3 Sinyal:</b>
"""
        
        # Son 3 sinyali ekle
        for i, signal in enumerate(last_signal[-3:], 1):
            time = signal.get('timestamp', 'N/A')
            price = signal.get('close', 0)
            direction = signal.get('Signal_Direction', 'HOLD')
            strength = signal.get('Signal_Strength', 0)
            
            emoji = direction_emoji.get(direction, '🟡')
            message += f"{i}. {emoji} {time} - {price:,.2f} ({direction}, {strength:.2f})\n"
        
        # Özet istatistikler
        summary = signal_data.get('summary', {})
        if summary:
            message += f"""
<b>📈 Özet İstatistikler:</b>
• Toplam Sinyal: {summary.get('total_signals', 0)}
• Alım: {summary.get('buy_signals', 0)} ({summary.get('buy_percentage', 0):.1f}%)
• Satım: {summary.get('sell_signals', 0)} ({summary.get('sell_percentage', 0):.1f}%)
• Bekle: {summary.get('hold_signals', 0)} ({summary.get('hold_percentage', 0):.1f}%)
• Ortalama Güç: {summary.get('avg_strength', 0):.2f}
"""
        
        message += "\n🤖 <i>Kripto Sinyal Bot</i>"
        return message
    
    async def send_signal_alert(self, symbol: str, signal_data: Dict) -> bool:
        """
        Sinyal uyarısı gönderir.
        
        Args:
            symbol (str): Kripto para sembolü
            signal_data (Dict): Sinyal verileri
            
        Returns:
            bool: Başarılı ise True
        """
        message = self.format_signal_message(symbol, signal_data)
        return await self.send_message(message)
    
    async def send_daily_summary(self, all_signals: Dict[str, Dict]) -> bool:
        """
        Günlük özet raporu gönderir.
        
        Args:
            all_signals (Dict): Tüm kripto paraların sinyal verileri
            
        Returns:
            bool: Başarılı ise True
        """
        message = f"""
<b>📊 Günlük Kripto Sinyal Özeti</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 <b>Tarih:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}

"""
        
        for symbol, signal_data in all_signals.items():
            summary = signal_data.get('summary', {})
            last_signal = signal_data.get('last_signals', [])
            
            if last_signal:
                latest = last_signal[-1]
                direction = latest.get('Signal_Direction', 'HOLD')
                price = latest.get('close', 0)
                
                direction_emoji = {
                    'BUY': '🟢',
                    'SELL': '🔴',
                    'HOLD': '🟡'
                }.get(direction, '🟡')
                
                message += f"""
<b>{symbol}</b> {direction_emoji}
💰 {price:,.2f} USDT | {direction}
📊 {summary.get('total_signals', 0)} sinyal | {summary.get('avg_strength', 0):.2f} güç
"""
        
        message += "\n🤖 <i>Kripto Sinyal Bot - Günlük Özet</i>"
        return await self.send_message(message)
    
    async def send_error_alert(self, error_message: str) -> bool:
        """
        Hata uyarısı gönderir.
        
        Args:
            error_message (str): Hata mesajı
            
        Returns:
            bool: Başarılı ise True
        """
        message = f"""
<b>⚠️ Bot Hatası</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ <b>Hata:</b> {error_message}
⏰ <b>Zaman:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔧 Lütfen sistemi kontrol edin.
"""
        return await self.send_message(message)
    
    async def send_startup_message(self) -> bool:
        """
        Bot başlangıç mesajı gönderir.
        
        Returns:
            bool: Başarılı ise True
        """
        message = f"""
<b>🚀 Kripto Sinyal Bot Başlatıldı</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Bot başarıyla çalışıyor
⏰ Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 Sinyal takibi aktif

🤖 <i>Kripto Sinyal Bot</i>
"""
        return await self.send_message(message)

# Test fonksiyonu
async def test_telegram_bot():
    """Telegram bot'unu test eder."""
    bot = TelegramSignalBot()
    
    if not bot.is_configured():
        print("❌ Telegram bot yapılandırılmamış!")
        print("TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID değişkenlerini .env dosyasına ekleyin.")
        return
    
    # Test mesajı gönder
    test_message = """
<b>🧪 Test Mesajı</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Telegram bot başarıyla çalışıyor!
⏰ Test zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🤖 <i>Kripto Sinyal Bot</i>
""".format(datetime=datetime)
    
    success = await bot.send_message(test_message)
    if success:
        print("✅ Test mesajı başarıyla gönderildi!")
    else:
        print("❌ Test mesajı gönderilemedi!")

if __name__ == "__main__":
    # Test çalıştır
    asyncio.run(test_telegram_bot()) 