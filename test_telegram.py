#!/usr/bin/env python3
"""
Telegram bot test dosyası
"""

import asyncio
import os
from dotenv import load_dotenv
from src.telegram_bot import TelegramSignalBot

load_dotenv()

async def test_telegram():
    """Telegram bot'unu test eder."""
    print("🧪 Telegram Bot Testi Başlatılıyor...")
    
    # Bot'u başlat
    bot = TelegramSignalBot()
    
    if not bot.is_configured():
        print("❌ Telegram bot yapılandırılmamış!")
        print("\n📋 Kurulum Adımları:")
        print("1. @BotFather ile bot oluşturun")
        print("2. Bot token'ını alın")
        print("3. Bot ile konuşmaya başlayın")
        print("4. Chat ID'nizi alın")
        print("5. .env dosyasına ekleyin:")
        print("   TELEGRAM_BOT_TOKEN=your_token")
        print("   TELEGRAM_CHAT_ID=your_chat_id")
        return
    
    print("✅ Bot yapılandırılmış!")
    
    # Test mesajı gönder
    test_message = """
<b>🧪 Test Mesajı</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Telegram bot başarıyla çalışıyor!
⏰ Test zamanı: {}

📊 Kripto sinyalleri yakında gelecek...

🤖 <i>Kripto Sinyal Bot</i>
""".format(asyncio.get_event_loop().time())
    
    print("📤 Test mesajı gönderiliyor...")
    success = await bot.send_message(test_message)
    
    if success:
        print("✅ Test mesajı başarıyla gönderildi!")
        print("📱 Telegram'ınızı kontrol edin.")
    else:
        print("❌ Test mesajı gönderilemedi!")
        print("🔧 Bot token ve chat ID'yi kontrol edin.")

if __name__ == "__main__":
    asyncio.run(test_telegram()) 