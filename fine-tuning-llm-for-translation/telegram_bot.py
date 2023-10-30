import telepot
import time
import csv

BOT_TOKEN = '6650593986:AAFohMgNrNPqehjGLF_nOWmPoYdegACk-t8'
# CHANNEL_ID = -1004073044929
CHANNEL_ID = -1004091103791

def handle(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)

    if content_type == 'text' and chat_id == CHANNEL_ID:
        message_text = msg['text']
        with open('telegram_messages.csv', 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([time.ctime(), message_text])

bot = telepot.Bot(BOT_TOKEN)
bot.message_loop(handle)