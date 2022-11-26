import telegram
from telegram.ext import *
from telegram import *
from telebot import functions
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path("./telebot/.env"))
TOKEN = os.getenv('TOKEN')


def start_command(update, context):
    context.bot.sendMessage(text='Bye👋', chat_id=update.effective_chat.id)


def messages(update, context):
    context.bot.sendMessage(text='Bye👋', chat_id=update.effective_chat.id)


def photo_handler(update, context):
    file = update.message.photo[0].file_id
    print('downloading')
    obj = context.bot.get_file(file)
    obj.download()


def main():
    updater = Updater(TOKEN, use_context=True)
    disp = updater.dispatcher

    disp.add_handler(CommandHandler('help', start_command))
    disp.add_handler(CommandHandler('start', start_command))
    disp.add_handler(CommandHandler('check', start_command))

    # disp.add_handler(MessageHandler(Filters.text, messages))
    disp.add_handler(MessageHandler(Filters.photo, photo_handler))

    updater.start_polling()
    print('running')
    updater.idle()

