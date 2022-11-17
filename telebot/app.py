import telegram
from telegram.ext import *
from telegram import *
from telebot import functions
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path("./telebot/.env"))
TOKEN = os.getenv('TOKEN')


def start_command():
    print('hello')


def messages():
    print('me')


def main():
    updater = Updater(TOKEN, use_context=True)
    disp = updater.dispatcher

    disp.add_handler(CommandHandler('help', start_command))
    disp.add_handler(CommandHandler('start', start_command))
    disp.add_handler(CommandHandler('check', start_command))

    disp.add_handler(MessageHandler(Filters.text, messages))

    updater.start_polling()
