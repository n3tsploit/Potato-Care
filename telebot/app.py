import telegram
from telegram.ext import *
from telegram import *
from telebot.functions import predict
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path("./telebot/.env"))
TOKEN = os.getenv('TOKEN')


def start_command(update, context):
    context.bot.sendMessage(text='welcome😊\n  Just send a picture of a potato leaf and let our AI do the rest😉.', chat_id=update.effective_chat.id)


def help_command(update, context):
    context.bot.sendMessage(text='Just send a picture of a potato leaf and let our AI do the rest😉.', chat_id=update.effective_chat.id)


def messages(update, context):
    context.bot.sendMessage(text='Sorry I cannot understand your message.\n Just send a picture of a potato leaf and '
                                 'let our AI do the rest😉. ',
                            chat_id=update.effective_chat.id)


def photo_handler(update, context):
    file = update.message.document.file_id
    print('downloading')
    obj = context.bot.get_file(file)
    file = obj.download('image.jpg')
    response = predict(file)
    print(file)
    update.message.reply_text(f'{response[0]}', parse_mode=telegram.ParseMode.HTML)
    update.message.reply_text(f'{response[1]}')


def main():
    updater = Updater(TOKEN, use_context=True)
    disp = updater.dispatcher

    disp.add_handler(CommandHandler('help', start_command))
    disp.add_handler(CommandHandler('start', start_command))

    disp.add_handler(MessageHandler(Filters.text, messages))
    disp.add_handler(MessageHandler(Filters.document, photo_handler))

    updater.start_polling()
    print('running')
    updater.idle()
