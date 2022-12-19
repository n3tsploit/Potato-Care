import os, re


def image_check():
    png = re.compile(r"\.png", re.IGNORECASE)
    jpg = re.compile(r"\.jpg", re.IGNORECASE)
    files = os.listdir('../')

    for file in files:
        if '.jpg' in file:
            return file
        elif '.png' in file:
            return file



