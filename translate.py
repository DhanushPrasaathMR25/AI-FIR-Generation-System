from deep_translator import GoogleTranslator
from langdetect import detect
import re

def protect_numbers(text):
    patterns = {
        r'₹\s?\d+(?:,\d+)*': 'MONEY',
        r'\d{1,2}:\d{2}': 'TIME',
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}': 'DATE',
        r'\d+(?:,\d+)*': 'NUMBER'
    }

    placeholders = {}
    counter = 0

    for pattern, label in patterns.items():
        matches = re.findall(pattern, text)
        for match in matches:
            key = f"<{label}_{counter}>"
            placeholders[key] = match
            text = text.replace(match, key, 1)
            counter += 1

    return text, placeholders


def restore_numbers(text, placeholders):
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text
def convert_to_english(text):

    try:
        text, placeholders = protect_numbers(text)

        lang = detect(text)

        if lang != "en":
            text = GoogleTranslator(source='auto', target='en').translate(text)

        text = restore_numbers(text, placeholders)

    except:
        pass

    return text