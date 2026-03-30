import re
from deep_translator import GoogleTranslator


# 🔒 Protect numbers
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


# 🔓 Restore numbers
def restore_numbers(text, placeholders):
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


# 🧹 Clean noisy input (VERY IMPORTANT)
def clean_text(text):
    # remove weird symbols but keep Tamil + English + numbers
    text = re.sub(r'[^\u0B80-\u0BFFa-zA-Z0-9\s.,:/₹-]', '', text)
    return text.strip()


# 🚀 MAIN FUNCTION
def tamil_to_english(text):

    if not text or not isinstance(text, str):
        return ""

    print("🔄 Translating Tamil → English...")

    try:
        # Step 0: clean text
        text = clean_text(text)

        # Step 1: protect numbers
        text, placeholders = protect_numbers(text)

        # Step 2: translate
        translated = GoogleTranslator(
            source='auto',   # 👈 IMPORTANT CHANGE
            target='en'
        ).translate(text)

        # Step 3: fallback safety
        if not translated:
            return text

        # Step 4: restore numbers
        translated = restore_numbers(translated, placeholders)

        return translated

    except Exception as e:
        print("❌ Translation Error:", str(e))
        return text