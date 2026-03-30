import torch
import indic_transliteration.sanscript as sanscript
from indic_transliteration.sanscript import transliterate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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

# -------------------------------------------------
# MODEL CONFIG
# -------------------------------------------------

MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------
# GLOBAL VARIABLES (model loads only once)
# -------------------------------------------------

tokenizer = None
model = None


# -------------------------------------------------
# LOAD MODEL FUNCTION
# -------------------------------------------------

def load_translation_model():
    global tokenizer, model

    if tokenizer is None or model is None:

        print("Loading ....")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_fast=False
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        ).to(DEVICE)

        model.eval()

        print("Translation model ready.")


# -------------------------------------------------
# CORE TRANSLATION FUNCTION
# -------------------------------------------------
def translate_batch(text_list, src_lang, tgt_lang):

    load_translation_model()

    protected_texts = []
    all_placeholders = []

    # 🔒 protect numbers
    for text in text_list:
        t, p = protect_numbers(text)
        protected_texts.append(t)
        all_placeholders.append(p)

    tagged = [f"{src_lang} {tgt_lang} {t}" for t in protected_texts]

    inputs = tokenizer(
        tagged,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)

    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    tamil_results = [
        transliterate(r, sanscript.DEVANAGARI, sanscript.TAMIL)
        for r in results
    ]

    # 🔓 restore numbers
    final_results = [
        restore_numbers(res, placeholders)
        for res, placeholders in zip(tamil_results, all_placeholders)
    ]

    return final_results

def translate_to_tamil(text):
    return translate_batch([text], "eng_Latn", "tam_Taml")[0]


def translate_to_english(text):
    return translate_batch([text], "tam_Taml", "eng_Latn")[0]