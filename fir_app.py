import torch
import sys
from deep_translator import GoogleTranslator
sys.stdout.reconfigure(encoding='utf-8')
import joblib
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

import warnings
warnings.filterwarnings("ignore")

import torch.nn.functional as F
from datetime import datetime
from jinja2 import Template
import subprocess
import os
import sys

# ======================================================
# TRANSLATION MODULES
# ======================================================
from translator import translate_to_tamil
from translate_ta_en import tamil_to_english


# ======================================================
# OTHER MODULES
# ======================================================

from audio_input import speech_to_text
from tamil_input import tamil_text_input


from transformers import AutoTokenizer
from langdetect import detect

from db_manager import init_db, insert_fir, get_next_fir_number
from auto_fill import auto_extract_fields
from model_definition import HybridInLegal, INLEGAL


# ===== ADD THIS RIGHT BELOW IMPORTS =====
def to_tamil(text):
    if not text:
        return ""
    
    text = str(text)

    # skip if already Tamil
    if any('\u0B80' <= ch <= '\u0BFF' for ch in text):
        return text

    try:
        return GoogleTranslator(source='en', target='ta').translate(text)
    except:
        return text
# ======================================================
# INITIAL SETUP
# ======================================================

def initialize():
    """Initialize database and device"""
    init_db()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Using device: {device.upper()}")
    return device

DEVICE = initialize()

# ======================================================
# LOAD TOKENIZER & ENCODERS
# ======================================================

try:

    tokenizer = AutoTokenizer.from_pretrained(INLEGAL, model_max_length=512)

    if not hasattr(tokenizer, "_pad_token"):
        tokenizer._pad_token = "[PAD]"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "[PAD]"

    le_cat = joblib.load("le_cat.pkl")
    le_section = joblib.load("le_section.pkl")

except Exception as e:
    print("❌ Error loading tokenizer or encoders:", e)
    sys.exit(1)

# ======================================================
# LOAD MODEL
# ======================================================

try:

    model = HybridInLegal(transformer_name=INLEGAL)


    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

except Exception as e:
    print("❌ Error loading model:", e)
    sys.exit(1)

# ======================================================
# INPUT SECTION
# ======================================================

print("\n=========================================")
print("   AI-POWERED FIR GENERATION ASSISTANT   ")
print("=========================================\n")

print("\nChoose Complaint Input Mode")
print("1 - Text Complaint")
print("2 - Audio File (MP3/WAV)")

mode = input("Enter choice: ").strip()

# ======================================================
# AUDIO INPUT
# ======================================================

if mode == "2":
    fir_type = "Oral"
    audio_path = input("Enter path of audio file: ").strip()

    if not os.path.exists(audio_path):
        print("Audio file not found.")
        sys.exit(1)

    complaint_text = speech_to_text(audio_path)

    print("\nTranscribed Complaint:")
    print(complaint_text)


# ======================================================
# TEXT INPUT (SMART SWITCH)
# ======================================================

else:
    fir_type = "Written"

    print("\nSelect Input Language")
    print("1 - English (Terminal)")
    print("2 - Tamil (Popup Window)")

    lang_choice = input("Enter choice: ").strip()

    # -------------------------
    # ENGLISH INPUT (Terminal)
    # -------------------------
    if lang_choice == "1":

        print("\nEnter Complaint (Press ENTER twice to finish):\n")

        lines = []
        blank_count = 0

        while True:
            line = input()

            if line.strip() == "":
                blank_count += 1
                if blank_count == 2:
                    break
            else:
                blank_count = 0

            lines.append(line)

        complaint_text = "\n".join(lines).strip()

    # -------------------------
    # TAMIL INPUT (Popup)
    # -------------------------
    elif lang_choice == "2":

        print("\nOpening Tamil input window...\n")

        complaint_text = tamil_text_input()

    else:
        print("❌ Invalid choice")
        sys.exit(1)

    # -------------------------
    # VALIDATION
    # -------------------------
    if not complaint_text or not complaint_text.strip():
        print("❌ Complaint cannot be empty.")
        sys.exit(1)

    print("\nComplaint Entered:\n")
    print(complaint_text)

    # 🔍 DEBUG START
    print("DEBUG TYPE:", type(complaint_text))
    print("DEBUG VALUE:", complaint_text)
# 🔍 DEBUG END
# ======================================================
# LANGUAGE DETECTION
# ======================================================

if not complaint_text:
    print("❌ Complaint cannot be empty.")
    sys.exit(1)

# Detect Tamil using Unicode range
if any('\u0B80' <= c <= '\u0BFF' for c in complaint_text):
    language = "ta"
else:
    language = "en"

print("Detected language:", language)
# ======================================================
# TAMIL → ENGLISH
# ======================================================

if language == "ta":

    print("\nTamil complaint detected.")
    print("Translating complaint to English...\n")

    complaint_english = tamil_to_english(complaint_text)

    print("------ ENGLISH TRANSLATION ------\n")
    print(complaint_english)

    input("\nPress ENTER to continue with AI prediction...")

else:

    complaint_english = complaint_text

print("\nUsing original complaint (no rewriting)...\n")
print(complaint_english)
input("\nPress ENTER to continue...")

# ======================================================
# AI PREDICTION
# ======================================================

enc = tokenizer(
    complaint_english,
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt"
)

enc = {k: v.to(DEVICE) for k, v in enc.items()}

with torch.no_grad():
    logits_cat, logits_sec = model(**enc)

fir = {}

# ======================================================
# CATEGORY PREDICTION
# ======================================================

cat_id = torch.argmax(logits_cat, dim=1).item()
fir["predicted_category"] = le_cat.inverse_transform([cat_id])[0]

# ======================================================
# SECTION PREDICTION
# ======================================================

top_ids = torch.topk(logits_sec, 5).indices[0].tolist()
fir["predicted_sections"] = ", ".join(
    le_section.inverse_transform(top_ids)
)

print("Predicted BNS Sections:", fir["predicted_sections"])

input("\nPress ENTER to continue...")

# ======================================================
# AUTO FILL FROM COMPLAINT
# ======================================================


auto_data = auto_extract_fields(complaint_english)
print("\n--- AUTO-EXTRACTED DATA ---")
print(auto_data)

for key, value in auto_data.items():
    if value:
        fir[key] = value

print("\n--- AUTO-FILLED DATA FROM COMPLAINT ---")

for k, v in fir.items():
    print(f"{k}: {v}")

# ======================================================
# ASK MISSING FIELDS
# ======================================================

HTML_FIELDS = {    
    "occurrence_day": "Occurrence Day",
    "date_from": "Date From",
    "date_to": "Date To",
    "time_from": "Time From",
    "time_to": "Time To",
    "time_period": "Time Period",

    "info_received_date": "Info Received Date",
    "info_received_time": "Info Received Time",
    "gd_entry_no": "GD Entry No",
    "GD_entry_time": "GD Entry Time",

    "distance_ps": "Distance from P.S",
    "direction_ps": "Direction from P.S",
    "beat_no": "Beat No",
    "address": "Address",

    "comp_name": "Complainant Name",
    "comp_father": "Father's Name",
    "comp_dob": "Age / DOB",
    "comp_nationality": "Nationality",
    "comp_address": "Complainant Address",

    "accused_details": "Accused Details",
    "weapon_used": "Weapon Used",
    "properties_stolen": "Properties Stolen",
    "total_property_value": "Total Value of Properties Stolen",

    "delay_reason": "Reasons for Delay in Reporting",
    "inquest_case_no": "Inquest / U.D. Case No.",

    "io_name": "Investigating Officer Name",
    "action_taken": "Action Taken",
    "court_dispatch_date": "Court Dispatch Date/Time"
}

print("\n--- ENTER ONLY MISSING FIR DETAILS ---")

for key, label in HTML_FIELDS.items():
    if not fir.get(key):
        fir[key] = input(f"{label}: ").strip()

# ======================================================
# SYSTEM AUTO FIELDS
# ======================================================

now = datetime.now()
# ======================================================
# FIXED SYSTEM FIELDS
# ======================================================

fir["district"] = "Salem"
fir["ps"] = "Sankari"
fir["type_of_info"] = fir_type

fir["year"] = now.year
fir["fir_no"] = get_next_fir_number(fir["ps"], fir["year"])
fir["date_now"] = now.strftime("%d/%m/%Y")
fir["time_now"] = now.strftime("%H:%M")

fir["original_complaint"] = complaint_text
fir["complaint_narrative"] = complaint_english
print("FIR KEYS:", fir.keys())
# ======================================================
# FINAL TAMIL TRANSLATION BLOCK (CLEAN & STABLE)
# ======================================================

# 1. FIR CONTENT (MAIN)
if len(complaint_english) < 300:
    try:
        fir["complaint_narrative_ta"] = translate_to_tamil(complaint_english)
    except:
        fir["complaint_narrative_ta"] = GoogleTranslator(source='en', target='ta').translate(complaint_english)
else:
    fir["complaint_narrative_ta"] = GoogleTranslator(source='en', target='ta').translate(complaint_english)
# ===== ADD ONLY THESE MISSING TAMIL FIELDS =====

fir["occurrence_day_ta"] = to_tamil(fir.get("occurrence_day"))
fir["distance_ps_ta"] = to_tamil(fir.get("distance_ps"))
fir["direction_ps_ta"] = to_tamil(fir.get("direction_ps"))
fir["year_ta"] = to_tamil(fir.get("year"))
fir["fir_no_ta"] = to_tamil(fir.get("fir_no"))
fir["io_name_ta"] = to_tamil(fir.get("io_name"))
# ===== ADD THIS EXACTLY BELOW complaint_narrative_ta =====


# Translate all fields
fir["comp_name_ta"] = to_tamil(fir.get("comp_name"))
fir["comp_father_ta"] = to_tamil(fir.get("comp_father"))
fir["comp_address_ta"] = to_tamil(fir.get("comp_address"))
fir["comp_nationality_ta"] = to_tamil(fir.get("comp_nationality"))

fir["accused_details_ta"] = to_tamil(fir.get("accused_details"))
fir["address_ta"] = to_tamil(fir.get("address"))

fir["type_of_info_ta"] = to_tamil(fir.get("type_of_info"))
fir["action_taken_ta"] = to_tamil(fir.get("action_taken"))

fir["properties_stolen_ta"] = to_tamil(fir.get("properties_stolen"))
fir["delay_reason_ta"] = to_tamil(fir.get("delay_reason"))
# 2. SIMPLE TEXT FIELDS
simple_fields = [
    "name",
    "father_name",
    "nationality",
    "district",
    "ps",
    "accused",
    "properties_stolen",
    "delay_reason"
]

for field in simple_fields:
    value = fir.get(field, "")
    if value:
        try:
            fir[field + "_ta"] = translate_to_tamil(str(value))
        except:
            fir[field + "_ta"] = value

# 3. ADDRESS (GOOGLE ONLY - SAFE)
address_val = fir.get("address", "")

if address_val.strip():
    try:
        fir["address_ta"] = GoogleTranslator(source='en', target='ta').translate(address_val)
    except:
        fir["address_ta"] = address_val
else:
    fir["address_ta"] = "முகவரி இல்லை"

if len(complaint_english) < 300:
    try:
        ta_text = translate_to_tamil(complaint_english)
    except:
        ta_text = GoogleTranslator(source='en', target='ta').translate(complaint_english)
else:
    ta_text = GoogleTranslator(source='en', target='ta').translate(complaint_english)
# remove repeated words (SAFE - no number damage)





# 2. Simple fields (safe translation)
simple_fields = [
    "comp_name",
    "comp_father",
    "district",
    "ps",
    "accused_details",
    "properties_stolen",
    "delay_reason"
]
# ✅ ADDRESS FIX (USE GOOGLE TRANSLATE - SAFE)

address_val = fir.get("address", "")
if address_val:
    try:
        fir["address_ta"] = GoogleTranslator(source='en', target='ta').translate(address_val)
    except:
        fir["address_ta"] = address_val
for field in simple_fields:
    value = fir.get(field, "")
    if value:
        try:
            fir[field + "_ta"] = translate_to_tamil(str(value))
        except:
            fir[field + "_ta"] = value


# ======================================================
# DEBUG OUTPUT
# ======================================================

print("\nDEBUG FIR DICTIONARY\n")

for k, v in fir.items():
    print(k, "=", v)

# ======================================================
# PDF GENERATION
# ======================================================

print("\n📄 Generating FIR PDF...\n")

output_dir = r"D:\AI_Driven_Legal_Section_Classification_and_F_I_R_Documentation\FIRs_Generated"
os.makedirs(output_dir, exist_ok=True)

timestamp = now.strftime("%Y%m%d_%H%M%S")
safe_fir_no = fir["fir_no"].replace("/", "_")

pdf_file = os.path.join(
    output_dir,
    f"FIR_{safe_fir_no}_{timestamp}.pdf"
)

print("Saving FIR to:", pdf_file)

try:

    with open("fir_template.html", "r", encoding="utf-8") as f:
        template = Template(f.read())

    html_content = template.render(**fir)

    temp_html = os.path.join(output_dir, "_temp_fir.html")

    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)

except Exception as e:
    print("❌ Error generating HTML:", e)
    sys.exit(1)

# ======================================================
# CHROME PDF PRINT
# ======================================================

chrome_paths = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
]

chrome_path = next((p for p in chrome_paths if os.path.exists(p)), None)

if not chrome_path:
    print("❌ Google Chrome not found.")
    sys.exit(1)

command = [
    chrome_path,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_file}",
    f"file:///{temp_html.replace(os.sep,'/')}"
]

result = subprocess.run(command, capture_output=True, text=True)

if os.path.exists(temp_html):
    os.remove(temp_html)

if result.returncode != 0:
    print("❌ PDF generation failed")
    print(result.stderr)
    sys.exit(1)

print("✅ FIR PDF GENERATED SUCCESSFULLY")
print("📂 Saved at:", pdf_file)

# ======================================================
# STORE IN DATABASE
# ======================================================

try:

    fir["pdf_path"] = pdf_file
    insert_fir(fir)

    print("✅ FIR STORED IN DATABASE")

except Exception as e:

    print("❌ Failed to store FIR:", e)