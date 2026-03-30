import re
import spacy

from datetime import datetime, timedelta

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("❌ spaCy model missing. Run: python -m spacy download en_core_web_sm")
    nlp = None

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------

def extract_day(text):

    text_lower = text.lower()

    days = [
        "monday","tuesday","wednesday","thursday",
        "friday","saturday","sunday"
    ]

    # Direct weekday detection
    for d in days:
        if d in text_lower:
            return d.capitalize()

    # Handle relative words
    today = datetime.now()

    if "yesterday" in text_lower:
        return (today - timedelta(days=1)).strftime("%A")

    if "today" in text_lower:
        return today.strftime("%A")

    if "last night" in text_lower:
        return (today - timedelta(days=1)).strftime("%A")

    if "this morning" in text_lower:
        return today.strftime("%A")

    return ""


def extract_dates(text):

    pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"

    dates = re.findall(pattern, text)

    if len(dates) >= 2:
        return dates[0], dates[1]

    if len(dates) == 1:
        return dates[0], ""

    return "", ""


def extract_times(text):

    # Only detect valid time formats
    pattern = r"\b\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)?\b|\b\d{1,2}\s?(?:AM|PM|am|pm)\b"

    matches = re.findall(pattern, text)

    times = [t.strip() for t in matches]

    if len(times) >= 2:
        return times[0], times[1]

    if len(times) == 1:
        return times[0], ""

    return "", ""


def extract_phone(text):

    m = re.search(r"\b[6-9]\d{9}\b", text)

    return m.group() if m else ""


def extract_vehicle(text):

    m = re.search(r"[A-Z]{2}[-\s]?\d{1,2}[-\s]?[A-Z]{1,2}[-\s]?\d{4}", text)

    return m.group() if m else ""


def extract_weapon(text):

    weapons = [
        "knife","rod","stick","iron rod",
        "gun","pistol","blade","stone","screwdriver"
    ]

    text_lower = text.lower()

    for w in weapons:
        if w in text_lower:
            return w

    return ""


def extract_property(text):

    items = [
        "mobile","phone","cash","money","gold",
        "chain","bike","motorcycle","car",
        "laptop","watch","wallet","bag"
    ]

    text_lower = text.lower()

    found = []

    for item in items:
        if item in text_lower:
            found.append(item)

    return ", ".join(found)


# -------------------------------------------------
# PERSON ROLE DETECTION
# -------------------------------------------------

def extract_people_roles(text, persons):

    comp = ""
    accused = []

    lower = text.lower()

    for p in persons:

        if f"my name is {p.lower()}" in lower or f"i am {p.lower()}" in lower:
            comp = p

        elif re.search(rf"i am {p.lower()}", lower):
            comp = p

        elif re.search(rf"by {p.lower()}", lower):
            accused.append(p)

        elif re.search(rf"person named {p.lower()}", lower):
            accused.append(p)

    if not comp and persons:
        comp = persons[0]

    if not accused and len(persons) > 1:
        accused = persons[1:]

    return comp, ", ".join(accused)


def format_fields(data):

    formatted = {}

    for key, value in data.items():

        if not value:
            formatted[key] = value
            continue

        # convert to string
        val = str(value)

        # keep vehicle numbers and phone numbers unchanged
        if key in ["vehicle_number", "phone_number"]:
            formatted[key] = val
            continue

        # convert to title case
        formatted[key] = val.title()

    return formatted

# -------------------------------------------------
# MAIN AUTO EXTRACTION
# -------------------------------------------------

def auto_extract_fields(complaint: str) -> dict:

    doc = nlp(complaint) if nlp else None

    data = {}

    persons = []
    locations = []

    # -------------------------
    # ENTITY EXTRACTION
    # -------------------------

    if doc:
        for ent in doc.ents:

            if ent.label_ == "PERSON":
                persons.append(ent.text)

            elif ent.label_ in ["GPE", "LOC", "FAC"]:
                locations.append(ent.text)
    # -------------------------
    # PEOPLE ROLES
    # -------------------------

    comp, accused = extract_people_roles(complaint, persons)

    if comp:
        data["comp_name"] = comp

    if accused:
        data["accused_details"] = accused

    # -------------------------
    # LOCATION
    # -------------------------

    address = extract_address(complaint, locations)

    if address:
        data["address"] = address

    # -------------------------
    # DATE
    # -------------------------

    date_from, date_to = extract_dates(complaint)

    if date_from:
        data["date_from"] = date_from

    if date_to:
        data["date_to"] = date_to

    # -------------------------
    # TIME
    # -------------------------

    time_from, time_to = extract_times(complaint)

    if time_from:
        data["time_from"] = time_from

    if time_to:
        data["time_to"] = time_to

    # -------------------------
    # DAY
    # -------------------------

    day = extract_day(complaint)

    if day:
        data["occurrence_day"] = day

    # -------------------------
    # WEAPON
    # -------------------------

    weapon = extract_weapon(complaint)

    if weapon:
        data["weapon_used"] = weapon

    # -------------------------
    # PROPERTY
    # -------------------------

    prop = extract_property(complaint)

    if prop:
        data["properties_stolen"] = prop

    # -------------------------
    # PHONE
    # -------------------------

    phone = extract_phone(complaint)

    if phone:
        data["phone_number"] = phone

    # -------------------------
    # VEHICLE
    # -------------------------

    vehicle = extract_vehicle(complaint)

    if vehicle:
        data["vehicle_number"] = vehicle

    return format_fields(data)

def extract_address(text, locations):

    text_lower = text.lower()

    # Remove intro phrases that describe the complainant
    intro_phrases = ["my name is", "i am", "i live in", "i reside at"]

    for phrase in intro_phrases:
        if phrase in text_lower:
            text_lower = text_lower.split(phrase)[-1]

    # Look for crime location keywords
    location_patterns = [
        r"near\s+([A-Za-z0-9\s]+)",
        r"at\s+([A-Za-z0-9\s]+)",
        r"in\s+([A-Za-z0-9\s]+)",
        r"opposite\s+([A-Za-z0-9\s]+)",
        r"behind\s+([A-Za-z0-9\s]+)"
    ]

    for pattern in location_patterns:

        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1).strip().title()

    # fallback to spaCy detected locations
    if locations:
        return ", ".join(dict.fromkeys(locations))

    return ""