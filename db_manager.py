import sqlite3
from datetime import datetime

DB_PATH = "fir_database.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS fir_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        fir_no TEXT,
        district TEXT,
        ps TEXT,
        year TEXT,
        date_now TEXT,
        time_now TEXT,

        predicted_category TEXT,
        predicted_sections TEXT,

        occurrence_day TEXT,
        date_from TEXT,
        date_to TEXT,
        time_period TEXT,
        time_from TEXT,
        time_to TEXT,

        info_received_date TEXT,
        info_received_time TEXT,
        gd_entry_no TEXT,
        type_of_info TEXT,

        direction_distance TEXT,
        beat_no TEXT,
        address TEXT,

        comp_name TEXT,
        comp_father TEXT,
        comp_dob TEXT,
        comp_nationality TEXT,
        comp_address TEXT,

        accused_details TEXT,
        weapon_used TEXT,
        properties_stolen TEXT,

        action_taken TEXT,
        court_dispatch_date TEXT,
        io_name TEXT,

        complaint_narrative TEXT,
        pdf_path TEXT,

        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()

def insert_fir(data):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO fir_records (
        fir_no, district, ps, year, date_now, time_now,
        predicted_category, predicted_sections,
        occurrence_day, date_from, date_to, time_period,
        time_from, time_to,
        info_received_date, info_received_time,
        gd_entry_no, type_of_info,
        direction_distance, beat_no, address,
        comp_name, comp_father, comp_dob,
        comp_nationality, comp_address,
        accused_details, weapon_used, properties_stolen,
        action_taken, court_dispatch_date, io_name,
        complaint_narrative, pdf_path, created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("fir_no"),
        data.get("district"),
        data.get("ps"),
        data.get("year"),
        data.get("date_now"),
        data.get("time_now"),

        data.get("predicted_category"),
        data.get("predicted_sections"),

        data.get("occurrence_day"),
        data.get("date_from"),
        data.get("date_to"),
        data.get("time_period"),
        data.get("time_from"),
        data.get("time_to"),

        data.get("info_received_date"),
        data.get("info_received_time"),
        data.get("gd_entry_no"),
        data.get("type_of_info"),

        data.get("direction_distance"),
        data.get("beat_no"),
        data.get("address"),

        data.get("comp_name"),
        data.get("comp_father"),
        data.get("comp_dob"),
        data.get("comp_nationality"),
        data.get("comp_address"),

        data.get("accused_details"),
        data.get("weapon_used"),
        data.get("properties_stolen"),

        data.get("action_taken"),
        data.get("court_dispatch_date"),
        data.get("io_name"),

        data.get("complaint_narrative"),
        data.get("pdf_path"),

        datetime.now().isoformat()
    ))

    
    conn.commit()
    conn.close()

def get_next_fir_number(ps, year):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT fir_no FROM fir_records
        WHERE ps = ? AND year = ?
        ORDER BY id DESC
    """, (ps, year))

    rows = cur.fetchall()
    conn.close()

    last_no = 0

    for row in rows:
        fir_no = row[0]

        if fir_no and "/" in fir_no:
            try:
                number = int(fir_no.split("/")[0])
                last_no = number
                break
            except ValueError:
                continue

    next_no = last_no + 1

    return f"{next_no:04d}/{year}"
