
import sqlite3
from pathlib import Path


DB_PATH = Path("plantai_history.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_table():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            image_name TEXT NOT NULL,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(date, image_name, disease, confidence):
    conn = get_connection()

    conn.execute("""
        INSERT INTO predictions
        (date, image_name, disease, confidence)
        VALUES (?, ?, ?, ?)
    """, (date, image_name, disease, confidence))

    conn.commit()
    conn.close()


def get_all_predictions():
    conn = get_connection()

    cursor = conn.execute("""
        SELECT id, date, image_name, disease, confidence
        FROM predictions
        ORDER BY id DESC
    """)

    records = cursor.fetchall()
    conn.close()

    return records


def clear_predictions():
    conn = get_connection()

    conn.execute("DELETE FROM predictions")

    conn.commit()
    conn.close()
