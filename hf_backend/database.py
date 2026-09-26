import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DB_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "supplychainiq.db"
)


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def init_database():
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,

            model TEXT NOT NULL,
            risk_type TEXT NOT NULL,

            prediction INTEGER,
            risk_probability REAL,
            risk_percentage REAL,
            risk_level TEXT,

            input_data TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_prediction(
    model,
    risk_type,
    prediction,
    risk_probability,
    risk_percentage,
    risk_level,
    input_data
):
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO prediction_history (
            created_at,
            model,
            risk_type,
            prediction,
            risk_probability,
            risk_percentage,
            risk_level,
            input_data
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            model,
            risk_type,
            prediction,
            risk_probability,
            risk_percentage,
            risk_level,
            input_data
        )
    )

    connection.commit()

    record_id = cursor.lastrowid

    connection.close()

    return record_id


def get_prediction_history(limit=100):
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM prediction_history
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()

    connection.close()

    return [dict(row) for row in rows]


def clear_prediction_history():
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("DELETE FROM prediction_history")

    connection.commit()
    connection.close()