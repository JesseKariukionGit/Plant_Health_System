import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "logs", "plant_health.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS DiagnosisLog (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            disease_predicted TEXT NOT NULL,
            confidence REAL NOT NULL,
            actual_disease TEXT,
            user_feedback TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS TreatmentLog (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            diagnosis_id INTEGER NOT NULL,
            retrieved_chunks TEXT NOT NULL,
            llm_response TEXT NOT NULL,
            safety_flag INTEGER NOT NULL,
            FOREIGN KEY (diagnosis_id) REFERENCES DiagnosisLog(log_id) ON DELETE CASCADE,
            UNIQUE(diagnosis_id)
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database initialized")

def log_diagnosis(disease_predicted: str, confidence: float, 
                  actual_disease: Optional[str] = None, 
                  user_feedback: Optional[str] = None) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO DiagnosisLog 
        (timestamp, disease_predicted, confidence, actual_disease, user_feedback)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, disease_predicted, confidence, actual_disease, user_feedback))
    log_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return log_id

def log_treatment(diagnosis_id: int, retrieved_chunks: list, 
                  llm_response: str, safety_flag: int = 0) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    chunks_json = json.dumps(retrieved_chunks)
    cursor.execute('''
        INSERT INTO TreatmentLog 
        (diagnosis_id, retrieved_chunks, llm_response, safety_flag)
        VALUES (?, ?, ?, ?)
    ''', (diagnosis_id, chunks_json, llm_response, safety_flag))
    log_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return log_id

def delete_old_records(hours: int = 24):
    """Delete records older than specified hours"""
    cutoff = datetime.now() - timedelta(hours=hours)
    cutoff_str = cutoff.isoformat()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Delete from TreatmentLog first (due to foreign key)
    cursor.execute('''
        DELETE FROM TreatmentLog 
        WHERE diagnosis_id IN (
            SELECT log_id FROM DiagnosisLog WHERE timestamp < ?
        )
    ''', (cutoff_str,))
    # Then delete from DiagnosisLog
    cursor.execute('''
        DELETE FROM DiagnosisLog WHERE timestamp < ?
    ''', (cutoff_str,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    if deleted:
        print(f"🧹 Deleted {deleted} old diagnosis records (older than {hours} hours)")
    return deleted

def get_diagnosis_history(limit: int = 20, offset: int = 0) -> tuple:
    """Return (rows, total_count)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM DiagnosisLog')
    total = cursor.fetchone()[0]
    cursor.execute('''
        SELECT log_id, timestamp, disease_predicted, confidence, actual_disease
        FROM DiagnosisLog
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    ''', (limit, offset))
    rows = cursor.fetchall()
    conn.close()
    return rows, total

def get_treatment_by_diagnosis(diagnosis_id: int) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT llm_response FROM TreatmentLog 
        WHERE diagnosis_id = ?
        ORDER BY log_id DESC LIMIT 1
    ''', (diagnosis_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]
    return "No treatment found."

init_db()