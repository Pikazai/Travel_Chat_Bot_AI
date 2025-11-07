"""
Logger service for SQLite database operations.
Handles interaction logging and analytics.
"""

import sqlite3
from datetime import datetime
from typing import Optional
from pathlib import Path

from config.settings import Settings


class LoggerService:
    """Service for logging interactions to SQLite database."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_path = settings.DB_PATH
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_input TEXT,
                city TEXT,
                start_date TEXT,
                end_date TEXT,
                intent TEXT,
                rag_used BOOLEAN DEFAULT 0,
                sources_count INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()
    
    def log_interaction(self, user_input: str, city: Optional[str] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       intent: Optional[str] = None,
                       rag_used: bool = False,
                       sources_count: int = 0):
        """Log user interaction."""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO interactions (timestamp, user_input, city, start_date, end_date, intent, rag_used, sources_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            user_input,
            city,
            start_date.isoformat() if start_date else None,
            end_date.isoformat() if end_date else None,
            intent,
            rag_used,
            sources_count
        ))
        conn.commit()
        conn.close()
    
    def get_recent_interactions(self, limit: int = 1000):
        """Get recent interactions."""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("SELECT * FROM interactions ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        conn.close()
        return rows
    
    def clear_all_interactions(self):
        """Clear all interaction logs."""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("DELETE FROM interactions")
        conn.commit()
        conn.close()

