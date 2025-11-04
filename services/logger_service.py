"""
SQLite logging service for chatbot interactions.
"""
import sqlite3
from datetime import datetime
from typing import Optional
from config.settings import get_settings


class LoggerService:
    """Service for logging chatbot interactions to SQLite database."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the logger service.
        
        Args:
            db_path: Path to SQLite database file
        """
        settings = get_settings()
        self.db_path = db_path or settings.DB_PATH
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_input TEXT,
                city TEXT,
                start_date TEXT,
                end_date TEXT,
                intent TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def log_interaction(
        self,
        user_input: str,
        city: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        intent: Optional[str] = None
    ):
        """
        Log an interaction to the database.
        
        Args:
            user_input: User's input text
            city: Extracted city name
            start_date: Start date of travel
            end_date: End date of travel
            intent: Detected intent
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO interactions (timestamp, user_input, city, start_date, end_date, intent)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            user_input,
            city,
            start_date.isoformat() if start_date else None,
            end_date.isoformat() if end_date else None,
            intent
        ))
        conn.commit()
        conn.close()
    
    def get_interactions(self, limit: int = 1000):
        """
        Retrieve recent interactions from the database.
        
        Args:
            limit: Maximum number of interactions to retrieve
            
        Returns:
            List of interaction records
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM interactions 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        results = cur.fetchall()
        conn.close()
        return results
    
    def clear_interactions(self):
        """Clear all interactions from the database."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM interactions")
        conn.commit()
        conn.close()

