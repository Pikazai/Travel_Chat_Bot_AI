"""
Application settings and configuration management.
Loads environment variables and provides centralized configuration.
"""

import os
from typing import Optional
from pathlib import Path


class Settings:
    """Centralized application settings."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    CHROMA_DIR = PROJECT_ROOT / "chromadb_data"
    DB_PATH = PROJECT_ROOT / "travel_chatbot_logs.db"
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_ENDPOINT: str = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
    DEPLOYMENT_NAME: str = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
    
    # External API Keys
    OPENWEATHERMAP_API_KEY: Optional[str] = os.getenv("OPENWEATHERMAP_API_KEY", "")
    GOOGLE_PLACES_KEY: Optional[str] = os.getenv("PLACES_API_KEY", "")
    PIXABAY_API_KEY: Optional[str] = os.getenv("PIXABAY_API_KEY", "")
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(CHROMA_DIR))
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Chatbot Configuration
    CHATBOT_NAME: str = "[Mây Lang Thang]"
    SYSTEM_PROMPT: str = """Bạn là Hướng dẫn viên du lịch ảo Alex - người kể chuyện, am hiểu văn hóa, lịch sử, ẩm thực và thời tiết Việt Nam.
Luôn đưa ra thông tin hữu ích, gợi ý lịch trình, món ăn, chi phí, thời gian lý tưởng, sự kiện và góc chụp ảnh."""
    
    # RAG Configuration
    RAG_TOP_K: int = 5
    INTENT_THRESHOLD: float = 0.18
    MEMORY_RECALL_K: int = 3
    
    # Voice Configuration
    ASR_LANGUAGE: str = "vi-VN"
    TTS_LANGUAGE: str = "vi"
    
    # Database Configuration
    DB_TABLE_NAME: str = "interactions"
    
    # File paths
    FOODS_CSV: Path = DATA_DIR / "vietnam_foods.csv"
    RESTAURANTS_CSV: Path = DATA_DIR / "restaurants_vn.csv"
    TRAVEL_DOCS_CSV: Path = DATA_DIR / "vietnam_travel_docs.csv"
    
    @classmethod
    def load_from_streamlit_secrets(cls) -> "Settings":
        """Load settings from Streamlit secrets if available."""
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                cls.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", cls.OPENAI_API_KEY)
                cls.OPENAI_ENDPOINT = st.secrets.get("OPENAI_ENDPOINT", cls.OPENAI_ENDPOINT)
                cls.DEPLOYMENT_NAME = st.secrets.get("DEPLOYMENT_NAME", cls.DEPLOYMENT_NAME)
                cls.OPENWEATHERMAP_API_KEY = st.secrets.get("OPENWEATHERMAP_API_KEY", cls.OPENWEATHERMAP_API_KEY)
                cls.GOOGLE_PLACES_KEY = st.secrets.get("PLACES_API_KEY", cls.GOOGLE_PLACES_KEY)
                cls.PIXABAY_API_KEY = st.secrets.get("PIXABAY_API_KEY", cls.PIXABAY_API_KEY)
        except Exception:
            pass  # Fallback to environment variables
        return cls


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings.load_from_streamlit_secrets()

