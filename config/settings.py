"""
Configuration management for Travel Chatbot.
Loads environment variables and provides centralized settings.
"""
import os
from typing import Optional
import streamlit as st


class Settings:
    """Centralized configuration settings."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_KEY_EMBEDDING: Optional[str] = None
    OPENAI_ENDPOINT: str = "https://api.openai.com/v1"
    DEPLOYMENT_NAME: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # External APIs
    OPENWEATHERMAP_API_KEY: Optional[str] = None
    GOOGLE_PLACES_KEY: Optional[str] = None
    PIXABAY_API_KEY: Optional[str] = None
    
    # ChromaDB Configuration
    CHROMA_PATH: str = "./.chroma"
    
    # Database Configuration
    DB_PATH: str = "travel_chatbot_logs.db"
    
    # Chatbot Configuration
    CHATBOT_NAME: str = "[Mây Lang Thang]"
    SYSTEM_PROMPT: str = """Bạn là Hướng dẫn viên du lịch ảo Alex - người kể chuyện, am hiểu văn hóa, lịch sử, ẩm thực và thời tiết Việt Nam.
Luôn đưa ra thông tin hữu ích, gợi ý lịch trình, món ăn, chi phí, thời gian lý tưởng, sự kiện và góc chụp ảnh."""
    
    # Data paths
    DATA_DIR: str = "data"
    FOODS_CSV: str = "data/vietnam_foods.csv"
    RESTAURANTS_CSV: str = "data/restaurants_vn.csv"
    
    def __init__(self):
        """Initialize settings from Streamlit secrets or environment variables."""
        self._load_secrets()
    
    def _load_secrets(self):
        """Load configuration from Streamlit secrets or environment variables."""
        # Try Streamlit secrets first, fallback to environment variables
        try:
            self.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            self.OPENAI_API_KEY_EMBEDDING = st.secrets.get("OPENAI_API_KEY_EMBEDDING") or os.getenv("OPENAI_API_KEY_EMBEDDING")
            self.OPENAI_ENDPOINT = st.secrets.get("OPENAI_ENDPOINT", self.OPENAI_ENDPOINT) or os.getenv("OPENAI_ENDPOINT", self.OPENAI_ENDPOINT)
            self.DEPLOYMENT_NAME = st.secrets.get("DEPLOYMENT_NAME", self.DEPLOYMENT_NAME) or os.getenv("DEPLOYMENT_NAME", self.DEPLOYMENT_NAME)
            self.EMBEDDING_MODEL = st.secrets.get("EMBEDDING_MODEL", self.EMBEDDING_MODEL) or os.getenv("EMBEDDING_MODEL", self.EMBEDDING_MODEL)
            
            self.OPENWEATHERMAP_API_KEY = st.secrets.get("OPENWEATHERMAP_API_KEY", "") or os.getenv("OPENWEATHERMAP_API_KEY", "")
            self.GOOGLE_PLACES_KEY = st.secrets.get("PLACES_API_KEY", "") or os.getenv("PLACES_API_KEY", "")
            self.PIXABAY_API_KEY = st.secrets.get("PIXABAY_API_KEY", "") or os.getenv("PIXABAY_API_KEY", "")
            
            self.CHROMA_PATH = st.secrets.get("CHROMA_PATH", self.CHROMA_PATH) or os.getenv("CHROMA_PATH", self.CHROMA_PATH)
        except Exception:
            # Fallback to environment variables only
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            self.OPENAI_API_KEY_EMBEDDING = os.getenv("OPENAI_API_KEY_EMBEDDING")
            self.OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", self.OPENAI_ENDPOINT)
            self.DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", self.DEPLOYMENT_NAME)
            self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", self.EMBEDDING_MODEL)
            
            self.OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
            self.GOOGLE_PLACES_KEY = os.getenv("PLACES_API_KEY", "")
            self.PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "")
            
            self.CHROMA_PATH = os.getenv("CHROMA_PATH", self.CHROMA_PATH)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

