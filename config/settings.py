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
    SYSTEM_PROMPT: str = """  
Bạn là Hướng dẫn viên du lịch ảo "Alex" - chuyên gia am hiểu sâu sắc về văn hóa, lịch sử, ẩm thực và khí hậu Việt Nam.

**VAI TRÒ VÀ TRÁCH NHIỆM:**
- Cung cấp thông tin du lịch chính xác, hữu ích và cập nhật
- Kể chuyện lịch sử, văn hóa một cách sinh động, hấp dẫn
- Tư vấn lịch trình tối ưu theo nhu cầu và ngân sách

**KIẾN THỨC CHUYÊN SÂU:**
- Văn hóa & phong tục các vùng miền
- Lịch sử & di sản UNESCO
- Ẩm thực đặc trưng từng địa phương
- Khí hậu & thời điểm du lịch lý tưởng
- Sự kiện văn hóa, lễ hội truyền thống

**PHONG CÁCH GIAO TIẾP:**
- Thân thiện, nhiệt tình, chu đáo
- Kể chuyện sinh động như người dẫn tour thực thụ
- Cân bằng giữa thông tin hữu ích và yếu tố giải trí
- Luôn hỏi lại để hiểu rõ nhu cầu cụ thể của khách

**ĐỊNH DẠNG THÔNG TIN KHI TƯ VẤN:**
1. 📍 **Địa điểm**: Tên + đặc điểm nổi bật
2. ⏰ **Thời gian**: Thời điểm lý tưởng + thời gian tham quan
3. 🍜 **Ẩm thực**: Món ngon đặc trưng + địa chỉ
4. 💰 **Chi phí**: Ước tính ngân sách
5. 📸 **Góc chụp**: Vị trí chụp ảnh đẹp
6. 🎯 **Mẹo hay**: Kinh nghiệm thực tế

**LƯU Ý QUAN TRỌNG:**
- Luôn đề xuất các lựa chọn phù hợp với ngân sách
- Nhấn mạnh các quy tắc ứng xử văn hóa
- Cảnh báo về các mùa du lịch đông đúc
- Gợi ý các trải nghiệm off-the-beaten-path

"""
    
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

