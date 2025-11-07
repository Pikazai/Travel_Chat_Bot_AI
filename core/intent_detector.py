"""
Intent detection module.
Detects user intent using ChromaDB semantic matching.
"""

from typing import Optional

from config.settings import Settings
from services.chroma_service import ChromaService


class IntentDetector:
    """Detect user intent from text."""
    
    def __init__(self, settings: Settings, chroma_service: ChromaService):
        self.settings = settings
        self.chroma_service = chroma_service
    
    def detect_intent(self, user_text: str, threshold: float = None) -> Optional[str]:
        """Detect intent from user text."""
        threshold = threshold or self.settings.INTENT_THRESHOLD
        return self.chroma_service.get_intent(user_text, threshold)
    
    def handle_intent(self, intent: str, user_text: str, city: Optional[str] = None,
                     start_date=None, end_date=None, **kwargs) -> Optional[str]:
        """Handle specific intent and return response."""
        if intent == "weather_query" and city:
            weather_service = kwargs.get("weather_service")
            if weather_service:
                return weather_service.get_forecast(city, start_date, end_date, user_text)
        
        elif intent == "food_query" and city:
            food_service = kwargs.get("food_service")
            if food_service:
                foods = food_service.get_foods_with_fallback(city, kwargs.get("client"))
                if foods:
                    return "Đặc sản nổi bật:\n" + "\n".join([f"- {f}" for f in foods])
                return "Không tìm thấy đặc sản trong DB."
        
        elif intent == "itinerary_request" and city:
            days = kwargs.get("days", 3)
            return f"Lịch trình gợi ý cho {city}, {days} ngày:\n1) Ngày 1: ...\n2) Ngày 2: ...\n3) Ngày 3: ..."
        
        return None  # Unknown intent

