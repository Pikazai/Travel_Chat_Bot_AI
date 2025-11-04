"""
Weather service for fetching weather forecasts from OpenWeatherMap API.
"""
import requests
from datetime import datetime, timedelta
from typing import Optional
from config.settings import get_settings
from utils.extractors import resolve_city_via_ai


class WeatherService:
    """Service for fetching weather forecasts."""
    
    def __init__(self):
        """Initialize the weather service."""
        settings = get_settings()
        self.api_key = settings.OPENWEATHERMAP_API_KEY
        self.client = None  # Will be set by set_openai_client if needed
    
    def set_openai_client(self, client):
        """Set OpenAI client for AI-based city resolution."""
        self.client = client
    
    def get_weather_forecast(
        self,
        city_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_text: Optional[str] = None
    ) -> str:
        """
        Get weather forecast for a city.
        
        Args:
            city_name: Name of the city
            start_date: Start date for forecast
            end_date: End date for forecast
            user_text: Original user text (for AI fallback)
            
        Returns:
            Weather forecast text
        """
        if not self.api_key:
            return "⚠️ Thiếu OpenWeatherMap API Key."
        
        try:
            def _fetch_weather(city):
                url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={self.api_key}&lang=vi&units=metric"
                response = requests.get(url, timeout=8)
                return response.json()
            
            data = _fetch_weather(city_name)
            
            # Try AI resolution if direct lookup fails
            if data.get("cod") != "200" and user_text and self.client:
                ai_city = resolve_city_via_ai(user_text, self.client)
                if ai_city and ai_city.lower() != city_name.lower():
                    data = _fetch_weather(f"{ai_city},VN")
                    city_name = ai_city
            
            if data.get("cod") != "200":
                return f"❌ Không tìm thấy thông tin dự báo thời tiết cho địa điểm: **{city_name}**."
            
            # Default to next 3 days if no dates provided
            if start_date is None or end_date is None:
                today = datetime.now().date()
                start_date = datetime.combine(today, datetime.min.time())
                end_date = datetime.combine(today + timedelta(days=3), datetime.min.time())
            
            forecast_text = f"🌤 **Dự báo thời tiết cho {city_name}:**\n"
            
            if start_date and end_date:
                current = start_date
                while current <= end_date:
                    date_str = current.strftime("%Y-%m-%d")
                    day_forecasts = [f for f in data['list'] if f['dt_txt'].startswith(date_str)]
                    if not day_forecasts:
                        forecast_text += f"\n📅 {current.strftime('%d/%m/%Y')}: Không có dữ liệu dự báo.\n"
                    else:
                        temps = [f['main']['temp'] for f in day_forecasts]
                        desc = day_forecasts[0]['weather'][0]['description']
                        forecast_text += (
                            f"\n📅 {current.strftime('%d/%m/%Y')} - {desc.capitalize()}\n"
                            f"🌡 Nhiệt độ trung bình: {sum(temps)/len(temps):.1f}°C\n"
                        )
                    current += timedelta(days=1)
            else:
                first_forecast = data['list'][0]
                desc = first_forecast['weather'][0]['description'].capitalize()
                temp = first_forecast['main']['temp']
                forecast_text += f"- Hiện tại: {desc}, {temp}°C\n"
            
            return forecast_text
        except Exception as e:
            return f"⚠️ Lỗi khi lấy dữ liệu thời tiết: {e}"

