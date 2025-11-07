"""
Weather service for OpenWeatherMap API integration.
"""

import requests
import json
from typing import Optional
from datetime import datetime, timedelta

from config.settings import Settings


class WeatherService:
    """Service for weather forecast data."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.OPENWEATHERMAP_API_KEY
    
    def resolve_city_via_ai(self, user_text: str, client=None) -> Optional[str]:
        """Resolve city name via AI fallback when not found."""
        if not client:
            return None
        try:
            prompt = f"""
Bạn là chuyên gia địa lý du lịch Việt Nam.
Phân tích câu sau để xác định:
1. 'place': địa danh cụ thể
2. 'province_or_city': tỉnh/thành của Việt Nam chứa địa danh đó.
Nếu không xác định được, trả về null.
JSON ví dụ: {{"place":"Phong Nha - Kẻ Bàng","province_or_city":"Quảng Bình"}}
Câu: "{user_text}"
"""
            response = client.chat.completions.create(
                model=self.settings.DEPLOYMENT_NAME,
                messages=[{"role": "system", "content": prompt}],
                max_tokens=200,
                temperature=0
            )
            text = response.choices[0].message.content.strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                return None
            data = json.loads(text[start:end+1])
            return data.get("province_or_city")
        except Exception:
            return None
    
    def get_forecast(self, city_name: str, start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None, user_text: Optional[str] = None,
                    client=None) -> str:
        """Get weather forecast for city."""
        if not self.api_key:
            return "⚠️ Thiếu OpenWeatherMap API Key."
        
        # Default to today if no dates provided
        if start_date is None or end_date is None:
            today = datetime.now().date()
            start_date = datetime.combine(today, datetime.min.time())
            end_date = datetime.combine(today, datetime.min.time())
        
        try:
            def _fetch_weather(city):
                url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={self.api_key}&lang=vi&units=metric"
                response = requests.get(url, timeout=8)
                return response.json()
            
            data = _fetch_weather(city_name)
            
            # Fallback to AI resolution if city not found
            if data.get("cod") != "200" and user_text and client:
                ai_city = self.resolve_city_via_ai(user_text, client)
                if ai_city and ai_city.lower() != city_name.lower():
                    data = _fetch_weather(f"{ai_city},VN")
                    city_name = ai_city
            
            if data.get("cod") != "200":
                return f"❌ Không tìm thấy thông tin dự báo thời tiết cho địa điểm: **{city_name}**."
            
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

