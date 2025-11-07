"""
Entity extraction module.
Extracts city names, dates, and other entities from user input.
"""

import json
from typing import Optional, Tuple
from datetime import datetime

from config.settings import Settings


class EntityExtractor:
    """Extract entities (city, dates) from user text."""
    
    def __init__(self, settings: Settings, openai_client=None):
        self.settings = settings
        self.client = openai_client
    
    def extract_city_and_dates(self, user_text: str) -> Tuple[Optional[str], Optional[datetime], Optional[datetime]]:
        """Extract city and date range from user text."""
        if not self.client:
            return None, None, None
        
        try:
            prompt = f"""
You are a multilingual travel information extractor.
Extract 'city','start_date','end_date' (YYYY-MM-DD). 
If only one date is provided, set both to that date.
If no date is mentioned, set both start_date and end_date to null.
Return JSON only.
Message: "{user_text}"
"""
            response = self.client.chat.completions.create(
                model=self.settings.DEPLOYMENT_NAME,
                messages=[{"role": "system", "content": prompt}],
                max_tokens=300,
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            start = content.find('{')
            end = content.rfind('}')
            if start == -1 or end == -1:
                return None, None, None
            
            data = json.loads(content[start:end+1])
            city = data.get("city")
            s = data.get("start_date")
            e = data.get("end_date")
            
            def _parse(d):
                if not d or d.lower() == 'null' or d == '':
                    return None
                try:
                    dt = datetime.strptime(d, "%Y-%m-%d")
                    return dt
                except ValueError:
                    return None
            
            start_dt = _parse(s)
            end_dt = _parse(e)
            
            # Validate dates
            today = datetime.now().date()
            if start_dt and start_dt.date() < today:
                start_dt = None
            if end_dt and end_dt.date() < today:
                end_dt = None
            if start_dt and not end_dt:
                end_dt = start_dt
            
            return city, start_dt, end_dt
        except Exception:
            return None, None, None
    
    def resolve_city_via_ai(self, user_text: str) -> Optional[str]:
        """Resolve city name using AI when geocoding fails."""
        if not self.client:
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
            response = self.client.chat.completions.create(
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
    
    def is_travel_related(self, user_text: str) -> bool:
        """Check if user text is travel-related."""
        if not self.client:
            return True  # Default to allow
        
        try:
            prompt = f"""
Bạn là bộ phân loại chủ đề thông minh.
Hãy xác định xem câu sau có liên quan đến lĩnh vực *du lịch Việt Nam* hay không.

Các chủ đề được coi là liên quan bao gồm:
- địa điểm, thành phố, tỉnh, danh lam thắng cảnh
- thời tiết, khí hậu
- lịch trình du lịch, tour, chi phí, gợi ý điểm đến
- món ăn địa phương, đặc sản, nhà hàng
- khách sạn, homestay, resort
- sự kiện, lễ hội, văn hoá vùng miền

Nếu KHÔNG thuộc những chủ đề trên, trả về JSON: {{"related": false}}
Nếu CÓ liên quan, trả về JSON: {{"related": true}}

Câu người dùng: "{user_text}"
"""
            response = self.client.chat.completions.create(
                model=self.settings.DEPLOYMENT_NAME,
                messages=[{"role": "system", "content": prompt}],
                temperature=0,
                max_tokens=30,
            )
            text = response.choices[0].message.content.strip().lower()
            if '"related": true' in text:
                return True
            if '"related": false' in text:
                return False
        except Exception as e:
            print(f"[WARN] Topic classification error: {e}")
        
        return True  # Fallback

