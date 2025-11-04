"""
Data extraction utilities for travel information.
Extracts cities, dates, and other information from user input.
"""
import re
import json
from datetime import datetime
from typing import Optional, Tuple
import openai
from config.settings import get_settings


def extract_city_and_dates(user_text: str, client: Optional[openai.OpenAI] = None) -> Tuple[Optional[str], Optional[datetime], Optional[datetime]]:
    """
    Extract city, start_date, and end_date from user text using AI.
    
    Args:
        user_text: User input text
        client: OpenAI client instance
        
    Returns:
        Tuple of (city, start_date, end_date)
    """
    settings = get_settings()
    if client is None:
        try:
            client = openai.OpenAI(base_url=settings.OPENAI_ENDPOINT, api_key=settings.OPENAI_API_KEY)
        except Exception:
            return None, None, None
    
    if not client:
        return None, None, None
    
    try:
        prompt = f"""
You are a multilingual travel information extractor.
The user message may be in Vietnamese or English.
Extract:
1. Destination city (field 'city') - if none, return null
2. Start date (field 'start_date') in YYYY-MM-DD format or null
3. End date (field 'end_date') in YYYY-MM-DD format or null
Rules:
- If user provides only one date, set both start_date and end_date to that date.
- If the user gives day/month without year, assume current year.
- Return valid JSON ONLY, for example:
{{"city":"Hanoi", "start_date":"2025-10-20", "end_date":"2025-10-22"}}
Message: "{user_text}"
"""
        response = client.chat.completions.create(
            model=settings.DEPLOYMENT_NAME,
            messages=[{"role": "system", "content": prompt}],
            max_tokens=200,
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
            if not d:
                return None
            return datetime.strptime(d, "%Y-%m-%d")
        
        start_dt = _parse(s)
        end_dt = _parse(e)
        if start_dt and not end_dt:
            end_dt = start_dt
        
        return city, start_dt, end_dt
    except Exception:
        return None, None, None


def extract_days_from_text(user_text: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, client: Optional[openai.OpenAI] = None) -> int:
    """
    Extract number of days from user text.
    
    Args:
        user_text: User input text
        start_date: Optional start date
        end_date: Optional end date
        client: OpenAI client instance
        
    Returns:
        Number of days (default: 3)
    """
    if start_date and end_date:
        try:
            delta = (end_date - start_date).days + 1
            return max(delta, 1)
        except Exception:
            pass
    
    # Try regex pattern matching
    m = re.search(r"(\d+)\s*(ngày|day|days|tuần|week|weeks)", user_text, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        unit = m.group(2).lower()
        if "tuần" in unit or "week" in unit:
            return num * 7
        return num
    
    # Fallback to AI extraction
    settings = get_settings()
    if client is None:
        try:
            client = openai.OpenAI(base_url=settings.OPENAI_ENDPOINT, api_key=settings.OPENAI_API_KEY)
        except Exception:
            return 3
    
    if client:
        try:
            prompt = f"""
Bạn là một bộ phân tích ngữ nghĩa tiếng Việt & tiếng Anh.
Xác định người dùng muốn nói bao nhiêu ngày trong câu sau, nếu không có thì mặc định 3:
Trả về JSON: {{"days": <số nguyên>}}
Câu: "{user_text}"
"""
            response = client.chat.completions.create(
                model=settings.DEPLOYMENT_NAME,
                messages=[{"role": "system", "content": prompt}],
                max_tokens=50,
                temperature=0
            )
            text = response.choices[0].message.content.strip()
            num_match = re.search(r'"days"\s*:\s*(\d+)', text)
            if num_match:
                return int(num_match.group(1))
        except Exception:
            pass
    
    return 3


def resolve_city_via_ai(user_text: str, client: Optional[openai.OpenAI] = None) -> Optional[str]:
    """
    Resolve city name from user text using AI when direct geocoding fails.
    
    Args:
        user_text: User input text
        client: OpenAI client instance
        
    Returns:
        City name or None
    """
    settings = get_settings()
    if client is None:
        try:
            client = openai.OpenAI(base_url=settings.OPENAI_ENDPOINT, api_key=settings.OPENAI_API_KEY)
        except Exception:
            return None
    
    if not client:
        return None
    
    try:
        prompt = f"""
Bạn là chuyên gia địa lý du lịch Việt Nam.
Phân tích câu sau để xác định:
1. 'place': địa danh cụ thể (khu du lịch, công viên, đảo, thắng cảnh,...)
2. 'province_or_city': tên tỉnh hoặc thành phố của Việt Nam mà địa danh đó thuộc về.
Nếu không xác định được, trả về null.
Kết quả JSON ví dụ: {{"place": "Phong Nha - Kẻ Bàng", "province_or_city": "Quảng Bình"}}
Câu: "{user_text}"
"""
        response = client.chat.completions.create(
            model=settings.DEPLOYMENT_NAME,
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

