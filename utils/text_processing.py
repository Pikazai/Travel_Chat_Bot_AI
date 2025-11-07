"""
Text processing utilities.
Functions for parsing and extracting information from text.
"""

import re
from typing import Optional
from datetime import datetime, timedelta


def extract_days_from_text(user_text: str, start_date: Optional[datetime] = None, 
                          end_date: Optional[datetime] = None) -> int:
    """
    Extract number of days from user text.
    
    Args:
        user_text: User input text
        start_date: Start date if available
        end_date: End date if available
        
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
    
    return 3  # Default


def re_split_foods(s: str) -> list[str]:
    """
    Split food string by common separators.
    
    Args:
        s: Food string (e.g., "phở, bún bò, bánh mì")
        
    Returns:
        List of food items
    """
    for sep in [",", "|", ";"]:
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s.strip()] if s.strip() else []


def clean_text(text: str) -> str:
    """Remove extra whitespace and normalize text."""
    return " ".join(text.split())


def extract_city_keywords(text: str) -> list[str]:
    """Extract potential city names from text."""
    # Common Vietnamese city patterns
    cities = ["Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Hội An", "Huế", "Sapa", 
              "Nha Trang", "Đà Lạt", "Phú Quốc", "Hạ Long"]
    found = []
    for city in cities:
        if city.lower() in text.lower():
            found.append(city)
    return found

