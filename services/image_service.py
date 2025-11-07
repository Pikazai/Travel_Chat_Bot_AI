"""
Image service for fetching images from Pixabay API.
"""

import requests
from typing import Optional, List, Dict

from config.settings import Settings


class ImageService:
    """Service for image retrieval."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.PIXABAY_API_KEY
    
    def get_image(self, query: str, per_page: int = 3) -> Optional[str]:
        """Get image URL from Pixabay."""
        if not self.api_key:
            return None
        try:
            url = "https://pixabay.com/api/"
            params = {
                "key": self.api_key,
                "q": query,
                "image_type": "photo",
                "orientation": "horizontal",
                "safesearch": "true",
                "per_page": per_page,
            }
            res = requests.get(url, params=params, timeout=8)
            data = res.json()
            if data.get("hits"):
                return data["hits"][0].get("largeImageURL") or data["hits"][0].get("webformatURL")
            return None
        except Exception:
            return None
    
    def get_city_image(self, city: str) -> Optional[str]:
        """Get city image."""
        if not city:
            return None
        queries = [
            f"{city} Vietnam landscape",
            f"{city} Vietnam city",
            f"{city} Vietnam travel",
            "Vietnam travel landscape"
        ]
        for q in queries:
            img = self.get_image(q)
            if img:
                return img
        return "https://via.placeholder.com/1200x800?text=No+Image"
    
    def get_food_images(self, food_list: List[str]) -> List[Dict]:
        """Get images for food items."""
        images = []
        for food in food_list[:5]:
            query = f"{food} Vietnam food"
            img_url = self.get_image(query)
            if not img_url:
                img_url = "https://via.placeholder.com/400x300?text=No+Image"
            images.append({"name": food, "image": img_url})
        return images

