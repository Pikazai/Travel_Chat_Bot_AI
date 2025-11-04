"""
Image service for fetching images from Pixabay API.
"""
import requests
from typing import Optional, List, Dict
from config.settings import get_settings


class ImageService:
    """Service for fetching images from Pixabay."""
    
    def __init__(self):
        """Initialize the image service."""
        settings = get_settings()
        self.api_key = settings.PIXABAY_API_KEY
    
    def get_pixabay_image(self, query: str, per_page: int = 3) -> Optional[str]:
        """
        Get an image URL from Pixabay.
        
        Args:
            query: Search query
            per_page: Number of results per page
            
        Returns:
            Image URL or None
        """
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
        """
        Get an image for a city.
        
        Args:
            city: City name
            
        Returns:
            Image URL or placeholder
        """
        if not city:
            return None
        
        queries = [
            f"{city} Vietnam landscape",
            f"{city} Vietnam city",
            f"{city} Vietnam travel",
            "Vietnam travel landscape"
        ]
        
        for q in queries:
            img = self.get_pixabay_image(q)
            if img:
                return img
        
        return "https://via.placeholder.com/1200x800?text=No+Image"
    
    def get_food_images(self, food_list: List[str]) -> List[Dict[str, str]]:
        """
        Get images for a list of foods.
        
        Args:
            food_list: List of food names
            
        Returns:
            List of dictionaries with 'name' and 'image' keys
        """
        images = []
        for food in food_list[:5]:
            query = f"{food} Vietnam food"
            img_url = self.get_pixabay_image(query)
            if not img_url:
                img_url = "https://via.placeholder.com/400x300?text=No+Image"
            images.append({"name": food, "image": img_url})
        return images

