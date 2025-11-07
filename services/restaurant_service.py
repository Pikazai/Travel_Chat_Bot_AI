"""
Restaurant service for retrieving restaurant recommendations.
"""

import pandas as pd
import requests
from typing import List, Dict, Optional

from config.settings import Settings


class RestaurantService:
    """Service for restaurant recommendations."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.GOOGLE_PLACES_KEY
    
    def get_restaurants_google(self, city: str, limit: int = 5) -> List[Dict]:
        """Get restaurants from Google Places API."""
        if not self.api_key:
            return []
        try:
            query = f"nhà hàng tại {city}, Việt Nam"
            url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            params = {"query": query, "key": self.api_key, "language": "vi"}
            res = requests.get(url, params=params, timeout=10).json()
            if "error_message" in res:
                return [{"error": res["error_message"]}]
            results = []
            for r in res.get("results", [])[:limit]:
                results.append({
                    "name": r.get("name"),
                    "rating": r.get("rating", "N/A"),
                    "address": r.get("formatted_address", ""),
                    "maps_url": f"https://www.google.com/maps/place/?q=place_id:{r.get('place_id')}"
                })
            return results
        except Exception as e:
            return [{"error": str(e)}]
    
    def get_local_restaurants(self, city: str, limit: int = 5) -> List[Dict]:
        """Get restaurants from CSV."""
        try:
            df = pd.read_csv(self.settings.RESTAURANTS_CSV)
            df_city = df[df["city"].str.lower().str.contains(str(city).lower(), na=False)]
            if df_city.empty:
                return []
            return df_city.head(limit).to_dict("records")
        except Exception:
            return []
    
    def get_restaurants(self, city: str, limit: int = 5) -> List[Dict]:
        """Get restaurants with Google Places and CSV fallback."""
        if not city:
            return []
        if self.api_key:
            data = self.get_restaurants_google(city, limit)
            if data and not (data[0].get("error") if data else False):
                return data
        return self.get_local_restaurants(city, limit)

