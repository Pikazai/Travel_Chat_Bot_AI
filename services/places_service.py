"""
Places service for restaurants and food recommendations.
Handles Google Places API and CSV fallback.
"""
import pandas as pd
import requests
from typing import List, Dict, Optional
import re
import openai
from config.settings import get_settings


class PlacesService:
    """Service for fetching restaurants and food recommendations."""
    
    def __init__(self):
        """Initialize the places service."""
        settings = get_settings()
        self.google_places_key = settings.GOOGLE_PLACES_KEY
        self.restaurants_csv = settings.RESTAURANTS_CSV
        self.foods_csv = settings.FOODS_CSV
        self.client = None  # OpenAI client for GPT fallback
    
    def set_openai_client(self, client):
        """Set OpenAI client for GPT-based food recommendations."""
        self.client = client
    
    def get_restaurants(self, city: str, limit: int = 5) -> List[Dict]:
        """
        Get restaurant recommendations for a city.
        
        Args:
            city: City name
            limit: Maximum number of restaurants to return
            
        Returns:
            List of restaurant dictionaries
        """
        if not city:
            return []
        
        # Try Google Places first
        if self.google_places_key:
            data = self._get_restaurants_google(city, self.google_places_key, limit)
            if data and not (data[0].get("error") if data else False):
                return data
        
        # Fallback to CSV
        return self._get_local_restaurants(city, limit)
    
    def _get_restaurants_google(self, city: str, api_key: str, limit: int = 5) -> List[Dict]:
        """Get restaurants from Google Places API."""
        try:
            query = f"nhà hàng tại {city}, Việt Nam"
            url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            params = {"query": query, "key": api_key, "language": "vi"}
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
    
    def _get_local_restaurants(self, city: str, limit: int = 5) -> List[Dict]:
        """Get restaurants from local CSV file."""
        try:
            df = pd.read_csv(self.restaurants_csv)
            df_city = df[df["city"].str.lower().str.contains(str(city).lower(), na=False)]
            if df_city.empty:
                return []
            return df_city.head(limit).to_dict("records")
        except Exception:
            return []
    
    def get_foods(self, city: str, max_items: int = 5) -> List[str]:
        """
        Get food recommendations for a city.
        
        Args:
            city: City name
            max_items: Maximum number of foods to return
            
        Returns:
            List of food names
        """
        foods = self._get_local_foods(city)
        if not foods and self.client:
            foods = self._get_foods_via_gpt(city, max_items)
        return foods
    
    def _get_local_foods(self, city: str) -> List[str]:
        """Get foods from local CSV file."""
        try:
            df = pd.read_csv(self.foods_csv, dtype=str)
            mask = df["city"].str.lower().str.contains(str(city).lower(), na=False)
            row = df[mask]
            if not row.empty:
                row0 = row.iloc[0]
                if "foods" in row0.index:
                    foods_cell = row0["foods"]
                    if pd.notna(foods_cell):
                        return self._re_split_foods(foods_cell)
                else:
                    vals = row0.dropna().tolist()
                    if len(vals) > 1:
                        return [v for v in vals[1:]]
        except Exception:
            pass
        return []
    
    def _get_foods_via_gpt(self, city: str, max_items: int = 5) -> List[str]:
        """Get food recommendations using GPT."""
        if not self.client:
            return []
        
        try:
            settings = get_settings()
            prompt = (
                f"You are an expert on Vietnamese cuisine.\n"
                f"List up to {max_items} iconic or must-try dishes from the city/region '{city}'.\n"
                "Return only a comma-separated list of dish names (no extra text)."
            )
            response = self.client.chat.completions.create(
                model=settings.DEPLOYMENT_NAME,
                messages=[{"role": "system", "content": prompt}],
                max_tokens=150,
                temperature=0.5
            )
            text = response.choices[0].message.content.strip()
            items = [t.strip() for t in text.split(",") if t.strip()]
            return items[:max_items]
        except Exception:
            return []
    
    @staticmethod
    def _re_split_foods(s: str) -> List[str]:
        """Split food string by various delimiters."""
        for sep in [",", "|", ";"]:
            if sep in s:
                return [p.strip() for p in s.split(sep) if p.strip()]
        return [s.strip()]

