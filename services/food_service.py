"""
Food service for retrieving local food recommendations.
"""

import pandas as pd
from typing import List, Optional

from config.settings import Settings
from utils.text_processing import re_split_foods


class FoodService:
    """Service for food recommendations."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    def get_local_foods(self, city: str) -> List[str]:
        """Get foods from CSV."""
        try:
            df = pd.read_csv(self.settings.FOODS_CSV, dtype=str)
            mask = df["city"].str.lower().str.contains(str(city).lower(), na=False)
            row = df[mask]
            if not row.empty:
                row0 = row.iloc[0]
                if "foods" in row0.index:
                    foods_cell = row0["foods"]
                    if pd.notna(foods_cell):
                        return re_split_foods(foods_cell)
                else:
                    vals = row0.dropna().tolist()
                    if len(vals) > 1:
                        return [v for v in vals[1:]]
        except Exception:
            pass
        return []
    
    def get_foods_via_gpt(self, city: str, max_items: int = 5, client=None) -> List[str]:
        """Get foods via GPT fallback."""
        if not client:
            return []
        try:
            prompt = (
                f"You are an expert on Vietnamese cuisine.\n"
                f"List up to {max_items} iconic or must-try dishes from the city/region '{city}'.\n"
                "Return only a comma-separated list of dish names (no extra text)."
            )
            response = client.chat.completions.create(
                model=self.settings.DEPLOYMENT_NAME,
                messages=[{"role": "system", "content": prompt}],
                max_tokens=150,
                temperature=0.5
            )
            text = response.choices[0].message.content.strip()
            items = [t.strip() for t in text.split(",") if t.strip()]
            return items[:max_items]
        except Exception:
            return []
    
    def get_foods_with_fallback(self, city: str, client=None) -> List[str]:
        """Get foods with CSV and GPT fallback."""
        foods = self.get_local_foods(city)
        if not foods:
            foods = self.get_foods_via_gpt(city, client=client)
        return foods

