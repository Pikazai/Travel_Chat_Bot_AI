"""
Geocoding service for location lookup and mapping.
"""

from typing import Optional, Tuple
from geopy.geocoders import Nominatim
import pydeck as pdk
import pandas as pd
import streamlit as st

from config.settings import Settings


class GeocodingService:
    """Service for geocoding and map display."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.geolocator = Nominatim(user_agent="travel_chatbot_app")
    
    def geocode_city(self, city_name: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """Get coordinates for city name."""
        try:
            loc = self.geolocator.geocode(city_name, country_codes="VN", timeout=10)
            if loc:
                return loc.latitude, loc.longitude, loc.address
            return None, None, None
        except Exception:
            return None, None, None
    
    def show_map(self, lat: float, lon: float, zoom: int = 8, title: str = ""):
        """Display map using PyDeck."""
        if lat is None or lon is None:
            st.info("Không có dữ liệu toạ độ để hiển thị bản đồ.")
            return
        
        st.write(f"**Vị trí:** {title} ({lat:.5f}, {lon:.5f})")
        
        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{"lat": lat, "lon": lon}]),
            get_position='[lon, lat]',
            get_radius=2000,
            get_fill_color=[255, 0, 0],
            get_line_color=[0, 0, 0],
            line_width_min_pixels=1,
            pickable=True,
            opacity=0.9,
        )
        
        marker_layer = pdk.Layer(
            "TextLayer",
            data=pd.DataFrame([{"lat": lat, "lon": lon, "name": "📍"}]),
            get_position='[lon, lat]',
            get_text="name",
            get_size=24,
            get_color=[200, 30, 30],
            billboard=True,
        )
        
        deck = pdk.Deck(
            layers=[layer, marker_layer],
            initial_view_state=view_state,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            tooltip={"text": title or "Vị trí"},
        )
        
        st.pydeck_chart(deck)

