"""
Geocoding utilities for location lookup and map display.
"""
from typing import Optional, Tuple
from geopy.geocoders import Nominatim
import pandas as pd
import pydeck as pdk
import streamlit as st


geolocator = Nominatim(user_agent="travel_chatbot_app")


def geocode_city(city_name: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Geocode a city name to coordinates.
    
    Args:
        city_name: Name of the city
        
    Returns:
        Tuple of (latitude, longitude, address) or (None, None, None) if not found
    """
    try:
        loc = geolocator.geocode(city_name, country_codes="VN", timeout=10)
        if loc:
            return loc.latitude, loc.longitude, loc.address
        return None, None, None
    except Exception:
        return None, None, None


def show_map(lat: float, lon: float, zoom: int = 8, title: str = ""):
    """
    Display a map using PyDeck.
    
    Args:
        lat: Latitude
        lon: Longitude
        zoom: Zoom level (4-15)
        title: Map title
    """
    if lat is None or lon is None:
        st.info("Không có dữ liệu toạ độ để hiển thị bản đồ.")
        return
    
    # Ensure float type
    lat, lon = float(lat), float(lon)
    
    st.write(f"**Vị trí:** {title} ({lat:.5f}, {lon:.5f})")
    
    # Set up view state
    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom)
    
    # Create scatterplot layer
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
    
    # Add text marker layer
    marker_layer = pdk.Layer(
        "TextLayer",
        data=pd.DataFrame([{"lat": lat, "lon": lon, "name": "📍"}]),
        get_position='[lon, lat]',
        get_text="name",
        get_size=24,
        get_color=[200, 30, 30],
        billboard=True,
    )
    
    # Create deck
    deck = pdk.Deck(
        layers=[layer, marker_layer],
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"text": title or "Vị trí"},
    )
    
    st.pydeck_chart(deck)

