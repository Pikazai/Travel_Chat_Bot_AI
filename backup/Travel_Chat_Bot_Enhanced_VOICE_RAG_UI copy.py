# Travel_Chat_Bot_Enhanced_VOICE_RAG_Modern.py
# =================================
# Mở rộng: RAG (ChromaDB) + long-term memory + intent quick-match + recommendations
# Giữ lại toàn bộ chức năng gốc (voice, TTS, weather, map, foods, restaurants...)
# MODERN UI VERSION - Enhanced Professional Interface

import streamlit as st
import openai
import json
import requests
import os
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
import pandas as pd
import sqlite3
import pydeck as pdk
import re
import time
import plotly.express as px

# === VOICE imports ===
import io
import base64
import tempfile
import subprocess
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS

# === RAG / Chroma imports ===
from chromadb import PersistentClient
import uuid

# === THÊM IMPORTS CHO EMBEDDING LOCAL ===
from sentence_transformers import SentenceTransformer
import numpy as np

# === KHAI BÁO MODEL EMBEDDING LOCAL ===
@st.cache_resource
def load_embedding_model():
    try:
        model_path = "data/all-MiniLM-L6-v2"
        if os.path.exists(model_path) and os.path.isdir(model_path):
            if any(file.endswith('model.safetensors') for file in os.listdir(model_path)):
                model = SentenceTransformer(model_path)
                print("✅ Đã tải model embedding local: all-MiniLM-L6-v2")
                return model
        
        print("📥 Đang tải model từ Hugging Face...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        os.makedirs(model_path, exist_ok=True)
        model.save(model_path)
        print(f"✅ Đã tải và lưu model vào: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Lỗi khi tải model embedding: {e}")
        return None

# Load model
embedding_model = load_embedding_model()

# === Chroma persistent directory ===
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chromadb_data")

# -------------------------
st.set_page_config(
    page_title="🌤️ Mây Lang Thang - AI Travel Assistant", 
    layout="wide", 
    page_icon="🌤️",
    initial_sidebar_state="expanded"
)

# ========================
# MODERN CSS STYLING
# ========================
st.markdown(
    """
    <style>
    :root {
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --secondary: #8b5cf6;
        --accent: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --background: #f8fafc;
        --surface: #ffffff;
        --text: #1e293b;
        --text-muted: #64748b;
    }

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f0fdf4 100%);
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
        line-height: 1.6;
    }

    /* Ẩn header mặc định của Streamlit */
    .stApp > header { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Main container improvements */
    .main .block-container {
        padding-top: 1rem;
        max-width: 1400px;
    }

    /* Compact Hero Section */
    .modern-hero {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 0.5rem 0 1.5rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.15);
        color: white;
        min-height: 120px;
        display: flex;
        align-items: center;
    }

    .modern-hero::before {
        content: '';
        position: absolute;
        top: -30%;
        right: -10%;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-content {
        flex: 1;
    }

    .hero-content h1 {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, #fff, #e0f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        font-size: 1rem;
        margin-bottom: 0;
        opacity: 0.9;
        font-weight: 400;
    }

    .hero-stats {
        display: flex;
        gap: 1.5rem;
        margin-top: 1rem;
    }

    .stat {
        text-align: center;
    }

    .stat-number {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }

    .stat-label {
        font-size: 0.75rem;
        opacity: 0.8;
    }

    .hero-visual {
        display: none; /* Ẩn visual để tiết kiệm không gian */
    }

    /* Modern Cards */
    .surface-card {
        background: var(--surface);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }

    /* Enhanced Chat Messages */
    .user-message {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .assistant-message {
        background: var(--surface);
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 80%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }

    /* Modern Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--surface) 0%, #f8fafc 100%);
        border-right: 1px solid rgba(0, 0, 0, 0.05);
    }

    .sidebar-card {
        background: var(--surface);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }

    .sidebar-header {
        padding: 1rem 0;
        border-bottom: 1px solid rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }

    .logo-title {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .logo {
        font-size: 2rem;
    }

    .sidebar-subtitle {
        color: var(--text-muted);
        font-size: 0.8rem;
        margin-top: -0.3rem;
    }

    /* Quick Actions - Giữ nguyên kiểu cũ nhưng chỉnh CSS */
    .quick-actions {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .quick-action-btn {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 0.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        font-weight: 600;
        font-size: 0.85rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 60px;
    }

    .quick-action-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(37, 99, 235, 0.3);
    }

    .quick-action-btn .emoji {
        font-size: 1.2rem;
        margin-bottom: 0.3rem;
    }

    .quick-action-btn .text {
        font-size: 0.75rem;
        line-height: 1.2;
    }

    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--surface);
        padding: 0.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: white !important;
    }

    /* Enhanced Status Cards */
    .status-card {
        background: var(--surface);
        padding: 0.8rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 4px solid var(--success);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }

    .status-card.warning {
        border-left-color: var(--warning);
    }

    .status-card.error {
        border-left-color: var(--error);
    }

    /* Source badges */
    .source-badge {
        background: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 10px;
        padding: 6px 10px;
        margin: 5px 0;
        font-size: 0.8em;
    }
    .source-chroma {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .source-intent {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .source-memory {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }

    /* Loading Animation */
    .stSpinner > div {
        border: 2px solid var(--primary);
        border-radius: 50%;
        border-top: 2px solid transparent;
        width: 20px;
        height: 20px;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .modern-hero {
            padding: 1rem;
            text-align: center;
        }
        
        .hero-content h1 {
            font-size: 1.5rem;
        }
        
        .hero-stats {
            justify-content: center;
            gap: 1rem;
        }
        
        .user-message,
        .assistant-message {
            max-width: 90%;
        }
        
        .quick-actions {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }

    /* Audio wrapper */
    .audio-wrapper {
        margin-top: 0.5rem;
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# -------------------------
# CONFIG / SECRETS
# -------------------------
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
    
OPENAI_ENDPOINT = st.secrets.get("OPENAI_ENDPOINT", "https://api.openai.com/v1") if hasattr(st, 'secrets') else os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
DEPLOYMENT_NAME = st.secrets.get("DEPLOYMENT_NAME", "gpt-4o-mini") if hasattr(st, 'secrets') else os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
OPENWEATHERMAP_API_KEY = st.secrets.get("OPENWEATHERMAP_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("OPENWEATHERMAP_API_KEY", "")
GOOGLE_PLACES_KEY = st.secrets.get("PLACES_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("PLACES_API_KEY", "")
PIXABAY_API_KEY = st.secrets.get("PIXABAY_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("PIXABAY_API_KEY", "")

# Initialize OpenAI client
if OPENAI_API_KEY:
    client = openai.OpenAI(base_url=OPENAI_ENDPOINT, api_key=OPENAI_API_KEY)
else:
    client = None

ChatBotName = "🌤️ Mây Lang Thang"
system_prompt = """
Bạn là Hướng dẫn viên du lịch ảo Alex - người kể chuyện, am hiểu văn hóa, lịch sử, ẩm thực và thời tiết Việt Nam.
Luôn đưa ra thông tin hữu ích, gợi ý lịch trình, món ăn, chi phí, thời gian lý tưởng, sự kiện và góc chụp ảnh.
"""

# -------------------------
# COMPACT HERO SECTION
# -------------------------
def render_compact_hero():
    st.markdown("""
    <div class="modern-hero">
        <div class="hero-content">
            <h1>🌤️ Mây Lang Thang</h1>
            <p class="hero-subtitle">Trợ lý du lịch AI - Khám phá Việt Nam thông minh</p>
            <div class="hero-stats">
                <div class="stat">
                    <div class="stat-number">500+</div>
                    <div class="stat-label">Điểm đến</div>
                </div>
                <div class="stat">
                    <div class="stat-number">AI</div>
                    <div class="stat-label">Thông minh</div>
                </div>
                <div class="stat">
                    <div class="stat-number">24/7</div>
                    <div class="stat-label">Hỗ trợ</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# QUICK ACTIONS - GIỮ NGUYÊN NHƯ CŨ
# -------------------------
def render_quick_actions():
    st.markdown("### 🚀 Hành động nhanh")
    
    # Sử dụng HTML/CSS để tạo layout grid cho quick actions
    st.markdown("""
    <div class="quick-actions">
        <button class="quick-action-btn" onclick="setQuickAction('weather')">
            <div class="emoji">🌤️</div>
            <div class="text">Thời tiết</div>
        </button>
        <button class="quick-action-btn" onclick="setQuickAction('food')">
            <div class="emoji">🍜</div>
            <div class="text">Đặc sản</div>
        </button>
        <button class="quick-action-btn" onclick="setQuickAction('map')">
            <div class="emoji">🗺️</div>
            <div class="text">Bản đồ</div>
        </button>
        <button class="quick-action-btn" onclick="setQuickAction('suggest')">
            <div class="emoji">💡</div>
            <div class="text">Gợi ý</div>
        </button>
    </div>
    
    <script>
    function setQuickAction(action) {
        const actions = {
            'weather': 'Thời tiết hiện tại ở đây',
            'food': 'Đặc sản địa phương', 
            'map': 'Hiển thị bản đồ',
            'suggest': 'Gợi ý lịch trình'
        };
        if (actions[action]) {
            // This would need to be handled via Streamlit's JavaScript integration
            // For now, we'll use a workaround with session state
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: actions[action]
            }, '*');
        }
    }
    </script>
    """, unsafe_allow_html=True)
    
    # Tạo các nút Streamlit thông thường như cũ
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🌤️\nThời tiết", use_container_width=True, help="Xem thời tiết hiện tại"):
            st.session_state.user_input = "Thời tiết hiện tại ở đây"
            st.rerun()
    
    with col2:
        if st.button("🍜\nĐặc sản", use_container_width=True, help="Xem đặc sản địa phương"):
            st.session_state.user_input = "Đặc sản địa phương"
            st.rerun()
    
    with col3:
        if st.button("🗺️\nBản đồ", use_container_width=True, help="Xem bản đồ du lịch"):
            st.session_state.user_input = "Hiển thị bản đồ"
            st.rerun()
    
    with col4:
        if st.button("💡\nGợi ý", use_container_width=True, help="Nhận gợi ý lịch trình"):
            st.session_state.user_input = "Gợi ý lịch trình"
            st.rerun()

# -------------------------
# DB LOGGING (SQLite)
# -------------------------
DB_PATH = "travel_chatbot_logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_input TEXT,
            city TEXT,
            start_date TEXT,
            end_date TEXT,
            intent TEXT,
            rag_used BOOLEAN DEFAULT 0,
            sources_count INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_interaction(user_input, city=None, start_date=None, end_date=None, intent=None, rag_used=False, sources_count=0):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO interactions (timestamp, user_input, city, start_date, end_date, intent, rag_used, sources_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), user_input, city,
          start_date.isoformat() if start_date else None,
          end_date.isoformat() if end_date else None,
          intent, rag_used, sources_count))
    conn.commit()
    conn.close()

# -------------------------
# EMBEDDING LOCAL FUNCTION
# -------------------------
def get_embedding_local(text):
    if embedding_model is None:
        return None
    try:
        if not text or not isinstance(text, str):
            return None
        embedding = embedding_model.encode(text)
        if hasattr(embedding, 'tolist'):
            return embedding.tolist()
        return list(embedding)
    except Exception as e:
        print(f"[WARN] Local embedding failed: {e}")
        return None

# -------------------------
# CHROMA INITIALIZATION
# -------------------------
def safe_get_collection(client, name, expected_dim=384):
    try:
        col = None
        try:
            col = client.get_collection(name)
        except Exception:
            try:
                col = client.get_or_create_collection(name=name)
            except Exception:
                pass
        if col is None:
            try:
                col = client.create_collection(name=name)
            except Exception:
                try:
                    col = client.get_or_create_collection(name=name)
                except Exception:
                    return None
        try:
            test_emb = [0.0] * expected_dim
            try:
                col.query(query_embeddings=[test_emb], n_results=1)
            except Exception as qe:
                msg = str(qe).lower()
                if "dimension" in msg or "expected" in msg:
                    try:
                        print(f"🧹 Deleting collection {name} due to embedding-dimension mismatch ({qe})")
                        client.delete_collection(name=name)
                        col = client.create_collection(name=name)
                    except Exception as de:
                        print(f"[WARN] Failed deleting/recreating collection {name}: {de}")
        except Exception:
            pass
        return col
    except Exception as e:
        print(f"[WARN] safe_get_collection failed for {name}: {e}")
        return None

def init_chroma():
    global chroma_client, chroma_travel_col, chroma_memory_col, chroma_intent_col
    EXPECTED_DIM = 384
    
    persist_dir = os.path.join(os.getcwd(), "chromadb_data")
    try:
        if "chroma_client" not in st.session_state or st.session_state.get("chroma_client") is None:
            st.session_state["chroma_client"] = PersistentClient(path=persist_dir)
            print("[INIT] Created PersistentClient for Chroma at", persist_dir)
        else:
            print("[DEBUG] Reusing existing PersistentClient in session_state")
        chroma_client = st.session_state["chroma_client"]
    except Exception as e:
        print(f"[WARN] Failed to init PersistentClient: {e}")
        try:
            chroma_client = PersistentClient(path=persist_dir)
        except Exception as e2:
            print(f"[ERROR] PersistentClient fallback failed: {e2}")
            return None, None, None, None
    
    try:
        os.makedirs(persist_dir, exist_ok=True)
    except Exception:
        pass

    travel_col = safe_get_collection(chroma_client, "vietnam_travel_v2", expected_dim=EXPECTED_DIM)
    memory_col = safe_get_collection(chroma_client, "chat_memory_v2", expected_dim=EXPECTED_DIM)
    intent_col = safe_get_collection(chroma_client, "intent_bank_v2", expected_dim=EXPECTED_DIM)

    print("✅ Chroma collections ready:", 
          f"travel={'OK' if travel_col else 'NO'}, memory={'OK' if memory_col else 'NO'}, intent={'OK' if intent_col else 'NO'}")
    return chroma_client, travel_col, memory_col, intent_col

try:
    chroma_client, chroma_travel_col, chroma_memory_col, chroma_intent_col = init_chroma()
except Exception as e:
    chroma_client = chroma_travel_col = chroma_memory_col = chroma_intent_col = None
    print(f"[WARN] init_chroma() failed: {e}")

# -------------------------
# UTILITY FUNCTIONS (giữ nguyên từ file gốc)
# -------------------------
def extract_days_from_text(user_text, start_date=None, end_date=None):
    if start_date and end_date:
        try:
            delta = (end_date - start_date).days + 1
            return max(delta, 1)
        except Exception:
            pass
    m = re.search(r"(\d+)\s*(ngày|day|days|tuần|week|weeks)", user_text, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        unit = m.group(2).lower()
        if "tuần" in unit or "week" in unit:
            return num * 7
        return num
    if client:
        try:
            prompt = f"""
Bạn là một bộ phân tích ngữ nghĩa tiếng Việt & tiếng Anh.
Xác định người dùng muốn nói bao nhiêu ngày trong câu sau, nếu không có thì mặc định 3:
Trả về JSON: {{"days": <số nguyên>}}
Câu: "{user_text}"
"""
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
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

geolocator = Nominatim(user_agent="travel_chatbot_app")

def geocode_city(city_name):
    try:
        loc = geolocator.geocode(city_name, country_codes="VN", timeout=10)
        if loc:
            return loc.latitude, loc.longitude, loc.address
        return None, None, None
    except Exception:
        return None, None, None

def show_map(lat, lon, zoom=8, title=""):
    if lat is None or lon is None:
        st.info("Không có dữ liệu toạ độ để hiển thị bản đồ.")
        return

    lat, lon = float(lat), float(lon)
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

def get_weather_forecast(city_name, start_date=None, end_date=None, user_text=None):
    if not OPENWEATHERMAP_API_KEY:
        return "⚠️ Thiếu OpenWeatherMap API Key."
    
    if start_date is None or end_date is None:
        today = datetime.now().date()
        start_date = datetime.combine(today, datetime.min.time())
        end_date = datetime.combine(today + timedelta(days=3), datetime.min.time())
    
    try:
        def _fetch_weather(city):
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHERMAP_API_KEY}&lang=vi&units=metric"
            response = requests.get(url, timeout=8)
            return response.json()
        data = _fetch_weather(city_name)
        if data.get("cod") != "200" and user_text:
            ai_city = resolve_city_via_ai(user_text)
            if ai_city and ai_city.lower() != city_name.lower():
                data = _fetch_weather(f"{ai_city},VN")
                city_name = ai_city
        if data.get("cod") != "200":
            return f"❌ Không tìm thấy thông tin dự báo thời tiết cho địa điểm: **{city_name}**."
        forecast_text = f"🌤 **Dự báo thời tiết cho {city_name}:**\n"
        if start_date and end_date:
            current = start_date
            while current <= end_date:
                date_str = current.strftime("%Y-%m-%d")
                day_forecasts = [f for f in data['list'] if f['dt_txt'].startswith(date_str)]
                if not day_forecasts:
                    forecast_text += f"\n📅 {current.strftime('%d/%m/%Y')}: Không có dữ liệu dự báo.\n"
                else:
                    temps = [f['main']['temp'] for f in day_forecasts]
                    desc = day_forecasts[0]['weather'][0]['description']
                    forecast_text += (
                        f"\n📅 {current.strftime('%d/%m/%Y')} - {desc.capitalize()}\n"
                        f"🌡 Nhiệt độ trung bình: {sum(temps)/len(temps):.1f}°C\n"
                    )
                current += timedelta(days=1)
        else:
            first_forecast = data['list'][0]
            desc = first_forecast['weather'][0]['description'].capitalize()
            temp = first_forecast['main']['temp']
            forecast_text += f"- Hiện tại: {desc}, {temp}°C\n"
        return forecast_text
    except Exception as e:
        return f"⚠️ Lỗi khi lấy dữ liệu thời tiết: {e}"

def resolve_city_via_ai(user_text):
    if not client:
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
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
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

def get_pixabay_image(query, per_page=3):
    if not PIXABAY_API_KEY:
        return None
    try:
        url = "https://pixabay.com/api/"
        params = {
            "key": PIXABAY_API_KEY,
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

def get_city_image(city):
    if not city:
        return None
    queries = [
        f"{city} Vietnam landscape",
        f"{city} Vietnam city",
        f"{city} Vietnam travel",
        "Vietnam travel landscape"
    ]
    for q in queries:
        img = get_pixabay_image(q)
        if img:
            return img
    return "https://via.placeholder.com/1200x800?text=No+Image"

def get_food_images(food_list):
    images = []
    for food in food_list[:5]:
        query = f"{food} Vietnam food"
        img_url = get_pixabay_image(query)
        if not img_url:
            img_url = "https://via.placeholder.com/400x300?text=No+Image"
        images.append({"name": food, "image": img_url})
    return images

def get_restaurants_google(city, api_key, limit=5):
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

def get_local_restaurants(city, limit=5):
    try:
        df = pd.read_csv("data/restaurants_vn.csv")
        df_city = df[df["city"].str.lower().str.contains(str(city).lower(), na=False)]
        if df_city.empty:
            return []
        return df_city.head(limit).to_dict("records")
    except Exception:
        return []

def get_restaurants(city, limit=5):
    if not city:
        return []
    if GOOGLE_PLACES_KEY:
        data = get_restaurants_google(city, GOOGLE_PLACES_KEY, limit)
        if data and not data[0].get("error"):
            return data
    return get_local_restaurants(city, limit)

def re_split_foods(s):
    for sep in [",", "|", ";"]:
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s.strip()]

def get_local_foods(city):
    try:
        df = pd.read_csv("data/vietnam_foods.csv", dtype=str)
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

def get_foods_via_gpt(city, max_items=5):
    if not client:
        return []
    try:
        prompt = (
            f"You are an expert on Vietnamese cuisine.\n"
            f"List up to {max_items} iconic or must-try dishes from the city/region '{city}'.\n"
            "Return only a comma-separated list of dish names (no extra text)."
        )
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role":"system","content":prompt}],
            max_tokens=150,
            temperature=0.5
        )
        text = response.choices[0].message.content.strip()
        items = [t.strip() for t in text.split(",") if t.strip()]
        return items[:max_items]
    except Exception:
        return []

def get_local_foods_with_fallback(city):
    foods = get_local_foods(city)
    if not foods:
        foods = get_foods_via_gpt(city)
    return foods

def estimate_cost(city, days=3, people=1, style="trung bình"):
    mapping = {"tiết kiệm": 400000, "trung bình": 800000, "cao cấp": 2000000}
    per_day = mapping.get(style, 800000)
    total = per_day * days * people
    return f"💸 Chi phí ước tính: khoảng {total:,} VNĐ cho {people} người, {days} ngày."

def suggest_local_food(city):
    return f"🍜 Gõ 'Đặc sản {city}' để nhận danh sách món ăn nổi bật."

def suggest_events(city):
    return f"🎉 Sự kiện ở {city}: lễ hội địa phương, chợ đêm, hội chợ ẩm thực (tuỳ mùa)."

def extract_city_and_dates(user_text):
    if not client:
        return None, None, None
    try:
        prompt = f"""
You are a multilingual travel information extractor.
Extract 'city','start_date','end_date' (YYYY-MM-DD). If only one date is provided, set both to that date.
Return JSON only.
Message: "{user_text}"
"""
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role":"system","content":prompt}],
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
            dt = datetime.strptime(d, "%Y-%m-%d")
            return dt
        start_dt = _parse(s)
        end_dt = _parse(e)
        if start_dt and not end_dt:
            end_dt = start_dt
        return city, start_dt, end_dt
    except Exception:
        return None, None, None

# -------------------------
# RAG FUNCTIONS
# -------------------------
def rag_query_top_k(user_text, k=5):
    if chroma_travel_col is None or client is None:
        return [], ""
    emb = get_embedding_local(user_text)
    if emb is None:
        return [], ""
    try:
        res = chroma_travel_col.query(query_embeddings=[emb], n_results=k, include=["documents","metadatas","distances"])
        docs = []
        
        docs_texts = res.get("documents", [[]])
        if docs_texts and isinstance(docs_texts, list):
            docs_texts = docs_texts[0] if docs_texts and isinstance(docs_texts[0], list) else docs_texts
            
        metadatas = res.get("metadatas", [[]])
        if metadatas and isinstance(metadatas, list):
            metadatas = metadatas[0] if metadatas and isinstance(metadatas[0], list) else metadatas
            
        ids = res.get("ids", [[]])
        if ids and isinstance(ids, list):
            ids = ids[0] if ids and isinstance(ids[0], list) else ids
            
        distances = res.get("distances", [[]])
        if distances and isinstance(distances, list):
            distances = distances[0] if distances and isinstance(distances[0], list) else distances

        docs_texts = docs_texts or []
        metadatas = metadatas or [{}] * len(docs_texts)
        ids = ids or [f"doc_{i}" for i in range(len(docs_texts))]
        distances = distances or [None] * len(docs_texts)

        for i, txt in enumerate(docs_texts):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc_id = ids[i] if i < len(ids) else f"doc_{i}"
            distance = distances[i] if i < len(distances) else None
            
            docs.append({
                "id": doc_id,
                "text": txt,
                "metadata": meta,
                "distance": distance
            })
            
        context_parts = []
        for d in docs:
            src = d["metadata"].get("source", "") if isinstance(d.get("metadata"), dict) else ""
            context_parts.append(f"[src:{d['id']}{('|' + src) if src else ''}] {d['text'][:1200]}")
        context = "\n\n".join(context_parts)
        st.session_state["last_rag_docs"] = docs
        return docs, context
    except Exception as e:
        print(f"[WARN] chroma query error: {e}")
        return [], ""

def add_to_memory_collection(text, role="user", city=None, extra_meta=None):
    if chroma_memory_col is None or client is None:
        return
    try:
        emb = get_embedding_local(text)
        doc_id = f"mem_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
        meta = {"role": role, "city": city or "", "timestamp": datetime.utcnow().isoformat()}
        if extra_meta and isinstance(extra_meta, dict):
            meta.update(extra_meta)
        try:
            chroma_memory_col.add(documents=[text], metadatas=[meta], ids=[doc_id], embeddings=[emb])
        except TypeError:
            chroma_memory_col.add(documents=[text], metadatas=[meta], ids=[doc_id])
    except Exception as e:
        print(f"[WARN] add to memory failed: {e}")

def recall_recent_memories(user_text, k=5):
    if chroma_memory_col is None or client is None:
        return []
    emb = get_embedding_local(user_text)
    if emb is None:
        return []
    try:
        res = chroma_memory_col.query(query_embeddings=[emb], n_results=k, include=["documents","metadatas","distances"])
        items = []
        
        docs_texts = res.get("documents", [[]])
        if docs_texts and isinstance(docs_texts, list):
            docs_texts = docs_texts[0] if docs_texts and isinstance(docs_texts[0], list) else docs_texts
        
        metadatas = res.get("metadatas", [[]])
        if metadatas and isinstance(metadatas, list):
            metadatas = metadatas[0] if metadatas and isinstance(metadatas[0], list) else metadatas
            
        ids = res.get("ids", [[]])
        if ids and isinstance(ids, list):
            ids = ids[0] if ids and isinstance(ids[0], list) else ids
            
        distances = res.get("distances", [[]])
        if distances and isinstance(distances, list):
            distances = distances[0] if distances and isinstance(distances[0], list) else distances

        docs_texts = docs_texts or []
        metadatas = metadatas or [{}] * len(docs_texts)
        ids = ids or [f"mem_{i}" for i in range(len(docs_texts))]
        distances = distances or [None] * len(docs_texts)

        for i, t in enumerate(docs_texts):
            meta = metadatas[i] if i < len(metadatas) else {}
            item_id = ids[i] if i < len(ids) else f"mem_{i}"
            distance = distances[i] if i < len(distances) else None
            
            items.append({
                "id": item_id, 
                "text": t, 
                "meta": meta, 
                "distance": distance
            })
        return items
    except Exception as e:
        print(f"[WARN] recall error: {e}")
        return []

def get_intent_via_chroma(user_text, threshold=0.2):
    if chroma_intent_col is None or client is None:
        return None
    emb = get_embedding_local(user_text)
    if emb is None:
        return None
    try:
        res = chroma_intent_col.query(query_embeddings=[emb], n_results=1, include=["metadatas","distances"])
        
        distances = res.get("distances", [[]])
        if distances and isinstance(distances, list):
            distances = distances[0] if distances and isinstance(distances[0], list) else distances
            
        metadatas = res.get("metadatas", [[]])
        if metadatas and isinstance(metadatas, list):
            metadatas = metadatas[0] if metadatas and isinstance(metadatas[0], list) else metadatas

        if distances and len(distances) > 0 and distances[0] is not None and distances[0] < threshold:
            meta = metadatas[0] if metadatas and len(metadatas) > 0 else {}
            return meta.get("intent") if isinstance(meta, dict) else None
    except Exception as e:
        print(f"[WARN] intent chroma error: {e}")
    return None

def recommend_similar_trips(city, k=3):
    if chroma_memory_col is None:
        return []
    emb = get_embedding_local(city)
    if emb is None:
        return []
    try:
        res = chroma_memory_col.query(query_embeddings=[emb], n_results=10, include=["documents","metadatas","distances"])
        
        docs_texts = res.get("documents", [[]])
        if docs_texts and isinstance(docs_texts, list):
            docs_texts = docs_texts[0] if docs_texts and isinstance(docs_texts[0], list) else docs_texts
            
        metadatas = res.get("metadatas", [[]])
        if metadatas and isinstance(metadatas, list):
            metadatas = metadatas[0] if metadatas and isinstance(metadatas[0], list) else metadatas
            
        ids = res.get("ids", [[]])
        if ids and isinstance(ids, list):
            ids = ids[0] if ids and isinstance(ids[0], list) else ids

        recommendations = []
        for i, m in enumerate(metadatas):
            rec_city = m.get("city") if isinstance(m, dict) else None
            if rec_city and rec_city.lower() != city.lower() and rec_city not in [r.get("city") for r in recommendations]:
                doc = docs_texts[i] if i < len(docs_texts) else ""
                rec_id = ids[i] if i < len(ids) else None
                recommendations.append({"city": rec_city, "meta": m, "doc": doc, "id": rec_id})
            if len(recommendations) >= k:
                break
        return recommendations
    except Exception as e:
        print(f"[WARN] recommend error: {e}")
        return []

def preload_intents():
    if chroma_intent_col is None or client is None:
        return
    try:
        samples = [
            ("Thời tiết ở {city} tuần tới?", {"intent":"weather_query"}),
            ("Lịch trình 3 ngày ở {city}", {"intent":"itinerary_request"}),
            ("Đặc sản {city}", {"intent":"food_query"}),
            ("Gợi ý nhà hàng ở {city}", {"intent":"restaurant_query"})
        ]
        docs = []
        metas = []
        ids = []
        for i, (t, meta) in enumerate(samples):
            docs.append(t)
            metas.append(meta)
            ids.append(f"intent_sample_{i}")
        chroma_intent_col.add(documents=docs, metadatas=metas, ids=ids)
    except Exception as e:
        print(f"[WARN] preload intents: {e}")

try:
    preload_intents()
except Exception:
    pass

def seed_vietnam_travel_from_csv(path="data/vietnam_travel_docs.csv"):
    if chroma_travel_col is None:
        print("Chroma travel collection not ready")
        return False
    if not os.path.exists(path):
        print("Seed file not found:", path)
        return False
    try:
        docs = []
        metas = []
        ids = []
        import csv
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text") or row.get("description") or ""
                docs.append(text)
                metas.append({"title": row.get("title",""), "city": row.get("city",""), "source": row.get("source","")})
                ids.append(row.get("id") or f"doc_{uuid.uuid4().hex[:8]}")
        chroma_travel_col.add(documents=docs, metadatas=metas, ids=ids)
        print(f"Seeded {len(docs)} docs to vietnam_travel")
        return True
    except Exception as e:
        print("Seed error:", e)
        return False

# -------------------------
# VOICE HELPERS
# -------------------------
def detect_audio_type_header(b):
    if len(b) < 12:
        return None
    if b[0:4] == b'RIFF' and b[8:12] == b'WAVE':
        return 'wav'
    if b[0:4] == b'fLaC':
        return 'flac'
    if b[0:4] == b'OggS':
        return 'ogg'
    if b[0:4] == b'\x1A\x45\xDF\xA3':
        return 'webm'
    if b[0:3] == b'ID3' or b[0] == 0xFF:
        return 'mp3'
    return None

def write_temp_file_and_convert_to_wav(audio_bytes):
    header = audio_bytes[:64]
    atype = detect_audio_type_header(header)
    ext_map = {'wav': '.wav', 'ogg': '.ogg', 'webm': '.webm', 'mp3': '.mp3', 'flac': '.flac'}
    ext = ext_map.get(atype, '.webm')
    tmp_dir = tempfile.mkdtemp()
    src_path = os.path.join(tmp_dir, "input" + ext)
    wav_path = os.path.join(tmp_dir, "converted.wav")
    with open(src_path, "wb") as f:
        f.write(audio_bytes)
    if atype == 'wav':
        return src_path
    try:
        audio = AudioSegment.from_file(src_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        try:
            cmd = ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return wav_path
        except Exception as e2:
            raise RuntimeError(f"Không thể chuyển đổi audio sang WAV: {e} | {e2}")

# -------------------------
# TOPIC CLASSIFIER
# -------------------------
def is_travel_related_via_gpt(user_text):
    if not client:
        return True

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

Nếu KHÔNG thuộc những chủ đề trên (ví dụ: lập trình, tài chính, thể thao, học tập...), hãy trả về JSON:
{{"related": false}}

Nếu CÓ liên quan, trả về JSON:
{{"related": true}}

Câu người dùng: "{user_text}"
"""
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
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
        print(f"[WARN] Lỗi phân loại chủ đề: {e}")
    return True

# ========================
# STREAMLIT UI LAYOUT
# ========================

# Render Compact Hero
render_compact_hero()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

# Main Tabs
main_tab, analytics_tab = st.tabs(["💬 Trò chuyện với AI", "📊 Analytics Dashboard"])

# ========================
# MODERN SIDEBAR
# ========================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="logo-title">
            <div class="logo">🌤️</div>
            <div>
                <h2>Mây Lang Thang</h2>
                <p class="sidebar-subtitle">AI Travel Assistant</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Settings Card
    with st.container():
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Cài đặt hiển thị")
        info_options = st.multiselect(
            "Thông tin hiển thị",
            ["Weather", "Food", "Map", "Photos", "Cost", "Events"],
            default=["Weather", "Map", "Food", "Photos"],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Voice Settings Card
    with st.container():
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.subheader("🎙️ Công cụ giọng nói")
        enable_voice = st.checkbox("Bật nhập liệu bằng giọng nói", value=True)
        tts_enable = st.checkbox("🔊 Đọc to phản hồi", value=False)
        asr_lang = st.selectbox("Ngôn ngữ nhận dạng", ["vi-VN", "en-US"], index=0, label_visibility="collapsed")
        tts_lang = st.selectbox("Ngôn ngữ đọc", ["vi", "en"], index=0, label_visibility="collapsed")
        st.caption("Yêu cầu: ffmpeg + internet cho gTTS.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Map Settings Card
    with st.container():
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.subheader("🗺️ Cài đặt bản đồ")
        map_zoom = st.slider("Mức zoom bản đồ", 4, 15, 8, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # System Status Card
    with st.container():
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.subheader("📊 Trạng thái hệ thống")
        
        def status_card(title, status, ok=True):
            icon = "✅" if ok else "⚠️"
            color = "var(--success)" if ok else "var(--warning)"
            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.5rem 0;">
                <span>{title}</span>
                <span style="color: {color}; font-weight: 600;">{icon} {status}</span>
            </div>
            """, unsafe_allow_html=True)
        
        status_card("OpenWeatherMap", "Connected" if OPENWEATHERMAP_API_KEY else "Disabled", bool(OPENWEATHERMAP_API_KEY))
        status_card("Pixabay Images", "Connected" if PIXABAY_API_KEY else "Disabled", bool(PIXABAY_API_KEY))
        status_card("ChromaDB RAG", "Connected" if chroma_client else "Disabled", chroma_client is not None)
        status_card("Embedding Model", "Loaded" if embedding_model else "Failed", embedding_model is not None)
        status_card("OpenAI API", "Connected" if client else "Disabled", client is not None)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Management Card
    with st.container():
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.subheader("🔄 Quản lý dữ liệu")
        if st.button("🔄 Tải dữ liệu du lịch", use_container_width=True):
            try:
                if seed_vietnam_travel_from_csv("data/vietnam_travel_docs.csv"):
                    st.success("✅ Đã tải dữ liệu thành công!")
                else:
                    st.error("❌ Lỗi khi tải dữ liệu")
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
        
        # Auto seed check
        try:
            if chroma_travel_col and chroma_travel_col.count() == 0:
                if seed_vietnam_travel_from_csv("data/vietnam_travel_docs.csv"):
                    st.success("✅ Đã tự động tải dữ liệu du lịch")
                else:
                    st.warning("⚠️ Chưa có dữ liệu du lịch. Vui lòng tải thủ công.")
        except Exception as e:
            st.warning(f"⚠️ Chưa tải được dữ liệu: {e}")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("🍜 Food AI: Dữ liệu địa phương + GPT")
    st.caption("Version: v2.0 • Modern UI")
    st.markdown("By [Mây Lang Thang Team](https://#) ❤️")

# ========================
# MAIN CHAT INTERFACE
# ========================
with main_tab:
    # Quick Actions - Giữ nguyên như cũ
    render_quick_actions()
    
    # Voice Input
    voice_text = None
    if enable_voice:
        audio = mic_recorder(
            start_prompt="🎙️ Nhấn để nói",
            stop_prompt="⏹️ Dừng thu âm",
            just_once=True,
            key="rec_chat"
        )
        if audio:
            st.info("Đã nhận dữ liệu âm thanh, đang xử lý...")
            try:
                wav_file = write_temp_file_and_convert_to_wav(audio["bytes"])
            except Exception as e:
                wav_file = None
                st.error(f"Không thể chuyển đổi audio: {e}")
            if wav_file:
                r = sr.Recognizer()
                try:
                    with sr.AudioFile(wav_file) as source:
                        audio_data = r.record(source)
                        voice_text = r.recognize_google(audio_data, language=asr_lang)
                        st.success(f"🗣️ Bạn nói: {voice_text}")
                        st.session_state.user_input = voice_text
                        st.rerun()
                except sr.UnknownValueError:
                    st.error("Không thể nhận diện giọng nói (UnknownValueError).")
                except Exception as e:
                    st.error(f"Lỗi nhận diện: {e}")

    # Chat History
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧭"):
                st.markdown(f"<div class='user-message'>{msg['content']}</div>", unsafe_allow_html=True)
        elif msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"<div class='assistant-message'>{msg['content']}</div>", unsafe_allow_html=True)

    # Chat Input
    user_input = st.chat_input("Nhập câu hỏi về du lịch Việt Nam...")
    if "user_input" in st.session_state and st.session_state.user_input:
        user_input = st.session_state.user_input
        del st.session_state.user_input

    if user_input:
        with st.chat_message("user", avatar="🧭"):
            st.markdown(f"<div class='user-message'>{user_input}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Topic classification
        try:
            if not is_travel_related_via_gpt(user_input):
                msg = "Xin lỗi 😅, tôi chỉ hỗ trợ các câu hỏi liên quan đến **du lịch Việt Nam**, như thời tiết, địa điểm, món ăn, lịch trình..."
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                try:
                    add_to_memory_collection(user_input, role="user")
                    add_to_memory_collection(msg, role="assistant")
                except Exception:
                    pass
                st.stop()
        except Exception:
            pass

        city_guess, start_date, end_date = extract_city_and_dates(user_input)
        days = extract_days_from_text(user_input, start_date, end_date)
        
        # Reset tracking variables
        rag_used = False
        sources_count = 0
        intent_used = None
        memory_used = False

        if start_date:
            today = datetime.now().date()
            max_forecast_date = today + timedelta(days=5)
            if start_date.date() > max_forecast_date:
                st.warning(f"⚠️ Lưu ý: OpenWeather chỉ cung cấp dự báo ~5 ngày. Bạn yêu cầu bắt đầu {start_date.strftime('%d/%m/%Y')}.")

        # Quick info blocks
        blocks = []
        if city_guess and "Weather" in info_options:
            blocks.append(get_weather_forecast(city_guess, start_date, end_date, user_input))
        if city_guess and "Cost" in info_options:
            blocks.append(estimate_cost(city_guess, days=days))
        if city_guess and "Events" in info_options:
            blocks.append(suggest_events(city_guess))

        for b in blocks:
            with st.chat_message("assistant", avatar="🤖"):
                if isinstance(b, str):
                    st.markdown(b.replace("\\n", "\n"))
                else:
                    st.write(b)

        # AI Response Generation
        with st.spinner("🤖 AI đang phân tích..."):
            try:
                progress_text = "Đang xử lý yêu cầu của bạn..."
                progress_bar = st.progress(0, text=progress_text)
                for percent_complete in range(0, 101, 20):
                    time.sleep(0.05)
                    progress_bar.progress(percent_complete, text=progress_text)
                progress_bar.empty()

                assistant_text = ""
                if client:
                    # RAG + Intent + Memory enhanced generation
                    try:
                        detected_intent = get_intent_via_chroma(user_input, threshold=0.18)
                        if detected_intent:
                            intent_used = detected_intent
                            if detected_intent == "weather_query" and city_guess:
                                assistant_text = get_weather_forecast(city_guess, start_date, end_date, user_input)
                                assistant_text += f"\n\n( Nguồn: OpenWeatherMap )"
                            elif detected_intent == "food_query" and city_guess:
                                foods = get_local_foods_with_fallback(city_guess)
                                assistant_text = "Đặc sản nổi bật:\n" + "\n".join([f"- {f}" for f in foods]) if foods else "Không tìm thấy đặc sản trong DB."
                            elif detected_intent == "itinerary_request" and city_guess:
                                days_local = extract_days_from_text(user_input, start_date, end_date)
                                assistant_text = f"Lịch trình gợi ý cho {city_guess}, {days_local} ngày:\n1) Ngày 1: ...\n2) Ngày 2: ...\n3) Ngày 3: ..."
                            else:
                                detected_intent = None
                                intent_used = None
                        if not detected_intent:
                            docs, rag_context = rag_query_top_k(user_input, k=5)
                            sources_count = len(docs)
                            recent_mem = recall_recent_memories(user_input, k=3)
                            memory_used = len(recent_mem) > 0
                            recall_text = ""
                            if recent_mem:
                                recall_parts = []
                                for m in recent_mem:
                                    ts = m.get("meta", {}).get("timestamp", "")
                                    role = m.get("meta", {}).get("role", "")
                                    recall_parts.append(f"[mem:{m.get('id')}] ({role} {ts}) {m.get('text')[:400]}")
                                recall_text = "\n\n".join(recall_parts)

                            augmentation = "\n\n--- Thông tin tham khảo nội bộ (trích dẫn): ---\n"
                            if rag_context:
                                rag_used = True
                                augmentation += rag_context + "\n\n"
                            if recall_text:
                                augmentation += "\n--- Nhớ gần đây ---\n" + recall_text + "\n\n"
                            augmentation += "--- Khi trả lời, nếu dùng thông tin từ phần trên hãy đánh dấu nguồn như [src:ID] hoặc [mem:ID]. ---\n"

                            temp_messages = [{"role":"system", "content": system_prompt + "\n\n" + augmentation}]
                            temp_messages.extend(st.session_state.messages[-12:])
                            response = client.chat.completions.create(
                                model=DEPLOYMENT_NAME,
                                messages=temp_messages,
                                max_tokens=900,
                                temperature=0.7
                            )
                            assistant_text = response.choices[0].message.content.strip()

                        # Save to memory
                        try:
                            add_to_memory_collection(user_input, role="user", city=city_guess)
                            add_to_memory_collection(assistant_text, role="assistant", city=city_guess)
                        except Exception:
                            pass

                    except Exception as e:
                        assistant_text = f"⚠️ Lỗi khi tạo phản hồi: {e}"
                else:
                    assistant_text = f"Xin chào! Tôi có thể giúp bạn với thông tin về {city_guess or 'địa điểm'} — thử hỏi 'Thời tiết', 'Đặc sản', hoặc 'Lịch trình 3 ngày'."

                if not assistant_text.endswith(("🌤️❤️", "😊", "🌸", "🌴", "✨")):
                    assistant_text += "\n\nChúc bạn có chuyến đi vui vẻ 🌤️❤️"

                st.session_state.messages.append({"role": "assistant", "content": assistant_text})

                # Display response with typing effect
                with st.chat_message("assistant", avatar="🤖"):
                    placeholder = st.empty()
                    display_text = ""
                    for char in assistant_text:
                        display_text += char
                        placeholder.markdown(f"<div class='assistant-message'>{display_text}▌</div>", unsafe_allow_html=True)
                        time.sleep(0.01)
                    time.sleep(0.3)
                    placeholder.empty()
                    
                    # Final display with source highlighting
                    display_text_processed = re.sub(r'(\[src:[^\]]+\])', r'**\1**', assistant_text)
                    display_text_processed = re.sub(r'(\[mem:[^\]]+\])', r'**\1**', display_text_processed)
                    st.markdown(f"<div class='assistant-message'>{display_text_processed}</div>", unsafe_allow_html=True)
                    
                    # Source references
                    if rag_used or intent_used or memory_used:
                        st.markdown("---")
                        st.subheader("🔍 Nguồn tham khảo")
                        
                        if rag_used and sources_count > 0:
                            st.markdown(f'<div class="source-badge source-chroma">📚 <b>ChromaDB RAG</b>: Sử dụng {sources_count} tài liệu từ cơ sở tri thức du lịch</div>', unsafe_allow_html=True)
                        
                        if intent_used:
                            st.markdown(f'<div class="source-badge source-intent">🎯 <b>Intent Matching</b>: Phát hiện intent "{intent_used}" từ ChromaDB</div>', unsafe_allow_html=True)
                        
                        if memory_used:
                            st.markdown(f'<div class="source-badge source-memory">💭 <b>Memory Recall</b>: Tham khảo hội thoại trước đó từ ChromaDB</div>', unsafe_allow_html=True)
                        
                        # Detailed RAG sources
                        if "last_rag_docs" in st.session_state and st.session_state["last_rag_docs"]:
                            sources = st.session_state["last_rag_docs"]
                            with st.expander(f"📖 Chi tiết {len(sources)} tài liệu tham khảo"):
                                for i, src in enumerate(sources, 1):
                                    meta = src.get("metadata", {}) or {}
                                    title = meta.get("title", "Không có tiêu đề")
                                    city = meta.get("city", "")
                                    srcname = meta.get("source", "Nội bộ")
                                    distance = src.get("distance")
                                    
                                    st.markdown(f"**{i}. {title}**")
                                    if city:
                                        st.caption(f"📍 {city}")
                                    if srcname:
                                        st.caption(f"📚 Nguồn: {srcname}")
                                    if distance is not None:
                                        st.caption(f"📊 Độ tương đồng: {1 - distance:.3f}")
                                    st.markdown(f"*{src['text'][:200]}...*")
                                    st.markdown("---")

                # Log interaction
                log_interaction(user_input, city_guess, start_date, end_date, intent_used, rag_used, sources_count)

                # TTS
                if tts_enable:
                    try:
                        tts = gTTS(assistant_text, lang=tts_lang)
                        bio = io.BytesIO()
                        tts.write_to_fp(bio)
                        bio.seek(0)
                        b64 = base64.b64encode(bio.read()).decode()
                        st.markdown(
                            f'<div class="audio-wrapper"><audio autoplay controls><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio></div>',
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.warning(f"Không thể tạo audio TTS: {e}")

                st.balloons()
            except Exception as e:
                st.error(f"⚠️ Lỗi khi xử lý yêu cầu: {e}")

        # Additional Info Display
        lat, lon, addr = (None, None, None)
        if city_guess:
            lat, lon, addr = geocode_city(city_guess)
        
        cols = st.columns([2, 3])
        with cols[0]:
            if "Map" in info_options and city_guess:
                st.markdown("### 🗺️ Bản đồ")
                show_map(lat, lon, zoom=map_zoom, title=addr or city_guess)
            
            if "Photos" in info_options and city_guess:
                st.markdown("### 🏞️ Hình ảnh")
                img = get_city_image(city_guess)
                if img:
                    st.image(img, caption=f"{city_guess}", use_container_width=True)
                else:
                    st.info("Không tìm thấy ảnh minh họa.")
        
        with cols[1]:
            if "Food" in info_options and city_guess:
                st.markdown("### 🍽️ Ẩm thực & Nhà hàng")
                
                # Local Foods
                foods = get_local_foods_with_fallback(city_guess)
                if foods:
                    st.markdown("#### 🥘 Đặc sản nổi bật")
                    food_images = get_food_images(foods)
                    if food_images:
                        img_cols = st.columns(min(len(food_images), 4))
                        for i, item in enumerate(food_images):
                            with img_cols[i % len(img_cols)]:
                                if item["image"]:
                                    st.image(item["image"], caption=item["name"], use_container_width=True)
                                else:
                                    st.write(f"- {item['name']}")
                    else:
                        for food in foods:
                            st.write(f"- {food}")
                else:
                    st.info("Không tìm thấy món đặc trưng.")
                
                # Restaurants
                st.markdown("#### 🍴 Nhà hàng gợi ý")
                restaurants = get_restaurants(city_guess, limit=5)
                if restaurants:
                    for r in restaurants:
                        if isinstance(r, dict) and r.get("error"):
                            st.write(f"⚠️ {r.get('error')}")
                        else:
                            name = r.get("name") or r.get("place_name") or str(r)
                            rating = r.get("rating", "")
                            addr_text = r.get("address", r.get("formatted_address", ""))
                            maps_url = r.get("maps_url", "")
                            st.markdown(f"- **{name}** {f'• ⭐ {rating}' if rating else ''}  \n  {addr_text}  " + (f"[Bản đồ]({maps_url})" if maps_url else ""))
                else:
                    st.info("Không có dữ liệu nhà hàng.")

# ========================
# ANALYTICS TAB
# ========================
with analytics_tab:
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.header("📊 Analytics Dashboard")
    
    # Metrics Row
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(interactions)")
        columns = [col[1] for col in cur.fetchall()]
        
        df_logs = pd.read_sql("SELECT * FROM interactions ORDER BY timestamp DESC LIMIT 1000", conn)
        conn.close()
        
        if not df_logs.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_interactions = len(df_logs)
                st.metric("Tổng tương tác", f"{total_interactions:,}")
            
            with col2:
                if 'rag_used' in columns:
                    rag_count = df_logs['rag_used'].sum()
                    rag_rate = (rag_count / total_interactions) * 100
                    st.metric("Tỷ lệ sử dụng RAG", f"{rag_rate:.1f}%")
                else:
                    st.metric("Tỷ lệ RAG", "N/A")
            
            with col3:
                if 'sources_count' in columns:
                    avg_sources = df_logs['sources_count'].mean()
                    st.metric("TB nguồn/RAG", f"{avg_sources:.1f}")
                else:
                    st.metric("TB nguồn", "N/A")
            
            with col4:
                unique_cities = df_logs['city'].nunique()
                st.metric("Điểm đến", unique_cities)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Hoạt động theo ngày")
                if not df_logs.empty:
                    df_logs['timestamp_dt'] = pd.to_datetime(df_logs['timestamp'])
                    df_logs['date'] = df_logs['timestamp_dt'].dt.date
                    daily_counts = df_logs.groupby('date').size().reset_index(name='count')
                    
                    fig_daily = px.line(daily_counts, x='date', y='count', 
                                      title='Số lượng truy vấn theo ngày',
                                      line_shape='spline')
                    fig_daily.update_traces(line=dict(color='#2563eb', width=3))
                    fig_daily.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_daily, use_container_width=True)
            
            with col2:
                st.markdown("#### 📍 Top điểm đến")
                if not df_logs.empty:
                    city_counts = df_logs['city'].value_counts().head(10).reset_index()
                    city_counts.columns = ['city', 'count']
                    
                    fig_cities = px.bar(city_counts, x='count', y='city', 
                                      orientation='h',
                                      title='Thành phố được hỏi nhiều nhất',
                                      color='count',
                                      color_continuous_scale='blues')
                    fig_cities.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_cities, use_container_width=True)
            
            # RAG Usage Chart
            if 'rag_used' in columns:
                st.markdown("#### 🔍 Sử dụng RAG")
                rag_daily = df_logs.groupby('date').agg({
                    'rag_used': 'sum',
                    'timestamp': 'count'
                }).reset_index()
                rag_daily.columns = ['date', 'rag_queries', 'total_queries']
                rag_daily['non_rag_queries'] = rag_daily['total_queries'] - rag_daily['rag_queries']
                
                fig_rag = px.area(rag_daily, x='date', y=['rag_queries', 'non_rag_queries'],
                                title='Phân bổ sử dụng RAG theo thời gian',
                                color_discrete_map={'rag_queries': '#2563eb', 'non_rag_queries': '#94a3b8'})
                fig_rag.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_rag, use_container_width=True)
            
            # Data Table
            st.markdown("#### 📋 Dữ liệu chi tiết")
            display_cols = ['timestamp', 'user_input', 'city']
            if 'rag_used' in columns:
                display_cols.extend(['rag_used', 'sources_count'])
            if 'intent' in columns:
                display_cols.append('intent')
                
            display_df = df_logs[display_cols].head(100)
            st.dataframe(display_df, use_container_width=True)
            
        else:
            st.info("Chưa có dữ liệu thống kê.")
            
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu analytics: {e}")
    
    # Data Management
    st.markdown("#### 🗃️ Quản lý dữ liệu")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Làm mới dữ liệu", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("📥 Xuất dữ liệu", use_container_width=True):
            try:
                conn = sqlite3.connect(DB_PATH)
                df_export = pd.read_sql("SELECT * FROM interactions", conn)
                conn.close()
                
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="📥 Tải file CSV",
                    data=csv,
                    file_name=f"travel_bot_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Lỗi khi xuất dữ liệu: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# FOOTER
# ========================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: var(--text-muted); padding: 1rem 0;">
        <p>💡 <strong>Mẹo sử dụng:</strong> Bạn có thể yêu cầu cụ thể như 'Lịch trình 3 ngày ở Hội An', 'Đặc sản Sapa', hoặc 'Thời tiết Đà Nẵng tuần tới'.</p>
        <p style="margin-top: 0.5rem;">🌤️ <strong>Mây Lang Thang</strong> - AI Travel Assistant • Phiên bản 2.0</p>
    </div>
    """, 
    unsafe_allow_html=True
)