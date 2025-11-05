# Travel_Chat_Bot_Enhanced_VOICE.py
# =================================
# Mở rộng: RAG (ChromaDB) + long-term memory + intent quick-match + recommendations
# Giữ lại toàn bộ chức năng gốc (voice, TTS, weather, map, foods, restaurants...)
#
# Yêu cầu:
#   pip install streamlit-mic-recorder SpeechRecognition pydub gTTS chromadb openai geopy pandas pydeck plotly
#   Cài ffmpeg cho pydub
#

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

# === VOICE imports (mới) ===
import io
import base64
import tempfile
import subprocess
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from pydub import AudioSegment   # yêu cầu ffmpeg
from gtts import gTTS

# === RAG / Chroma imports ===
from chromadb import PersistentClient
# NOTE: Replaced Client->PersistentClient for Chroma v1.2+# Chroma v1.2+ no longer uses chromadb.config.Settings
import uuid

# === Ensure single persistent Chroma client in Streamlit session ===
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chromadb_data")
# We'll create chroma_client lazily in init_chroma(), but ensure session key exists placeholder
# Actual ChromaClient will be created inside init_chroma using this CHROMA_PERSIST_DIR.

# -------------------------
st.set_page_config(page_title="[Mây Lang Thang] - Travel Assistant (Voice + RAG)", layout="wide", page_icon="🤖")

# Global CSS + UI tweaks
st.markdown(
    """
    <style>
    :root{
      --primary:#2b4c7e;
      --accent:#e7f3ff;
      --muted:#f2f6fa;
    }
    body {
      background: linear-gradient(90deg, #f8fbff 0%, #eef5fa 100%);
      font-family: 'Segoe UI', Roboto, Arial, sans-serif;
    }
    .stApp > header {visibility: hidden;}
    h1, h2, h3 { color: var(--primary); }
    .sidebar-card { background-color:#f1f9ff; padding:10px; border-radius:10px; margin-bottom:8px;}
    .user-message { background: #f2f2f2; padding:10px; border-radius:12px; }
    .assistant-message { background: #e7f3ff; padding:10px; border-radius:12px; }
    .pill-btn { border-radius:999px !important; background:#e3f2fd !important; color:var(--primary) !important; padding:6px 12px; border: none; }
    .status-ok { background:#d4edda; padding:8px; border-radius:8px; }
    .status-bad { background:#f8d7da; padding:8px; border-radius:8px; }
    .small-muted { color: #6b7280; font-size:12px; }
    .logo-title { display:flex; align-items:center; gap:10px; }
    .logo-title h1 { margin:0; }
    .assistant-bubble {
      background-color: #e7f3ff;
      padding: 12px 16px;
      border-radius: 15px;
      margin-bottom: 6px;
    }
    .user-message {
      background-color: #f2f2f2;
      padding: 12px 16px;
      border-radius: 15px;
      margin-bottom: 6px;
    }
    /* HERO */
    .hero {
      position: relative;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 8px 30px rgba(43,76,126,0.12);
      margin-bottom: 18px;
    }
    .hero__bg {
      width: 100%;
      height: 320px;
      object-fit: cover;
      filter: brightness(0.65) saturate(1.05);
    }
    .hero__overlay {
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .hero__card {
      background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.08));
      backdrop-filter: blur(6px);
      border-radius: 12px;
      padding: 18px;
      width: 100%;
      max-width: 980px;
      color: white;
    }
    .hero__title { font-size: 28px; font-weight:700; margin:0 0 6px 0; color: #fff; }
    .hero__subtitle { margin:0 0 12px 0; color: #f0f6ff; }
    .hero__cta { display:flex; gap:8px; align-items:center; }
    @media (max-width: 768px) {
      .hero__bg { height: 220px; }
      .hero__title { font-size: 20px; }
    }
    .audio-wrapper {margin-top: 6px;}
    </style>
    """, unsafe_allow_html=True
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
OPENAI_API_KEY_EMBEDDING = st.secrets["OPENAI_API_KEY_EMBEDDING"]

# Chroma persistent dir (tùy chọn)
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chromadb_data")

# Initialize OpenAI client (using openai Python SDK modern interface)
if OPENAI_API_KEY:
    client = openai.OpenAI(base_url=OPENAI_ENDPOINT, api_key=OPENAI_API_KEY)
else:
    client = None

# --- Separate Embedding client (dùng key riêng) ---
try:
    OPENAI_API_KEY_EMBEDDING = st.secrets.get("OPENAI_API_KEY_EMBEDDING", None) if hasattr(st, 'secrets') else os.getenv("OPENAI_API_KEY_EMBEDDING", None)
except Exception:
    OPENAI_API_KEY_EMBEDDING = os.getenv("OPENAI_API_KEY_EMBEDDING", None)

if OPENAI_API_KEY_EMBEDDING:
    embedding_client = openai.OpenAI(base_url=OPENAI_ENDPOINT, api_key=OPENAI_API_KEY_EMBEDDING)
    # print("OPENAI_API_KEY_EMBEDDING: " + OPENAI_API_KEY_EMBEDDING)
else:
    embedding_client = client  # fallback nếu chưa có key riêng

ChatBotName = "[Mây Lang Thang]"  # display name
system_prompt = """
Bạn là Hướng dẫn viên du lịch ảo Alex - người kể chuyện, am hiểu văn hóa, lịch sử, ẩm thực và thời tiết Việt Nam.
Luôn đưa ra thông tin hữu ích, gợi ý lịch trình, món ăn, chi phí, thời gian lý tưởng, sự kiện và góc chụp ảnh.
"""

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
            intent TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_interaction(user_input, city=None, start_date=None, end_date=None, intent=None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO interactions (timestamp, user_input, city, start_date, end_date, intent)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), user_input, city,
          start_date.isoformat() if start_date else None,
          end_date.isoformat() if end_date else None,
          intent))
    conn.commit()
    conn.close()

# -------------------------
# CHROMA (RAG + Memory + Intent) INIT
# -------------------------


def safe_get_collection(client, name, expected_dim=1536):
    """
    Create or get a Chroma collection safely.
    Auto recreate collection if dimension mismatch or corruption occurs.
    Compatible with Chroma v1.2+ PersistentClient.
    """
    try:
        col = None
        try:
            col = client.get_collection(name)
        except Exception:
            # if get_collection not available, try get_or_create_collection
            try:
                col = client.get_or_create_collection(name=name)
            except Exception:
                pass
        if col is None:
            try:
                col = client.create_collection(name=name)
            except Exception:
                # fallback to get_or_create
                try:
                    col = client.get_or_create_collection(name=name)
                except Exception:
                    return None
        # Probe dimension by attempting a harmless query with expected_dim; delete if mismatch
        try:
            test_emb = [0.0] * expected_dim
            try:
                # newer chroma expects embeddings param name 'query_embeddings' or 'embeddings' depending on version
                col.query(query_embeddings=[test_emb], n_results=1)
            except Exception as qe:
                msg = str(qe).lower()
                if "dimension" in msg or "expected" in msg:
                    try:
                        print(f"🧹 Deleting collection {name} due to embedding-dimension mismatch ({qe})")
                        client.delete_collection(name=name)
                        # recreate
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
    """
    Initialize Chroma persistent client and ensure collections exist.
    Compatible with Chroma v1.2+ using PersistentClient.
    """
    global chroma_client, chroma_travel_col, chroma_memory_col, chroma_intent_col
    EXPECTED_DIM = 1536
    # persist dir (project-local)
    persist_dir = os.path.join(os.getcwd(), "chromadb_data")
    try:
        # Use PersistentClient for new Chroma versions
        from chromadb import PersistentClient
        # create or reuse client in st.session_state
        if "chroma_client" not in st.session_state or st.session_state.get("chroma_client") is None:
            st.session_state["chroma_client"] = PersistentClient(path=persist_dir)
            print("[INIT] Created PersistentClient for Chroma at", persist_dir)
        else:
            print("[DEBUG] Reusing existing PersistentClient in session_state")
        chroma_client = st.session_state["chroma_client"]
    except Exception as e:
        print(f"[WARN] Failed to init PersistentClient: {e}")
        try:
            # fallback: try to use PersistentClient directly without session_state
            chroma_client = PersistentClient(path=persist_dir)
        except Exception as e2:
            print(f"[ERROR] PersistentClient fallback failed: {e2}")
            return None, None, None, None
    # --- Force scan and delete any 384-dimension collections ---
    try:
        for col in chroma_client.list_collections():
            cname = getattr(col, "name", str(col))
            try:
                emb = [0.0] * EXPECTED_DIM
                col.query(query_embeddings=[emb], n_results=1)
            except Exception as qe:
                if "384" in str(qe):
                    print(f"🧹 Force deleting old collection {cname} (384-dim detected)")
                    try:
                        chroma_client.delete_collection(name=cname)
                    except Exception as de:
                        print(f"[WARN] Could not delete old collection {cname}: {de}")
    except Exception as e:
        print(f"[WARN] Force cleanup skipped: {e}")

    except Exception as e:
        print(f"[WARN] Failed to init PersistentClient: {e}")
        try:
            # fallback: try to use PersistentClient directly without session_state
            chroma_client = PersistentClient(path=persist_dir)
        except Exception as e2:
            print(f"[ERROR] PersistentClient fallback failed: {e2}")
            return None, None, None, None

    # ensure persist dir exists
    try:
        os.makedirs(persist_dir, exist_ok=True)
    except Exception:
        pass
    try:
        st.sidebar.markdown(f"🧠 **Chroma DB:** `{os.path.abspath(persist_dir)}`")
    except Exception:
        pass

    # Attempt to cleanup any existing collections with mismatched dimension
    try:
        existing = []
        try:
            existing = chroma_client.list_collections()
        except Exception:
            try:
                existing = [c.name for c in chroma_client.get_collections()]  # older API
            except Exception:
                existing = []
        names = []
        for item in existing:
            if isinstance(item, dict) and item.get("name"):
                names.append(item["name"])
            elif isinstance(item, str):
                names.append(item)
            else:
                try:
                    n = getattr(item, "name", None)
                    if n:
                        names.append(n)
                except Exception:
                    pass
        candidate_names = set(names)
        candidate_names.update(["vietnam_travel", "chat_memory", "intent_bank",
                                "vietnam_travel_v2", "chat_memory_v2", "intent_bank_v2"])
        for cname in list(candidate_names):
            try:
                col = chroma_client.get_collection(name=cname)
            except Exception:
                # some APIs use different signature
                try:
                    col = chroma_client.get_collection(cname)
                except Exception:
                    continue
            try:
                test_emb = [0.0] * EXPECTED_DIM
                try:
                    col.query(query_embeddings=[test_emb], n_results=1)
                except Exception as qe:
                    msg = str(qe).lower()
                    if "dimension" in msg or "expected" in msg:
                        try:
                            print(f"🧹 Deleting collection {cname} due to embedding-dimension mismatch ({qe})")
                            chroma_client.delete_collection(name=cname)
                        except Exception as de:
                            print(f"[WARN] Failed deleting collection {cname}: {de}")
            except Exception:
                # fallback: inspect attribute
                try:
                    col_dim = getattr(col, "dimension", None)
                    if col_dim and col_dim != EXPECTED_DIM:
                        try:
                            chroma_client.delete_collection(name=cname)
                        except Exception:
                            pass
                except Exception:
                    pass
    except Exception as e:
        print(f"[WARN] Error when scanning collections: {e}")

    # create/get our important collections
    travel_col = safe_get_collection(chroma_client, "vietnam_travel_v2", expected_dim=EXPECTED_DIM)
    memory_col = safe_get_collection(chroma_client, "chat_memory_v2", expected_dim=EXPECTED_DIM)
    intent_col = safe_get_collection(chroma_client, "intent_bank_v2", expected_dim=EXPECTED_DIM)

    print("✅ Chroma collections ready (or created):", 
          f"travel={'OK' if travel_col else 'NO'}, memory={'OK' if memory_col else 'NO'}, intent={'OK' if intent_col else 'NO'}")
    print(f"✅ Chroma initialized: travel={bool(travel_col)}, memory={bool(memory_col)}, intent={bool(intent_col)}")
    return chroma_client, travel_col, memory_col, intent_col

    # Scan existing collections and delete those with mismatched embedding dimension
    try:
        existing = []
        try:
            existing = chroma_client.list_collections()
        except Exception:
            existing = []

        names = []
        for item in existing:
            if isinstance(item, dict) and item.get("name"):
                names.append(item["name"])
            elif isinstance(item, str):
                names.append(item)
            else:
                try:
                    n = getattr(item, "name", None)
                    if n:
                        names.append(n)
                except Exception:
                    pass

        # include legacy names we care about
        candidate_names = set(names)
        candidate_names.update(["vietnam_travel", "chat_memory", "intent_bank",
                                "vietnam_travel_v2", "chat_memory_v2", "intent_bank_v2"])

        for cname in list(candidate_names):
            try:
                col = chroma_client.get_collection(cname)
            except Exception:
                continue
            try:
                # probe with a test embedding of EXPECTED_DIM
                test_emb = [0.0] * EXPECTED_DIM
                try:
                    col.query(query_embeddings=[test_emb], n_results=1, include=["documents"])
                    # if no exception -> likely dimension matches
                except Exception as qe:
                    msg = str(qe).lower()
                    if "dimension" in msg or "expected" in msg or "384" in msg:
                        print(f"🧹 Deleting collection {cname} due to embedding-dimension mismatch (error: {qe})")
                        try:
                            chroma_client.delete_collection(cname)
                        except Exception as de:
                            print(f"[WARN] Failed deleting collection {cname}: {de}")
                    else:
                        # unknown error - ignore
                        pass
            except Exception:
                try:
                    col_dim = getattr(col, "dimension", None)
                    if col_dim and col_dim != EXPECTED_DIM:
                        print(f"🧹 Deleting collection {cname} (col.dimension={col_dim} != {EXPECTED_DIM})")
                        try:
                            chroma_client.delete_collection(cname)
                        except Exception as de:
                            print(f"[WARN] Failed deleting collection {cname}: {de}")
                except Exception:
                    pass
    except Exception as e:
        print(f"[WARN] Error when scanning/deleting old collections: {e}")

    # create/get collections safely
    travel_col = safe_get_collection(chroma_client, "vietnam_travel_v2", expected_dim=EXPECTED_DIM)
    memory_col = safe_get_collection(chroma_client, "chat_memory_v2", expected_dim=EXPECTED_DIM)
    intent_col = safe_get_collection(chroma_client, "intent_bank_v2", expected_dim=EXPECTED_DIM)

    print("✅ Chroma collections ready (or created):",
          f"travel={'OK' if travel_col else 'NO'}, memory={'OK' if memory_col else 'NO'}, intent={'OK' if intent_col else 'NO'}")
    print(f"✅ Chroma initialized: travel={bool(travel_col)}, memory={bool(memory_col)}, intent={bool(intent_col)}")
    return chroma_client, travel_col, memory_col, intent_col

# --- Initialize Chroma client and collections once (and store in session_state/global) ---
try:
    chroma_client, chroma_travel_col, chroma_memory_col, chroma_intent_col = init_chroma()
except Exception as e:
    chroma_client = chroma_travel_col = chroma_memory_col = chroma_intent_col = None
    print(f"[WARN] init_chroma() failed: {e}")

# Safe preload intents if function exists
if 'preload_intents' in globals() and callable(globals()['preload_intents']):
    try:
        preload_intents()
    except Exception as e:
        print(f"[WARN] preload_intents failed: {e}")

# -------------------------
# UTILITIES: days extraction (original logic)
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

# -------------------------
# GEOCODING & MAPS
# -------------------------
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

# -------------------------
# WEATHER (OpenWeatherMap) with AI fallback on location
# -------------------------
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

# -------------------------
# PIXABAY IMAGE FUNCTIONS
# -------------------------
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

# -------------------------
# RESTAURANTS HYBRID (Google Places + CSV fallback)
# -------------------------
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

# -------------------------
# FOOD AI ASSISTANT (CSV + GPT fallback)
# -------------------------
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

# -------------------------
# SUGGESTIONS / COST / PHOTOSPOTS
# -------------------------
def estimate_cost(city, days=3, people=1, style="trung bình"):
    mapping = {"tiết kiệm": 400000, "trung bình": 800000, "cao cấp": 2000000}
    per_day = mapping.get(style, 800000)
    total = per_day * days * people
    return f"💸 Chi phí ước tính: khoảng {total:,} VNĐ cho {people} người, {days} ngày."

def suggest_local_food(city):
    return f"🍜 Gõ 'Đặc sản {city}' để nhận danh sách món ăn nổi bật."

def suggest_events(city):
    return f"🎉 Sự kiện ở {city}: lễ hội địa phương, chợ đêm, hội chợ ẩm thực (tuỳ mùa)."

def suggest_photospots(city):
    return f"📸 Gợi ý check-in: trung tâm lịch sử, bờ sông/biển, quán cà phê có view đẹp."

# -------------------------
# BILINGUAL CITY & DATE EXTRACTION
# -------------------------
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
# RAG / Chroma helper functions
# -------------------------
def get_embedding_openai(text):
    """
    Trả về embedding list bằng model text-embedding-3-small.
    Sử dụng embedding_client (có key riêng).
    """
    if not embedding_client:
        return None
    try:
        emb_resp = embedding_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return emb_resp.data[0].embedding
    except Exception as e:
        print(f"[WARN] embedding failed: {e}")
        return None

def rag_query_top_k(user_text, k=5):
    """
    Lấy top-k đoạn văn từ collection vietnam_travel bằng embedding.
    Trả về list dict và context string.
    """
    if chroma_travel_col is None or client is None:
        return [], ""
    emb = get_embedding_openai(user_text)
    if emb is None:
        return [], ""
    try:
        res = chroma_travel_col.query(query_embeddings=[emb], n_results=k, include=["documents","metadatas","distances"])
        docs = []
        # robust parsing for different chroma versions
        try:
            docs_texts = res["documents"][0]
            metadatas = res.get("metadatas",[[]])[0] if res.get("metadatas") else [None]*len(docs_texts)
            ids = res.get([[]])[0] if res.get("ids") else [None]*len(docs_texts)
            distances = res.get("distances",[[]])[0] if res.get("distances") else [None]*len(docs_texts)
            for i, txt in enumerate(docs_texts):
                docs.append({"id": ids[i] or str(uuid.uuid4()),
                             "text": txt,
                             "metadata": metadatas[i] or {},
                             "distance": distances[i] if i < len(distances) else None})
        except Exception:
            # fallback if different shape
            try:
                docs_texts = res["documents"]
                for i, txt in enumerate(docs_texts):
                    md = res.get("metadatas",[{}])[i] if res.get("metadatas") else {}
                    _id = res.get([None])[i] if res.get("ids") else str(uuid.uuid4())
                    dist = res.get("distances",[None])[i] if res.get("distances") else None
                    docs.append({"id": _id, "text": txt, "metadata": md, "distance": dist})
            except Exception:
                pass
        context_parts = []
        for d in docs:
            src = d["metadata"].get("source", "") if isinstance(d.get("metadata"), dict) else ""
            context_parts.append(f"[src:{d['id']}{('|' + src) if src else ''}] {d['text'][:1200]}")
        context = "\n\n".join(context_parts)
        st.session_state["last_rag_docs"] = docs  # lưu nguồn vào session
        return docs, context
    except Exception as e:
        print(f"[WARN] chroma query error: {e}")
        return [], ""

def add_to_memory_collection(text, role="user", city=None, extra_meta=None):
    """
    Lưu embedding + text vào collection chat_memory.
    """
    if chroma_memory_col is None or client is None:
        return
    try:
        emb = get_embedding_openai(text)
        doc_id = f"mem_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
        meta = {"role": role, "city": city or "", "timestamp": datetime.utcnow().isoformat()}
        if extra_meta and isinstance(extra_meta, dict):
            meta.update(extra_meta)
        # Some chroma versions accept embeddings param, others compute embedding on add
        try:
            chroma_memory_col.add(documents=[text], metadatas=[meta], ids=[doc_id], embeddings=[emb])
        except TypeError:
            # fallback without embeddings param
            chroma_memory_col.add(documents=[text], metadatas=[meta], ids=[doc_id])
    except Exception as e:
        print(f"[WARN] add to memory failed: {e}")

def recall_recent_memories(user_text, k=5):
    """
    Truy vấn chat_memory bằng embedding user_text để lấy các đoạn hội thoại gần nhất.
    """
    if chroma_memory_col is None or client is None:
        return []
    emb = get_embedding_openai(user_text)
    if emb is None:
        return []
    try:
        res = chroma_memory_col.query(query_embeddings=[emb], n_results=k, include=["documents","metadatas","distances"])
        items = []
        docs_texts = res.get("documents", [[]])[0] if res.get("documents") else []
        metadatas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
        ids = res.get( [[]])[0] if res.get("ids") else []
        distances = res.get("distances", [[]])[0] if res.get("distances") else [None]*len(docs_texts)
        for i, t in enumerate(docs_texts):
            items.append({"id": ids[i] if i < len(ids) else None, "text": t, "meta": metadatas[i] if i < len(metadatas) else {}, "distance": distances[i] if i < len(distances) else None})
        return items
    except Exception as e:
        print(f"[WARN] recall error: {e}")
        return []

def get_intent_via_chroma(user_text, threshold=0.2):
    """
    Truy vấn intent_bank tìm intent gần nhất. Nếu distance < threshold => trả về intent id.
    """
    if chroma_intent_col is None or client is None:
        return None
    emb = get_embedding_openai(user_text)
    if emb is None:
        return None
    try:
        res = chroma_intent_col.query(query_embeddings=[emb], n_results=1, include=["metadatas","distances"])
        distances = res.get("distances", [[]])[0] if res.get("distances") else []
        metadatas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
        if distances and distances[0] is not None and distances[0] < threshold:
            return metadatas[0].get("intent") if isinstance(metadatas[0], dict) else None
    except Exception as e:
        print(f"[WARN] intent chroma error: {e}")
    return None

def recommend_similar_trips(city, k=3):
    """
    Tìm trong chat_memory các trips tương tự dựa trên city (hoặc mô tả).
    """
    if chroma_memory_col is None:
        return []
    emb = get_embedding_openai(city)
    if emb is None:
        return []
    try:
        res = chroma_memory_col.query(query_embeddings=[emb], n_results=10, include=["documents","metadatas","distances"])
        docs = res.get("documents", [[]])[0] if res.get("documents") else []
        metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
        ids = res.get( [[]])[0] if res.get("ids") else []
        recommendations = []
        for i, m in enumerate(metas):
            rec_city = m.get("city") if isinstance(m, dict) else None
            if rec_city and rec_city.lower() != city.lower() and rec_city not in [r.get("city") for r in recommendations]:
                recommendations.append({"city": rec_city, "meta": m, "doc": docs[i] if i < len(docs) else "", "id": ids[i] if i < len(ids) else None})
            if len(recommendations) >= k:
                break
        return recommendations
    except Exception as e:
        print(f"[WARN] recommend error: {e}")
        return []

# preload intents samples
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

# -------------------------
# HERO / HEADER SECTION
# -------------------------
def render_hero_section(default_city_hint="Hội An, Đà Nẵng, Hà Nội..."):
    with st.form(key='hero_search_form', clear_on_submit=False):
        cols = st.columns([3,2,1,1])
        dest = cols[0].text_input("Điểm đến", placeholder=default_city_hint)
        dates = cols[1].date_input("Ngày (bắt đầu / kết thúc)", [])
        people = cols[2].selectbox("Người", [1,2,3,4,5,6], index=0)
        style = cols[3].selectbox("Mức chi", ["trung bình", "tiết kiệm", "cao cấp"], index=0)
        submitted = st.form_submit_button("Gợi ý nhanh", use_container_width=True)
        if submitted:
            if len(dates) == 2:
                s = dates[0].strftime("%Y-%m-%d")
                e = dates[1].strftime("%Y-%m-%d")
                q = f"Lịch trình { ( (dates[1]-dates[0]).days +1 ) } ngày ở {dest} từ {s} đến {e}"
            elif len(dates) == 1:
                s = dates[0].strftime("%Y-%m-%d")
                q = f"Lịch trình 1 ngày ở {dest} vào {s}"
            else:
                q = f"Lịch trình 3 ngày ở {dest}"
            q += f" • người: {people} • mức: {style}"
            st.session_state.user_input = q
            st.rerun()

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
# TOPIC CLASSIFIER (OpenAI) - dùng để kiểm tra nếu bạn muốn reject non-travel
# -------------------------
def is_travel_related_via_gpt(user_text):
    """
    Dùng OpenAI để xác định xem câu hỏi có liên quan đến du lịch không.
    Trả về True nếu liên quan, False nếu không.
    """
    if not client:
        return True  # nếu không có API key thì cho qua luôn

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
    return True  # fallback

# -------------------------
# STREAMLIT UI LAYOUT
# -------------------------
render_hero_section()
main_tab, analytics_tab = st.tabs(["💬 Chatbot Du lịch", "📊 Thống kê truy vấn"])

with st.sidebar:
    st.markdown("<div class='logo-title'><img src='https://img.icons8.com/emoji/48/000000/cloud-emoji.png'/> <h2>Mây Lang Thang</h2></div>", unsafe_allow_html=True)
    st.header("Cài đặt")
    info_options = st.multiselect("Hiển thị thông tin",
                                  ["Weather", "Food", "Map", "Photos", "Cost", "Events"],
                                  default=["Weather", "Map","Food", "Photos"])
    st.markdown("---")
    st.write("Chọn mức zoom bản đồ:")
    map_zoom = st.slider("Zoom (4 = xa, 15 = gần)", 4, 15, 8)
    st.markdown("---")
    st.subheader("🎙️ Voice")
    enable_voice = st.checkbox("Bật nhập liệu bằng giọng nói", value=True)
    asr_lang = st.selectbox("Ngôn ngữ nhận dạng", ["vi-VN", "en-US"], index=0)
    tts_enable = st.checkbox("🔊 Đọc to phản hồi", value=False)
    tts_lang = st.selectbox("Ngôn ngữ TTS", ["vi", "en"], index=0)
    st.caption("Yêu cầu: ffmpeg + internet cho gTTS.")
    st.markdown("---")
    def status_card(title, ok=True):
        cls = "status-ok" if ok else "status-bad"
        icon = "✅" if ok else "⚠️"
        st.markdown(f"<div class='{cls}'>{icon} {title}</div>", unsafe_allow_html=True)
    status_card("OpenWeatherMap", bool(OPENWEATHERMAP_API_KEY))
    status_card("Google Places", bool(GOOGLE_PLACES_KEY))
    status_card("Pixabay", bool(PIXABAY_API_KEY))
    st.markdown("---")
    st.caption("🍜 Food AI: CSV local dữ liệu + GPT fallback")
    st.markdown("Version: v1.3 + Voice + RAG")

# initialize session messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

with main_tab:
    today = datetime.now().date()
    if "quicksearch" in st.session_state:
        qs = st.session_state.quicksearch
        city_qs = qs["city"]; start_qs = qs["start"]; end_qs = qs["end"]
        people_qs = qs["people"]; style_qs = qs["style"]
        st.markdown(f"### ✈️ Gợi ý cho chuyến đi {city_qs} ({start_qs} – {end_qs})")
        weather_qs = get_weather_forecast(city_qs, start_qs, end_qs)
        cost_qs = estimate_cost(city_qs, (end_qs - start_qs).days + 1, people_qs, style_qs)
        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"**{weather_qs}**")
            st.markdown(f"**{cost_qs}**")
        with colB:
            img = get_city_image(city_qs)
            if img:
                st.image(img, caption=f"🏞️ {city_qs}", use_container_width=True)
            lat, lon, addr = geocode_city(city_qs)
            if lat and lon:
                show_map(lat, lon, zoom=map_zoom, title=addr or city_qs)
        st.markdown("---")

    # === VOICE INPUT BAR ===
    voice_text = None
    if enable_voice:
        audio = mic_recorder(
            start_prompt="🎙️ [Chat voice] Nói để nhập câu hỏi",
            stop_prompt="✋Dừng nhận diện giọng nói",
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

    # --- Hiển thị lại toàn bộ lịch sử cũ ---
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧭"):
                st.markdown(f"<div class='user-message'>{msg['content']}</div>", unsafe_allow_html=True)
        elif msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"<div class='assistant-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

    # Chat input (gõ phím)
    user_input = st.chat_input("Mời bạn đặt câu hỏi:")
    if "user_input" in st.session_state and st.session_state.user_input:
        user_input = st.session_state.user_input

    if user_input:
        with st.chat_message("user", avatar="🧭"):
            st.markdown(f"<div class='user-message'>{user_input}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Optional: reject non-travel topics via GPT classifier
        try:
            if not is_travel_related_via_gpt(user_input):
                msg = "Xin lỗi 😅, tôi chỉ hỗ trợ các câu hỏi liên quan đến **du lịch Việt Nam**, như thời tiết, địa điểm, món ăn, lịch trình..."
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                # add to memory (optional)
                try:
                    add_to_memory_collection(user_input, role="user")
                    add_to_memory_collection(msg, role="assistant")
                except Exception:
                    pass
                st.stop()
        except Exception:
            # nếu classifier lỗi thì tiếp tục bình thường
            pass

        city_guess, start_date, end_date = extract_city_and_dates(user_input)
        days = extract_days_from_text(user_input, start_date, end_date)
        log_interaction(user_input, city_guess, start_date, end_date)

        if start_date:
            today = datetime.now().date()
            max_forecast_date = today + timedelta(days=5)
            if start_date.date() > max_forecast_date:
                st.warning(f"⚠️ Lưu ý: OpenWeather chỉ cung cấp dự báo ~5 ngày. Bạn yêu cầu bắt đầu {start_date.strftime('%d/%m/%Y')}.")

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

        with st.spinner("⏳ Đang soạn phản hồi..."):
            try:
                progress_text = "AI đang phân tích dữ liệu du lịch..."
                progress_bar = st.progress(0, text=progress_text)
                for percent_complete in range(0, 101, 20):
                    time.sleep(0.08)
                    progress_bar.progress(percent_complete, text=progress_text)
                progress_bar.empty()

                assistant_text = ""
                if client:
                    # --- RAG + Intent + Memory enhanced generation ---
                    try:
                        detected_intent = get_intent_via_chroma(user_input, threshold=0.18)
                        if detected_intent:
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
                                # Unknown or not handled intent -> fallback to full generation
                                detected_intent = None
                        if not detected_intent:
                            docs, rag_context = rag_query_top_k(user_input, k=5)
                            recent_mem = recall_recent_memories(user_input, k=3)
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
                                augmentation += rag_context + "\n\n"
                            if recall_text:
                                augmentation += "\n--- Nhớ gần đây ---\n" + recall_text + "\n\n"
                            augmentation += "--- Khi trả lời, nếu dùng thông tin từ phần trên hãy đánh dấu nguồn như [src:ID] hoặc [mem:ID]. ---\n"

                            temp_messages = [{"role":"system", "content": system_prompt + "\n\n" + augmentation}]
                            temp_messages.extend(st.session_state.messages[-12:])  # keep last 12 msgs
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

                with st.chat_message("assistant", avatar="🤖"):
                    placeholder = st.empty()
                    display_text = ""
                    for char in assistant_text:
                        display_text += char
                        placeholder.markdown(display_text + "▌")
                        time.sleep(0.01)
                    time.sleep(0.3)
                    placeholder.empty()
                    with st.container():
                        # highlight citations like [src:...] or [mem:...]
                        display_text_processed = re.sub(r'(\[src:[^\]]+\])', r'**\1**', assistant_text)
                        display_text_processed = re.sub(r'(\[mem:[^\]]+\])', r'**\1**', display_text_processed)
                        st.markdown("<div class='assistant-bubble'>", unsafe_allow_html=True)
                        st.markdown(display_text_processed)
                        st.markdown("</div>", unsafe_allow_html=True)
                        # --- Hiển thị nguồn trích dẫn (RAG metadata) ---
                        if "last_rag_docs" in st.session_state and st.session_state["last_rag_docs"]:
                            sources = st.session_state["last_rag_docs"]
                            st.markdown("##### 📚 Nguồn dữ liệu tham khảo:")
                                # Tạo expander hiển thị danh sách nguồn
                            with st.expander("📚 Nguồn dữ liệu tham khảo"):
                                for src in sources:
                                    meta = src.get("metadata", {}) or {}
                                    title = meta.get("title", "")
                                    city = meta.get("city", "")
                                    srcname = meta.get("source", "")
                                    display_line = f"- **{src['id']}**"
                                    if title:
                                        display_line += f": *{title}*"
                                    if city:
                                        display_line += f" – {city}"
                                    if srcname:
                                        display_line += f" _(nguồn: {srcname})_"
                                    st.markdown(display_line)


                    # === TTS (đọc to phản hồi) ===
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
                st.error(f"⚠️ Lỗi khi gọi OpenAI: {e}")

        lat, lon, addr = (None, None, None)
        if city_guess:
            lat, lon, addr = geocode_city(city_guess)
        cols = st.columns([2, 3])
        with cols[0]:
            if "Map" in info_options:
                show_map(lat, lon, zoom=map_zoom, title=addr or city_guess)
            if "Photos" in info_options:
                img = get_city_image(city_guess)
                if img:
                    st.image(img, caption=f"🏞️ {city_guess}", use_container_width=True)
                else:
                    st.info("Không tìm thấy ảnh minh họa.")
        with cols[1]:
            if "Food" in info_options:
                st.subheader(f"🍽️ Ẩm thực & Nhà hàng tại {city_guess or 'địa điểm'}")
                foods = get_local_foods_with_fallback(city_guess) if city_guess else []
                if foods:
                    st.markdown("#### 🥘 Đặc sản nổi bật")
                    food_images = get_food_images(foods)
                    img_cols = st.columns(min(len(food_images), 4))
                    for i, item in enumerate(food_images):
                        with img_cols[i % len(img_cols)]:
                            if item["image"]:
                                st.image(item["image"], caption=item["name"], use_container_width=True)
                            else:
                                st.write(f"- {item['name']}")
                else:
                    st.info("Không tìm thấy món đặc trưng (CSV/GPT fallback không trả kết quả).")
            if city_guess:
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
                    st.info("Không có dữ liệu nhà hàng (CSV/Google Places fallback).")

with analytics_tab:
    st.header("📊 Thống kê truy vấn (gần đây)")
    with st.expander("🗑️ Xóa lịch sử truy vấn"):
        st.warning("⚠️ Thao tác này sẽ xóa toàn bộ lịch sử truy vấn đã lưu trong cơ sở dữ liệu (SQLite). Không thể hoàn tác.")
        confirm_delete = st.checkbox("Tôi hiểu và muốn xóa toàn bộ lịch sử truy vấn", value=False)
        if confirm_delete:
            if st.button("✅ Xác nhận xóa toàn bộ lịch sử"):
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    cur.execute("DELETE FROM interactions")
                    conn.commit()
                    conn.close()
                    st.success("✅ Đã xóa toàn bộ lịch sử truy vấn.")
                except Exception as e:
                    st.error(f"⚠️ Lỗi khi xóa dữ liệu: {e}")
        else:
            st.info("👉 Hãy tick vào ô xác nhận trước khi xóa lịch sử.")
    try:
        conn = sqlite3.connect(DB_PATH)
        df_logs = pd.read_sql("SELECT * FROM interactions ORDER BY timestamp DESC LIMIT 1000", conn)
        conn.close()
        total = int(df_logs.shape[0]) if not df_logs.empty else 0
        st.metric("Tổng tương tác", total)
        if not df_logs.empty:
            df_logs['timestamp_dt'] = pd.to_datetime(df_logs['timestamp'])
            df_logs['date'] = df_logs['timestamp_dt'].dt.date
            series = df_logs.groupby('date').size().reset_index(name='queries')
            fig = px.bar(series, x='date', y='queries', title='📈 Số truy vấn mỗi ngày', color='queries', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
            top_cities = df_logs['city'].fillna("Unknown").value_counts().reset_index()
            top_cities.columns = ['city', 'count']
            if not top_cities.empty:
                fig2 = px.bar(top_cities.head(10), x='city', y='count', title='📍 Top địa điểm được hỏi', color='count', color_continuous_scale='Viridis')
                st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(df_logs[["timestamp", "user_input", "city"]])
        else:
            st.info("Chưa có truy vấn nào được ghi nhận.")
    except Exception as e:
        st.warning(f"Lỗi đọc dữ liệu: {e}")

st.markdown("---")
st.markdown("<div class='small-muted'>Tip: Bạn có thể yêu cầu cụ thể như 'Lịch trình 3 ngày ở Hội An', 'Đặc sản Sapa', hoặc 'Thời tiết Đà Nẵng 2025-10-20 đến 2025-10-22'.</div>", unsafe_allow_html=True)


# -------------------------
# Optional: seeding vietnam_travel collection from CSV
# -------------------------
def seed_vietnam_travel_from_csv(path="data/vietnam_travel_docs.csv"):
    if chroma_travel_col is None:
        print("Chroma travel collection not ready")
        return
    if not os.path.exists(path):
        print("Seed file not found:", path)
        return
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
    except Exception as e:
        print("Seed error:", e)

# You can call seed_vietnam_travel_from_csv() manually in a session if needed.
# Example: seed_vietnam_travel_from_csv("data/vietnam_travel_docs.csv")

