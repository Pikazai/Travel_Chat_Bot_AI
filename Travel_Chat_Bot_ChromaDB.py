# Travel_Chat_Bot_ChromaDB.py
# =======================================
# Ứng dụng Streamlit "Mây Lang Thang" + ChromaDB (RAG + Semantic Cache + Memory)
# - Giữ nguyên trải nghiệm & logic từ bản UI_FULL_with_Hero, bổ sung:
#   * ChromaDB PersistentClient
#   * Collection: travel_kb, answer_cache, conversations
#   * Seed dữ liệu từ CSV (vietnam_foods.csv, restaurants_vn.csv)
#   * RAG: truy xuất ngữ cảnh theo thành phố + loại (food/restaurant/tip)
#   * Semantic Answer Cache: giảm chi phí, tăng tốc
#   * Conversation Memory: nhớ Q/A theo city
#   * Debugger UI trong tab Analytics
#
# Chạy thử:
#   1) pip install -r requirements.txt
#   2) export OPENAI_API_KEY=... (hoặc đặt trong .streamlit/secrets.toml)
#   3) streamlit run Travel_Chat_Bot_ChromaDB.py
#
# Tham khảo kiến trúc & cấu trúc UI gốc từ file của bạn. (see README)
# =======================================

import streamlit as st
import json, os, re, time, sqlite3
from datetime import datetime, timedelta
import pandas as pd
import requests
import pydeck as pdk
import plotly.express as px
from chromadb.utils import embedding_functions

# OpenAI client (modern SDK)
import openai

# Geocoding
from geopy.geocoders import Nominatim

# ---- ChromaDB (optional, graceful fallback) ----
CHROMA_AVAILABLE = True
try:
    import chromadb
except Exception:
    CHROMA_AVAILABLE = False

# -------------------------
# PAGE CONFIG & THEME
# -------------------------
st.set_page_config(page_title="🤖 [Mây Lang Thang] - Travel Assistant + ChromaDB", layout="wide", page_icon="🌴")

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
    </style>
    """, unsafe_allow_html=True
)

# -------------------------
# CONFIG / SECRETS
# -------------------------
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    OPENAI_API_KEY_EMBEDDING = st.secrets["OPENAI_API_KEY_EMBEDDING"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
    OPENAI_API_KEY_EMBEDDING = st.secrets["OPENAI_API_KEY_EMBEDDING"]

OPENAI_ENDPOINT = st.secrets.get("OPENAI_ENDPOINT", "https://api.openai.com/v1") if hasattr(st, 'secrets') else os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
DEPLOYMENT_NAME = st.secrets.get("DEPLOYMENT_NAME", "gpt-4o-mini") if hasattr(st, 'secrets') else os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
OPENWEATHERMAP_API_KEY = st.secrets.get("OPENWEATHERMAP_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("OPENWEATHERMAP_API_KEY", "")
GOOGLE_PLACES_KEY = st.secrets.get("PLACES_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("PLACES_API_KEY", "")
PIXABAY_API_KEY = st.secrets.get("PIXABAY_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("PIXABAY_API_KEY", "")

EMBEDDING_MODEL = st.secrets.get("EMBEDDING_MODEL", "text-embedding-3-small") if hasattr(st, 'secrets') else os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_PATH = st.secrets.get("CHROMA_PATH", "./.chroma") if hasattr(st, 'secrets') else os.getenv("CHROMA_PATH", "./.chroma")

# Initialize OpenAI client (modern)
if OPENAI_API_KEY:
    client = openai.OpenAI(base_url=OPENAI_ENDPOINT, api_key=OPENAI_API_KEY)
else:
    client = None

ChatBotName = "[Mây Lang Thang]"
system_prompt = """
Bạn là Hướng dẫn viên du lịch ảo Alex - người kể chuyện, am hiểu văn hóa, lịch sử, ẩm thực và thời tiết Việt Nam.
Luôn đưa ra thông tin hữu ích, gợi ý lịch trình, món ăn, chi phí, thời gian lý tưởng, sự kiện và góc chụp ảnh.
"""

# -------------------------
# DB LOGGING (SQLite)
# -------------------------
DB_PATH = "travel_chatbot_logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
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
    conn.commit(); conn.close()

init_db()

def log_interaction(user_input, city=None, start_date=None, end_date=None, intent=None):
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("""
        INSERT INTO interactions (timestamp, user_input, city, start_date, end_date, intent)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), user_input, city,
          start_date.isoformat() if start_date else None,
          end_date.isoformat() if end_date else None,
          intent))
    conn.commit(); conn.close()

# -------------------------
# GEOCODING & MAPS
# -------------------------
geolocator = Nominatim(user_agent="travel_chatbot_app")

def geocode_city(city_name):
    try:
        loc = geolocator.geocode(city_name, timeout=10)
        if loc: return loc.latitude, loc.longitude, loc.address
        return None, None, None
    except Exception:
        return None, None, None

def show_map(lat, lon, zoom=8, title=""):
    if lat is None or lon is None:
        st.info("Không có dữ liệu toạ độ để hiển thị bản đồ."); return
    st.write(f"**Vị trí:** {title} ({lat:.5f}, {lon:.5f})")
    view = pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame([{"lat": lat, "lon": lon}]),
        get_position='[lon, lat]',
        get_radius=3000,
        get_fill_color=[220, 40, 40],
        get_opacity=0.9,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"text": title or "Vị trí"}
    )
    st.pydeck_chart(deck)

# -------------------------
# WEATHER (OpenWeatherMap) with AI fallback
# -------------------------
def resolve_city_via_ai(user_text):
    if not client: return None
    try:
        prompt = f"""
Bạn là chuyên gia địa lý du lịch Việt Nam.
Phân tích câu sau để xác định:
1. 'place': địa danh cụ cụ thể (khu du lịch, công viên, đảo, thắng cảnh,...)
2. 'province_or_city': tên tỉnh hoặc thành phố của Việt Nam mà địa danh đó thuộc về.
Nếu không xác định được, trả về null.
Kết quả JSON ví dụ: {{"place": "Phong Nha - Kẻ Bàng", "province_or_city": "Quảng Bình"}}
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
        if start == -1 or end == -1: return None
        data = json.loads(text[start:end+1])
        return data.get("province_or_city")
    except Exception:
        return None

def get_weather_forecast(city_name, start_date=None, end_date=None, user_text=None):
    if not OPENWEATHERMAP_API_KEY:
        return "⚠️ Thiếu OpenWeatherMap API Key."
    try:
        def _fetch(city):
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHERMAP_API_KEY}&lang=vi&units=metric"
            return requests.get(url, timeout=8).json()
        data = _fetch(city_name)
        if data.get("cod") != "200" and user_text:
            ai_city = resolve_city_via_ai(user_text)
            if ai_city and ai_city.lower() != city_name.lower():
                data = _fetch(f"{ai_city},VN"); city_name = ai_city
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
                    forecast_text += f"\n📅 {current.strftime('%d/%m/%Y')} - {desc.capitalize()}\n🌡 Nhiệt độ trung bình: {sum(temps)/len(temps):.1f}°C\n"
                current += timedelta(days=1)
        else:
            first = data['list'][0]
            desc = first['weather'][0]['description'].capitalize()
            temp = first['main']['temp']
            forecast_text += f"- Hiện tại: {desc}, {temp}°C\n"
        return forecast_text
    except Exception as e:
        return f"⚠️ Lỗi khi lấy dữ liệu thời tiết: {e}"

# -------------------------
# PIXABAY IMAGE FUNCTIONS
# -------------------------
def get_pixabay_image(query, per_page=3):
    if not PIXABAY_API_KEY: return None
    try:
        url = "https://pixabay.com/api/"
        params = {"key": PIXABAY_API_KEY, "q": query, "image_type": "photo", "orientation": "horizontal", "safesearch": "true", "per_page": per_page}
        res = requests.get(url, params=params, timeout=8); data = res.json()
        if data.get("hits"):
            return data["hits"][0].get("largeImageURL") or data["hits"][0].get("webformatURL")
        return None
    except Exception:
        return None

def get_city_image(city):
    if not city: return None
    queries = [f"{city} Vietnam landscape", f"{city} Vietnam city", f"{city} Vietnam travel", "Vietnam travel landscape"]
    for q in queries:
        img = get_pixabay_image(q)
        if img: return img
    return "https://via.placeholder.com/1200x800?text=No+Image"

def get_food_images(food_list):
    images = []
    for food in food_list[:5]:
        query = f"{food} Vietnam food"
        img_url = get_pixabay_image(query) or "https://via.placeholder.com/400x300?text=No+Image"
        images.append({"name": food, "image": img_url})
    return images

# -------------------------
# RESTAURANTS HYBRID
# -------------------------
def get_restaurants_google(city, api_key, limit=5):
    try:
        query = f"nhà hàng tại {city}, Việt Nam"
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {"query": query, "key": api_key, "language": "vi"}
        res = requests.get(url, params=params, timeout=10).json()
        if "error_message" in res: return [{"error": res["error_message"]}]
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
        if df_city.empty: return []
        return df_city.head(limit).to_dict("records")
    except Exception:
        return []

def get_restaurants(city, limit=5):
    if not city: return []
    if GOOGLE_PLACES_KEY:
        data = get_restaurants_google(city, GOOGLE_PLACES_KEY, limit)
        if data and not data[0].get("error"): return data
    return get_local_restaurants(city, limit)

# -------------------------
# FOOD AI ASSISTANT
# -------------------------
def re_split_foods(s):
    for sep in [",", "|", ";"]:
        if sep in s: return [p.strip() for p in s.split(sep) if p.strip()]
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
                if pd.notna(foods_cell): return re_split_foods(foods_cell)
            else:
                vals = row0.dropna().tolist()
                if len(vals) > 1: return [v for v in vals[1:]]
    except Exception:
        pass
    return []

def get_foods_via_gpt(city, max_items=5):
    if not client: return []
    try:
        prompt = (
            f"You are an expert on Vietnamese cuisine.\n"
            f"List up to {max_items} iconic or must-try dishes from the city/region '{city}'.\n"
            "Return only a comma-separated list of dish names (no extra text)."
        )
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role":"system","content":prompt}],
            max_tokens=150, temperature=0.5
        )
        text = response.choices[0].message.content.strip()
        items = [t.strip() for t in text.split(",") if t.strip()]
        return items[:max_items]
    except Exception:
        return []

def get_local_foods_with_fallback(city):
    foods = get_local_foods(city)
    if not foods: foods = get_foods_via_gpt(city)
    return foods

# -------------------------
# SUGGESTIONS / COST / PHOTOSPOTS
# -------------------------
def estimate_cost(city, days=3, people=1, style="trung bình"):
    mapping = {"tiết kiệm": 400000, "trung bình": 800000, "cao cấp": 2000000}
    per_day = mapping.get(style, 800000); total = per_day * days * people
    return f"💸 Chi phí ước tính: khoảng {total:,} VNĐ cho {people} người, {days} ngày."

def suggest_events(city): return f"🎉 Sự kiện ở {city}: lễ hội địa phương, chợ đêm, hội chợ ẩm thực (tuỳ mùa)."

# -------------------------
# BILINGUAL CITY & DATE EXTRACTION
# -------------------------
def extract_city_and_dates(user_text):
    if not client: return None, None, None
    try:
        prompt = f"""
You are a multilingual travel information extractor.
The user message may be in Vietnamese or English.
Extract:
1. Destination city (field 'city') - if none, return null
2. Start date (field 'start_date') in YYYY-MM-DD format or null
3. End date (field 'end_date') in YYYY-MM-DD format or null
Rules:
- If user provides only one date, set both start_date and end_date to that date.
- If the user gives day/month without year, assume current year.
- Return valid JSON ONLY, for example:
{{"city":"Hanoi", "start_date":"2025-10-20", "end_date":"2025-10-22"}}
Message: "{user_text}"
"""
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role":"system","content":prompt}],
            max_tokens=200, temperature=0
        )
        content = response.choices[0].message.content.strip()
        start = content.find('{'); end = content.rfind('}')
        if start == -1 or end == -1: return None, None, None
        data = json.loads(content[start:end+1])
        city = data.get("city"); s = data.get("start_date"); e = data.get("end_date")
        def _parse(d):
            if not d: return None
            return datetime.strptime(d, "%Y-%m-%d")
        start_dt = _parse(s); end_dt = _parse(e)
        if start_dt and not end_dt: end_dt = start_dt
        return city, start_dt, end_dt
    except Exception:
        return None, None, None

def extract_days_from_text(user_text, start_date=None, end_date=None):
    if start_date and end_date:
        try:
            delta = (end_date - start_date).days + 1
            return max(delta, 1)
        except Exception:
            pass
    m = re.search(r"(\d+)\s*(ngày|day|days|tuần|week|weeks)", user_text, re.IGNORECASE)
    if m:
        num = int(m.group(1)); unit = m.group(2).lower()
        if "tuần" in unit or "week" in unit: return num * 7
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
                max_tokens=50, temperature=0
            )
            text = response.choices[0].message.content.strip()
            num_match = re.search(r'"days"\s*:\s*(\d+)', text)
            if num_match: return int(num_match.group(1))
        except Exception:
            pass
    return 3

# -------------------------
# === ChromaDB INTEGRATION ===
# -------------------------
@st.cache_resource
def get_chroma():
    if not CHROMA_AVAILABLE:
        return None, None, None, None
    try:
        client_chroma = chromadb.PersistentClient(path=CHROMA_PATH)

        class OpenAIEmbedder:
            def __init__(self, _client, model):
                self.client = _client; self.model = model
            def __call__(self, texts):
                if isinstance(texts, str): texts = [texts]
                try:
                    resp = client.embeddings.create(model=self.model, input=texts)
                    return [d.embedding for d in resp.data]
                except Exception as e:
                    # fallback: zeros
                    return [[0.0]*1536 for _ in texts]

        # embedder = OpenAIEmbedder(client, EMBEDDING_MODEL)
        embedder = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name= OPENAI_API_KEY_EMBEDDING
        )


        kb = client_chroma.get_or_create_collection(
            name="travel_kb",
            embedding_function=embedder,
            metadata={"hnsw:space": "cosine"}
        )
        cache = client_chroma.get_or_create_collection(
            name="answer_cache",
            embedding_function=embedder,
            metadata={"hnsw:space": "cosine"}
        )
        conv = client_chroma.get_or_create_collection(
            name="conversations",
            embedding_function=embedder,
            metadata={"hnsw:space": "cosine"}
        )
        return client_chroma, kb, cache, conv
    except Exception as e:
        st.warning(f"Không thể khởi tạo ChromaDB: {e}")
        return None, None, None, None

chroma_client, chroma_kb, chroma_cache, chroma_conv = get_chroma()
CHROMA_READY = CHROMA_AVAILABLE and (chroma_kb is not None)

def seed_kb_from_csvs(kb):
    if not kb: return 0
    docs, metas, ids = [], [], []
    # foods
    try:
        df_food = pd.read_csv("data/vietnam_foods.csv", dtype=str)
        for _, r in df_food.iterrows():
            city = (r.get("city") or "").strip()
            foods_cell = (r.get("foods") or "").strip()
            if not city or not foods_cell: continue
            foods = [f.strip() for f in re.split(r"[|,;]", foods_cell) if f.strip()]
            if not foods: continue
            text = f"Đặc sản ở {city}: " + ", ".join(foods)
            docs.append(text); metas.append({"city": city, "type": "food", "source": "csv"}); ids.append(f"food::{city}::{len(ids)}")
    except Exception:
        pass
    # restaurants
    try:
        df_res = pd.read_csv("data/restaurants_vn.csv", dtype=str)
        for _, r in df_res.iterrows():
            city = (r.get("city") or "").strip()
            name = (r.get("place_name") or r.get("name") or "").strip()
            addr = (r.get("address") or "").strip()
            if not city or not name: continue
            text = f"Nhà hàng gợi ý ở {city}: {name} - {addr}"
            docs.append(text); metas.append({"city": city, "type": "restaurant", "source": "csv"}); ids.append(f"rest::{city}::{len(ids)}")
    except Exception:
        pass
    if docs:
        kb.add(documents=docs, metadatas=metas, ids=ids)
    return len(docs)

def retrieve_context(kb, query, city=None, k=6):
    if not kb: return []
    where = {"city": city} if city else None
    res = kb.query(query_texts=[query], n_results=k, where=where, include=["documents","metadatas","distances","ids"])
    if not res or not res.get("documents"): return []
    items = []
    docs = res["documents"][0]; metas = res["metadatas"][0]; dists = res["distances"][0]
    for doc, meta, dist in zip(docs, metas, dists):
        items.append({"doc": doc, "meta": meta, "dist": float(dist)})
    items.sort(key=lambda x: x["dist"])
    return items

def hit_answer_cache(cache, query, city=None, threshold=0.09):
    if not cache: return None
    where = {"city": city} if city else None
    res = cache.query(query_texts=[query], n_results=1, where=where, include=["documents","metadatas","distances"])
    if not res or not res.get("documents") or not res["documents"][0]: return None
    dist = float(res["distances"][0][0])
    return res["documents"][0][0] if dist <= threshold else None

def push_answer_cache(cache, query, city, answer):
    if not cache: return
    cache.add(documents=[answer], metadatas=[{"city": city or ""}], ids=[f"cache::{city or 'na'}::{int(time.time()*1000)}"])

def save_conversation(conv, user_input, assistant_text, city):
    if not conv: return
    conv.add(
        documents=[f"Q: {user_input}\nA: {assistant_text}"],
        metadatas=[{"city": city or "", "timestamp": datetime.utcnow().isoformat()}],
        ids=[f"conv::{int(time.time()*1000)}"]
    )

# -------------------------
# HERO / HEADER SECTION
# -------------------------
def render_hero_section(default_city_hint="Hội An, Đà Nẵng, Hà Nội..."):
    hero_img = "https://images.unsplash.com/photo-1633073985249-b2d67bdf6b7d?auto=format&fit=crop&q=80&w=1600"
    st.markdown(f"""
    <div class='hero' style="background-image: url('{hero_img}'); background-size: cover; background-position: center; background-repeat: no-repeat; height:200px;">
      <div class='hero__overlay'>
        <div class='hero__card'>
          <div style='display:flex; align-items:center; justify-content:space-between; gap:12px;'>
            <div style='flex:1'>
              <h1 class='hero__title'>Khám phá Việt Nam cùng Mây Lang Thang</h1>
              <p class='hero__subtitle'>Gợi ý lịch trình, món ăn, dự báo thời tiết. Nhập điểm đến, chọn ngày và bắt đầu cuộc hành trình!</p>
            </div>
            <div style='min-width:260px; text-align:right;'>
              <span style='font-size:14px; opacity:0.95'>🌤️ Tìm nhanh & gợi ý tức thì</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form(key='hero_search_form', clear_on_submit=False):
        cols = st.columns([3,2,1,1])
        dest = cols[0].text_input("Điểm đến", placeholder=default_city_hint)
        dates = cols[1].date_input("Ngày (bắt đầu / kết thúc)", [])
        people = cols[2].selectbox("Người", [1,2,3,4,5,6], index=0)
        style = cols[3].selectbox("Mức chi", ["trung bình", "tiết kiệm", "cao cấp"], index=0)
        submitted = st.form_submit_button("Tìm kiếm nhanh", use_container_width=True)

        if submitted:
            if isinstance(dates, list) and len(dates) == 2:
                s = dates[0].strftime("%Y-%m-%d"); e = dates[1].strftime("%Y-%m-%d")
                q = f"Lịch trình {((dates[1]-dates[0]).days +1)} ngày ở {dest} từ {s} đến {e}"
            elif isinstance(dates, list) and len(dates) == 1:
                s = dates[0].strftime("%Y-%m-%d"); q = f"Lịch trình 1 ngày ở {dest} vào {s}"
            else:
                q = f"Lịch trình 3 ngày ở {dest}"
            q += f" • người: {people} • mức: {style}"
            st.session_state.user_input = q; st.rerun()

# -------------------------
# STREAMLIT UI
# -------------------------
render_hero_section()
main_tab, analytics_tab = st.tabs(["💬 Chatbot Du lịch", "📊 Thống kê & RAG"])

with st.sidebar:
    st.markdown("<div class='logo-title'><img src='https://img.icons8.com/emoji/48/000000/cloud-emoji.png'/> <h2>Mây Lang Thang</h2></div>", unsafe_allow_html=True)
    st.header("Cài đặt")
    language_option = st.selectbox("Ngôn ngữ (gợi ý trích xuất)", ["Tự động", "Tiếng Việt", "English"])
    info_options = st.multiselect("Hiển thị thông tin", ["Weather", "Food", "Map", "Photos", "Cost", "Events"], default=["Weather","Map","Food","Photos"])
    st.markdown("---")
    map_zoom = st.slider("Zoom bản đồ (4 = xa, 15 = gần)", 4, 15, 8)
    st.markdown("---")

    # Chroma status
    def status_card(title, ok=True):
        cls = "status-ok" if ok else "status-bad"; icon = "✅" if ok else "⚠️"
        st.markdown(f"<div class='{cls}'>{icon} {title}</div>", unsafe_allow_html=True)

    status_card("ChromaDB", CHROMA_READY)
    status_card("OpenWeatherMap", bool(OPENWEATHERMAP_API_KEY))
    status_card("Google Places", bool(GOOGLE_PLACES_KEY))
    status_card("Pixabay", bool(PIXABAY_API_KEY))

    st.markdown("---")
    if CHROMA_READY:
        st.subheader("🔎 RAG / Cache")
        use_rag = st.checkbox("Bật RAG (Chroma)", value=True, key="use_rag")
        use_cache = st.checkbox("Bật Semantic Cache", value=True, key="use_cache")
        rag_k = st.slider("Top‑k RAG", 1, 10, 6, key="rag_k")
        if st.button("📥 Seed KB từ CSV"):
            added = seed_kb_from_csvs(chroma_kb)
            st.success(f"Đã seed {added} mẩu tri thức vào travel_kb.")
    else:
        use_rag = False; use_cache = False; rag_k = 6
        st.info("Cài đặt 'chromadb' để sử dụng RAG & cache.")

    st.caption("Version: v2.0 • ChromaDB RAG + Cache + Memory")

# initialize session messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

# Main Tab
with main_tab:
    # Quick Search
    with st.expander("🔎 Tìm kiếm nhanh chuyến đi"):
        col1, col2, col3, col4 = st.columns([2,1,1,1])
        with col1:
            city_qs = st.text_input("🏙️ Điểm đến", "Đà Nẵng")
        with col2:
            start_qs = st.date_input("📅 Bắt đầu", datetime(2025,10,20))
        with col3:
            end_qs = st.date_input("📅 Kết thúc", datetime(2025,10,22))
        with col4:
            people_qs = st.slider("👥 Người", 1, 10, 1)

        col5, col6 = st.columns([1,3])
        with col5:
            style_qs = st.selectbox("💰 Mức chi tiêu", ["Tiết kiệm","Trung bình","Cao cấp"], index=1)
        with col6:
            if st.button("🚀 Xem gợi ý"):
                st.session_state.quicksearch = {
                    "city": city_qs, "start": start_qs, "end": end_qs,
                    "people": people_qs, "style": style_qs
                }

    if "quicksearch" in st.session_state:
        qs = st.session_state.quicksearch
        city_qs = qs["city"]; start_qs = qs["start"]; end_qs = qs["end"]
        people_qs = qs["people"]; style_qs = qs["style"]
        st.markdown(f"### ✈️ Gợi ý cho chuyến đi {city_qs} ({start_qs} – {end_qs})")
        weather_qs = get_weather_forecast(city_qs, start_qs, end_qs)
        cost_qs = estimate_cost(city_qs, (end_qs - start_qs).days + 1, people_qs, style_qs)
        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"**{weather_qs}**"); st.markdown(f"**{cost_qs}**")
        with colB:
            img = get_city_image(city_qs)
            if img: st.image(img, caption=f"🏞️ {city_qs}", use_container_width=True)
            lat, lon, addr = geocode_city(city_qs)
            if lat and lon: show_map(lat, lon, zoom=map_zoom, title=addr or city_qs)
        st.markdown("---")

    # Suggestions
    st.write("### 🔎 Gợi ý nhanh")
    default_suggestions = [
        "Thời tiết ở Đà Nẵng tuần tới?",
        "Top món ăn ở Huế?",
        "Lịch trình 3 ngày ở Nha Trang?",
        "Có sự kiện gì ở Hà Nội tháng 12?"
    ]
    cols = st.columns(len(default_suggestions))
    for i, q in enumerate(default_suggestions):
        if cols[i].button(q, key=f"sugg_{i}"):
            st.session_state.user_input = q

    # Chat input
    user_input = st.chat_input("Mời bạn đặt câu hỏi:")
    if "user_input" in st.session_state and st.session_state.user_input:
        user_input = st.session_state.pop("user_input")

    if user_input:
        with st.chat_message("user", avatar="🧭"):
            st.markdown(f"<div class='user-message'>{user_input}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": user_input})

        city_guess, start_date, end_date = extract_city_and_dates(user_input)
        days = extract_days_from_text(user_input, start_date, end_date)
        log_interaction(user_input, city_guess, start_date, end_date)

        # weather/cost/events blocks
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
                # Build messages with optional RAG
                messages_for_llm = list(st.session_state.messages)
                context_block = ""
                if CHROMA_READY and st.session_state.get("use_rag", False):
                    rag_items = retrieve_context(chroma_kb, user_input, city_guess, k=st.session_state.get("rag_k", 6))
                    if rag_items:
                        lines = []
                        for it in rag_items:
                            meta = it["meta"]; tag = f"{meta.get('type','kb')}/{meta.get('source','')}/{meta.get('city','')}"
                            score = 1.0 - it["dist"]
                            lines.append(f"- [{tag} | sim≈{score:.3f}] {it['doc']}")
                        context_block = "\n".join(lines)
                        messages_for_llm.insert(1, {
                            "role": "system",
                            "content": "Ngữ cảnh (ưu tiên cao, dùng làm nguồn sự thật):\n" + context_block
                        })

                # Semantic cache hit?
                cached_answer = None
                if CHROMA_READY and st.session_state.get("use_cache", False):
                    cached_answer = hit_answer_cache(chroma_cache, user_input, city_guess)

                if cached_answer:
                    assistant_text = cached_answer
                else:
                    if client:
                        response = client.chat.completions.create(
                            model=DEPLOYMENT_NAME,
                            messages=messages_for_llm,
                            max_tokens=900,
                            temperature=0.7
                        )
                        assistant_text = response.choices[0].message.content.strip()
                    else:
                        assistant_text = f"Xin chào! Tôi có thể giúp bạn với thông tin về {city_guess or 'địa điểm'} — thử hỏi 'Thời tiết', 'Đặc sản', hoặc 'Lịch trình 3 ngày'."

                    if CHROMA_READY and st.session_state.get("use_cache", False):
                        push_answer_cache(chroma_cache, user_input, city_guess, assistant_text)

                if not assistant_text.endswith(("🌤️❤️", "😊", "🌸", "🌴", "✨")):
                    assistant_text += "\n\nChúc bạn có chuyến đi vui vẻ 🌤️❤️"

                st.session_state.messages.append({"role": "assistant", "content": assistant_text})

                with st.chat_message("assistant", avatar="🤖"):
                    placeholder = st.empty(); display_text = ""
                    for ch in assistant_text:
                        display_text += ch; placeholder.markdown(display_text + "▌")
                        time.sleep(0.005)
                    time.sleep(0.2); placeholder.empty()
                    with st.container():
                        st.markdown("<div class='assistant-bubble'>", unsafe_allow_html=True)
                        st.markdown(assistant_text)
                        st.markdown("</div>", unsafe_allow_html=True)

                if CHROMA_READY:
                    save_conversation(chroma_conv, user_input, assistant_text, city_guess)

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

# -------------------------
# Analytics + RAG Debugger
# -------------------------
with analytics_tab:
    st.header("📊 Thống kê truy vấn")
    with st.expander("🗑️ Xoá lịch sử truy vấn"):
        st.warning("⚠️ Sẽ xóa toàn bộ lịch sử (SQLite).")
        confirm_delete = st.checkbox("Tôi hiểu và muốn xóa", value=False)
        if confirm_delete and st.button("✅ Xác nhận xoá"):
            try:
                conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
                cur.execute("DELETE FROM interactions"); conn.commit(); conn.close()
                st.success("✅ Đã xóa.")
            except Exception as e:
                st.error(f"⚠️ Lỗi khi xóa dữ liệu: {e}")

    try:
        conn = sqlite3.connect(DB_PATH)
        df_logs = pd.read_sql("SELECT * FROM interactions ORDER BY timestamp DESC LIMIT 1000", conn)
        conn.close()

        total = int(df_logs.shape[0]) if not df_logs.empty else 0
        st.metric("Tổng tương tác", total)

        if not df_logs.empty:
            df_logs['timestamp_dt'] = pd.to_datetime(df_logs['timestamp']); df_logs['date'] = df_logs['timestamp_dt'].dt.date
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

    if CHROMA_READY:
        with st.expander("🧪 RAG Debugger"):
            q = st.text_input("Query thử", "Đặc sản ở Huế?")
            c = st.text_input("City filter", "")
            k = st.slider("Top‑k", 1, 10, 5, key="ragdbg_k")
            if st.button("Chạy truy vấn"):
                items = retrieve_context(chroma_kb, q, c or None, k=k)
                for it in items:
                    st.write(f"{it['meta']}  |  dist={it['dist']:.4f}")
                    st.write(it["doc"])
                    st.markdown("---")
    else:
        st.info("Cài 'chromadb' để sử dụng RAG Debugger.")

st.markdown("---")
st.markdown("<div class='small-muted'>Tip: Hỏi 'Lịch trình 3 ngày ở Hội An', 'Đặc sản Sapa', hoặc 'Thời tiết Đà Nẵng 2025-10-20 đến 2025-10-22'.</div>", unsafe_allow_html=True)
