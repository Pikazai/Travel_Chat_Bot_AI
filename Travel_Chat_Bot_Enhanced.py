# Travel_Chat_Bot_Enhanced_UI_FULL_with_Hero.py
# Phiên bản hoàn chỉnh (giữ nguyên logic gốc) + cải tiến UX/UI
# Thêm Header / Hero Section để tạo cảm giác "App du lịch thật sự"
# Yêu cầu: streamlit, openai, requests, geopy, pandas, pydeck, plotly, sqlite3

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

# -------------------------
# PAGE CONFIG & THEME
# -------------------------
st.set_page_config(page_title="🤖 [Mây Lang Thang] - Travel Assistant", layout="wide", page_icon="🌴")

# Global CSS + UI tweaks (including hero styles)
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
    background-color: #e7f3ff; /* xanh nhạt */
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

    </style>
    """, unsafe_allow_html=True
)

# -------------------------
# CONFIG / SECRETS
# -------------------------
# These must be set in Streamlit secrets (or env) exactly as used previously
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_ENDPOINT = st.secrets.get("OPENAI_ENDPOINT", "https://api.openai.com/v1") if hasattr(st, 'secrets') else os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
DEPLOYMENT_NAME = st.secrets.get("DEPLOYMENT_NAME", "gpt-4o-mini") if hasattr(st, 'secrets') else os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
OPENWEATHERMAP_API_KEY = st.secrets.get("OPENWEATHERMAP_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("OPENWEATHERMAP_API_KEY", "")
GOOGLE_PLACES_KEY = st.secrets.get("PLACES_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("PLACES_API_KEY", "")
PIXABAY_API_KEY = st.secrets.get("PIXABAY_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("PIXABAY_API_KEY", "")

# Initialize OpenAI client (using openai Python SDK modern interface)
if OPENAI_API_KEY:
    client = openai.OpenAI(base_url=OPENAI_ENDPOINT, api_key=OPENAI_API_KEY)
else:
    client = None

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
# UTILITIES: days extraction (original logic)
# -------------------------
def extract_days_from_text(user_text, start_date=None, end_date=None):
    """
    Phân tích số ngày từ câu hỏi người dùng.
    Ưu tiên tính từ start_date, end_date nếu có.
    Nếu không, dùng regex / AI fallback.
    """
    # If start & end provided, compute days
    if start_date and end_date:
        try:
            delta = (end_date - start_date).days + 1
            return max(delta, 1)
        except Exception:
            pass

    # Try regex for 'X ngày' or 'X tuần'
    m = re.search(r"(\d+)\s*(ngày|day|days|tuần|week|weeks)", user_text, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        unit = m.group(2).lower()
        if "tuần" in unit or "week" in unit:
            return num * 7
        return num

    # AI fallback (if client available)
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

    return 3  # fallback

# -------------------------
# GEOCODING & MAPS
# -------------------------
geolocator = Nominatim(user_agent="travel_chatbot_app")

def geocode_city(city_name):
    try:
        loc = geolocator.geocode(city_name, timeout=10)
        if loc:
            return loc.latitude, loc.longitude, loc.address
        return None, None, None
    except Exception:
        return None, None, None

def show_map(lat, lon, zoom=8, title=""):
    if lat is None or lon is None:
        st.info("Không có dữ liệu toạ độ để hiển thị bản đồ.")
        return

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
# WEATHER (OpenWeatherMap) with AI fallback on location
# -------------------------
def resolve_city_via_ai(user_text):
    """
    Dùng AI để xác định tỉnh/thành nếu OpenWeather không trả kết quả
    """
    if not client:
        return None
    try:
        prompt = f"""
Bạn là chuyên gia địa lý du lịch Việt Nam.
Phân tích câu sau để xác định:
1. 'place': địa danh cụ cụ thể (khu du lịch, công viên, đảo, thắng cảnh,...)
2. 'province_or_city': tên tỉnh hoặc thành phố của Việt Nam mà địa danh đó thuộc về.

Nếu không xác định được, trả về null.

Kết quả JSON ví dụ:
{{"place": "Phong Nha - Kẻ Bàng", "province_or_city": "Quảng Bình"}}

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
    """
    Lấy dự báo thời tiết (5 ngày) từ OpenWeatherMap.
    Nếu không tìm thấy, thử dùng AI để đoán tỉnh/thành rồi thử lại.
    """
    if not OPENWEATHERMAP_API_KEY:
        return "⚠️ Thiếu OpenWeatherMap API Key."

    try:
        def _fetch_weather(city):
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHERMAP_API_KEY}&lang=vi&units=metric"
            response = requests.get(url, timeout=8)
            return response.json()

        data = _fetch_weather(city_name)

        # Try fallback via AI if not found
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
    """
    Dùng AI để trích xuất city, start_date, end_date.
    Nếu thất bại, trả về None.
    """
    if not client:
        return None, None, None
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
            # if past year given without context, keep as is
            return dt
        start_dt = _parse(s)
        end_dt = _parse(e)
        if start_dt and not end_dt:
            end_dt = start_dt
        return city, start_dt, end_dt
    except Exception:
        return None, None, None

# -------------------------
# AI suggestions generator (keeps as python list)
# -------------------------
def generate_ai_suggestions():
    try:
        prompt = f"""
Bạn là {ChatBotName} – {system_prompt.strip()}
Hãy tạo 4 câu hỏi gợi ý (ngắn gọn, thân thiện) để người dùng có thể hỏi bạn.
Trả về dưới dạng danh sách (list) các chuỗi.
"""
        if client:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[{"role":"system","content":prompt}],
                max_tokens=200,
                temperature=0.7
            )
            text = response.choices[0].message.content.strip()

            # Try parse JSON list
            try:
                data = json.loads(text)
                if isinstance(data, list) and all(isinstance(x, str) for x in data):
                    return [s.strip() for s in data][:4]
            except Exception:
                pass

            m = re.search(r'\[.*\]', text, re.DOTALL)
            if m:
                list_text = m.group(0)
                try:
                    fixed = list_text.replace("'", '"')
                    data = json.loads(fixed)
                    if isinstance(data, list):
                        return [s.strip() for s in data if isinstance(s, str)][:4]
                except Exception:
                    inner = list_text[1:-1]
                    parts = [p.strip().strip(' "\'') for p in inner.split(',') if p.strip()]
                    parts = [p if p.endswith('?') else p + '?' for p in parts]
                    return parts[:4]

            # fallback parsing by lines
            lines = [l.strip() for l in re.split(r'[\r\n]+', text) if l.strip()]
            parts = []
            if len(lines) > 1:
                for l in lines:
                    if ',' in l and len(l.split(',')) > 1:
                        for p in l.split(','):
                            p = p.strip().strip('"-• ')
                            if p:
                                parts.append(p if p.endswith('?') else p + '?')
                    else:
                        subs = [s.strip() for s in l.split('?') if s.strip()]
                        for s in subs:
                            parts.append(s + '?')
            else:
                single = lines[0] if lines else text
                if ',' in single:
                    items = [p.strip().strip('"\'') for p in single.split(',') if p.strip()]
                    parts = [p if p.endswith('?') else p + '?' for p in items]
                else:
                    subs = [s.strip() for s in re.split(r'\?|•|-', single) if s.strip()]
                    parts = [s + '?' for s in subs]

            clean = []
            for p in parts:
                if p and p not in clean:
                    clean.append(p)
                if len(clean) >= 4:
                    break
            if clean:
                return clean[:4]
    except Exception:
        pass

    # fallback static suggestions
    return [
        "Thời tiết ở Đà Nẵng tuần tới?",
        "Top món ăn ở Huế?",
        "Lịch trình 3 ngày ở Nha Trang?",
        "Có sự kiện gì ở Hà Nội tháng 12?"
    ]

# ensure suggestions stored
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = generate_ai_suggestions()

# -------------------------
# HERO / HEADER SECTION
# -------------------------

def render_hero_section(default_city_hint="Hội An, Đà Nẵng, Hà Nội..."):
    # Use a fixed Unsplash hero image (static fallback only)
    hero_img = "https://images.unsplash.com/photo-1633073985249-b2d67bdf6b7d?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=1074"

    # Hero markup using background-image on a full-width div so the image covers entire hero area
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

    # form below hero for inputs
    with st.form(key='hero_search_form', clear_on_submit=False):
        cols = st.columns([3,2,1,1])
        dest = cols[0].text_input("Điểm đến", placeholder=default_city_hint)
        dates = cols[1].date_input("Ngày (bắt đầu / kết thúc)", [])
        people = cols[2].selectbox("Người", [1,2,3,4,5,6], index=0)
        style = cols[3].selectbox("Mức chi", ["trung bình", "tiết kiệm", "cao cấp"], index=0)
        submitted = st.form_submit_button("Tìm kiếm nhanh", use_container_width=True)

        if submitted:
            # Build a friendly query that existing pipeline understands
            if isinstance(dates, list) and len(dates) == 2:
                s = dates[0].strftime("%Y-%m-%d")
                e = dates[1].strftime("%Y-%m-%d")
                q = f"Lịch trình { ( (dates[1]-dates[0]).days +1 ) } ngày ở {dest} từ {s} đến {e}"
            elif isinstance(dates, list) and len(dates) == 1:
                s = dates[0].strftime("%Y-%m-%d")
                q = f"Lịch trình 1 ngày ở {dest} vào {s}"
            else:
                q = f"Lịch trình 3 ngày ở {dest}"

            # attach preferences
            q += f" • người: {people} • mức: {style}"

            # set into session so main chat logic picks it up
            st.session_state.user_input = q
            # After submit we rerun to process immediately
            st.rerun()


# -------------------------
# STREAMLIT UI LAYOUT
# -------------------------

# Render hero at the very top of the app
render_hero_section()

main_tab, analytics_tab = st.tabs(["💬 Chatbot Du lịch", "📊 Thống kê truy vấn"])

with st.sidebar:
    st.markdown("<div class='logo-title'><img src='https://img.icons8.com/emoji/48/000000/cloud-emoji.png'/> <h2>Mây Lang Thang</h2></div>", unsafe_allow_html=True)
    st.header("Cài đặt")
    language_option = st.selectbox("Ngôn ngữ (gợi ý trích xuất)", ["Tự động", "Tiếng Việt", "English"])
    info_options = st.multiselect("Hiển thị thông tin",
                                  ["Weather", "Food", "Map", "Photos", "Cost", "Events"],
                                  default=["Weather", "Map","Food", "Photos"])
    st.markdown("---")
    st.write("Chọn mức zoom bản đồ:")
    map_zoom = st.slider("Zoom (4 = xa, 15 = gần)", 4, 15, 8)
    st.markdown("---")
    # status cards
    def status_card(title, ok=True):
        cls = "status-ok" if ok else "status-bad"
        icon = "✅" if ok else "⚠️"
        st.markdown(f"<div class='{cls}'>{icon} {title}</div>", unsafe_allow_html=True)
    status_card("OpenWeatherMap", bool(OPENWEATHERMAP_API_KEY))
    status_card("Google Places", bool(GOOGLE_PLACES_KEY))
    status_card("Pixabay", bool(PIXABAY_API_KEY))
    st.markdown("---")
    st.caption("🍜 Food AI: CSV local dữ liệu + GPT fallback")
    st.markdown("Version: v1.2")

# initialize session messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

# Main Tab
with main_tab:
    # --- Quick Search Form ---
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
                    "city": city_qs,
                    "start": start_qs,
                    "end": end_qs,
                    "people": people_qs,
                    "style": style_qs
                }

    # Nếu người dùng vừa thực hiện tìm kiếm nhanh
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

    st.write("### 🔎 Gợi ý nhanh")
    cols = st.columns(len(st.session_state.suggested_questions))
    for i, q in enumerate(st.session_state.suggested_questions):
        if cols[i].button(q, key=f"sugg_{i}"):
            st.session_state.user_input = q

    # Chat input
    user_input = st.chat_input("Mời bạn đặt câu hỏi:")
    # preserve suggestion override
    if "user_input" in st.session_state and st.session_state.user_input:
        user_input = st.session_state.pop("user_input")

    if user_input:
        # display user message
        with st.chat_message("user", avatar="🧭"):
            st.markdown(f"<div class='user-message'>{user_input}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # analyze input for city and dates
        city_guess, start_date, end_date = extract_city_and_dates(user_input)
        days = extract_days_from_text(user_input, start_date, end_date)
        log_interaction(user_input, city_guess, start_date, end_date)

        # weather forecast check
        if start_date:
            today = datetime.now().date()
            max_forecast_date = today + timedelta(days=5)
            if start_date.date() > max_forecast_date:
                st.warning(f"⚠️ Lưu ý: OpenWeather chỉ cung cấp dự báo ~5 ngày. Bạn yêu cầu bắt đầu {start_date.strftime('%d/%m/%Y')}.")

        # quick blocks (Weather, Cost, Events)
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

        # main assistant response with spinner + progress bar
        with st.spinner("⏳ Đang soạn phản hồi..."):
            try:
                # Simulate progressive feedback
                progress_text = "AI đang phân tích dữ liệu du lịch..."
                progress_bar = st.progress(0, text=progress_text)

                for percent_complete in range(0, 101, 20):
                    time.sleep(0.08)
                    progress_bar.progress(percent_complete, text=progress_text)

                progress_bar.empty()

                # Call OpenAI for full response if client available
                assistant_text = ""
                if client:
                    response = client.chat.completions.create(
                        model=DEPLOYMENT_NAME,
                        messages=st.session_state.messages,
                        max_tokens=900,
                        temperature=0.7
                    )
                    assistant_text = response.choices[0].message.content.strip()
                else:
                    # fallback summary if no OpenAI key
                    assistant_text = f"Xin chào! Tôi có thể giúp bạn với thông tin về {city_guess or 'địa điểm'} — thử hỏi 'Thời tiết', 'Đặc sản', hoặc 'Lịch trình 3 ngày'."

                # ensure friendly closing
                if not assistant_text.endswith(("🌤️❤️", "😊", "🌸", "🌴", "✨")):
                    assistant_text += "\n\nChúc bạn có chuyến đi vui vẻ 🌤️❤️"

                st.session_state.messages.append({"role": "assistant", "content": assistant_text})

                # show assistant message with styled bubble
                with st.chat_message("assistant", avatar="🤖"):
                    # Typing giả
                    placeholder = st.empty()
                    display_text = ""
                    for char in assistant_text:
                        display_text += char
                        placeholder.markdown(display_text + "▌")  # con trỏ giả
                        time.sleep(0.01)
                    time.sleep(0.3)
                    placeholder.empty()

                    # Hiển thị Markdown gốc với khung màu bằng container
                    with st.container():
                        st.markdown(
                            f"<div class='assistant-bubble'>", unsafe_allow_html=True
                        )
                        st.markdown(assistant_text)  # Markdown giữ nguyên
                        st.markdown("</div>", unsafe_allow_html=True)


                st.balloons()

            except Exception as e:
                st.error(f"⚠️ Lỗi khi gọi OpenAI: {e}")

        # Display map and photos and food/restaurants in two columns
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

            # Restaurants (hybrid)
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

    # --- Delete history with confirmation ---
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

    # --- Show analytics: metric, chart, table ---
    try:
        conn = sqlite3.connect(DB_PATH)
        df_logs = pd.read_sql("SELECT * FROM interactions ORDER BY timestamp DESC LIMIT 1000", conn)
        conn.close()

        total = int(df_logs.shape[0]) if not df_logs.empty else 0
        st.metric("Tổng tương tác", total)

        if not df_logs.empty:
            # convert timestamp
            df_logs['timestamp_dt'] = pd.to_datetime(df_logs['timestamp'])
            df_logs['date'] = df_logs['timestamp_dt'].dt.date

            # chart: queries per day
            series = df_logs.groupby('date').size().reset_index(name='queries')
            fig = px.bar(series, x='date', y='queries', title='📈 Số truy vấn mỗi ngày', color='queries', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

            # top cities
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

# Footer tip
st.markdown("---")
st.markdown("<div class='small-muted'>Tip: Bạn có thể yêu cầu cụ thể như 'Lịch trình 3 ngày ở Hội An', 'Đặc sản Sapa', hoặc 'Thời tiết Đà Nẵng 2025-10-20 đến 2025-10-22'.</div>", unsafe_allow_html=True)
