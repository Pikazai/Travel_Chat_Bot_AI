# Travel_Chat_Bot_Enhanced_Hybrid_FoodAI_Pixabay.py
# Streamlit travel chatbot (final tabs version)
# - Two tabs: Chatbot Du lịch & Thống kê truy vấn
# - Pixabay for images (requires PIXABAY_API_KEY in .streamlit/secrets.toml)
# - Hybrid restaurants: Google Places + CSV fallback
# - Food AI Assistant: CSV local foods + GPT fallback
# - Weather: OpenWeatherMap
# - Map: Carto Positron style (no API key required)
# - Zoom slider in sidebar, use_container_width for images
# - Logging to SQLite, analytics tab with metric + table
#
# Requirements:
# pip install streamlit openai requests geopy pandas pydeck

# Travel_Chat_Bot_Enhanced.py (bản đã cập nhật)
# Bổ sung AI tự động nhận diện số ngày từ người dùng (extract_days_from_text)

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

# -------------------------
# CONFIG / SECRETS
# -------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_ENDPOINT = st.secrets.get("OPENAI_ENDPOINT", "https://api.openai.com/v1")
DEPLOYMENT_NAME = st.secrets.get("DEPLOYMENT_NAME", "gpt-4o-mini")
OPENWEATHERMAP_API_KEY = st.secrets["OPENWEATHERMAP_API_KEY"]
GOOGLE_PLACES_KEY = st.secrets["PLACES_API_KEY"]
PIXABAY_API_KEY = st.secrets["PIXABAY_API_KEY"]

client = openai.OpenAI(base_url=OPENAI_ENDPOINT, api_key=OPENAI_API_KEY)

ChatBotName = "[Mây Lang Thang]" #Mây Lang Thang
system_prompt = """
Bạn là Hướng dẫn viên du lịch ảo Alex - người kể chuyện, am hiểu văn hóa, lịch sử, ẩm thực và thời tiết Việt Nam.
Luôn đưa ra thông tin hữu ích, gợi ý lịch trình, món ăn, chi phí, thời gian lý tưởng, sự kiện và góc chụp ảnh.
"""

# -------------------------
# DB LOGGING
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

init_db()

# -------------------------
# Hàm mới: Phân tích số ngày tự động
# -------------------------
def extract_days_from_text(user_text, start_date=None, end_date=None):
    """
    Phân tích số ngày từ câu hỏi người dùng.
    Ưu tiên tính từ start_date, end_date nếu có.
    Nếu không, dùng AI hoặc regex để suy luận ('1 tuần', '3 ngày', ...)
    """
    # Nếu có ngày bắt đầu & kết thúc => tính chênh lệch
    if start_date and end_date:
        delta = (end_date - start_date).days + 1
        return max(delta, 1)

    # Thử tìm "X ngày" hoặc "X tuần"
    m = re.search(r"(\d+)\s*(ngày|day|days|tuần|week|weeks)", user_text, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        unit = m.group(2).lower()
        if "tuần" in unit or "week" in unit:
            return num * 7
        return num

    # Nếu không có, thử hỏi AI
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
# GEOCODING & MAPS, WEATHER, ... (các phần còn lại giữ nguyên)
# -------------------------
# (Phần code gốc từ file Travel_Chat_Bot_Enhanced.py sẽ tiếp tục như ban đầu, chỉ cần thay đổi đoạn gọi estimate_cost)

# Trong phần xử lý user_input:
# city_guess, start_date, end_date = extract_city_and_dates(user_input)
# days = extract_days_from_text(user_input, start_date, end_date)
# if city_guess and "Cost" in info_options:
#     blocks.append(estimate_cost(city_guess, days=days))
init_db()

# -------------------------
# GEOCODING & MAPS (Carto Positron style)
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
# WEATHER (OpenWeatherMap)
# -------------------------
# def get_weather_forecast(city_name, start_date=None, end_date=None):
#     try:
#         url = f"http://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={OPENWEATHERMAP_API_KEY}&lang=vi&units=metric"
#         response = requests.get(url, timeout=10)
#         data = response.json()
#         if data.get("cod") != "200":
#             return f"❌ Không tìm thấy thông tin dự báo thời tiết cho địa điểm: **{city_name}**."
#         forecast_text = f"🌤 **Dự báo thời tiết cho {city_name}:**\n"
#         if start_date and end_date:
#             current = start_date
#             while current <= end_date:
#                 date_str = current.strftime("%Y-%m-%d")
#                 day_forecasts = [f for f in data['list'] if f['dt_txt'].startswith(date_str)]
#                 if not day_forecasts:
#                     forecast_text += f"\n📅 {current.strftime('%d/%m/%Y')}: Không có dữ liệu dự báo.\n"
#                 else:
#                     temps = [f['main']['temp'] for f in day_forecasts]
#                     desc = day_forecasts[0]['weather'][0]['description']
#                     forecast_text += (
#                         f"\n📅 {current.strftime('%d/%m/%Y')} - {desc.capitalize()}\n"
#                         f"🌡 Nhiệt độ trung bình: {sum(temps)/len(temps):.1f}°C\n"
#                     )
#                 current += timedelta(days=1)
#         else:
#             first_forecast = data['list'][0]
#             desc = first_forecast['weather'][0]['description'].capitalize()
#             temp = first_forecast['main']['temp']
#             forecast_text += f"- Hiện tại: {desc}, {temp}°C\n"
#         return forecast_text
#     except Exception as e:
#         return f"⚠️ Lỗi khi lấy dữ liệu thời tiết: {e}"

def get_weather_forecast(city_name, start_date=None, end_date=None, user_text=None):
    """
    Lấy dự báo thời tiết. Nếu không tìm thấy địa điểm, dùng AI để xác định
    tỉnh/thành tương ứng rồi thử lại.
    """
    try:
        def _fetch_weather(city):
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={city},VN&appid={OPENWEATHERMAP_API_KEY}&lang=vi&units=metric"
            response = requests.get(url, timeout=10)
            return response.json()

        data = _fetch_weather(city_name)

        # Nếu không có dữ liệu, thử dùng AI xác định lại tỉnh/thành
        if data.get("cod") != "200" and user_text:
            ai_city = resolve_city_via_ai(user_text)
            if ai_city and ai_city.lower() != city_name.lower():
                st.info(f"🤖 Không tìm thấy thời tiết cho '{city_name}', thử với '{ai_city}' ...")
                data = _fetch_weather(ai_city)
                city_name = ai_city

        if data.get("cod") != "200":
            return f"❌ Không tìm thấy thông tin dự báo thời tiết cho địa điểm: **{city_name}**."

        forecast_text = f"🌤 **Dự báo thời tiết cho {city_name}:**\\n"

        if start_date and end_date:
            current = start_date
            while current <= end_date:
                date_str = current.strftime("%Y-%m-%d")
                day_forecasts = [f for f in data['list'] if f['dt_txt'].startswith(date_str)]
                if not day_forecasts:
                    forecast_text += f"\\n📅 {current.strftime('%d/%m/%Y')}: Không có dữ liệu dự báo.\\n"
                else:
                    temps = [f['main']['temp'] for f in day_forecasts]
                    desc = day_forecasts[0]['weather'][0]['description']
                    forecast_text += (
                        f"\\n📅 {current.strftime('%d/%m/%Y')} - {desc.capitalize()}\\n"
                        f"🌡 Nhiệt độ trung bình: {sum(temps)/len(temps):.1f}°C\\n"
                    )
                current += timedelta(days=1)
        else:
            first_forecast = data['list'][0]
            desc = first_forecast['weather'][0]['description'].capitalize()
            temp = first_forecast['main']['temp']
            forecast_text += f"- Hiện tại: {desc}, {temp}°C\\n"

        return forecast_text

    except Exception as e:
        return f"⚠️ Lỗi khi lấy dữ liệu thời tiết: {e}"



# -------------------------
# PIXABAY IMAGE SYSTEM
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
# RESTAURANTS HYBRID (Google Places + CSV)
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
# SUGGESTIONS / COST / PHOTOSPOTS (no itinerary)
# -------------------------
def estimate_cost(city, days=3, people=1, style="trung bình"):
    mapping = {"tiết kiệm": 400000, "trung bình": 800000, "cao cấp": 2000000}
    per_day = mapping.get(style, 800000)
    total = per_day * days * people
    return f"💸 Chi phí ước tính: khoảng {total:,} VNĐ cho {people} người, {days} ngày."

def suggest_local_food(city):
    return f"🍜 Yêu cầu 'Đặc sản' để nhận danh sách món ăn nổi bật của {city}."

def suggest_events(city):
    return f"🎉 Sự kiện ở {city}: lễ hội địa phương, chợ đêm, hội chợ ẩm thực (tuỳ mùa)."

def suggest_photospots(city):
    return f"📸 Gợi ý check-in: trung tâm lịch sử, bờ sông/biển, quán cà phê có view đẹp."

# -------------------------
# BILINGUAL CITY & DATE EXTRACTION
# -------------------------
def extract_city_and_dates(user_text):
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
            if dt.year < datetime.now().year:
                dt = dt.replace(year=datetime.now().year)
            return dt
        start_dt = _parse(s)
        end_dt = _parse(e)
        if start_dt and not end_dt:
            end_dt = start_dt
        return city, start_dt, end_dt
    except Exception:
        return None, None, None

def resolve_city_via_ai(user_text):
    """
    Phân tích địa danh trong câu người dùng và xác định tỉnh/thành tương ứng.
    Dùng AI để nhận diện — không cần danh sách thủ công.
    """
    try:
        prompt = f"""
Bạn là chuyên gia địa lý du lịch Việt Nam.
Phân tích câu sau để xác định:
1. 'place': địa danh cụ thể (khu du lịch, công viên, đảo, thắng cảnh,...)
2. 'province_or_city': tên tỉnh hoặc thành phố của Việt Nam mà địa danh đó thuộc về.

Nếu không xác định được, trả về null.

Kết quả JSON ví dụ:
{{"place": "Phong Nha - Kẻ Bàng", "province_or_city": "Quảng Trị"}}

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


# -------------------------
# STREAMLIT UI (TABS)
# -------------------------
st.set_page_config(page_title=f"🤖 {ChatBotName} - Travel Assistant", layout="wide")
st.title(f"🤖 Chatbot Trợ lý du lịch {ChatBotName}")

with st.sidebar:
    st.header("Cài đặt")
    language_option = st.selectbox("Ngôn ngữ (gợi ý trích xuất)", ["Tự động", "Tiếng Việt", "English"])
    info_options = st.multiselect("Hiển thị thông tin",
                                  ["Weather", "Food", "Map", "Photos", "Cost", "Events"],
                                  default=["Weather", "Map","Food", "Photos"])
    st.markdown("---")
    st.write("Chọn mức zoom bản đồ:")
    map_zoom = st.slider("Zoom (4 = xa, 15 = gần)", 4, 15, 8)
    st.markdown("---")
    if OPENWEATHERMAP_API_KEY:
        st.success("✅ OpenWeatherMap OK")
    else:
        st.error("Thiếu OpenWeatherMap API Key")
    if GOOGLE_PLACES_KEY:
        st.success("✅ Google Places API Key found (using Google for restaurants)")
    else:
        st.info("📂 Google Places API Key not found — using CSV fallback for restaurants")
    if PIXABAY_API_KEY:
        st.success("✅ Pixabay API sẵn sàng (hiển thị ảnh minh họa)")
    else:
        st.warning("⚠️ Thiếu Pixabay API key — sẽ hiển thị placeholder cho ảnh")
    st.caption("🍜 Food AI: CSV local dữ liệu + GPT fallback")
    st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
# if "suggested_questions" not in st.session_state:
#     st.session_state.suggested_questions = [
#         "Thời tiết ở Đà Nẵng tuần tới?",
#         "Top món ăn ở Huế?",
#         "Lịch trình 3 ngày ở Nha Trang",
#         "Có sự kiện gì ở Hà Nội tháng 12?"
#     ]
# -------------------------
# AI sinh gợi ý nhanh (trả về luôn Python list của strings)
# -------------------------
def generate_ai_suggestions():
    try:
        prompt = f"""
Bạn là {ChatBotName} – {system_prompt.strip()}
Hãy tạo 4 câu hỏi gợi ý (ngắn gọn, thân thiện) để người dùng có thể hỏi bạn.
Ví dụ: “Thời tiết ở Huế tuần tới?”, “Đặc sản nổi tiếng ở Hà Nội?”, ...
Trả về dưới dạng danh sách (list) các chuỗi. Ví dụ:
["Thời tiết ở Huế tuần tới?", "Món ăn ngon ở Đà Nẵng?", "Chi phí du lịch Sapa 3 ngày?", "Sự kiện nổi bật tháng này?"]
"""
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "system", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        text = response.choices[0].message.content.strip()

        # 1) Thử parse JSON trực tiếp (ví dụ: ["...","..."])
        try:
            data = json.loads(text)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return [s.strip() for s in data][:4]
        except Exception:
            pass

        # 2) Nếu model trả về 1 phần của Python list hoặc có assignment: tìm nội dung giữa dấu ngoặc vuông
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            list_text = m.group(0)
            # Thử json.loads sau khi chuyển ' -> " nếu cần
            try:
                fixed = list_text.replace("'", '"')
                data = json.loads(fixed)
                if isinstance(data, list):
                    return [s.strip() for s in data if isinstance(s, str)][:4]
            except Exception:
                # fallback: tách bằng dấu phẩy thủ công
                inner = list_text[1:-1]
                parts = [p.strip().strip('"\'' ) for p in inner.split(',') if p.strip()]
                parts = [p if p.endswith('?') else p + '?' for p in parts]
                return parts[:4]

        # 3) Nếu không có ngoặc vuông, tách theo dòng / dấu phẩy / dấu hỏi
        # tách theo newline
        lines = [l.strip() for l in re.split(r'[\r\n]+', text) if l.strip()]
        parts = []
        if len(lines) > 1:
            for l in lines:
                # nếu dòng có nhiều câu phân tách bằng comma -> split
                if ',' in l and len(l.split(',')) > 1:
                    for p in l.split(','):
                        p = p.strip().strip('"-• ')
                        if p:
                            parts.append(p if p.endswith('?') else p + '?')
                else:
                    # split các câu trong cùng 1 dòng bằng dấu '?'
                    subs = [s.strip() for s in l.split('?') if s.strip()]
                    for s in subs:
                        parts.append(s + '?')
        else:
            # chỉ 1 dòng: split bằng comma hoặc bằng dấu '?'
            single = lines[0] if lines else text
            if ',' in single:
                items = [p.strip().strip('"\'' ) for p in single.split(',') if p.strip()]
                parts = [p if p.endswith('?') else p + '?' for p in items]
            else:
                subs = [s.strip() for s in re.split(r'\?|•|-', single) if s.strip()]
                parts = [s + '?' for s in subs]

        # dọn, loại bỏ trùng, giữ tối đa 4
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

    # fallback cố định (luôn list)
    return [
        "Thời tiết ở Đà Nẵng tuần tới?",
        "Top món ăn ở Huế?",
        "Lịch trình 3 ngày ở Nha Trang",
        "Có sự kiện gì ở Hà Nội tháng 12?"
    ]


# Khởi tạo gợi ý nhanh trong session (đảm bảo luôn là Python list)
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = generate_ai_suggestions()



main_tab, analytics_tab = st.tabs(["💬 Chatbot Du lịch", "📊 Thống kê truy vấn"])

with main_tab:
    st.write("### 🔎 Gợi ý nhanh")
    cols = st.columns(len(st.session_state.suggested_questions))
    for i, q in enumerate(st.session_state.suggested_questions):
        if cols[i].button(q):
            st.session_state.user_input = q

    user_input = st.chat_input("Mời bạn đặt câu hỏi:")
    if "user_input" in st.session_state and st.session_state.user_input:
        user_input = st.session_state.pop("user_input")

    if user_input:
        # Ghi lại tin nhắn người dùng
        with st.chat_message("user", avatar="🧭"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Phân tích thông tin
        city_guess, start_date, end_date = extract_city_and_dates(user_input)
        log_interaction(user_input, city_guess, start_date, end_date)

        # Kiểm tra ngày vượt giới hạn dự báo
        if start_date:
            today = datetime.now().date()
            max_forecast_date = today + timedelta(days=5)
            if start_date.date() > max_forecast_date:
                st.warning(f"⚠️ Lưu ý: OpenWeather chỉ cung cấp dự báo ~5 ngày. "
                        f"Bạn yêu cầu bắt đầu {start_date.strftime('%d/%m/%Y')}.")

        # ✅ Gợi ý nhanh (emoji + chat style)
        blocks = []
        if city_guess and "Weather" in info_options:
            blocks.append(get_weather_forecast(city_guess, start_date, end_date, user_input))
        # if city_guess and "Food" in info_options:
        #     blocks.append(f"🍜 Đặc sản nổi bật của {city_guess}:")
        if city_guess and "Cost" in info_options:
            blocks.append(estimate_cost(city_guess))
        if city_guess and "Events" in info_options:
            blocks.append(suggest_events(city_guess))
        # if city_guess and "Photos" in info_options:
        #     blocks.append(f"📸 Điểm check-in nổi bật ở {city_guess}:")

        for b in blocks:
            if isinstance(b, str):
                b = b.replace("\\n", "\n")
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(b)

        # ✅ Phần trả lời AI chính — có spinner
        with st.spinner("⏳ Đang soạn phản hồi..."):
            try:
                progress_text = "AI đang phân tích dữ liệu du lịch..."
                progress_bar = st.progress(0, text=progress_text)

                # Hiệu ứng tiến trình giả lập
                for percent_complete in range(0, 101, 15):
                    time.sleep(0.1)
                    progress_bar.progress(percent_complete, text=progress_text)

                # ✅ Ẩn progress bar sau khi hoàn tất
                progress_bar.empty()

                # Gọi OpenAI để sinh phản hồi
                response = client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=st.session_state.messages,
                    max_tokens=800,
                    temperature=0.7
                )

                assistant_text = response.choices[0].message.content.strip()

                # 💕 Thêm lời chúc kết thúc nếu chưa có emoji
                if not assistant_text.endswith(("🌤️❤️", "😊", "🌸", "🌴", "✨")):
                    assistant_text += "\n\nChúc bạn có chuyến đi vui vẻ 🌤️❤️"

                st.session_state.messages.append({"role": "assistant", "content": assistant_text})

                # 🎈 Hiệu ứng khi hoàn thành
                st.balloons()

                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(assistant_text)

            except Exception as e:
                st.error(f"⚠️ Lỗi khi gọi OpenAI: {e}")



        # ✅ Hiển thị thêm phần bản đồ & ảnh
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


with analytics_tab:
    st.header("📊 Thống kê truy vấn (gần đây)")

    # --- Xác nhận bảo vệ trước khi xóa ---
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

    # --- Hiển thị thống kê truy vấn ---
    try:
        conn = sqlite3.connect(DB_PATH)
        df_logs = pd.read_sql("SELECT * FROM interactions ORDER BY timestamp DESC LIMIT 200", conn)
        total = int(df_logs.shape[0])
        st.metric("Tổng tương tác", total)

        if not df_logs.empty:
            st.dataframe(df_logs[["timestamp", "user_input", "city"]])
        else:
            st.info("Chưa có truy vấn nào được ghi nhận.")
        conn.close()
    except Exception as e:
        st.warning(f"Lỗi đọc dữ liệu: {e}")


# End of file
