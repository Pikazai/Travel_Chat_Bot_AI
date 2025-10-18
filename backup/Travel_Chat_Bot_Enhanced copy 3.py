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

# -------------------------
# CONFIG / SECRETS
# -------------------------
OPENAI_API_KEY = st.secrets["openai"]["OPENAI_API_KEY"]
OPENAI_ENDPOINT = st.secrets["openai"].get("OPENAI_ENDPOINT", "https://api.openai.com/v1")
DEPLOYMENT_NAME = st.secrets["openai"].get("DEPLOYMENT_NAME", "gpt-4o-mini")
OPENWEATHERMAP_API_KEY = st.secrets["weather"]["OPENWEATHERMAP_API_KEY"]
GOOGLE_PLACES_KEY = st.secrets.get("google", {}).get("PLACES_API_KEY", None)
PIXABAY_API_KEY = st.secrets.get("pixabay", {}).get("PIXABAY_API_KEY", None)

client = openai.OpenAI(base_url=OPENAI_ENDPOINT, api_key=OPENAI_API_KEY)

ChatBotName = "Team_XYZ"
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
def get_weather_forecast(city_name, start_date=None, end_date=None):
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={OPENWEATHERMAP_API_KEY}&lang=vi&units=metric"
        response = requests.get(url, timeout=10)
        data = response.json()
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

# -------------------------
# STREAMLIT UI (TABS)
# -------------------------
st.set_page_config(page_title=f"🤖 {ChatBotName} - Travel Assistant", layout="wide")
st.title(f"🤖 Chatbot Hướng dẫn viên du lịch {ChatBotName}")

with st.sidebar:
    st.header("Cài đặt")
    language_option = st.selectbox("Ngôn ngữ (gợi ý trích xuất)", ["Tự động", "Tiếng Việt", "English"])
    info_options = st.multiselect("Hiển thị thông tin",
                                  ["Weather", "Food", "Map", "Photos", "Cost", "Events", "Analytics"],
                                  default=["Weather", "Food", "Map", "Photos"])
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
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = [
        "Thời tiết ở Đà Nẵng tuần tới?",
        "Top món ăn ở Huế?",
        "Lịch trình 3 ngày ở Nha Trang",
        "Có sự kiện gì ở Hà Nội tháng 12?"
    ]

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
        st.session_state.messages.append({"role":"user","content":user_input})
        city_guess, start_date, end_date = extract_city_and_dates(user_input)
        log_interaction(user_input, city_guess, start_date, end_date)

        # Weather warning for OpenWeather range
        if start_date:
            today = datetime.now().date()
            max_forecast_date = today + timedelta(days=5)
            if start_date.date() > max_forecast_date:
                st.warning(f"⚠️ Lưu ý: OpenWeather chỉ cung cấp dự báo ~5 ngày. Bạn yêu cầu bắt đầu {start_date.strftime('%d/%m/%Y')}.")

        # Immediate quick info blocks (no itinerary)
        blocks = []
        if city_guess and "Weather" in info_options:
            blocks.append(get_weather_forecast(city_guess, start_date, end_date))
        if city_guess and "Food" in info_options:
            blocks.append(suggest_local_food(city_guess))
        if city_guess and "Cost" in info_options:
            blocks.append(estimate_cost(city_guess))
        if city_guess and "Events" in info_options:
            blocks.append(suggest_events(city_guess))
        if city_guess and "Photos" in info_options:
            blocks.append(suggest_photospots(city_guess))

        for b in blocks:
            st.chat_message("assistant").markdown(b)

        # Conversational reply via OpenAI
        with st.spinner("AI đang soạn trả lời..."):
            try:
                prompt_messages = st.session_state.messages.copy()
                response = client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=prompt_messages,
                    max_tokens=800,
                    temperature=0.7
                )
                assistant_text = response.choices[0].message.content
                st.session_state.messages.append({"role":"assistant","content":assistant_text})
                st.chat_message("assistant").markdown(assistant_text)
            except Exception as e:
                st.error(f"Lỗi khi gọi OpenAI: {e}")

        # Detailed panels: Map / Photos / Food & Restaurants
        lat, lon, addr = (None, None, None)
        if city_guess:
            lat, lon, addr = geocode_city(city_guess)

        cols = st.columns([2,3])
        with cols[0]:
            if "Map" in info_options:
                show_map(lat, lon, zoom=map_zoom, title=addr or city_guess)
            if "Photos" in info_options:
                img = get_city_image(city_guess) if city_guess else None
                if img:
                    st.image(img, caption=f"Hình ảnh gợi ý cho {city_guess}", use_container_width=True)
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

                st.markdown("#### 🍴 Gợi ý nhà hàng")
                restaurants = get_restaurants(city_guess, limit=6)
                if restaurants:
                    df = pd.DataFrame(restaurants)
                    if "maps_url" in df.columns:
                        df["name"] = df.apply(lambda x: f'<a href="{x.get("maps_url")}" target="_blank">{x["name"]}</a>', axis=1)
                        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    else:
                        st.dataframe(df)
                else:
                    st.info("Không tìm thấy nhà hàng (Google/CSV).")

with analytics_tab:
    st.header("📊 Thống kê truy vấn (gần đây)")
    try:
        conn = sqlite3.connect(DB_PATH)
        df_logs = pd.read_sql("SELECT * FROM interactions ORDER BY timestamp DESC LIMIT 200", conn)
        st.metric("Tổng tương tác", int(df_logs.shape[0]))
        if not df_logs.empty:
            st.dataframe(df_logs[["timestamp","user_input","city"]])
        conn.close()
    except Exception:
        st.info("Chưa có dữ liệu phân tích hoặc lỗi đọc DB.")

# End of file
