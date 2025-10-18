# Travel_Chat_Bot_Enhanced_Hybrid.py
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

# ===============================
# 🔐 CONFIG / SECRETS
# ===============================
OPENAI_API_KEY = st.secrets["openai"]["OPENAI_API_KEY"]
OPENAI_ENDPOINT = st.secrets["openai"].get("OPENAI_ENDPOINT", "https://api.openai.com/v1")
DEPLOYMENT_NAME = st.secrets["openai"].get("DEPLOYMENT_NAME", "gpt-4o-mini")
OPENWEATHERMAP_API_KEY = st.secrets["weather"]["OPENWEATHERMAP_API_KEY"]
GOOGLE_PLACES_KEY = st.secrets.get("google", {}).get("PLACES_API_KEY", None)

client = openai.OpenAI(base_url=OPENAI_ENDPOINT, api_key=OPENAI_API_KEY)

ChatBotName = "Team_XYZ"

system_prompt = """
Bạn là Hướng dẫn viên du lịch ảo Alex - người kể chuyện, am hiểu văn hóa, lịch sử, ẩm thực và thời tiết Việt Nam.
Luôn đưa ra thông tin hữu ích, gợi ý lịch trình, món ăn, chi phí, thời gian lý tưởng, sự kiện và góc chụp ảnh.
"""

# ===============================
# 🧠 DATABASE LOGGING
# ===============================
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

# ===============================
# 🌍 GEOCODING & MAP
# ===============================
geolocator = Nominatim(user_agent="travel_chatbot_app")

def geocode_city(city_name):
    try:
        loc = geolocator.geocode(city_name, timeout=10)
        if loc:
            return loc.latitude, loc.longitude, loc.address
        return None, None, None
    except Exception:
        return None, None, None

def show_map(lat, lon, zoom=10, title=""):
    if lat is None or lon is None:
        st.info("Không có dữ liệu toạ độ để hiển thị bản đồ.")
        return
    st.write(f"**Vị trí:** {title} ({lat:.5f}, {lon:.5f})")
    view = pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame([{"lat": lat, "lon": lon}]),
        get_position='[lon, lat]',
        get_radius=2000,
    )
    r = pdk.Deck(layers=[layer], initial_view_state=view)
    st.pydeck_chart(r)

# ===============================
# ☁️ WEATHER
# ===============================
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

# ===============================
# 🖼️ UNSPLASH IMAGE
# ===============================
def get_unsplash_image_for_city(city):
    try:
        return f"https://source.unsplash.com/1200x800/?{requests.utils.requote_uri(city)}"
    except Exception:
        return None

# ===============================
# 🍜 RESTAURANT HYBRID MODE
# ===============================
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
        df_city = df[df["city"].str.lower().str.contains(city.lower())]
        if df_city.empty:
            return []
        return df_city.head(limit).to_dict("records")
    except Exception:
        return []

def get_restaurants(city, limit=5):
    if GOOGLE_PLACES_KEY:
        data = get_restaurants_google(city, GOOGLE_PLACES_KEY, limit)
        if data and not data[0].get("error"):
            return data
    return get_local_restaurants(city, limit)

# ===============================
# ✈️ SUGGESTIONS / ITINERARY
# ===============================
def suggest_trip_plan(city, days=3):
    return f"📅 **Lịch trình gợi ý {days} ngày tại {city}:**\n- Ngày 1: Khám phá trung tâm thành phố, chợ và di tích.\n- Ngày 2: Tham quan thiên nhiên và ẩm thực.\n- Ngày 3: Mua sắm, chụp ảnh và thưởng thức món đặc sản."

def estimate_cost(city, days=3, people=1):
    total = 800000 * days * people
    return f"💸 Ước tính chi phí cho {people} người {days} ngày ở {city}: khoảng {total:,} VNĐ."

def suggest_local_food(city):
    return f"🍲 Ẩm thực đặc trưng tại {city}: món địa phương, hải sản, cà phê và quán ăn đường phố."

def suggest_events(city):
    return f"🎉 Sự kiện tiêu biểu tại {city}: lễ hội văn hóa, chợ đêm, sự kiện ẩm thực hằng tháng."

def suggest_photospots(city):
    return f"📸 Điểm check-in nổi bật: trung tâm thành phố, bờ sông/biển, quán cà phê view đẹp."

# ===============================
# 🤖 AI CITY & DATE EXTRACTION (Bilingual)
# ===============================
def extract_city_and_dates(user_text):
    try:
        prompt = f"""
You are a multilingual travel information extractor.
The user message may be in **Vietnamese or English**.
Extract:
1. Destination city
2. Start date (YYYY-MM-DD)
3. End date (YYYY-MM-DD)
Return valid JSON only:
{{"city": "...", "start_date": "...", "end_date": "..."}}
Message: "{user_text}"
"""
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "system", "content": prompt}],
            max_tokens=200,
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        start, end = content.find('{'), content.rfind('}')
        data = json.loads(content[start:end+1])
        city, start_date, end_date = data.get("city"), data.get("start_date"), data.get("end_date")

        def _parse(d):
            if not d: return None
            dt = datetime.strptime(d, "%Y-%m-%d")
            if dt.year < datetime.now().year:
                dt = dt.replace(year=datetime.now().year)
            return dt
        start_dt, end_dt = _parse(start_date), _parse(end_date)
        if start_dt and not end_dt:
            end_dt = start_dt
        return city, start_dt, end_dt
    except Exception:
        return None, None, None

# ===============================
# 💬 STREAMLIT UI
# ===============================
st.set_page_config(page_title=f"🤖 {ChatBotName} - Travel Assistant", layout="wide")
st.title(f"🤖 Chatbot Hướng dẫn viên du lịch {ChatBotName}")

with st.sidebar:
    st.header("⚙️ Cài đặt")
    language = st.selectbox("Ngôn ngữ", ["Tự động", "Tiếng Việt", "English"])
    info_options = st.multiselect("Hiển thị thông tin", 
                                  ["Weather", "Itinerary", "Food", "Map", "Photos", "Cost", "Events", "Analytics"],
                                  default=["Weather", "Itinerary", "Food", "Map", "Photos"])
    st.markdown("---")
    if GOOGLE_PLACES_KEY:
        st.success("✅ Google Places API đang hoạt động")
    else:
        st.info("📂 Sử dụng dữ liệu mẫu (CSV fallback)")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = [
        "Thời tiết ở Đà Nẵng tuần tới?",
        "Top món ăn ở Huế?",
        "Lịch trình 3 ngày ở Nha Trang",
        "Có sự kiện gì ở Hà Nội tháng 12?"
    ]

st.write("### 🔎 Gợi ý nhanh")
cols = st.columns(len(st.session_state.suggested_questions))
for i, q in enumerate(st.session_state.suggested_questions):
    if cols[i].button(q):
        st.session_state.user_input = q

user_input = st.chat_input("Mời bạn đặt câu hỏi:")
if "user_input" in st.session_state and st.session_state.user_input:
    user_input = st.session_state.pop("user_input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    city_guess, start_date, end_date = extract_city_and_dates(user_input)
    log_interaction(user_input, city_guess, start_date, end_date)

    responses = []
    if city_guess and "Weather" in info_options:
        responses.append(get_weather_forecast(city_guess, start_date, end_date))
    if city_guess and "Itinerary" in info_options:
        responses.append(suggest_trip_plan(city_guess))
    if city_guess and "Food" in info_options:
        responses.append(suggest_local_food(city_guess))
    if city_guess and "Cost" in info_options:
        responses.append(estimate_cost(city_guess))
    if city_guess and "Events" in info_options:
        responses.append(suggest_events(city_guess))
    if city_guess and "Photos" in info_options:
        responses.append(suggest_photospots(city_guess))

    for r in responses:
        st.chat_message("assistant").markdown(r)

    lat, lon, full_address = geocode_city(city_guess) if city_guess else (None, None, None)
    cols = st.columns([2, 3])
    with cols[0]:
        if "Map" in info_options:
            show_map(lat, lon, title=full_address or city_guess)
        if "Photos" in info_options:
            img_url = get_unsplash_image_for_city(city_guess)
            if img_url:
                st.image(img_url, caption=f"Hình ảnh gợi ý cho {city_guess}")
    with cols[1]:
        if "Food" in info_options:
            st.subheader("🍽️ Gợi ý ẩm thực & nhà hàng")
            restaurants = get_restaurants(city_guess, limit=5)
            if restaurants:
                df = pd.DataFrame(restaurants)
                if "maps_url" in df.columns:
                    df["name"] = df.apply(lambda x: f'<a href="{x.get("maps_url")}" target="_blank">{x["name"]}</a>', axis=1)
                    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
                else:
                    st.dataframe(df)
            else:
                st.info("Không tìm thấy quán ăn phù hợp.")

# ===============================
# 📊 ANALYTICS
# ===============================
st.markdown("---")
st.header("📊 Thống kê truy vấn gần đây")
try:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM interactions ORDER BY timestamp DESC LIMIT 200", conn)
    st.metric("Tổng lượt tương tác", int(df.shape[0]))
    if not df.empty:
        st.write(df[["timestamp", "user_input", "city"]])
    conn.close()
except Exception:
    st.info("Chưa có dữ liệu tương tác.")
