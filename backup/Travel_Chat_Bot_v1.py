import streamlit as st
import openai
import json
import requests
from datetime import datetime

# ================== TÍNH NĂNG: LẤY THÔNG TIN THỜI TIẾT ==================

OPENWEATHER_API_KEY = "89fb4e8431599d5dba5e5d53a3fe4162"

def get_weather(city_name):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={OPENWEATHER_API_KEY}&lang=vi&units=metric"
        response = requests.get(url)
        data = response.json()

        if data.get("cod") != 200:
            return f"❌ Không tìm thấy thông tin thời tiết cho địa điểm: **{city_name}**."

        weather = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]

        suggestion = ""
        if "mưa" in weather.lower():
            suggestion = "Trời đang có mưa, bạn nên mang theo áo mưa và tránh các hoạt động ngoài trời."
        elif temp >= 33:
            suggestion = "Trời khá nóng, hãy mang theo nước và tránh đi vào buổi trưa."
        elif temp <= 20:
            suggestion = "Thời tiết se lạnh, bạn nên mang theo áo khoác."
        else:
            suggestion = "Thời tiết dễ chịu, rất phù hợp cho các hoạt động du lịch ngoài trời."

        return (
            f"🌤 **Thời tiết tại {city_name}**:\n"
            f"- Trạng thái: {weather}\n"
            f"- Nhiệt độ: {temp}°C (Cảm giác như {feels_like}°C)\n"
            f"- Độ ẩm: {humidity}%\n"
            f"- Tốc độ gió: {wind_speed} m/s\n\n"
            f"💡 **Gợi ý du lịch:** {suggestion}"
        )
    except Exception as e:
        return f"⚠️ Lỗi khi lấy dữ liệu thời tiết: {e}"

# ================== CÁC TÍNH NĂNG MỞ RỘNG ==================

def suggest_best_time(city):
    return f"🗓 Gợi ý: Thời điểm lý tưởng để du lịch {city} là từ tháng 3 đến tháng 8, khi thời tiết khô ráo và biển xanh đẹp nhất!"

def suggest_local_food(city):
    return f"🍜 Ẩm thực đặc trưng tại {city}: bánh mì, bún bò, hải sản tươi sống và cà phê địa phương. Đừng bỏ lỡ các quán nổi tiếng gần trung tâm!"

def suggest_trip_plan(city, days=3):
    plans = [
        f"📅 **Lịch trình gợi ý {days} ngày tại {city}:**",
        "- Ngày 1: Khám phá trung tâm thành phố, các điểm văn hóa, chợ địa phương.",
        "- Ngày 2: Tham quan thiên nhiên, danh lam thắng cảnh, trải nghiệm ẩm thực đặc trưng.",
        "- Ngày 3: Mua sắm, chụp hình lưu niệm và thưởng thức món đặc sản trước khi rời đi."
    ]
    return "\n".join(plans)

def estimate_cost(city, days=3):
    cost = days * 2000000
    return f"💸 Chi phí ước tính cho chuyến đi {days} ngày tại {city}: khoảng {cost:,} VNĐ/người (bao gồm ăn, ở, di chuyển và vé tham quan)."

def suggest_events(city):
    return f"🎉 Sự kiện nổi bật tại {city}: Lễ hội văn hóa, các show âm nhạc và hội chợ du lịch địa phương diễn ra hằng tháng."

def suggest_photospots(city):
    return f"📸 Gợi ý điểm check-in tại {city}: trung tâm thành phố, khu vực bờ biển, cầu nổi tiếng, và các quán cà phê view đẹp lúc hoàng hôn."

def personalized_memory(user_name, preference):
    return f"Xin chào {user_name}! Lần trước bạn thích du lịch {preference}, lần này bạn có muốn khám phá một nơi tương tự không?"

# ================== CẤU HÌNH OPENAI CHATBOT ==================

client = openai.OpenAI(
    base_url="https://aiportalapi.stu-platform.live/jpe",
    api_key="sk-DXNNoBnsC6GG_GePaPVBLg"
)

ChatBotName = "Team_XYZ"

system_prompt = """
## VAI TRÒ HỆ THỐNG ##
Bạn là Hướng dẫn viên du lịch ảo Alex - người kể chuyện, am hiểu văn hóa, lịch sử, ẩm thực và thời tiết Việt Nam.
Luôn đưa ra thông tin hữu ích, gợi ý lịch trình, món ăn, chi phí, thời gian lý tưởng, sự kiện và góc chụp ảnh.
"""

# ================== GIAO DIỆN STREAMLIT ==================

st.title(f"🤖 Chatbot Hướng dẫn viên du lịch {ChatBotName}")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

# Hàm tách địa điểm thông minh bằng AI
def extract_city_from_input(user_text):
    try:
        prompt = f"Hãy trích xuất tên thành phố hoặc địa điểm du lịch từ câu sau: '{user_text}'. Chỉ trả về tên địa điểm duy nhất, không thêm gì khác."
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=20,
            temperature=0
        )
        city_name = response.choices[0].message.content.strip()
        return city_name
    except Exception as e:
        return None

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])

user_input = st.chat_input("Mời bạn đặt câu hỏi:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    city_guess = extract_city_from_input(user_input)
    if city_guess:
        weather_info = get_weather(city_guess)
        st.session_state.messages.append({"role": "assistant", "content": weather_info})

        st.session_state.messages.append({"role": "assistant", "content": suggest_best_time(city_guess)})
        st.session_state.messages.append({"role": "assistant", "content": suggest_local_food(city_guess)})
        st.session_state.messages.append({"role": "assistant", "content": suggest_trip_plan(city_guess)})
        st.session_state.messages.append({"role": "assistant", "content": estimate_cost(city_guess)})
        st.session_state.messages.append({"role": "assistant", "content": suggest_events(city_guess)})
        st.session_state.messages.append({"role": "assistant", "content": suggest_photospots(city_guess)})

    with st.spinner("Đang suy nghĩ..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages,
                max_tokens=2000,
                temperature=0.7
            )
            assistant_response = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

            st.session_state.suggested_questions.clear()
            for line in assistant_response.split('\n'):
                if line.strip().startswith(('1.', '2.', '3.')):
                    st.session_state.suggested_questions.append(line.split('.', 1)[1].strip())

            st.rerun()
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
            st.session_state.messages.pop()
