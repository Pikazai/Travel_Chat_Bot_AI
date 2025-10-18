import streamlit as st
import openai
import json
import requests
import os
from datetime import datetime, timedelta

# ================== LẤY CÁC BIẾN MÔI TRƯỜNG ==================
OPENAI_API_KEY = st.secrets["openai"]["OPENAI_API_KEY"]
OPENAI_ENDPOINT = st.secrets["openai"].get("OPENAI_ENDPOINT", "https://api.openai.com/v1")
DEPLOYMENT_NAME = st.secrets["openai"].get("DEPLOYMENT_NAME", "gpt-4o-mini")
OPENWEATHERMAP_API_KEY = st.secrets["weather"]["OPENWEATHERMAP_API_KEY"]

# ================== CẤU HÌNH OPENAI CHATBOT ==================
client = openai.OpenAI(
    base_url=OPENAI_ENDPOINT,
    api_key=OPENAI_API_KEY
)

ChatBotName = "Team_XYZ"

system_prompt = """
## VAI TRÒ HỆ THỐNG ##
Bạn là Hướng dẫn viên du lịch ảo Alex - người kể chuyện, am hiểu văn hóa, lịch sử, ẩm thực và thời tiết Việt Nam.
Luôn đưa ra thông tin hữu ích, gợi ý lịch trình, món ăn, chi phí, thời gian lý tưởng, sự kiện và góc chụp ảnh.
"""

# ================== TÍNH NĂNG: LẤY THÔNG TIN THỜI TIẾT ==================

# API dự báo 5 ngày của OpenWeatherMap (mỗi 3 tiếng)
def get_weather_forecast(city_name, start_date=None, end_date=None):
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={OPENWEATHERMAP_API_KEY}&lang=vi&units=metric"
        response = requests.get(url)
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

# ================== GIAO DIỆN STREAMLIT ==================

st.title(f"🤖 Chatbot Hướng dẫn viên du lịch {ChatBotName}")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

# Hàm trích xuất địa điểm và thời gian thông minh bằng AI
def extract_city_and_dates(user_text):
    try:
        prompt = (
            "Hãy phân tích câu sau và trích xuất: \n"
            "1. Tên địa điểm du lịch (city)\n"
            "2. Ngày bắt đầu và ngày kết thúc (nếu có).\n"
            "Nếu chỉ có ngày/tháng mà không có năm, hãy tự động dùng năm hiện tại.\n"
            "Định dạng trả về JSON: {\"city\": \"...\", \"start_date\": \"YYYY-MM-DD\", \"end_date\": \"YYYY-MM-DD\" hoặc null nếu không có}.\n"
            f"\nCâu: '{user_text}'"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=200,
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        # Đảm bảo nội dung trả về là JSON hợp lệ. Nếu AI trả văn bản thêm, ta cố gắng trích xuất phần JSON.
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # tìm phần bắt đầu '{' và kết thúc '}' để cắt
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                data = json.loads(content[start:end+1])
            else:
                return None, None, None

        city = data.get("city")
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        current_year = datetime.now().year
        today = datetime.now().date()

        def _parse_and_normalize(d):
            if not d:
                return None
            # parse
            dt = datetime.strptime(d, "%Y-%m-%d")
            # nếu AI trả năm là 1900 (khi chỉ có ngày-tháng), hoặc năm nhỏ hơn năm hiện tại -> đặt về năm hiện tại
            if dt.year == 1900 or dt.year < current_year:
                try:
                    dt = dt.replace(year=current_year)
                except ValueError:
                    # ví dụ 29/02 với năm không nhuận, bỏ về 28/02
                    dt = dt.replace(year=current_year, day=28)
            # Nếu sau khi đặt về năm hiện tại mà ngày đã qua (trong tương lai người dùng có thể muốn mốc kế tiếp),
            # ta giữ nguyên (không tự nhảy năm), nhưng sẽ có cảnh báo sau.
            return dt

        start_dt = _parse_and_normalize(start_date)
        end_dt = _parse_and_normalize(end_date)

        # Nếu chỉ có start_date nhưng không có end_date, đặt end_date = start_date
        if start_dt and not end_dt:
            end_dt = start_dt

        return city, start_dt, end_dt
    except Exception as e:
        return None, None, None

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

    city_guess, start_date, end_date = extract_city_and_dates(user_input)

    if city_guess:
        # Kiểm tra phạm vi dự báo của OpenWeather (khoảng 5 ngày từ hôm nay)
        today = datetime.now().date()
        max_forecast_date = today + timedelta(days=5)
        warning_msgs = []
        if start_date and start_date.date() > max_forecast_date:
            warning_msgs.append(f"⚠️ Lưu ý: Dữ liệu dự báo của OpenWeather chỉ cung cấp tối đa 5 ngày tới (tính từ hôm nay {today.strftime('%d/%m/%Y')}). Bạn yêu cầu bắt đầu từ {start_date.strftime('%d/%m/%Y')} > phạm vi dự báo.")
        if end_date and end_date.date() > max_forecast_date:
            warning_msgs.append(f"⚠️ Lưu ý: Dữ liệu dự báo của OpenWeather chỉ cung cấp tối đa 5 ngày tới. Bạn yêu cầu kết thúc vào {end_date.strftime('%d/%m/%Y')} > phạm vi dự báo.")

        for w in warning_msgs:
            st.session_state.messages.append({"role": "assistant", "content": w})

        weather_info = get_weather_forecast(city_guess, start_date, end_date)
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
                model=DEPLOYMENT_NAME,
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
