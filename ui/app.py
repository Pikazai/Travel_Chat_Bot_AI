"""
Main Streamlit UI application for Travel Chatbot.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import time
from datetime import datetime, timedelta
from typing import List, Dict

from config.settings import get_settings
from core.chat_engine import ChatEngine
from services.logger_service import LoggerService
from services.weather_service import WeatherService
from services.places_service import PlacesService
from services.image_service import ImageService
from services.voice_service import VoiceService
from services.chroma_service import ChromaService
from utils.geocoding import geocode_city, show_map
from ui.styles import STYLES
from ui.components import (
    render_hero_section,
    render_sidebar,
    render_quick_search,
    render_suggestions,
    render_voice_input
)


def initialize_session_state():
    """Initialize session state variables."""
    settings = get_settings()
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": settings.SYSTEM_PROMPT}]
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = ChatEngine()
    if "logger" not in st.session_state:
        st.session_state.logger = LoggerService()
    if "weather_service" not in st.session_state:
        st.session_state.weather_service = WeatherService()
        st.session_state.weather_service.set_openai_client(st.session_state.chat_engine.client)
    if "places_service" not in st.session_state:
        st.session_state.places_service = PlacesService()
        st.session_state.places_service.set_openai_client(st.session_state.chat_engine.client)
    if "image_service" not in st.session_state:
        st.session_state.image_service = ImageService()
    if "voice_service" not in st.session_state:
        st.session_state.voice_service = VoiceService()
    if "chroma_service" not in st.session_state:
        st.session_state.chroma_service = ChromaService()
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = st.session_state.chat_engine.generate_suggestions()


def render_analytics_tab():
    """Render analytics tab with statistics and ChromaDB debugger."""
    st.header("📊 Thống kê truy vấn (gần đây)")
    
    # Delete history option
    with st.expander("🗑️ Xóa lịch sử truy vấn"):
        st.warning("⚠️ Thao tác này sẽ xóa toàn bộ lịch sử truy vấn đã lưu trong cơ sở dữ liệu (SQLite). Không thể hoàn tác.")
        confirm_delete = st.checkbox("Tôi hiểu và muốn xóa toàn bộ lịch sử truy vấn", value=False)
        if confirm_delete:
            if st.button("✅ Xác nhận xóa toàn bộ lịch sử"):
                try:
                    st.session_state.logger.clear_interactions()
                    st.success("✅ Đã xóa toàn bộ lịch sử truy vấn.")
                except Exception as e:
                    st.error(f"⚠️ Lỗi khi xóa dữ liệu: {e}")
        else:
            st.info("👉 Hãy tick vào ô xác nhận trước khi xóa lịch sử.")
    
    # Display statistics
    try:
        interactions = st.session_state.logger.get_interactions(limit=1000)
        if interactions:
            df_logs = pd.DataFrame(interactions, columns=["id", "timestamp", "user_input", "city", "start_date", "end_date", "intent"])
            total = len(df_logs)
            st.metric("Tổng tương tác", total)
            
            df_logs['timestamp_dt'] = pd.to_datetime(df_logs['timestamp'])
            df_logs['date'] = df_logs['timestamp_dt'].dt.date
            series = df_logs.groupby('date').size().reset_index(name='queries')
            
            fig = px.bar(
                series,
                x='date',
                y='queries',
                title='📈 Số truy vấn mỗi ngày',
                color='queries',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            top_cities = df_logs['city'].fillna("Unknown").value_counts().reset_index()
            top_cities.columns = ['city', 'count']
            if not top_cities.empty:
                fig2 = px.bar(
                    top_cities.head(10),
                    x='city',
                    y='count',
                    title='📍 Top địa điểm được hỏi',
                    color='count',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            st.dataframe(df_logs[["timestamp", "user_input", "city"]])
        else:
            st.info("Chưa có truy vấn nào được ghi nhận.")
    except Exception as e:
        st.warning(f"Lỗi đọc dữ liệu: {e}")
    
    # ChromaDB RAG Debugger
    if st.session_state.chroma_service.is_available():
        with st.expander("🧪 RAG Debugger"):
            q = st.text_input("Query thử", "Đặc sản ở Huế?")
            c = st.text_input("City filter", "")
            k = st.slider("Top‑k", 1, 10, 5, key="ragdbg_k")
            if st.button("Chạy truy vấn"):
                items = st.session_state.chroma_service.retrieve_context(q, c or None, k=k)
                for it in items:
                    st.write(f"{it['meta']}  |  dist={it['dist']:.4f}")
                    st.write(it["doc"])
                    st.markdown("---")
    else:
        st.info("Cài 'chromadb' để sử dụng RAG Debugger.")


def main():
    """Main application entry point."""
    settings = get_settings()
    
    # Page configuration
    st.set_page_config(
        page_title=f"🤖 {settings.CHATBOT_NAME} - Travel Assistant",
        layout="wide",
        page_icon="🎙️"
    )
    
    # Apply styles
    st.markdown(STYLES, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar FIRST (before hero and tabs, like original)
    sidebar_config = render_sidebar()
    
    # Render hero section
    render_hero_section()
    
    # Create tabs
    main_tab, analytics_tab = st.tabs(["💬 Chatbot Du lịch", "📊 Thống kê truy vấn"])
    
    # Main tab
    with main_tab:
        # Quick search
        quicksearch = render_quick_search()
        
        if quicksearch:
            city_qs = quicksearch["city"]
            start_qs = quicksearch["start"]
            end_qs = quicksearch["end"]
            people_qs = quicksearch["people"]
            style_qs = quicksearch["style"]
            
            st.markdown(f"### ✈️ Gợi ý cho chuyến đi {city_qs} ({start_qs} – {end_qs})")
            
            weather_qs = st.session_state.weather_service.get_weather_forecast(
                city_qs, start_qs, end_qs
            )
            cost_qs = st.session_state.chat_engine.estimate_cost(
                city_qs,
                (end_qs - start_qs).days + 1,
                people_qs,
                style_qs
            )
            
            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"**{weather_qs}**")
                st.markdown(f"**{cost_qs}**")
            with colB:
                img = st.session_state.image_service.get_city_image(city_qs)
                if img:
                    st.image(img, caption=f"🏞️ {city_qs}", use_container_width=True)
                lat, lon, addr = geocode_city(city_qs)
                if lat and lon:
                    show_map(lat, lon, zoom=sidebar_config["map_zoom"], title=addr or city_qs)
            st.markdown("---")
        
        # Suggestions
        render_suggestions(st.session_state.suggested_questions)
        
        # Voice input
        render_voice_input(
            st.session_state.voice_service,
            sidebar_config["enable_voice"],
            sidebar_config["asr_lang"]
        )
        
        # Display conversation history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="🧭"):
                    st.markdown(f"<div class='user-message'>{msg['content']}</div>", unsafe_allow_html=True)
            elif msg["role"] == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(f"<div class='assistant-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
        
        # Chat input
        user_input = st.chat_input("Mời bạn đặt câu hỏi:")
        if "user_input" in st.session_state and st.session_state.user_input:
            user_input = st.session_state.user_input
            del st.session_state.user_input
        
        if user_input:
            # Display user message
            with st.chat_message("user", avatar="🧭"):
                st.markdown(f"<div class='user-message'>{user_input}</div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Process message
            result = st.session_state.chat_engine.process_message(
                user_input,
                st.session_state.messages,
                use_rag=sidebar_config.get("use_rag", False) and st.session_state.chroma_service.is_available(),
                use_cache=sidebar_config.get("use_cache", False) and st.session_state.chroma_service.is_available(),
                rag_k=sidebar_config.get("rag_k", 6)
            )
            
            # Log interaction
            st.session_state.logger.log_interaction(
                user_input,
                result["city"],
                result["start_date"],
                result["end_date"]
            )
            
            # Display info blocks
            blocks = []
            if result["city"] and "Weather" in sidebar_config["info_options"]:
                blocks.append(
                    st.session_state.weather_service.get_weather_forecast(
                        result["city"],
                        result["start_date"],
                        result["end_date"],
                        user_input
                    )
                )
            if result["city"] and "Cost" in sidebar_config["info_options"]:
                blocks.append(
                    st.session_state.chat_engine.estimate_cost(
                        result["city"],
                        days=result["days"]
                    )
                )
            if result["city"] and "Events" in sidebar_config["info_options"]:
                blocks.append(
                    st.session_state.chat_engine.suggest_events(result["city"])
                )
            
            for b in blocks:
                with st.chat_message("assistant", avatar="🤖"):
                    if isinstance(b, str):
                        st.markdown(b.replace("\\n", "\n"))
                    else:
                        st.write(b)
            
            # Generate AI response
            with st.spinner("⏳ Đang soạn phản hồi..."):
                try:
                    progress_text = "AI đang phân tích dữ liệu du lịch..."
                    progress_bar = st.progress(0, text=progress_text)
                    for percent_complete in range(0, 101, 20):
                        time.sleep(0.08)
                        progress_bar.progress(percent_complete, text=progress_text)
                    progress_bar.empty()
                    
                    assistant_text = result["response"]
                    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                    
                    # Display response with typing animation
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
                            st.markdown("<div class='assistant-bubble'>", unsafe_allow_html=True)
                            st.markdown(assistant_text)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # TTS
                            if sidebar_config["tts_enable"]:
                                audio_b64 = st.session_state.voice_service.text_to_speech(
                                    assistant_text,
                                    sidebar_config["tts_lang"]
                                )
                                if audio_b64:
                                    st.markdown(
                                        f'<div class="audio-wrapper"><audio autoplay controls><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio></div>',
                                        unsafe_allow_html=True
                                    )
                    
                    st.balloons()
                except Exception as e:
                    st.error(f"⚠️ Lỗi khi gọi OpenAI: {e}")
            
            # Display maps, photos, and food
            lat, lon, addr = (None, None, None)
            if result["city"]:
                lat, lon, addr = geocode_city(result["city"])
            
            cols = st.columns([2, 3])
            with cols[0]:
                if "Map" in sidebar_config["info_options"]:
                    show_map(lat, lon, zoom=sidebar_config["map_zoom"], title=addr or result["city"])
                if "Photos" in sidebar_config["info_options"]:
                    img = st.session_state.image_service.get_city_image(result["city"])
                    if img:
                        st.image(img, caption=f"🏞️ {result['city']}", use_container_width=True)
                    else:
                        st.info("Không tìm thấy ảnh minh họa.")
            
            with cols[1]:
                if "Food" in sidebar_config["info_options"]:
                    st.subheader(f"🍽️ Ẩm thực & Nhà hàng tại {result['city'] or 'địa điểm'}")
                    
                    # Foods
                    foods = st.session_state.places_service.get_foods(result["city"]) if result["city"] else []
                    if foods:
                        st.markdown("#### 🥘 Đặc sản nổi bật")
                        food_images = st.session_state.image_service.get_food_images(foods)
                        img_cols = st.columns(min(len(food_images), 4))
                        for i, item in enumerate(food_images):
                            with img_cols[i % len(img_cols)]:
                                if item["image"]:
                                    st.image(item["image"], caption=item["name"], use_container_width=True)
                                else:
                                    st.write(f"- {item['name']}")
                    else:
                        st.info("Không tìm thấy món đặc trưng.")
                    
                    # Restaurants
                    if result["city"]:
                        st.markdown("#### 🍴 Nhà hàng gợi ý")
                        restaurants = st.session_state.places_service.get_restaurants(result["city"], limit=5)
                        if restaurants:
                            for r in restaurants:
                                if isinstance(r, dict) and r.get("error"):
                                    st.write(f"⚠️ {r.get('error')}")
                                else:
                                    name = r.get("name") or r.get("place_name") or str(r)
                                    rating = r.get("rating", "")
                                    addr_text = r.get("address", r.get("formatted_address", ""))
                                    maps_url = r.get("maps_url", "")
                                    st.markdown(
                                        f"- **{name}** {f'• ⭐ {rating}' if rating else ''}  \n  {addr_text}  " +
                                        (f"[Bản đồ]({maps_url})" if maps_url else "")
                                    )
                        else:
                            st.info("Không có dữ liệu nhà hàng.")
    
    # Analytics tab
    with analytics_tab:
        render_analytics_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div class='small-muted'>Tip: Bạn có thể yêu cầu cụ thể như 'Lịch trình 3 ngày ở Hội An', 'Đặc sản Sapa', hoặc 'Thời tiết Đà Nẵng 2025-10-20 đến 2025-10-22'.</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

