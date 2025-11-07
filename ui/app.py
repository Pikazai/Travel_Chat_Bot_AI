"""
Streamlit UI module for Travel Chat Bot AI.
Main interface using modular architecture.
"""

import streamlit as st
import time
import re
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
from streamlit_mic_recorder import mic_recorder

from config.settings import Settings, get_settings
from core.chat_engine import ChatEngine
from services.chroma_service import ChromaService
from services.voice_service import VoiceService
from services.logger_service import LoggerService
from services.weather_service import WeatherService
from services.geocoding_service import GeocodingService
from services.image_service import ImageService
from services.food_service import FoodService
from services.restaurant_service import RestaurantService
from utils.text_processing import extract_days_from_text


def render_css():
    """Render custom CSS styles."""
    st.markdown("""
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
    .user-message { background: #f2f2f2; padding:10px; border-radius:12px; }
    .assistant-message { background: #e7f3ff; padding:10px; border-radius:12px; }
    .assistant-bubble {
      background-color: #e7f3ff;
      padding: 12px 16px;
      border-radius: 15px;
      margin-bottom: 6px;
    }
    .status-ok { background:#d4edda; padding:8px; border-radius:8px; }
    .status-bad { background:#f8d7da; padding:8px; border-radius:8px; }
    .source-badge {
      background: #e8f5e8;
      border: 1px solid #4caf50;
      border-radius: 12px;
      padding: 8px 12px;
      margin: 5px 0;
      font-size: 0.85em;
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
    .hero-banner {
      position: relative;
      width: 100%;
      height: 400px;
      border-radius: 16px;
      overflow: hidden;
      margin-bottom: 24px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }
    .hero-banner::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(135deg, rgba(43, 76, 126, 0.6) 0%, rgba(0, 0, 0, 0.3) 100%);
      z-index: 1;
    }
    .hero-content {
      position: relative;
      z-index: 2;
      height: 100%;
      display: flex;
      flex-direction: column;
      justify-content: center;
      padding: 40px;
      color: white;
    }
    .hero-title {
      font-size: 3.5rem;
      font-weight: 700;
      margin-bottom: 16px;
      line-height: 1.2;
      text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.5);
    }
    .hero-subtitle {
      font-size: 1.2rem;
      margin-bottom: 24px;
      opacity: 0.95;
      line-height: 1.6;
      max-width: 700px;
    }
    .hero-feature {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 1rem;
      margin-top: 16px;
    }
    .hero-search-form {
      background: white;
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
      margin-top: -60px;
      position: relative;
      z-index: 3;
      margin-bottom: 32px;
    }
    @media (max-width: 768px) {
      .hero-title {
        font-size: 2.5rem;
      }
      .hero-subtitle {
        font-size: 1rem;
      }
      .hero-banner {
        height: 300px;
      }
      .hero-search-form {
        margin-top: -40px;
        padding: 16px;
      }
    }
    </style>
    """, unsafe_allow_html=True)


def render_hero_section():
    """Render hero search section with banner and search form."""
    # Hero Banner với background image
    hero_image_url = "https://images.unsplash.com/photo-1633073985249-b2d67bdf6b7d?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=1074"
    
    st.markdown(f"""
    <div class="hero-banner" style="background-image: url('{hero_image_url}'); background-size: cover; background-position: center;">
        <div class="hero-content">
            <div class="hero-title">Khám phá Việt Nam cùng Mây<br>Lang Thang</div>
            <div class="hero-subtitle">Gợi ý lịch trình, món ăn, dự báo thời tiết. Nhập điểm đến, chọn ngày và bắt đầu cuộc hành trình!</div>
            <div class="hero-feature">
                <span>☀️</span>
                <span>Tìm nhanh & gợi ý tức thì</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Search Form với wrapper
    st.markdown("""
    <div class="hero-search-form">
    """, unsafe_allow_html=True)
    
    with st.form(key='hero_search_form', clear_on_submit=False):
        cols = st.columns([3, 2, 1, 1])
        dest = cols[0].text_input("Điểm đến", placeholder="Hội An, Đà Nẵng, Hà Nội...", key="hero_dest")
        dates = cols[1].date_input("Ngày (bắt đầu / kết thúc)", [], key="hero_dates")
        people = cols[2].selectbox("Người", [1, 2, 3, 4, 5, 6], index=0, key="hero_people")
        style = cols[3].selectbox("Mức chi", ["trung bình", "tiết kiệm", "cao cấp"], index=0, key="hero_style")
        submitted = st.form_submit_button("Tìm kiếm nhanh", use_container_width=True)
        if submitted and dest:
            from datetime import datetime
            if len(dates) == 2:
                s = dates[0]
                e = dates[1]
                q = f"Lịch trình {((dates[1]-dates[0]).days +1)} ngày ở {dest} từ {s.strftime('%Y-%m-%d')} đến {e.strftime('%Y-%m-%d')}"
            elif len(dates) == 1:
                s = dates[0]
                e = dates[0]
                q = f"Lịch trình 1 ngày ở {dest} vào {s.strftime('%Y-%m-%d')}"
            else:
                from datetime import timedelta
                today = datetime.now().date()
                s = today
                e = today + timedelta(days=2)
                q = f"Lịch trình 3 ngày ở {dest}"
            
            # Set quicksearch state (for display) and user_input (for chat)
            if len(dates) >= 1:
                st.session_state.quicksearch = {
                    "city": dest,
                    "start": datetime.combine(s, datetime.min.time()) if isinstance(s, type(datetime.now().date())) else s,
                    "end": datetime.combine(e, datetime.min.time()) if isinstance(e, type(datetime.now().date())) else e,
                    "people": people,
                    "style": style
                }
            else:
                from datetime import timedelta
                today = datetime.now().date()
                st.session_state.quicksearch = {
                    "city": dest,
                    "start": datetime.combine(today, datetime.min.time()),
                    "end": datetime.combine(today + timedelta(days=2), datetime.min.time()),
                    "people": people,
                    "style": style
                }
            
            q += f" • người: {people} • mức: {style}"
            st.session_state.user_input = q
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)


def estimate_cost(city: str, days: int = 3, people: int = 1, style: str = "trung bình") -> str:
    """Estimate travel cost."""
    mapping = {"tiết kiệm": 400000, "trung bình": 800000, "cao cấp": 2000000}
    per_day = mapping.get(style, 800000)
    total = per_day * days * people
    return f"💸 Chi phí ước tính: khoảng {total:,} VNĐ cho {people} người, {days} ngày."


def suggest_events(city: str) -> str:
    """Suggest events for a city."""
    return f"🎉 Sự kiện ở {city}: lễ hội địa phương, chợ đêm, hội chợ ẩm thực (tuỳ mùa)."


def suggest_photospots(city: str) -> str:
    """Suggest photo spots for a city."""
    return f"📸 Gợi ý check-in: trung tâm lịch sử, bờ sông/biển, quán cà phê có view đẹp."


def suggest_local_food(city: str) -> str:
    """Suggest local food for a city."""
    return f"🍜 Gõ 'Đặc sản {city}' để nhận danh sách món ăn nổi bật."


def render_sidebar(settings: Settings, chroma_service: ChromaService):
    """Render sidebar with settings."""
    st.markdown("<div class='logo-title'><h2>Mây Lang Thang</h2></div>", unsafe_allow_html=True)
    st.header("⚙️ Cài đặt")
    
    info_options = st.multiselect(
        "Hiển thị thông tin",
        ["Weather", "Food", "Map", "Photos", "Cost", "Events"],
        default=["Weather", "Map", "Food", "Photos"]
    )
    
    st.markdown("---")
    st.subheader("🎙️ Voice")
    enable_voice = st.checkbox("Bật nhập liệu bằng giọng nói", value=True)
    tts_enable = st.checkbox("🔊 Đọc to phản hồi", value=False)
    
    st.subheader("🗺️ Chọn mức zoom bản đồ:")
    map_zoom = st.slider("Zoom (4 = xa, 15 = gần)", 4, 15, 8)
    
    if st.button("🔄 Seed dữ liệu du lịch", width='stretch'):
        if chroma_service.seed_from_csv(str(settings.TRAVEL_DOCS_CSV)):
            st.success("✅ Đã seed dữ liệu thành công!")
        else:
            st.error("❌ Lỗi khi seed dữ liệu")
    
    st.markdown("---")
    
    def status_card(title, ok=True):
        cls = "status-ok" if ok else "status-bad"
        icon = "✅" if ok else "⚠️"
        st.markdown(f"<div class='{cls}'>{icon} {title}</div>", unsafe_allow_html=True)
    
    status_card("OpenWeatherMap", bool(settings.OPENWEATHERMAP_API_KEY))
    status_card("Pixabay", bool(settings.PIXABAY_API_KEY))
    status_card("ChromaDB RAG", chroma_service.travel_col is not None)
    status_card("Embedding Model", chroma_service.embedding_model is not None)
    
    st.caption("Version: v2.0 • ChromaDB RAG + Cache + Memory")
    
    return info_options, enable_voice, tts_enable, map_zoom


def render_main_chat(settings: Settings, chat_engine: ChatEngine, voice_service: VoiceService,
                     weather_service: WeatherService, geocoding_service: GeocodingService,
                     image_service: ImageService, food_service: FoodService,
                     restaurant_service: RestaurantService, info_options: list,
                     enable_voice: bool, tts_enable: bool, map_zoom: int, openai_client,
                     chroma_service: ChromaService):
    """Render main chat interface."""
    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": settings.SYSTEM_PROMPT}]
    
    # Quicksearch display (from hero section) - chỉ hiển thị khi không có user_input đang chờ xử lý
    # Kiểm tra user_input trước để tránh hiển thị quicksearch khi đang xử lý chat
    has_pending_input = "user_input" in st.session_state and st.session_state.user_input
    
    if not has_pending_input and "quicksearch" in st.session_state and st.session_state.quicksearch:
        qs = st.session_state.quicksearch
        city_qs = qs.get("city")
        start_qs = qs.get("start")
        end_qs = qs.get("end")
        people_qs = qs.get("people", 1)
        style_qs = qs.get("style", "trung bình")
        
        if city_qs and start_qs and end_qs:
            st.markdown(f"### ✈️ Gợi ý cho chuyến đi {city_qs} ({start_qs} – {end_qs})")
            weather_qs = weather_service.get_forecast(city_qs, start_qs, end_qs)
            cost_qs = estimate_cost(city_qs, (end_qs - start_qs).days + 1, people_qs, style_qs)
            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"**{weather_qs}**")
                st.markdown(f"**{cost_qs}**")
            with colB:
                img = image_service.get_city_image(city_qs)
                if img:
                    st.image(img, caption=f"🏞️ {city_qs}", width='stretch')
                lat, lon, addr = geocoding_service.geocode_city(city_qs)
                if lat and lon:
                    geocoding_service.show_map(lat, lon, zoom=map_zoom, title=addr or city_qs)
            st.markdown("---")
            # Xóa quicksearch sau khi hiển thị để tránh hiển thị lại
            del st.session_state.quicksearch
    
    # Voice input
    voice_text = None
    if enable_voice:
        audio = mic_recorder(
            start_prompt="🎙️ [Chat voice] Nói để nhập câu hỏi",
            stop_prompt="✋ Dừng nhận diện giọng nói",
            just_once=True,
            key="rec_chat"
        )
        if audio:
            st.info("Đã nhận dữ liệu âm thanh, đang xử lý...")
            voice_text = voice_service.speech_to_text(audio["bytes"])
            if voice_text:
                st.success(f"🗣️ Bạn nói: {voice_text}")
                st.session_state.user_input = voice_text
                st.rerun()
    
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧭"):
                st.markdown(f"<div class='user-message'>{msg['content']}</div>", unsafe_allow_html=True)
        elif msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                assistant_text = msg.get("content", "")
                # Highlight citations like [src:...] or [mem:...]
                display_text_processed = re.sub(r'(\[src:[^\]]+\])', r'**\1**', assistant_text)
                display_text_processed = re.sub(r'(\[mem:[^\]]+\])', r'**\1**', display_text_processed)
                st.markdown("<div class='assistant-bubble'>", unsafe_allow_html=True)
                st.markdown(display_text_processed)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Hiển thị nguồn tham khảo nếu có metadata
                rag_used = msg.get("rag_used", False)
                sources_count = msg.get("sources_count", 0)
                rag_docs = msg.get("rag_docs", [])
                intent_used = msg.get("intent_used")
                memory_used = msg.get("memory_used", False)
                
                if rag_used or intent_used or memory_used:
                    st.markdown("---")
                    st.subheader("🔍 Nguồn tham khảo")
                    
                    # Hiển thị thông tin về ChromaDB
                    if rag_used and sources_count > 0:
                        st.markdown(f'<div class="source-badge source-chroma">📚 <b>ChromaDB RAG</b>: Sử dụng {sources_count} tài liệu từ cơ sở tri thức du lịch</div>', unsafe_allow_html=True)
                    
                    if intent_used:
                        st.markdown(f'<div class="source-badge source-intent">🎯 <b>Intent Matching</b>: Phát hiện intent "{intent_used}" từ ChromaDB</div>', unsafe_allow_html=True)
                    
                    if memory_used:
                        st.markdown(f'<div class="source-badge source-memory">💭 <b>Memory Recall</b>: Tham khảo hội thoại trước đó từ ChromaDB</div>', unsafe_allow_html=True)
                    
                    # Hiển thị chi tiết các tài liệu RAG
                    if rag_docs:
                        with st.expander(f"📖 Chi tiết {len(rag_docs)} tài liệu tham khảo"):
                            for i, src in enumerate(rag_docs, 1):
                                # Xử lý cả format từ LangChain (meta) và ChromaService (metadata)
                                if isinstance(src, dict):
                                    # LangChain format: {"text": ..., "meta": {...}, "id": ...}
                                    # ChromaService format: {"text": ..., "metadata": {...}, "id": ..., "distance": ...}
                                    meta = src.get("metadata") or src.get("meta") or {}
                                    title = meta.get("title", meta.get("name", "Không có tiêu đề"))
                                    city = meta.get("city", "")
                                    srcname = meta.get("source", "Nội bộ")
                                    distance = src.get("distance")
                                    text = src.get("text", src.get("content", ""))
                                    doc_id = src.get("id", f"doc_{i}")
                                else:
                                    meta = {}
                                    title = "Không có tiêu đề"
                                    city = ""
                                    srcname = "Nội bộ"
                                    distance = None
                                    text = str(src)
                                    doc_id = f"doc_{i}"
                                
                                st.markdown(f"**{i}. {title}**")
                                if city:
                                    st.caption(f"📍 {city}")
                                if srcname and srcname != "Nội bộ":
                                    st.caption(f"📚 Nguồn: {srcname}")
                                if distance is not None:
                                    st.caption(f"📊 Độ tương đồng: {1 - distance:.3f}")
                                if doc_id:
                                    st.caption(f"🆔 ID: {doc_id}")
                                if text:
                                    st.markdown(f"*{text[:200]}...*")
                                st.markdown("---")
    
    # Chat input
    user_input = st.chat_input("Mời bạn đặt câu hỏi:")
    if "user_input" in st.session_state and st.session_state.user_input:
        user_input = st.session_state.user_input
        del st.session_state.user_input
    
    if user_input:
        with st.chat_message("user", avatar="🧭"):
            st.markdown(f"<div class='user-message'>{user_input}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Process message
        with st.spinner("⏳ Đang soạn phản hồi..."):
            result = chat_engine.process_message(
                user_input,
                st.session_state.messages,
                weather_service=weather_service,
                food_service=food_service,
                client=openai_client
            )
            
            assistant_text = result["response"]
            city = result.get("city")
            start_date = result.get("start_date")
            end_date = result.get("end_date")
            rag_used = result.get("rag_used", False)
            sources_count = result.get("sources_count", 0)
            rag_docs = result.get("rag_docs", [])
            intent_used = result.get("intent_used") or result.get("intent")
            memory_used = result.get("memory_used", False)
            
            # Lưu city và dates vào session_state để hiển thị lại sau khi rerun
            if city:
                st.session_state["last_city"] = city
                st.session_state["last_start_date"] = start_date
                st.session_state["last_end_date"] = end_date
                st.session_state["last_user_input"] = user_input
            
            # Lưu RAG documents vào session_state để hiển thị sau (giống code cũ)
            if rag_docs:
                st.session_state["last_rag_docs"] = rag_docs
            
            # Display response - lưu message với metadata RAG để hiển thị lại sau khi rerun
            message_data = {
                "role": "assistant", 
                "content": assistant_text,
                "rag_used": rag_used,
                "sources_count": sources_count,
                "rag_docs": rag_docs if rag_docs else [],
                "intent_used": intent_used,
                "memory_used": memory_used
            }
            st.session_state.messages.append(message_data)
            
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
                    
                    # === HIỂN THỊ NGUỒN THAM KHẢO CHI TIẾT ===
                    if rag_used or intent_used or memory_used:
                        st.markdown("---")
                        st.subheader("🔍 Nguồn tham khảo")
                        
                        # Hiển thị thông tin về ChromaDB
                        if rag_used and sources_count > 0:
                            st.markdown(f'<div class="source-badge source-chroma">📚 <b>ChromaDB RAG</b>: Sử dụng {sources_count} tài liệu từ cơ sở tri thức du lịch</div>', unsafe_allow_html=True)
                        
                        if intent_used:
                            st.markdown(f'<div class="source-badge source-intent">🎯 <b>Intent Matching</b>: Phát hiện intent "{intent_used}" từ ChromaDB</div>', unsafe_allow_html=True)
                        
                        if memory_used:
                            st.markdown(f'<div class="source-badge source-memory">💭 <b>Memory Recall</b>: Tham khảo hội thoại trước đó từ ChromaDB</div>', unsafe_allow_html=True)
                        
                        # Hiển thị chi tiết các tài liệu RAG
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
            
            # TTS
            if tts_enable:
                audio_b64 = voice_service.text_to_speech(assistant_text)
                if audio_b64:
                    st.markdown(
                        f'<div class="audio-wrapper"><audio autoplay controls><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio></div>',
                        unsafe_allow_html=True
                    )
            
            st.balloons()
    
    # Display additional info - sử dụng city từ result hoặc từ session_state (đặt ngoài block if user_input để luôn hiển thị)
    city_to_display = None
    start_date_to_display = None
    end_date_to_display = None
    user_input_to_display = ""
    
    if user_input:
        # Nếu có user_input mới, sử dụng city từ result
        city_to_display = city if city else st.session_state.get("last_city")
        start_date_to_display = start_date if start_date else st.session_state.get("last_start_date")
        end_date_to_display = end_date if end_date else st.session_state.get("last_end_date")
        user_input_to_display = user_input if user_input else st.session_state.get("last_user_input", "")
    else:
        # Nếu không có user_input mới, sử dụng city từ session_state
        city_to_display = st.session_state.get("last_city")
        start_date_to_display = st.session_state.get("last_start_date")
        end_date_to_display = st.session_state.get("last_end_date")
        user_input_to_display = st.session_state.get("last_user_input", "")
    
    if city_to_display:
        days = extract_days_from_text(user_input_to_display, start_date_to_display, end_date_to_display)
        
        # Weather
        if "Weather" in info_options:
            weather_text = weather_service.get_forecast(city_to_display, start_date_to_display, end_date_to_display, user_input_to_display, openai_client)
            st.markdown(f"**{weather_text}**")
        
        # Cost
        if "Cost" in info_options:
            st.markdown(f"**{estimate_cost(city_to_display, days)}**")
        
        # Events
        if "Events" in info_options:
            st.markdown(f"**{suggest_events(city_to_display)}**")
        
        # Map and images
        cols = st.columns([2, 3])
        with cols[0]:
            if "Map" in info_options:
                lat, lon, addr = geocoding_service.geocode_city(city_to_display)
                if lat and lon:
                    geocoding_service.show_map(lat, lon, zoom=map_zoom, title=addr or city_to_display)
            
            if "Photos" in info_options:
                img = image_service.get_city_image(city_to_display)
                if img:
                    st.image(img, caption=f"🏞️ {city_to_display}", width='stretch')
                else:
                    st.info("Không tìm thấy ảnh minh họa.")
        
        with cols[1]:
            if "Food" in info_options:
                st.subheader(f"🍽️ Ẩm thực & Nhà hàng tại {city_to_display}")
                foods = food_service.get_foods_with_fallback(city_to_display, openai_client)
                if foods:
                    st.markdown("#### 🥘 Đặc sản nổi bật")
                    food_images = image_service.get_food_images(foods)
                    img_cols = st.columns(min(len(food_images), 4))
                    for i, item in enumerate(food_images):
                        with img_cols[i % len(img_cols)]:
                            if item["image"]:
                                st.image(item["image"], caption=item["name"], width='stretch')
                            else:
                                st.write(f"- {item['name']}")
                else:
                    st.info("Không tìm thấy món đặc trưng (CSV/GPT fallback không trả kết quả).")
                
                st.markdown("#### 🍴 Nhà hàng gợi ý")
                restaurants = restaurant_service.get_restaurants(city_to_display, limit=5)
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


def render_analytics(logger_service: LoggerService):
    """Render analytics tab."""
    st.header("📊 Thống kê truy vấn")
    
    try:
        import sqlite3
        conn = sqlite3.connect(str(logger_service.db_path))
        
        # Check if columns exist
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(interactions)")
        columns = [col[1] for col in cur.fetchall()]
        
        df_logs = pd.read_sql("SELECT * FROM interactions ORDER BY timestamp DESC LIMIT 1000", conn)
        conn.close()
        
        if not df_logs.empty:
            total = len(df_logs)
            st.metric("Tổng tương tác", total)
            
            # RAG statistics (if columns exist)
            if 'rag_used' in columns:
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_rag = df_logs['rag_used'].sum()
                    st.metric("Truy vấn sử dụng RAG", f"{total_rag}/{total}")
                with col2:
                    avg_sources = df_logs['sources_count'].mean()
                    st.metric("Trung bình nguồn/RAG", f"{avg_sources:.1f}")
                with col3:
                    rag_rate = (total_rag / total) * 100 if total > 0 else 0
                    st.metric("Tỷ lệ sử dụng RAG", f"{rag_rate:.1f}%")
            
            df_logs['timestamp_dt'] = pd.to_datetime(df_logs['timestamp'])
            df_logs['date'] = df_logs['timestamp_dt'].dt.date
            
            series = df_logs.groupby('date').size().reset_index(name='queries')
            fig = px.bar(series, x='date', y='queries', title='📈 Số truy vấn mỗi ngày', 
                        color='queries', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
            
            top_cities = df_logs['city'].fillna("Unknown").value_counts().reset_index()
            top_cities.columns = ['city', 'count']
            if not top_cities.empty:
                fig2 = px.bar(top_cities.head(10), x='city', y='count', 
                            title='📍 Top địa điểm được hỏi', color='count', 
                            color_continuous_scale='Viridis')
                st.plotly_chart(fig2, use_container_width=True)
            
            st.dataframe(df_logs)
            
            # Delete history option
            with st.expander("🗑️ Xóa lịch sử truy vấn"):
                st.warning("⚠️ Thao tác này sẽ xóa toàn bộ lịch sử truy vấn đã lưu trong cơ sở dữ liệu (SQLite). Không thể hoàn tác.")
                confirm_delete = st.checkbox("Tôi hiểu và muốn xóa toàn bộ lịch sử truy vấn", value=False)
                if confirm_delete:
                    if st.button("✅ Xác nhận xóa toàn bộ lịch sử"):
                        try:
                            conn = sqlite3.connect(str(logger_service.db_path))
                            conn.execute("DELETE FROM interactions")
                            conn.commit()
                            conn.close()
                            st.success("✅ Đã xóa toàn bộ lịch sử truy vấn!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Lỗi khi xóa lịch sử: {e}")
        else:
            st.info("Chưa có truy vấn nào được ghi nhận.")
    except Exception as e:
        st.warning(f"Lỗi đọc dữ liệu: {e}")


def run_app():
    """Main application entry point."""
    st.set_page_config(
        page_title="[Mây Lang Thang] - Travel Assistant (Voice + RAG)",
        layout="wide",
        page_icon="🌤️"
    )
    
    render_css()
    render_hero_section()
    
    # Initialize settings
    settings = get_settings()
    
    # Initialize OpenAI client
    import openai
    openai_client = None
    if settings.OPENAI_API_KEY:
        openai_client = openai.OpenAI(
            base_url=settings.OPENAI_ENDPOINT,
            api_key=settings.OPENAI_API_KEY
        )
    
    # Initialize services
    chroma_service = ChromaService(settings)
    voice_service = VoiceService(settings)
    logger_service = LoggerService(settings)
    weather_service = WeatherService(settings)
    geocoding_service = GeocodingService(settings)
    image_service = ImageService(settings)
    food_service = FoodService(settings)
    restaurant_service = RestaurantService(settings)
    
    # Initialize chat engine
    chat_engine = ChatEngine(settings, openai_client, chroma_service, logger_service)
    
    # Create tabs
    main_tab, analytics_tab = st.tabs(["💬 Trò chuyện với [Mây lang thang]", "📊 Thống kê truy vấn"])
    
    # Sidebar
    with st.sidebar:
        info_options, enable_voice, tts_enable, map_zoom = render_sidebar(settings, chroma_service)
    
    # Main tab
    with main_tab:
        render_main_chat(
            settings, chat_engine, voice_service, weather_service,
            geocoding_service, image_service, food_service, restaurant_service,
            info_options, enable_voice, tts_enable, map_zoom, openai_client,
            chroma_service
        )
    
    # Analytics tab
    with analytics_tab:
        render_analytics(logger_service)
    
    st.markdown("---")
    st.markdown("<div class='small-muted'>Tip: Bạn có thể yêu cầu cụ thể như 'Lịch trình 3 ngày ở Hội An', 'Đặc sản Sapa', hoặc 'Thời tiết Đà Nẵng'.</div>", unsafe_allow_html=True)

