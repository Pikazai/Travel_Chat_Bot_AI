"""
UI components for Streamlit interface.
"""
import streamlit as st
from datetime import datetime
from streamlit_mic_recorder import mic_recorder
from config.settings import get_settings


def render_hero_section(default_city_hint: str = "Hội An, Đà Nẵng, Hà Nội..."):
    """Render the hero section with search form."""
    hero_img = "https://images.unsplash.com/photo-1633073985249-b2d67bdf6b7d?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=1074"
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
        cols = st.columns([3, 2, 1, 1])
        dest = cols[0].text_input("Điểm đến", placeholder=default_city_hint)
        dates = cols[1].date_input("Ngày (bắt đầu / kết thúc)", [])
        people = cols[2].selectbox("Người", [1, 2, 3, 4, 5, 6], index=0)
        style = cols[3].selectbox("Mức chi", ["trung bình", "tiết kiệm", "cao cấp"], index=0)
        submitted = st.form_submit_button("Tìm kiếm nhanh", use_container_width=True)
        
        if submitted:
            if isinstance(dates, list) and len(dates) == 2:
                s = dates[0].strftime("%Y-%m-%d")
                e = dates[1].strftime("%Y-%m-%d")
                q = f"Lịch trình {((dates[1]-dates[0]).days +1)} ngày ở {dest} từ {s} đến {e}"
            elif isinstance(dates, list) and len(dates) == 1:
                s = dates[0].strftime("%Y-%m-%d")
                q = f"Lịch trình 1 ngày ở {dest} vào {s}"
            else:
                q = f"Lịch trình 3 ngày ở {dest}"
            q += f" • người: {people} • mức: {style}"
            st.session_state.user_input = q
            st.rerun()


def render_sidebar():
    """Render the sidebar with settings and status."""
    settings = get_settings()
    
    with st.sidebar:
        st.markdown(
            "<div class='logo-title'><img src='https://img.icons8.com/emoji/48/000000/cloud-emoji.png'/> <h2>Mây Lang Thang</h2></div>",
            unsafe_allow_html=True
        )
        
        st.header("Cài đặt")
        language_option = st.selectbox("Ngôn ngữ (gợi ý trích xuất)", ["Tự động", "Tiếng Việt", "English"])
        info_options = st.multiselect(
            "Hiển thị thông tin",
            ["Weather", "Food", "Map", "Photos", "Cost", "Events"],
            default=["Weather", "Map", "Food", "Photos"]
        )
        st.markdown("---")
        st.write("Chọn mức zoom bản đồ:")
        map_zoom = st.slider("Zoom (4 = xa, 15 = gần)", 4, 15, 8)
        st.markdown("---")
        
        # Voice settings
        st.subheader("🎙️ Voice")
        enable_voice = st.checkbox("Bật nhập liệu bằng giọng nói", value=True)
        asr_lang = st.selectbox("Ngôn ngữ nhận dạng", ["vi-VN", "en-US"], index=0)
        tts_enable = st.checkbox("🔊 Đọc to phản hồi", value=False)
        tts_lang = st.selectbox("Ngôn ngữ TTS", ["vi", "en"], index=0)
        st.caption("Yêu cầu: ffmpeg + internet cho gTTS.")
        st.markdown("---")
        
        # RAG / ChromaDB section (if available)
        # Note: chroma_service is initialized in session state, so we check it here
        if "chroma_service" in st.session_state:
            chroma_service = st.session_state.chroma_service
            if chroma_service.is_available():
                st.subheader("🔎 RAG / Cache")
                use_rag = st.checkbox("Bật RAG (Chroma)", value=True, key="use_rag")
                use_cache = st.checkbox("Bật Semantic Cache", value=True, key="use_cache")
                rag_k = st.slider("Top‑k RAG", 1, 10, 6, key="rag_k")
                if st.button("📥 Seed KB từ CSV"):
                    try:
                        added = chroma_service.seed_kb_from_csvs(
                            "data/vietnam_foods.csv",
                            "data/restaurants_vn.csv"
                        )
                        st.success(f"Đã seed {added} mẩu tri thức vào travel_kb.")
                    except Exception as e:
                        st.error(f"Lỗi khi seed: {e}")
                st.markdown("---")
        
        # Status cards
        def status_card(title, ok=True):
            cls = "status-ok" if ok else "status-bad"
            icon = "✅" if ok else "⚠️"
            st.markdown(f"<div class='{cls}'>{icon} {title}</div>", unsafe_allow_html=True)
        
        status_card("ChromaDB", "chroma_service" in st.session_state and st.session_state.chroma_service.is_available())
        status_card("OpenWeatherMap", bool(settings.OPENWEATHERMAP_API_KEY))
        status_card("Google Places", bool(settings.GOOGLE_PLACES_KEY))
        status_card("Pixabay", bool(settings.PIXABAY_API_KEY))
        st.markdown("---")
        st.caption("🍜 Food AI: CSV local dữ liệu + GPT fallback")
        st.markdown("Version: v2.0 • Modular Architecture")
        
        # Get RAG settings from session state if available (inside sidebar context)
        use_rag = st.session_state.get("use_rag", True)
        use_cache = st.session_state.get("use_cache", True)
        rag_k = st.session_state.get("rag_k", 6)
    
    return {
        "language_option": language_option,
        "info_options": info_options,
        "map_zoom": map_zoom,
        "enable_voice": enable_voice,
        "asr_lang": asr_lang,
        "tts_enable": tts_enable,
        "tts_lang": tts_lang,
        "use_rag": use_rag,
        "use_cache": use_cache,
        "rag_k": rag_k
    }


def render_quick_search():
    """Render quick search form."""
    with st.expander("🔎 Tìm kiếm nhanh chuyến đi"):
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            city_qs = st.text_input("🏙️ Điểm đến", "Đà Nẵng")
        with col2:
            start_qs = st.date_input("📅 Bắt đầu", datetime(2025, 10, 20))
        with col3:
            end_qs = st.date_input("📅 Kết thúc", datetime(2025, 10, 22))
        with col4:
            people_qs = st.slider("👥 Người", 1, 10, 1)
        
        col5, col6 = st.columns([1, 3])
        with col5:
            style_qs = st.selectbox("💰 Mức chi tiêu", ["Tiết kiệm", "Trung bình", "Cao cấp"], index=1)
        with col6:
            if st.button("🚀 Xem gợi ý"):
                st.session_state.quicksearch = {
                    "city": city_qs,
                    "start": start_qs,
                    "end": end_qs,
                    "people": people_qs,
                    "style": style_qs
                }
                return {
                    "city": city_qs,
                    "start": start_qs,
                    "end": end_qs,
                    "people": people_qs,
                    "style": style_qs
                }
    return None


def render_suggestions(suggestions: list):
    """Render suggested questions."""
    st.write("### 🔎 Gợi ý nhanh")
    cols = st.columns(len(suggestions))
    for i, q in enumerate(suggestions):
        if cols[i].button(q, key=f"sugg_{i}"):
            st.session_state.user_input = q
            st.rerun()


def render_voice_input(voice_service, enable_voice: bool, asr_lang: str):
    """Render voice input interface."""
    if not enable_voice:
        return None
    
    st.write("### 🎙️ Nói để nhập câu hỏi")
    audio = mic_recorder(
        start_prompt="Bấm để nói",
        stop_prompt="Dừng",
        just_once=True,
        key="rec_chat"
    )
    
    if audio:
        st.info("Đã nhận dữ liệu âm thanh, đang xử lý...")
        try:
            audio_bytes = audio["bytes"]
            text = voice_service.speech_to_text(audio_bytes, language=asr_lang)
            if text:
                st.success(f"🗣️ Bạn nói: {text}")
                st.session_state.user_input = text
                st.rerun()
            else:
                st.error("Không thể nhận diện giọng nói.")
        except Exception as e:
            st.error(f"Lỗi xử lý giọng nói: {e}")
    
    return None

