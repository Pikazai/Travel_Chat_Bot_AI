# Travel_Chat_Bot_Enhanced_with_Chroma.py
# Bản gốc được mở rộng: tích hợp ChromaDB để hỗ trợ RAG, memory, semantic search
# Yêu cầu: streamlit, openai, requests, geopy, pandas, pydeck, plotly, sqlite3, chromadb

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

# Chroma imports
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

# -------------------------
# PAGE CONFIG & THEME
# -------------------------
st.set_page_config(page_title="🤖 [Mây Lang Thang] - Travel Assistant (Chroma)", layout="wide", page_icon="🌴")

# (CSS omitted for brevity — keep same UI as original)
st.markdown("""
<style>
:root{ --primary:#2b4c7e; --accent:#e7f3ff; --muted:#f2f6fa; }
body { background: linear-gradient(90deg, #f8fbff 0%, #eef5fa 100%); font-family: 'Segoe UI', Roboto, Arial, sans-serif; }
.stApp > header {visibility: hidden;} h1, h2, h3 { color: var(--primary); }
.sidebar-card { background-color:#f1f9ff; padding:10px; border-radius:10px; margin-bottom:8px;}
.user-message { background: #f2f2f2; padding:10px; border-radius:12px; }
.assistant-message { background: #e7f3ff; padding:10px; border-radius:12px; }
.pill-btn { border-radius:999px !important; background:#e3f2fd !important; color:var(--primary) !important; padding:6px 12px; border: none; }
.status-ok { background:#d4edda; padding:8px; border-radius:8px; }
.status-bad { background:#f8d7da; padding:8px; border-radius:8px; }
.small-muted { color: #6b7280; font-size:12px; }
.logo-title { display:flex; align-items:center; gap:10px; }
.logo-title h1 { margin:0; }
.hero { position: relative; border-radius: 16px; overflow: hidden; box-shadow: 0 8px 30px rgba(43,76,126,0.12); margin-bottom: 18px; }
.hero__bg { width: 100%; height: 320px; object-fit: cover; filter: brightness(0.65) saturate(1.05); }
.hero__overlay { position: absolute; top: 0; left: 0; right: 0; bottom: 0; display: flex; align-items: center; justify-content: center; padding: 24px; }
.hero__card { background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.08)); backdrop-filter: blur(6px); border-radius: 12px; padding: 18px; width: 100%; max-width: 980px; color: white; }
.hero__title { font-size: 28px; font-weight:700; margin:0 0 6px 0; color: #fff; }
.hero__subtitle { margin:0 0 12px 0; color: #f0f6ff; }
.hero__cta { display:flex; gap:8px; align-items:center; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# CONFIG / SECRETS
# -------------------------
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_ENDPOINT = st.secrets.get("OPENAI_ENDPOINT", "https://api.openai.com/v1") if hasattr(st, 'secrets') else os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
DEPLOYMENT_NAME = st.secrets.get("DEPLOYMENT_NAME", "gpt-4o-mini") if hasattr(st, 'secrets') else os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
OPENWEATHERMAP_API_KEY = st.secrets.get("OPENWEATHERMAP_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("OPENWEATHERMAP_API_KEY", "")
GOOGLE_PLACES_KEY = st.secrets.get("PLACES_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("PLACES_API_KEY", "")
PIXABAY_API_KEY = st.secrets.get("PIXABAY_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("PIXABAY_API_KEY", "")

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
# CHROMA SETUP / HELPERS
# -------------------------
CHROMA_CLIENT = None
CHROMA_EMBED_FN = None
TRAVEL_COLLECTION_NAME = "travel_docs"
MEMORY_COLLECTION_NAME = "chat_memory"

def init_chroma():
    global CHROMA_CLIENT, CHROMA_EMBED_FN
    if not CHROMA_AVAILABLE:
        return None
    if CHROMA_CLIENT:
        return CHROMA_CLIENT

    # Initialize Chroma client (defaults to in-memory or env-configured server)
    try:
        CHROMA_CLIENT = chromadb.Client()
        # Use OpenAI embeddings via chroma utils (requires OPENAI_API_KEY set)
        if OPENAI_API_KEY:
            CHROMA_EMBED_FN = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-3-small"
            )
        else:
            CHROMA_EMBED_FN = None

        # Ensure collections exist
        if CHROMA_CLIENT is not None:
            try:
                CHROMA_CLIENT.get_collection(TRAVEL_COLLECTION_NAME)
            except Exception:
                CHROMA_CLIENT.create_collection(TRAVEL_COLLECTION_NAME, embedding_function=CHROMA_EMBED_FN)
            try:
                CHROMA_CLIENT.get_collection(MEMORY_COLLECTION_NAME)
            except Exception:
                CHROMA_CLIENT.create_collection(MEMORY_COLLECTION_NAME, embedding_function=CHROMA_EMBED_FN)

        return CHROMA_CLIENT
    except Exception as e:
        st.warning(f"Chroma init failed: {e}")
        return None

# Index travel docs from CSV/text source into Chroma
def index_travel_docs_from_csv(path="data/vietnam_travel_docs.csv", batch_size=64):
    if not CHROMA_AVAILABLE:
        st.error("ChromaDB package không được cài đặt.")
        return
    client = init_chroma()
    if not client:
        st.error("Không thể khởi tạo Chroma client.")
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Không mở được file index: {e}")
        return

    collection = client.get_collection(TRAVEL_COLLECTION_NAME)

    docs = []
    metadatas = []
    ids = []

    for i, row in df.iterrows():
        text = row.get('text') or row.get('description') or json.dumps(row.dropna().to_dict())
        doc_id = f"doc_{i}_{int(time.time())}"
        docs.append(text)
        metadatas.append(row.dropna().to_dict())
        ids.append(doc_id)

        # batch add
        if len(docs) >= batch_size:
            collection.add(documents=docs, metadatas=metadatas, ids=ids)
            docs, metadatas, ids = [], [], []

    if docs:
        collection.add(documents=docs, metadatas=metadatas, ids=ids)

    st.success("✅ Đã index dữ liệu du lịch vào ChromaDB")

# Semantic search travel docs
def semantic_search_docs(query, k=3):
    if not CHROMA_AVAILABLE:
        return []
    client = init_chroma()
    if not client:
        return []
    collection = client.get_collection(TRAVEL_COLLECTION_NAME)
    try:
        res = collection.query(query_texts=[query], n_results=k)
        docs = []
        for i in range(len(res['ids'][0])):
            docs.append({
                'id': res['ids'][0][i],
                'score': res['distances'][0][i] if 'distances' in res else None,
                'text': res['documents'][0][i],
                'metadata': res['metadatas'][0][i] if 'metadatas' in res else {}
            })
        return docs
    except Exception as e:
        st.warning(f"Chroma query failed: {e}")
        return []

# Memory: save chat snippet
def save_chat_memory(text, metadata=None, id_prefix="mem"):
    if not CHROMA_AVAILABLE:
        return
    client = init_chroma()
    if not client:
        return
    collection = client.get_collection(MEMORY_COLLECTION_NAME)
    doc_id = f"{id_prefix}_{int(time.time()*1000)}"
    try:
        collection.add(documents=[text], metadatas=[metadata or {}], ids=[doc_id])
    except Exception as e:
        st.warning(f"Lưu memory thất bại: {e}")

# Query memory for context
def query_memory(query, k=3):
    if not CHROMA_AVAILABLE:
        return []
    client = init_chroma()
    if not client:
        return []
    collection = client.get_collection(MEMORY_COLLECTION_NAME)
    try:
        res = collection.query(query_texts=[query], n_results=k)
        results = []
        for i in range(len(res['ids'][0])):
            results.append({
                'id': res['ids'][0][i],
                'text': res['documents'][0][i],
                'metadata': res['metadatas'][0][i] if 'metadatas' in res else {}
            })
        return results
    except Exception as e:
        st.warning(f"Chroma memory query failed: {e}")
        return []

# -------------------------
# DB LOGGING (SQLite) - keep as original
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
# Remaining helpers (geocoding, weather, pixabay, restaurants, foods...)
# Largely copied from original with minimal changes to call Chroma where appropriate
# -------------------------

def extract_days_from_text(user_text, start_date=None, end_date=None):
    if start_date and end_date:
        try:
            delta = (end_date - start_date).days + 1
            return max(delta, 1)
        except Exception:
            pass
    m = re.search(r"(\d+)\s*(ngày|day|days|tuần|week|weeks)", user_text, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        unit = m.group(2).lower()
        if "tuần" in unit or "week" in unit:
            return num * 7
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
                max_tokens=50,
                temperature=0
            )
            text = response.choices[0].message.content.strip()
            num_match = re.search(r'"days"\s*:\s*(\d+)', text)
            if num_match:
                return int(num_match.group(1))
        except Exception:
            pass
    return 3

geolocator = Nominatim(user_agent="travel_chatbot_app")

def geocode_city(city_name):
    try:
        loc = geolocator.geocode(city_name, timeout=10)
        if loc:
            return loc.latitude, loc.longitude, loc.address
        return None, None, None
    except Exception:
        return None, None, None

# ... (Keep show_map, get_weather_forecast, get_pixabay_image, get_city_image, restaurants, foods functions)
# For brevity in this code sample, we will re-use the original implementations. In production keep them here.

# -------------------------
# AI FLOW: integrate semantic search and memory into chat loop
# -------------------------

# initialize chroma quietly
if CHROMA_AVAILABLE:
    init_chroma()

# Suggestion generator, hero, UI and entire Streamlit layout remains mostly identical to original.
# Key integration points explained below:

# 1) Before calling OpenAI for a long assistant response, we will:
#    - Run semantic_search_docs(user_input, k=3) to get supporting context
#    - Run query_memory(user_input, k=3) to retrieve recent personal memory
#    - Prepend those contexts to system/user messages as "retrieved_context"

# 2) After sending user query + receiving assistant_text, we will save a memory entry to Chroma
#    (only a short snippet with metadata: city, dates, intent) so later sessions can recall.

# Note: To avoid unbounded growth, consider TTL / prune policy for memory collection in production.

# -------------------------
# For the rest of the file, reuse original app UI and logic but insert RAG + memory
# Due to file length, we include the critical chat handling portion here — adapt into full app.

# --- Minimal reproduction of the essential chat handling with RAG ---

# initialize session
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

st.header("Mây Lang Thang (Chroma-enabled)")

user_input = st.text_input("Mời bạn đặt câu hỏi:")

if user_input:
    with st.spinner("Xử lý..."):
        # display user
        st.write(f"**Bạn:** {user_input}")
        st.session_state.messages.append({"role": "user", "content": user_input})

        # extract city/dates using previous functions (not re-included here) — placeholder None
        city_guess = None
        start_date = None
        end_date = None

        # Log into SQLite
        log_interaction(user_input, city_guess, start_date, end_date)

        # RAG: get supporting docs
        supporting = []
        if CHROMA_AVAILABLE:
            try:
                docs = semantic_search_docs(user_input, k=3)
                for d in docs:
                    supporting.append(d['text'])
            except Exception:
                supporting = []

            # memory recall
            try:
                memories = query_memory(user_input, k=3)
                for m in memories:
                    supporting.append(m['text'])
            except Exception:
                pass

        # Build context block
        context_block = ""
        if supporting:
            context_block = "\n\n=== Trích dẫn liên quan (dùng để hỗ trợ trả lời) ===\n"
            for i, s in enumerate(supporting, 1):
                context_block += f"[{i}] {s}\n\n"

        # Prepare messages to OpenAI: inject context as system message
        messages_for_ai = [m.copy() for m in st.session_state.messages]
        if context_block:
            messages_for_ai.insert(1, {"role": "system", "content": "Thông tin tham khảo từ cơ sở tri thức nội bộ:\n" + context_block})

        assistant_text = None
        if client:
            try:
                response = client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=messages_for_ai,
                    max_tokens=900,
                    temperature=0.7
                )
                assistant_text = response.choices[0].message.content.strip()
            except Exception as e:
                st.error(f"Gọi OpenAI thất bại: {e}")
                assistant_text = "Xin lỗi, hiện tại không thể liên lạc với OpenAI."
        else:
            assistant_text = "Xin chào! (fallback)"

        # Save memory (short): user's question + small assistant summary
        try:
            mem_meta = {"city": city_guess, "timestamp": datetime.utcnow().isoformat()}
            save_chat_memory(user_input, metadata=mem_meta, id_prefix="q")
            if assistant_text:
                save_chat_memory(assistant_text[:800], metadata={"reply_of": "assistant", "timestamp": datetime.utcnow().isoformat()}, id_prefix="a")
        except Exception:
            pass

        # show assistant
        st.write(f"**Assistant:** {assistant_text}")

# Sidebar: tools for developer to index docs
with st.sidebar:
    st.header("Chroma Tools")
    st.write(f"Chroma installed: {CHROMA_AVAILABLE}")
    if CHROMA_AVAILABLE:
        if st.button("Index travel docs from CSV (data/vietnam_travel_docs.csv)"):
            index_travel_docs_from_csv(path="data/vietnam_travel_docs.csv")
        if st.button("Prune memory (delete)"):
            try:
                client = init_chroma()
                coll = client.get_collection(MEMORY_COLLECTION_NAME)
                coll.delete()  # removes all
                st.success("Đã xóa memory collection")
            except Exception as e:
                st.error(f"Xoá memory thất bại: {e}")

# Footer note
st.markdown("---")
st.markdown("**Lưu ý:** Đây là bản mở rộng demo để tích hợp ChromaDB. Trong môi trường production, cân nhắc: pruning policy, chunking document text, embedding model selection, rate-limiting và xử lý lỗi kỹ lưỡng.")
