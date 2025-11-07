
project/
├── chromadb_data/          # ChromaDB storage
├── data/
│   ├── vietnam_travel_docs.csv
│   ├── restaurants_vn.csv
│   └── vietnam_foods.csv
├── travel_chatbot_logs.db  # SQLite database
└── Travel_Chat_Bot_Enhanced_VOICE_RAG.py
# Phân tích luồng thực thi: Travel_Chat_Bot_Enhanced_VOICE_RAG.py

Đây là tài liệu phân tích luồng hoạt động của ứng dụng chatbot du lịch Streamlit. Tài liệu này liệt kê các hàm, chức năng của chúng, và mô tả luồng thực thi chính (call flow) khi ứng dụng chạy.

## 1. Tóm tắt chung

File `Travel_Chat_Bot_Enhanced_VOICE_RAG.py` là một ứng dụng chatbot du lịch (Mây Lang Thang) xây dựng bằng Streamlit. Ứng dụng tích hợp nhiều tính năng nâng cao:

* **Chat cơ bản:** Sử dụng OpenAI (GPT) để trả lời.
* **Voice:** Nhận diện giọng nói (Speech-to-Text) và phát âm thanh (Text-to-Speech).
* **RAG (Retrieval-Augmented Generation):** Sử dụng ChromaDB và model embedding (Local) để truy vấn cơ sở tri thức (thông tin du lịch Việt Nam) và bổ sung cho câu trả lời của AI.
* **Memory:** Sử dụng ChromaDB để lưu trữ và truy vấn lịch sử hội thoại (long-term memory).
* **Intent Matching:** Sử dụng ChromaDB để nhận diện nhanh các ý định (intent) phổ biến.
* **Tích hợp APIs:** Lấy thông tin thời tiết (OpenWeatherMap), bản đồ (Geopy/Pydeck), hình ảnh (Pixabay), và nhà hàng (Google Places/CSV).
* **Logging:** Ghi lại lịch sử truy vấn vào database SQLite.

---

## 2. Danh sách Methods và Chức năng

Các hàm được nhóm theo chức năng để dễ theo dõi.

### 2.1. Khởi tạo & Cấu hình (Initialization & Config)

* **`load_embedding_model()`**:
    * **Chức năng:** Tải và cache (sử dụng `@st.cache_resource`) model embedding (all-MiniLM-L6-v2) từ thư mục local (`data/all-MiniLM-L6-v2`) hoặc tải về từ Hugging Face nếu chưa có.
    * **Được gọi bởi:** Luồng chính (ngay khi script chạy).
* **`init_db()`**:
    * **Chức năng:** Kết nối tới SQLite (`travel_chatbot_logs.db`) và tạo bảng `interactions` nếu chưa tồn tại.
    * **Được gọi bởi:** Luồng chính (ngay khi script chạy).
* **`init_chroma()`**:
    * **Chức năng:** Khởi tạo `PersistentClient` của ChromaDB (dữ liệu lưu ở `chromadb_data`). Hàm này cũng kiểm tra và dọn dẹp các collection cũ (nếu có lỗi dimension).
    * **Được gọi bởi:** Luồng chính (ngay khi script chạy).
    * **Gọi (gián tiếp):** `safe_get_collection()`.
* **`safe_get_collection(client, name, expected_dim)`**:
    * **Chức năng:** (Helper) Lấy hoặc tạo một collection trong ChromaDB một cách an toàn. Tự động xóa và tạo lại nếu phát hiện collection bị lỗi hoặc sai-dimension.
    * **Được gọi bởi:** `init_chroma()`.
* **`preload_intents()`**:
    * **Chức năng:** Thêm các mẫu câu (ví dụ: "Thời tiết ở...", "Đặc sản...") vào collection `intent_bank` của ChromaDB để phục vụ việc nhận diện intent nhanh.
    * **Được gọi bởi:** Luồng chính (ngay khi script chạy).

### 2.2. Giao diện (Streamlit UI)

* **`render_hero_section(default_city_hint)`**:
    * **Chức năng:** Vẽ (render) form tìm kiếm nhanh (`st.form`) ở đầu trang (Hero section).
    * **Được gọi bởi:** Luồng chính (trong `main_tab`).
* **`show_map(lat, lon, zoom, title)`**:
    * **Chức năng:** Hiển thị bản đồ Pydeck tại tọa độ (lat, lon) được cung cấp.
    * **Được gọi bởi:** Luồng chính (khi xử lý `quicksearch` hoặc sau khi AI trả lời).
* **`status_card(title, ok)`**:
    * **Chức năng:** (Helper) Hiển thị một thẻ trạng thái (màu xanh/đỏ) trong sidebar để báo tình trạng API (OpenWeather, ChromaDB...).
    * **Được gọi bởi:** Luồng chính (trong `st.sidebar`).

### 2.3. Xử lý Dữ liệu & APIs bên ngoài

* **`log_interaction(...)`**:
    * **Chức năng:** Ghi log một tương tác (input, city, intent, RAG status) vào bảng `interactions` trong SQLite.
    * **Được gọi bởi:** Luồng chính (sau mỗi lần AI trả lời).
* **`seed_vietnam_travel_from_csv(path)`**:
    * **Chức năng:** Đọc file `vietnam_travel_docs.csv` và nạp (seed) dữ liệu vào collection `vietnam_travel` của ChromaDB.
    * **Được gọi bởi:** `st.sidebar` (khi nhấn nút "Seed dữ liệu" hoặc khi tự động phát hiện collection trống).
* **`geocode_city(city_name)`**:
    * **Chức năng:** Lấy (latitude, longitude, address) từ tên thành phố (sử dụng Geopy/Nominatim).
    * **Được gọi bởi:** Luồng chính (khi xử lý `quicksearch` hoặc sau khi AI trả lời).
* **`resolve_city_via_ai(user_text)`**:
    * **Chức năng:** (Helper) Sử dụng GPT để phân tích text và tìm ra Tỉnh/Thành phố của Việt Nam (VD: "Sapa" -> "Lào Cai"). Dùng làm fallback cho `get_weather_forecast`.
    * **Được gọi bởi:** `get_weather_forecast()`.
* **`get_weather_forecast(city_name, ...)`**:
    * **Chức năng:** Lấy dự báo thời tiết cho một thành phố trong khoảng ngày (sử dụng OpenWeatherMap API).
    * **Được gọi bởi:** Luồng chính (khi xử lý `quicksearch`, khi AI trả lời, hoặc khi intent "weather_query" được kích hoạt).
* **`get_pixabay_image(query, per_page)`**:
    * **Chức năng:** Tìm và trả về URL ảnh từ Pixabay API.
    * **Được gọi bởi:** `get_city_image()`, `get_food_images()`.
* **`get_city_image(city)`**:
    * **Chức năng:** (Wrapper) Gọi `get_pixabay_image` để lấy ảnh đại diện cho thành phố.
    * **Được gọi bởi:** Luồng chính (khi xử lý `quicksearch` hoặc sau khi AI trả lời).
* **`get_food_images(food_list)`**:
    * **Chức năng:** (Wrapper) Gọi `get_pixabay_image` cho từng món ăn trong list.
    * **Được gọi bởi:** Luồng chính (sau khi AI trả lời, trong phần "Ẩm thực").
* **`get_restaurants_google(city, api_key, limit)`**:
    * **Chức năng:** Lấy danh sách nhà hàng từ Google Places API.
    * **Được gọi bởi:** `get_restaurants()`.
* **`get_local_restaurants(city, limit)`**:
    * **Chức năng:** Lấy danh sách nhà hàng từ file CSV `data/restaurants_vn.csv` (dùng làm fallback).
    * **Được gọi bởi:** `get_restaurants()`.
* **`get_restaurants(city, limit)`**:
    * **Chức năng:** (Wrapper) Thử `get_restaurants_google` trước, nếu thất bại (hoặc không có API key) thì dùng `get_local_restaurants`.
    * **Được gọi bởi:** Luồng chính (sau khi AI trả lời, trong phần "Ẩm thực").
* **`re_split_foods(s)`**:
    * **Chức năng:** (Helper) Tách chuỗi món ăn (VD: "Phở, Bún, Chả") thành list `['Phở', 'Bún', 'Chả']`.
    * **Được gọi bởi:** `get_local_foods()`.
* **`get_local_foods(city)`**:
    * **Chức năng:** Lấy danh sách đặc sản từ file CSV `data/vietnam_foods.csv`.
    * **Được gọi bởi:** `get_local_foods_with_fallback()`.
* **`get_foods_via_gpt(city, max_items)`**:
    * **Chức năng:** Dùng GPT để liệt kê đặc sản (dùng làm fallback).
    * **Được gọi bởi:** `get_local_foods_with_fallback()`.
* **`get_local_foods_with_fallback(city)`**:
    * **Chức năng:** (Wrapper) Thử `get_local_foods` trước, nếu không có kết quả thì dùng `get_foods_via_gpt`.
    * **Được gọi bởi:** Luồng chính (khi intent "food_query" kích hoạt hoặc sau khi AI trả lời).

### 2.4. Xử lý Ngôn ngữ & AI (NLP & AI)

* **`get_embedding_local(text)`**:
    * **Chức năng:** Tạo vector embedding cho một đoạn text (sử dụng model đã tải ở `load_embedding_model`).
    * **Được gọi bởi:** Tất cả các hàm truy vấn/thêm dữ liệu của ChromaDB (`rag_query_top_k`, `add_to_memory_collection`, `recall_recent_memories`, `get_intent_via_chroma`, `recommend_similar_trips`).
* **`extract_days_from_text(user_text, ...)`**:
    * **Chức năng:** Trích xuất số ngày (VD: "3 ngày", "1 tuần") từ text (dùng Regex + GPT fallback).
    * **Được gọi bởi:** Luồng chính (sau khi nhận `user_input`).
* **`extract_city_and_dates(user_text)`**:
    * **Chức năng:** Dùng GPT để trích xuất `city`, `start_date`, `end_date` từ `user_input`.
    * **Được gọi bởi:** Luồng chính (sau khi nhận `user_input`).
* **`is_travel_related_via_gpt(user_text)`**:
    * **Chức năng:** Dùng GPT như một bộ phân loại (classifier) để xác định câu hỏi có liên quan đến du lịch Việt Nam hay không.
    * **Được gọi bởi:** Luồng chính (ngay sau khi nhận `user_input` để lọc off-topic).
* **`estimate_cost(city, days, people, style)`**:
    * **Chức năng:** Ước tính chi phí chuyến đi (dựa trên logic `mapping` đơn giản).
    * **Được gọi bởi:** Luồng chính (khi xử lý `quicksearch` hoặc sau khi AI trả lời).
* **`suggest_local_food(city)`**, **`suggest_events(city)`**, **`suggest_photospots(city)`**:
    * **Chức năng:** Trả về các chuỗi gợi ý mẫu (template).
    * **Được gọi bởi:** Luồng chính (sau khi AI trả lời).

### 2.5. Hệ thống RAG & Memory (ChromaDB)

* **`rag_query_top_k(user_text, k)`**:
    * **Chức năng:** Tìm (query) K tài liệu (`documents`) liên quan nhất từ collection `vietnam_travel` dựa trên `user_text`.
    * **Được gọi bởi:** Luồng chính (khi AI cần tạo câu trả lời RAG).
* **`add_to_memory_collection(text, role, city, ...)`**:
    * **Chức năng:** Thêm 1 tin nhắn (user hoặc assistant) vào collection `chat_memory` của ChromaDB.
    * **Được gọi bởi:** Luồng chính (sau khi AI tạo xong câu trả lời).
* **`recall_recent_memories(user_text, k)`**:
    * **Chức năng:** Tìm (query) K ký ức (tin nhắn) liên quan nhất từ collection `chat_memory` để bổ sung ngữ cảnh cho AI.
    * **Được gọi bởi:** Luồng chính (khi AI cần tạo câu trả lời RAG).
* **`get_intent_via_chroma(user_text, threshold)`**:
    * **Chức năng:** Tìm (query) intent gần nhất (VD: `weather_query`) từ collection `intent_bank`. Chỉ trả về intent nếu độ tương đồng (distance) đủ thấp.
    * **Được gọi bởi:** Luồng chính (ngay trước khi gọi RAG).
* **`recommend_similar_trips(city, k)`**:
    * **Chức năng:** Tìm các chuyến đi (thành phố) tương tự đã được thảo luận trong `chat_memory`. *(Lưu ý: Hàm này được định nghĩa nhưng **không được gọi** ở bất kỳ đâu trong luồng chính của file này)*.

### 2.6. Xử lý Âm thanh (Voice)

* **`detect_audio_type_header(b)`**:
    * **Chức năng:** (Helper) Đọc header (bytes) của file audio để đoán định dạng (wav, ogg, webm, mp3...).
    * **Được gọi bởi:** `write_temp_file_and_convert_to_wav()`.
* **`write_temp_file_and_convert_to_wav(audio_bytes)`**:
    * **Chức năng:** Nhận bytes âm thanh (từ `mic_recorder`), lưu ra file tạm, và dùng `pydub` (hoặc `ffmpeg` trực tiếp) để chuyển đổi sang định dạng WAV 16kHz (yêu cầu của `speech_recognition`).
    * **Được gọi bởi:** Luồng chính (khi `mic_recorder` có dữ liệu).

---

## 3. Luồng thực thi chính (Call Flow)

Đây là mô tả tuần tự các sự kiện khi người dùng tương tác với ứng dụng.

### Giai đoạn 1: Khởi tạo (Khi mới tải trang)

1.  Script chạy từ trên xuống.
2.  `st.set_page_config()`: Cài đặt trang.
3.  Nạp API keys (OpenAI, OpenWeatherMap...).
4.  `client = openai.OpenAI()`: Khởi tạo OpenAI client.
5.  `init_db()`: Được gọi. Tạo file `travel_chatbot_logs.db`.
6.  `embedding_model = load_embedding_model()`: Được gọi. Tải model embedding vào cache.
7.  `chroma_client, ... = init_chroma()`: Được gọi. Khởi tạo ChromaDB client và các collection (`vietnam_travel`, `chat_memory`, `intent_bank`) vào `st.session_state`.
8.  `preload_intents()`: Được gọi. Nạp các mẫu intent vào ChromaDB.

### Giai đoạn 2: Vẽ Giao diện (UI Rendering)

1.  `render_hero_section()`: Được gọi. Vẽ form tìm kiếm hero.
2.  `main_tab, analytics_tab = st.tabs(...)`: Vẽ 2 tab chính.
3.  **Sidebar (`st.sidebar`)**:
    * Vẽ các tùy chọn (multiselect, checkbox, slider).
    * **Nút "Seed dữ liệu"**: Nếu được nhấn -> `seed_vietnam_travel_from_csv()` được gọi.
    * **Tự động Seed**: Kiểm tra `chroma_travel_col.count()`. Nếu == 0 -> `seed_vietnam_travel_from_csv()` được gọi.
    * `status_card()`: Được gọi nhiều lần để hiển thị trạng thái API, ChromaDB.
4.  **Tab Chính (`main_tab`)**:
    * Hiển thị lịch sử chat (từ `st.session_state.messages`).
    * **Voice Input** (Nếu `enable_voice`):
        * `audio = mic_recorder()`: Hiển thị nút Ghi âm.
        * Nếu người dùng nói ( `if audio:` ):
            * `wav_file = write_temp_file_and_convert_to_wav(audio["bytes"])`.
            * `voice_text = r.recognize_google(audio_data, ...)`: Nhận diện giọng nói.
            * `st.session_state.user_input = voice_text`: Gán text vào session.
            * `st.rerun()`: **Tải lại trang**.
    * **Text Input**:
        * `user_input = st.chat_input(...)`: Hiển thị ô nhập liệu.
        * (Nếu `st.rerun` từ Voice): `user_input` được gán giá trị từ `st.session_state.user_input`.

### Giai đoạn 3: Xử lý Phản hồi (Khi `user_input` tồn tại)

Đây là luồng chính khi người dùng gửi một câu hỏi (bằng text hoặc voice).

1.  Hiển thị tin nhắn của người dùng.
2.  `st.session_state.messages.append(...)`: Lưu tin nhắn user.
3.  `is_travel_related_via_gpt(user_input)`: **[Checkpoint 1]** Lọc off-topic. Nếu `False` -> Trả lời từ chối và `st.stop()`.
4.  `city_guess, start_date, end_date = extract_city_and_dates(user_input)`: Trích xuất thông tin.
5.  `days = extract_days_from_text(user_input, ...)`: Trích xuất số ngày.
6.  **Hiển thị thông tin phụ (Info Blocks)**:
    * Nếu `city_guess` tồn tại và "Weather" được chọn -> `get_weather_forecast(...)` được gọi và hiển thị.
    * (Tương tự cho "Cost" -> `estimate_cost()` và "Events" -> `suggest_events()`).
7.  **`with st.spinner(...)` (Lõi xử lý AI chính):**
    8.  `detected_intent = get_intent_via_chroma(user_input)`: **[Checkpoint 2]** Thử tìm Intent nhanh.
    9.  **IF (Phát hiện Intent)** (VD: `weather_query`):
        * `assistant_text = get_weather_forecast(...)` (Hoặc `get_local_foods_with_fallback` nếu là `food_query`).
    10. **ELSE (Không phát hiện Intent / Chạy RAG)**:
        * `docs, rag_context = rag_query_top_k(user_input)`: Lấy bối cảnh RAG từ `vietnam_travel`.
        * `recent_mem = recall_recent_memories(user_input)`: Lấy bối cảnh từ `chat_memory`.
        * `response = client.chat.completions.create(...)`: Gọi OpenAI với (Prompt + RAG Context + Memory Context + History).
        * `assistant_text = response.choices[0].message.content`.
    11. **Lưu trữ Memory**:
        * `add_to_memory_collection(user_input, ...)`: Lưu câu hỏi vào ChromaDB.
        * `add_to_memory_collection(assistant_text, ...)`: Lưu câu trả lời vào ChromaDB.
8.  `st.session_state.messages.append(...)`: Lưu tin nhắn của trợ lý.
9.  Hiển thị `assistant_text` (với hiệu ứng gõ chữ).
10. **Hiển thị Nguồn (Sources)**: Nếu `rag_used` hoặc `intent_used` -> hiển thị các nguồn đã dùng (lấy từ `st.session_state["last_rag_docs"]`).
11. `log_interaction(...)`: Ghi log kết quả vào SQLite.
12. **TTS (Text-to-Speech)** (Nếu `tts_enable`):
    * `tts = gTTS(assistant_text, ...)`: Tạo file âm thanh.
    * Hiển thị thẻ `<audio>` để tự động phát.
13. **Hiển thị thông tin bổ sung (Contextual Widgets)**:
    * `lat, lon, addr = geocode_city(city_guess)`.
    * (Nếu "Map" được chọn) -> `show_map(lat, lon, ...)`.
    * (Nếu "Photos" được chọn) -> `img = get_city_image(city_guess)`.
    * (Nếu "Food" được chọn):
        * `foods = get_local_foods_with_fallback(city_guess)`.
        * `food_images = get_food_images(foods)`.
        * `restaurants = get_restaurants(city_guess)`.
        * Hiển thị tất cả thông tin ẩm thực và nhà hàng.

### Giai đoạn 4: Tab Thống kê (Khi chuyển sang `analytics_tab`)

1.  `conn = sqlite3.connect(DB_PATH)`: Kết nối SQLite.
2.  `df_logs = pd.read_sql("SELECT ...")`: Đọc toàn bộ lịch sử log.
3.  **Xóa Lịch sử (Nếu nhấn nút)**:
    * Nếu `st.button("...Xác nhận xóa...")` -> `cur.execute("DELETE FROM interactions")` được gọi.
4.  `st.metric()`: Hiển thị các số liệu (Tổng, RAG, ...).
5.  `fig = px.bar(...)`: Tạo biểu đồ (dùng Plotly Express).
6.  `st.plotly_chart(fig)`: Hiển thị biểu đồ.
7.  `st.dataframe(df_logs)`: Hiển thị bảng dữ liệu log chi tiết.