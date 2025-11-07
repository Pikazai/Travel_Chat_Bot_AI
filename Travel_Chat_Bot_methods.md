
# Phân tích file `Travel_Chat_Bot_Enhanced_VOICE_RAG.py`
project/
├── chromadb_data/          # ChromaDB storage
├── data/
│   ├── vietnam_travel_docs.csv
│   ├── restaurants_vn.csv
│   └── vietnam_foods.csv
├── travel_chatbot_logs.db  # SQLite database
└── Travel_Chat_Bot_Enhanced_VOICE_RAG.py
---

## Tóm tắt nhanh
File là một ứng dụng **Streamlit** (Travel assistant) mở rộng với RAG (ChromaDB), embedding local, voice input, TTS, mapping, weather, đồ ăn, nhà hàng, và logging. Nó chứa nhiều hàm tiện ích (geocoding, weather, images, RAG helpers, memory, intents, UI helpers, voice helpers). fileciteturn1file0

---

## Danh sách hàm (method) chính — mô tả & nơi được gọi
> Ghi chú: "Called by" liệt kê các vị trí trong file nơi hàm đó được invoke (UI flow, các hàm khác, hay handlers). Tất cả thông tin trích xuất trực tiếp từ file. fileciteturn1file0

### 1. `load_embedding_model()`
**Chức năng:** Tải model embedding `all-MiniLM-L6-v2` từ thư mục local hoặc Hugging Face, lưu về `data/` để tái sử dụng. (Được cache bằng `@st.cache_resource`.)  
**Called by:** trực tiếp gọi khi file được load để gán `embedding_model = load_embedding_model()`. fileciteturn1file0

---

### 2. `init_db()`
**Chức năng:** Tạo schema SQLite (`interactions`) nếu chưa tồn tại.  
**Called by:** được gọi ngay sau định nghĩa để khởi tạo DB khi module load. (`init_db()`) và `log_interaction()` sử dụng DB. fileciteturn1file16

### 3. `log_interaction(user_input, city=None, start_date=None, end_date=None, intent=None, rag_used=False, sources_count=0)`
**Chức năng:** Ghi một dòng log interaction vào SQLite DB `travel_chatbot_logs.db`.  
**Called by:** gọi trong luồng xử lý khi trả lời user (sau khi tạo assistant_text). cite fileciteturn1file15

---

### 4. `seed_vietnam_travel_from_csv(path="data/vietnam_travel_docs.csv")`
**Chức năng:** đọc CSV seed dữ liệu du lịch và add vào Chroma collection `vietnam_travel`.  
**Called by:** sidebar button (seed data manual) và logic tự-seed khi collection rỗng. fileciteturn1file10

---

### 5. `get_embedding_local(text)`
**Chức năng:** Tạo embedding từ `embedding_model` local; trả về list float (384-dim).  
**Called by:** nhiều hàm RAG/memory/intents: `rag_query_top_k`, `add_to_memory_collection`, `recall_recent_memories`, `get_intent_via_chroma`, `recommend_similar_trips`, ... fileciteturn1file2

---

### 6. `safe_get_collection(client, name, expected_dim=384)`
**Chức năng:** Tạo hoặc lấy collection Chroma một cách "an toàn" — kiểm tra mismatch dimension và tái tạo nếu cần.  
**Called by:** `init_chroma()` khi khởi tạo các collection (travel, memory, intent). fileciteturn1file7

---

### 7. `init_chroma()`
**Chức năng:** Khởi tạo `PersistentClient` Chroma (path=`chromadb_data`), kiểm tra & xoá các collection cũ (1536-dim), và trả về `(chroma_client, travel_col, memory_col, intent_col)`.  
**Called by:** Được gọi tại module-level để khởi tạo `chroma_client, chroma_travel_col, chroma_memory_col, chroma_intent_col`. fileciteturn1file11

---

### 8. `extract_days_from_text(user_text, start_date=None, end_date=None)`
**Chức năng:** Tách số ngày từ câu người dùng (regex cho "ngày/tuần", hoặc fallback gọi model OpenAI để parse). Trả về số ngày (int), mặc định 3.  
**Called by:** luồng xử lý user input để tính `days` (khi build blocks/estimate cost, itinerary). fileciteturn1file11

---

### 9. `geocode_city(city_name)`
**Chức năng:** Geocode tên thành phố bằng `geopy.Nominatim` (lat, lon, address).  
**Called by:** UI để hiển thị map (`show_map`) và phần hiển thị city image/map sau khi trả lời. fileciteturn1file5

### 10. `show_map(lat, lon, zoom=8, title="")`
**Chức năng:** Hiển thị bản đồ pydeck trong Streamlit (marker + text layer).  
**Called by:** UI (sau khi trả lời user, trong sidebar quicksearch area, và nơi khác gọi map). fileciteturn1file5

---

### 11. `resolve_city_via_ai(user_text)`
**Chức năng:** Nếu OpenWeather trả lỗi, gọi GPT (OpenAI client) để suy luận ra province/city từ `user_text`.  
**Called by:** `get_weather_forecast()` như fallback khi OpenWeather không tìm thấy. fileciteturn1file5

### 12. `get_weather_forecast(city_name, start_date=None, end_date=None, user_text=None)`
**Chức năng:** Lấy dự báo thời tiết từ OpenWeatherMap (5-day forecast), xử lý ngày nếu missing, fallback sang `resolve_city_via_ai` khi cần, build và trả về chuỗi văn bản forecast.  
**Called by:** nhiều nơi: hero quicksearch summary, chat handling blocks, intent-specific response (when `weather_query`), và UI previews. (Ví dụ: `weather_qs = get_weather_forecast(...)`, `blocks.append(get_weather_forecast(...))`, and detected-intent branch). fileciteturn1file6fileciteturn1file9

---

### 13. `get_pixabay_image(query, per_page=3)`
**Chức năng:** Gọi Pixabay API để lấy ảnh (first hit).  
**Called by:** `get_city_image()` và `get_food_images()`. fileciteturn1file12

### 14. `get_city_image(city)`
**Chức năng:** Thử nhiều query (city landscape/city/travel) để lấy ảnh đại diện cho thành phố; fallback placeholder nếu không có.  
**Called by:** UI (hero/quicksearch and right-hand panel after answer). fileciteturn1file12

### 15. `get_food_images(food_list)`
**Chức năng:** Lấy ảnh cho một danh sách món ăn (dùng `get_pixabay_image`).  
**Called by:** phần hiển thị ẩm thực trong UI. fileciteturn1file12

---

### 16. `get_restaurants_google(city, api_key, limit=5)`
**Chức năng:** Kêu Google Places Text Search API để lấy nhà hàng; trả về list dict.  
**Called by:** `get_restaurants()` khi `GOOGLE_PLACES_KEY` có giá trị; UI hiển thị kết quả. fileciteturn1file12

### 17. `get_local_restaurants(city, limit=5)`
**Chức năng:** Fallback đọc CSV `data/restaurants_vn.csv` nếu Google Places không khả dụng.  
**Called by:** `get_restaurants()` fallback. fileciteturn1file12

### 18. `get_restaurants(city, limit=5)`
**Chức năng:** Wrapper: dùng Google nếu có API key, else fallback local CSV.  
**Called by:** UI để hiển thị "Nhà hàng gợi ý" sau khi trả lời. fileciteturn1file12

---

### 19. `re_split_foods(s)`
**Chức năng:** Tách chuỗi món ăn bằng dấu phân cách (`,` `|` `;`) — helper.  
**Called by:** `get_local_foods()`. fileciteturn1file12

### 20. `get_local_foods(city)`
**Chức năng:** Đọc CSV `data/vietnam_foods.csv` để trả về danh sách món của city (nếu có).  
**Called by:** `get_local_foods_with_fallback()` và UI hiển thị food. fileciteturn1file12

### 21. `get_foods_via_gpt(city, max_items=5)`
**Chức năng:** Gọi OpenAI để lấy danh sách món ăn (fallback nếu CSV không có).  
**Called by:** `get_local_foods_with_fallback()`. fileciteturn1file12

### 22. `get_local_foods_with_fallback(city)`
**Chức năng:** Trả foods từ CSV, nếu rỗng -> gọi GPT fallback.  
**Called by:** UI (các phần hiển thị ẩm thực / intent handling). fileciteturn1file12

---

### 23. `estimate_cost(city, days=3, people=1, style="trung bình")`
**Chức năng:** Ước tính chi phí dựa trên mapping theo phong cách (tiết kiệm/trung bình/cao cấp).  
**Called by:** hero quicksearch and chat blocks (blocks.append(estimate_cost(...))). fileciteturn1file14

### 24. `suggest_local_food(city)`, `suggest_events(city)`, `suggest_photospots(city)`
**Chức năng:** Các helper trả chuỗi gợi ý nhanh (food/events/photospots).  
**Called by:** UI blocks when building info (blocks list) and in assistant responses. fileciteturn1file14

---

### 25. `extract_city_and_dates(user_text)`
**Chức năng:** Gọi OpenAI để extract JSON `{city, start_date, end_date}` (multilingual). Trả về `city, start_dt, end_dt` (datetime hoặc None).  
**Called by:** Nơi xử lý user_input trong chat flow để xác định city và ngày. fileciteturn1file14

---

### 26. `rag_query_top_k(user_text, k=5)`
**Chức năng:** Lấy top-k documents từ Chroma `vietnam_travel_v2` bằng embedding local; trả về list `docs` (id/text/metadata/distance) và `context` (chuỗi tham khảo). Lưu kết quả vào `st.session_state["last_rag_docs"]`.  
**Called by:** Luồng tạo phản hồi khi không detect intent cụ thể — `docs, rag_context = rag_query_top_k(user_input, k=5)` và sau đó được thêm vào augmentation trước khi gọi OpenAI. fileciteturn1file3fileciteturn1file8

---

### 27. `add_to_memory_collection(text, role="user", city=None, extra_meta=None)`
**Chức năng:** Lưu embedding + text vào Chroma `chat_memory_v2` (document id `mem_...`). Hỗ trợ thêm `extra_meta`.  
**Called by:** Sau khi tạo assistant_text trong chat flow để lưu cả user và assistant messages; đôi khi được gọi ở chỗ khác (fallback). fileciteturn1file4fileciteturn1file15

---

### 28. `recall_recent_memories(user_text, k=5)`
**Chức năng:** Truy vấn `chat_memory_v2` bằng embedding từ user_text để lấy các đoạn hội thoại gần nhất (k items).  
**Called by:** Chat flow để hiển thị/đưa vào augmentation khi sinh phản hồi (`recent_mem = recall_recent_memories(user_input, k=3)`). fileciteturn1file4

---

### 29. `get_intent_via_chroma(user_text, threshold=0.2)`
**Chức năng:** Truy vấn `intent_bank_v2` để lấy intent gần nhất; nếu distance < threshold trả về `meta['intent']`.  
**Called by:** Chat flow đầu tiên khi xử lý input (để xử lý nhanh các intent phổ biến: weather_query, food_query, itinerary_request). fileciteturn1file18

---

### 30. `recommend_similar_trips(city, k=3)`
**Chức năng:** Dùng embedding của `city` để truy vấn `chat_memory_v2` và gợi ý các trips tương tự (dựa trên metadata `city`).  
**Called by:** Có thể gọi từ UI/analytics; hiện code cung cấp hàm nhưng không thấy gọi trực tiếp ở luồng chính (helper). fileciteturn1file18

---

### 31. `preload_intents()`
**Chức năng:** Thêm một số văn bản mẫu intent vào `chroma_intent_col` (seed).  
**Called by:** gọi ở module-level (try: preload_intents()). fileciteturn1file18

---

### 32. `render_hero_section(default_city_hint="Hội An, Đà Nẵng, Hà Nội...")`
**Chức năng:** Vẽ form hero (điểm đến, ngày, số người, mức chi) ở đầu trang; khi submit, set `st.session_state.user_input` và `st.rerun()`.  
**Called by:** được gọi tại module-level ở phần UI: `render_hero_section()`. fileciteturn1file10

---

### 33. `detect_audio_type_header(b)`
**Chức năng:** Kiểm tra header bytes audio để phát hiện loại (wav, flac, ogg, webm, mp3...).  
**Called by:** `write_temp_file_and_convert_to_wav()` để biết extension/give conversion behavior. fileciteturn1file10

### 34. `write_temp_file_and_convert_to_wav(audio_bytes)`
**Chức năng:** Ghi bytes audio tạm vào file, phát hiện loại, convert sang WAV (16kHz mono) bằng pydub hoặc ffmpeg fallback; trả về path WAV.  
**Called by:** Voice handler (mic_recorder) khi audio được ghi — luồng voice input trong UI. fileciteturn1file10

---

### 35. `is_travel_related_via_gpt(user_text)`
**Chức năng:** Gọi OpenAI để phân loại xem user_text có liên quan tới du lịch VN không; nếu không liên quan => reject early. Fallback True nếu không có API.  
**Called by:** Chat flow ngay sau thêm message user: `if not is_travel_related_via_gpt(user_input): ...` để reject non-travel queries. fileciteturn1file11

---

## Call-graph (sơ đồ gọi) — tổng quan (ASCII)
Dưới đây là sơ đồ gọi rút gọn (những luồng chính):
```
App load
 ├─> load_embedding_model()
 ├─> init_db()
 ├─> init_chroma() -> safe_get_collection()
 └─> preload_intents()

User input -> main chat flow
 ├─> is_travel_related_via_gpt(user_input)  (may STOP)
 ├─> extract_city_and_dates(user_input) -> (city, start_date, end_date)
 ├─> extract_days_from_text(user_input,...)
 ├─> get_weather_forecast(...)  (used in blocks & detected intent)
 ├─> get_local_foods_with_fallback() -> get_local_foods() or get_foods_via_gpt()
 ├─> get_restaurants() -> get_restaurants_google() or get_local_restaurants()
 ├─> get_city_image() -> get_pixabay_image()
 ├─> detected_intent = get_intent_via_chroma(user_input)
 │     └─ if intent -> shortcut handlers (weather_query, food_query, itinerary_request)
 └─> if not intent:
       ├─> rag_query_top_k(user_input) -> get_embedding_local()
       ├─> recall_recent_memories(user_input) -> get_embedding_local()
       └─> build augmentation -> call OpenAI -> assistant_text
             └─> add_to_memory_collection(user_input), add_to_memory_collection(assistant_text)
                     └─> get_embedding_local() (on add)

Voice path:
 mic_recorder -> write_temp_file_and_convert_to_wav() -> speech_recognition -> set st.session_state.user_input
```

