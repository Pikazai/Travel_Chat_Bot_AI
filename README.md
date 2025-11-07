# Mây Lang Thang - Travel Chatbot - [Leader AI]

![Hero Image](https://i.postimg.cc/P5M1XPvT/May-lang-thang.png)

**Mây Lang Thang** là một chatbot du lịch thông minh được xây dựng bằng [Streamlit](https://streamlit.io/), giúp người dùng khám phá Việt Nam qua các gợi ý lịch trình, dự báo thời tiết, đặc sản ẩm thực, nhà hàng, bản đồ, và ước tính chi phí. Ứng dụng tích hợp AI ([OpenAI](https://openai.com/)) để xử lý câu hỏi tự nhiên bằng tiếng Việt hoặc tiếng Anh, kết hợp với RAG (Retrieval-Augmented Generation) sử dụng ChromaDB, các API như OpenWeatherMap, Google Places, và Pixabay để mang đến trải nghiệm du lịch liền mạch.

## ✨ Tính Năng Chính

- **🤖 Chatbot thông minh với RAG**: Sử dụng ChromaDB và LangChain để truy xuất thông tin từ cơ sở tri thức và tạo phản hồi chính xác
- **🔗 LangChain Integration**: ConversationalRetrievalChain cho RAG nâng cao với memory management tự động
- **🎯 Phát hiện Intent tự động**: Tự động nhận diện ý định người dùng (thời tiết, ẩm thực, lịch trình)
- **💬 Trò chuyện tự nhiên**: Xử lý ngôn ngữ tự nhiên bằng tiếng Việt hoặc tiếng Anh
- **🎤 Chat giọng nói**: Hỗ trợ Speech-to-Text và Text-to-Speech (STT/TTS)
- **🧠 Trí nhớ ngữ cảnh kép**: LangChain memory (conversation context) + ChromaDB memory (long-term recall)
- **📅 Gợi ý lịch trình cá nhân hóa**: Tạo lịch trình dựa trên điểm đến, ngày đi, và mức chi tiêu
- **🌤️ Dự báo thời tiết**: Lấy dữ liệu thời tiết 5 ngày từ OpenWeatherMap
- **🍜 Ẩm thực & nhà hàng**: Gợi ý đặc sản từ dữ liệu CSV hoặc AI, cùng danh sách nhà hàng
- **🗺️ Bản đồ tương tác**: Hiển thị vị trí bằng PyDeck và Geopy
- **📸 Hình ảnh minh họa**: Lấy ảnh từ Pixabay cho điểm đến và món ăn
- **📊 Thống kê truy vấn**: Biểu đồ truy vấn hàng ngày và top địa điểm (SQLite + Plotly)

## 🏗️ Cấu Trúc Dự Án

Dự án được tổ chức theo kiến trúc modular, dễ bảo trì và mở rộng:

```
TRAVEL_CHAT_BOT_AI/
├── core/                          # Core chatbot logic
│   ├── chat_engine.py            # Main conversation orchestration
│   ├── intent_detector.py        # Intent detection using ChromaDB
│   └── entity_extractor.py       # Extract city, dates from text
│
├── services/                      # External services & integrations
│   ├── chroma_service.py         # ChromaDB RAG operations
│   ├── langchain_service.py     # LangChain RAG chains & memory
│   ├── voice_service.py          # Speech-to-Text & Text-to-Speech
│   ├── logger_service.py         # SQLite logging
│   ├── weather_service.py        # OpenWeatherMap API
│   ├── geocoding_service.py      # Location lookup & maps
│   ├── image_service.py          # Pixabay image API
│   ├── food_service.py           # Food recommendations
│   └── restaurant_service.py     # Restaurant recommendations
│
├── ui/                            # Streamlit user interface
│   └── app.py                    # Main UI application
│
├── config/                        # Configuration management
│   └── settings.py               # Environment variables & constants
│
├── utils/                         # Utility functions
│   ├── text_processing.py        # Text parsing utilities
│   └── date_utils.py             # Date handling utilities
│
├── data/                          # Data files (CSV, etc.)
│   ├── vietnam_foods.csv
│   ├── restaurants_vn.csv
│   └── vietnam_travel_docs.csv
│
├── chromadb_data/                 # ChromaDB persistent storage
├── main.py                        # Application entry point
├── requirements.txt               # Python dependencies
└── travel_chatbot_logs.db        # SQLite database
```

## 📋 Yêu Cầu Hệ Thống

### Python Version
- Python 3.8 trở lên

### Thư Viện Python
Cài đặt các thư viện qua `pip`:
```bash
pip install -r requirements.txt
```

Hoặc cài đặt thủ công các thư viện chính:
```bash
pip install streamlit openai chromadb sentence-transformers requests geopy pandas pydeck plotly gTTS SpeechRecognition python-dotenv
```

## 🔑 Cấu Hình API Keys

### Cách 1: Sử dụng file `.env` (Khuyến nghị)
Tạo file `.env` trong thư mục gốc của dự án:
```env
OPENAI_API_KEY=your_openai_key
OPENAI_ENDPOINT=https://api.openai.com/v1
DEPLOYMENT_NAME=gpt-4o-mini
OPENWEATHERMAP_API_KEY=your_weather_key
PLACES_API_KEY=your_google_key
PIXABAY_API_KEY=your_pixabay_key
```

### Cách 2: Sử dụng Streamlit Secrets
Tạo file `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your_openai_key"
OPENAI_ENDPOINT = "https://api.openai.com/v1"
DEPLOYMENT_NAME = "gpt-4o-mini"
OPENWEATHERMAP_API_KEY = "your_weather_key"
PLACES_API_KEY = "your_google_key"
PIXABAY_API_KEY = "your_pixabay_key"
```

### Lấy API Keys
- **OpenAI API Key**: Đăng ký tại [OpenAI Platform](https://platform.openai.com/api-keys)
- **OpenWeatherMap API Key**: Đăng ký tại [OpenWeatherMap](https://openweathermap.org/api)
- **Google Places API Key**: Đăng ký tại [Google Cloud Console](https://console.cloud.google.com/)
- **Pixabay API Key**: Đăng ký tại [Pixabay](https://pixabay.com/api/docs/)

## 🚀 Cài Đặt và Chạy Ứng Dụng

### Bước 1: Clone repository
```bash
git clone https://github.com/your-repo/Travel_Chat_Bot_AI.git
cd Travel_Chat_Bot_AI
```

### Bước 2: Tạo môi trường ảo (Khuyến nghị)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 4: Cấu hình API Keys
Tạo file `.env` hoặc `.streamlit/secrets.toml` như hướng dẫn ở trên.

### Bước 5: Chạy ứng dụng
```bash
streamlit run main.py
```

Ứng dụng sẽ tự động mở trong trình duyệt tại `http://localhost:8501`

## 📖 Hướng Dẫn Sử Dụng

### 1. Tìm kiếm nhanh
Nhập điểm đến, ngày đi, số người, mức chi trên Hero section để nhận gợi ý nhanh.

### 2. Chat tự nhiên
Đặt câu hỏi như:
- "Lịch trình 3 ngày ở Hội An"
- "Đặc sản Sapa là gì?"
- "Thời tiết Đà Nẵng tuần tới?"
- "Top món ăn ở Huế?"
- "Lịch trình 3 ngày ở Nha Trang?"

### 3. Chat giọng nói
- Bật tính năng voice trong sidebar
- Nhấn nút microphone để ghi âm
- Chatbot sẽ tự động chuyển đổi giọng nói thành văn bản và phản hồi

### 4. Tùy chỉnh hiển thị
Chọn thông tin hiển thị (thời tiết, ẩm thực, bản đồ,...) qua sidebar.

### 5. Xem thống kê
Xem biểu đồ truy vấn và top địa điểm trong tab "Thống kê truy vấn".

## 🔄 Luồng Hoạt Động

1. **User Input**: Người dùng nhập text hoặc ghi âm giọng nói
2. **Voice Processing**: Nếu là giọng nói, chuyển đổi sang text (STT)
3. **Entity Extraction**: Trích xuất thông tin (thành phố, ngày tháng)
4. **Intent Detection**: Phát hiện ý định người dùng
5. **RAG Retrieval & Generation**:
   - **Ưu tiên LangChain**: Sử dụng ConversationalRetrievalChain nếu LangChain khả dụng
   - **Fallback**: Sử dụng traditional RAG + LLM nếu LangChain không khả dụng
6. **Memory Recall**: 
   - LangChain ConversationBufferWindowMemory (conversation context)
   - ChromaDB memory recall (similar past conversations)
7. **LLM Generation**: Tạo phản hồi với ngữ cảnh được bổ sung (qua LangChain chain hoặc direct API)
8. **Memory Storage**: Lưu trữ cuộc trò chuyện vào cả LangChain memory và ChromaDB
9. **Logging**: Ghi log vào SQLite
10. **Additional Services**: Lấy thời tiết, bản đồ, hình ảnh, ẩm thực
11. **UI Display**: Hiển thị phản hồi và thông tin bổ sung

## 📊 Dữ Liệu

Dự án sử dụng các file CSV trong thư mục `data/`:
- `vietnam_foods.csv`: Danh sách đặc sản theo tỉnh/thành
- `restaurants_vn.csv`: Danh sách nhà hàng (fallback)
- `vietnam_travel_docs.csv`: Tài liệu du lịch Việt Nam (cho RAG)

## 🐛 Xử Lý Lỗi

### Lỗi kết nối API
- Kiểm tra API keys trong file `.env` hoặc `secrets.toml`
- Đảm bảo kết nối internet ổn định

### Lỗi ChromaDB
- Xóa thư mục `chromadb_data/` và chạy lại để khởi tạo lại database
- Kiểm tra quyền ghi file trong thư mục dự án

### Lỗi Voice Service
- Đảm bảo microphone hoạt động bình thường
- Kiểm tra kết nối internet (cần cho Google Speech Recognition)

## 🚢 Triển Khai

### Streamlit Cloud
1. Đẩy code lên GitHub
2. Đăng nhập [Streamlit Cloud](https://streamlit.io/cloud)
3. Kết nối repository
4. Cấu hình secrets trong Streamlit Cloud dashboard
5. Deploy!

### Heroku
```bash
# Tạo Procfile
echo "web: streamlit run main.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy
git push heroku main
```

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🔮 Kế Hoạch Cải Tiến

- [ ] Mở rộng dữ liệu ẩm thực và nhà hàng trong CSV
- [ ] Chế độ "Gợi ý cá nhân hóa" (có nhớ người dùng)
- [ ] Tạo Profile: lưu lịch sử từng người dùng
- [ ] Hỗ trợ đa ngôn ngữ hoàn thiện hơn
- [ ] FastAPI REST API backend
- [ ] WebSocket cho real-time chat
- [ ] Docker containerization
- [ ] CI/CD pipeline

## 📝 Hạn Chế

- Phụ thuộc vào API (có thể chậm nếu mạng kém)
- Dự báo thời tiết giới hạn 5 ngày (OpenWeatherMap)
- Hỗ trợ đa ngôn ngữ chưa hoàn thiện (chủ yếu tiếng Việt/Anh)
- Voice service cần kết nối internet (Google Speech Recognition)

## 🤝 Đóng Góp

Chúng tôi hoan nghênh mọi đóng góp! Để đóng góp:

1. Fork repository
2. Tạo branch (`git checkout -b feature/your-feature`)
3. Commit thay đổi (`git commit -m "Add your feature"`)
4. Push lên branch (`git push origin feature/your-feature`)
5. Tạo Pull Request

## 📄 License

[MIT License](LICENSE) - Xem chi tiết trong file LICENSE.

## 📚 Tài Liệu Tham Khảo

- [ARCHITECTURE.md](ARCHITECTURE.md) - Tài liệu kiến trúc chi tiết
- [README_ARCHITECTURE.md](README_ARCHITECTURE.md) - Hướng dẫn nhanh về kiến trúc
- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - Tóm tắt refactoring
- [LANGCHAIN_INTEGRATION.md](LANGCHAIN_INTEGRATION.md) - Hướng dẫn tích hợp LangChain

## 📞 Liên Hệ

Nếu có thắc mắc, tạo [issue](https://github.com/your-repo/Travel_Chat_Bot_AI/issues)

---

**Chúc bạn có những chuyến đi tuyệt vời cùng Mây Lang Thang! 🌴✨**

