

# Mây Lang Thang - Travel Chatbot - [Leader AI]

![Hero Image](https://i.postimg.cc/P5M1XPvT/May-lang-thang.png)

**Mây Lang Thang** là một chatbot du lịch ảo được xây dựng bằng [Streamlit](https://streamlit.io/), giúp người dùng khám phá Việt Nam qua các gợi ý lịch trình, dự báo thời tiết, đặc sản ẩm thực, nhà hàng, bản đồ, và ước tính chi phí. Ứng dụng tích hợp AI ([OpenAI](https://openai.com/)) để xử lý câu hỏi tự nhiên bằng tiếng Việt hoặc tiếng Anh, kết hợp với các API như OpenWeatherMap, Google Places, và Pixabay để mang đến trải nghiệm du lịch liền mạch.

## Tính Năng Chính
- **Gợi ý du lịch thông minh**: Tạo lịch trình cá nhân hóa dựa trên điểm đến, ngày đi, và mức chi tiêu.
- **Dự báo thời tiết**: Lấy dữ liệu thời tiết 5 ngày từ OpenWeatherMap, với AI fallback để đoán tỉnh/thành.
- **Ẩm thực & nhà hàng**: Gợi ý đặc sản từ dữ liệu CSV hoặc AI, cùng danh sách nhà hàng từ Google Places.
- **Bản đồ tương tác**: Hiển thị vị trí bằng PyDeck và Geopy.
- **Hình ảnh minh họa**: Lấy ảnh từ Pixabay cho điểm đến và món ăn.
- **Thống kê truy vấn**: Biểu đồ truy vấn hàng ngày và top địa điểm (SQLite + Plotly).
- **Giao diện thân thiện**: Hero section, chat bubbles, typing animation, và tùy chỉnh sidebar.

## Yêu Cầu Hệ Thống
### Thư Viện Python
Cài đặt các thư viện qua `pip`:
```bash
pip install streamlit openai requests geopy pandas pydeck plotly
```

### API Keys
Cấu hình qua file `.streamlit/secrets.toml` hoặc biến môi trường:
```toml
OPENAI_API_KEY = "your_openai_key"
OPENWEATHERMAP_API_KEY = "your_weather_key"
PLACES_API_KEY = "your_google_key"
PIXABAY_API_KEY = "your_pixabay_key"
OPENAI_ENDPOINT = "https://api.openai.com/v1"
DEPLOYMENT_NAME = "gpt-4o-mini"
```

### Dữ Liệu Local
- `data/vietnam_foods.csv`: Danh sách đặc sản theo tỉnh/thành.
- `data/restaurants_vn.csv`: Danh sách nhà hàng (fallback).

## Cài Đặt và Chạy
1. **Clone repository**:
   ```bash
   git clone https://github.com/your-repo/may-lang-thang.git
   cd may-lang-thang
   ```

2. **Cấu hình secrets**:
   Tạo file `.streamlit/secrets.toml` như trên.

3. **Chạy ứng dụng**:
   ```bash
   streamlit run Travel_Chat_Bot_Enhanced.py
   ```
   Truy cập tại `http://localhost:8501`.

4. **Triển khai**:
   - Sử dụng [Streamlit Cloud](https://streamlit.io/cloud) hoặc Heroku.
   - Đảm bảo cấu hình secrets trên nền tảng đám mây.

## Cấu Trúc Dự Án
- **Travel_Chat_Bot_Enhanced.py**: File chính chứa toàn bộ logic (UI, API, DB, AI).
- **data/**: Thư mục chứa `vietnam_foods.csv` và `restaurants_vn.csv`.
- **travel_chatbot_logs.db**: SQLite database lưu lịch sử truy vấn.

## Hướng Dẫn Sử Dụng
1. **Tìm kiếm nhanh**: Nhập điểm đến, ngày đi, số người, mức chi trên Hero section.
2. **Chat tự nhiên**: Đặt câu hỏi như "Lịch trình 3 ngày ở Hội An" hoặc "Đặc sản Sapa".
3. **Tùy chỉnh**: Chọn thông tin hiển thị (thời tiết, ẩm thực, bản đồ,...) qua sidebar.
4. **Thống kê**: Xem biểu đồ truy vấn và top địa điểm trong tab "Thống kê truy vấn".

Ví dụ câu hỏi:
- "Thời tiết Đà Nẵng tuần tới?"
- "Top món ăn ở Huế?"
- "Lịch trình 3 ngày ở Nha Trang?"

## Hạn Chế
- Phụ thuộc vào API (có thể chậm nếu mạng kém).
- Dự báo thời tiết giới hạn 5 ngày (OpenWeatherMap).
- Hỗ trợ đa ngôn ngữ chưa hoàn thiện (chủ yếu tiếng Việt/Anh).

## Kế Hoạch Cải Tiến
- Mở rộng dữ liệu ẩm thực và nhà hàng trong CSV.
- Chế độ “Gợi ý cá nhân hóa” (có nhớ người dùng)
Lưu thông tin người dùng:
Thành phố đang sống, sở thích (biển, núi, ẩm thực…)
Khi hỏi: “Cuối tuần này nên đi đâu?” → AI dựa vào sở thích để gợi ý.
Tạo Profile: lưu lịch sử từng người dùng
- Chat giọng nói (Speech-to-Text & Text-to-Speech)
Người dùng nói chuyện trực tiếp với chatbot.

## Đóng Góp
1. Fork repository.
2. Tạo branch (`git checkout -b feature/your-feature`).
3. Commit thay đổi (`git commit -m "Add your feature"`).
4. Push lên branch (`git push origin feature/your-feature`).
5. Tạo Pull Request.

## License
[MIT License](LICENSE) - Xem chi tiết trong file LICENSE.

## Liên Hệ
Nếu có thắc mắc, tạo [issue](https://github.com/your-repo/may-lang-thang/issues)

**Chúc bạn có những chuyến đi tuyệt vời cùng Mây Lang Thang! 🌴**

