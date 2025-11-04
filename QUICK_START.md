# 🚀 Quick Start Guide

## Prerequisites

- Python 3.13
- pip
- ffmpeg (for voice features)

## Installation

```bash
# Clone or navigate to project directory
cd Travel_Chat_Bot_AI

# Install dependencies
pip install -r requirements.txt

# Install ffmpeg (if using voice features)
# Windows: Download from https://ffmpeg.org/download.html
# macOS: brew install ffmpeg
# Linux: sudo apt-get install ffmpeg
```

## Configuration

### Option 1: Streamlit Secrets (Recommended)

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-your-key-here"
OPENAI_API_KEY_EMBEDDING = "sk-your-key-here"  # Optional, for ChromaDB embeddings
OPENWEATHERMAP_API_KEY = "your-weather-key"
PLACES_API_KEY = "your-google-places-key"  # Optional
PIXABAY_API_KEY = "your-pixabay-key"  # Optional

# Optional: ChromaDB settings
CHROMA_PATH = "./.chroma"
EMBEDDING_MODEL = "text-embedding-3-small"
```

### Option 2: Environment Variables

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-your-key-here"
$env:OPENWEATHERMAP_API_KEY = "your-weather-key"

# Linux/Mac
export OPENAI_API_KEY="sk-your-key-here"
export OPENWEATHERMAP_API_KEY="your-weather-key"
```

## Running the Application

```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

## First-Time Setup (ChromaDB)

1. Open the application
2. In the sidebar, scroll to "🔎 RAG / Cache" section
3. Click "📥 Seed KB từ CSV" button
4. This populates the knowledge base with food and restaurant data

## Features

### 💬 Chat Interface
- Type questions naturally in Vietnamese or English
- Example: "Lịch trình 3 ngày ở Hội An"

### 🎙️ Voice Input
- Enable in sidebar: "Bật nhập liệu bằng giọng nói"
- Click microphone button to record
- Speech-to-text converts your voice to text

### 🔊 Text-to-Speech
- Enable in sidebar: "🔊 Đọc to phản hồi"
- Responses are read aloud automatically

### 📊 Analytics
- View interaction statistics
- See top cities and query trends
- RAG debugger for ChromaDB testing

## Example Queries

- "Thời tiết ở Đà Nẵng tuần tới?"
- "Top món ăn ở Huế?"
- "Lịch trình 3 ngày ở Nha Trang?"
- "Đặc sản Sapa?"
- "Nhà hàng ở Hà Nội?"

## Troubleshooting

### Import Errors
```bash
# Verify Python path
python -c "import sys; print(sys.path)"

# Reinstall in development mode
pip install -e .
```

### ChromaDB Not Working
```bash
# Install ChromaDB
pip install chromadb

# Verify installation
python -c "import chromadb; print('OK')"
```

### Voice Not Working
- Ensure ffmpeg is installed and in PATH
- Check browser microphone permissions
- Verify internet connection (for Google Speech Recognition)

### API Errors
- Check API keys in `.streamlit/secrets.toml`
- Verify API quotas and limits
- Check internet connection

## Project Structure

```
Travel_Chat_Bot_AI/
├── main.py              # Entry point
├── core/                # Business logic
├── services/            # External APIs
├── ui/                  # Streamlit UI
├── config/              # Configuration
├── utils/               # Utilities
└── data/                # CSV data files
```

## Next Steps

- Read `ARCHITECTURE.md` for detailed documentation
- Check `MIGRATION_GUIDE.md` if migrating from old structure
- Customize `config/settings.py` for your needs
- Add new features following the modular structure

## Support

For issues or questions:
1. Check `ARCHITECTURE.md` for module documentation
2. Review error messages in Streamlit console
3. Check service status (sidebar shows API status)

---

**Happy Traveling! 🌴**

