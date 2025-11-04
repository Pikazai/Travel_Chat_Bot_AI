# 🏗️ Travel Chatbot Architecture

## 📁 Project Structure

```
TRAVEL_CHAT_BOT_AI/
│
├── core/                          # Core business logic
│   ├── __init__.py
│   └── chat_engine.py            # Main conversation engine (LLM, RAG, memory)
│
├── services/                      # External service integrations
│   ├── __init__.py
│   ├── chroma_service.py         # ChromaDB vector store (RAG, cache, memory)
│   ├── logger_service.py         # SQLite logging service
│   ├── weather_service.py        # OpenWeatherMap API integration
│   ├── places_service.py         # Google Places API + CSV fallback
│   ├── image_service.py          # Pixabay image API
│   └── voice_service.py          # Speech-to-text & Text-to-speech
│
├── ui/                           # Streamlit user interface
│   ├── __init__.py
│   ├── app.py                    # Main Streamlit application
│   ├── components.py             # Reusable UI components
│   └── styles.py                 # CSS styles
│
├── config/                       # Configuration management
│   ├── __init__.py
│   └── settings.py               # Environment variables & constants
│
├── utils/                        # Utility functions
│   ├── __init__.py
│   ├── extractors.py             # Data extraction (city, dates, days)
│   └── geocoding.py              # Geocoding & map utilities
│
├── tests/                        # Unit & integration tests (future)
│
├── data/                         # Data files
│   ├── vietnam_foods.csv
│   ├── restaurants_vn.csv
│   └── vietnam_travel_docs.csv
│
├── main.py                       # Entry point (run: streamlit run main.py)
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── ARCHITECTURE.md               # This file
```

---

## 🧩 Module Overview

### **core/** - Core Business Logic

#### `chat_engine.py`
**Purpose**: Central conversation engine that orchestrates AI interactions.

**Key Responsibilities**:
- Manages OpenAI client initialization
- Processes user messages with RAG (Retrieval-Augmented Generation)
- Implements semantic caching for cost optimization
- Generates conversation suggestions
- Coordinates with services (ChromaDB, Weather, Places)

**Key Methods**:
- `process_message()`: Main method to process user input and generate responses
- `generate_suggestions()`: Generate suggested questions for users
- `estimate_cost()`: Calculate travel cost estimates
- `suggest_events()`: Generate event suggestions

**Flow**:
```
User Input → Extract Info → Check Cache → RAG Retrieval → LLM Generation → Cache Answer → Save Conversation
```

---

### **services/** - External Service Integrations

#### `chroma_service.py`
**Purpose**: Vector database operations for RAG, semantic caching, and conversation memory.

**Collections**:
- `travel_kb`: Knowledge base (foods, restaurants from CSV)
- `answer_cache`: Semantic answer cache (reduces API costs)
- `conversations`: Conversation history for context

**Key Methods**:
- `retrieve_context()`: Retrieve relevant context for RAG
- `hit_answer_cache()`: Check if similar query was answered before
- `push_answer_cache()`: Cache new answers
- `seed_kb_from_csvs()`: Populate knowledge base from CSV files

#### `logger_service.py`
**Purpose**: SQLite database logging for analytics.

**Database Schema**:
```sql
interactions (
    id, timestamp, user_input, city, start_date, end_date, intent
)
```

**Key Methods**:
- `log_interaction()`: Log user interactions
- `get_interactions()`: Retrieve interaction history
- `clear_interactions()`: Clear all logs

#### `weather_service.py`
**Purpose**: Weather forecast integration via OpenWeatherMap API.

**Key Methods**:
- `get_weather_forecast()`: Get weather for a city/date range
- `set_openai_client()`: Set OpenAI client for AI-based city resolution

#### `places_service.py`
**Purpose**: Restaurant and food recommendations.

**Key Methods**:
- `get_restaurants()`: Get restaurants (Google Places → CSV fallback)
- `get_foods()`: Get food recommendations (CSV → GPT fallback)

#### `image_service.py`
**Purpose**: Image fetching from Pixabay API.

**Key Methods**:
- `get_city_image()`: Get city landscape images
- `get_food_images()`: Get food images

#### `voice_service.py`
**Purpose**: Voice input/output processing.

**Key Methods**:
- `speech_to_text()`: Convert audio to text (Google Speech Recognition)
- `text_to_speech()`: Convert text to audio (gTTS)
- `convert_to_wav()`: Audio format conversion (supports multiple formats)

---

### **config/** - Configuration Management

#### `settings.py`
**Purpose**: Centralized configuration management.

**Loads from**:
1. Streamlit secrets (`.streamlit/secrets.toml`)
2. Environment variables (`.env` or system env)

**Configuration**:
- OpenAI API keys & endpoints
- External API keys (Weather, Places, Pixabay)
- ChromaDB path
- Database paths
- Chatbot settings (name, system prompt)

**Usage**:
```python
from config.settings import get_settings
settings = get_settings()
api_key = settings.OPENAI_API_KEY
```

---

### **utils/** - Utility Functions

#### `extractors.py`
**Purpose**: Extract structured information from user text.

**Key Functions**:
- `extract_city_and_dates()`: Extract city, start_date, end_date using AI
- `extract_days_from_text()`: Extract number of days (regex + AI fallback)
- `resolve_city_via_ai()`: AI-based city name resolution

#### `geocoding.py`
**Purpose**: Geocoding and map display.

**Key Functions**:
- `geocode_city()`: Convert city name to coordinates
- `show_map()`: Display interactive map using PyDeck

---

### **ui/** - Streamlit User Interface

#### `app.py`
**Purpose**: Main Streamlit application.

**Key Features**:
- Hero section with quick search
- Chat interface with voice input
- Sidebar with settings
- Analytics tab with statistics
- RAG debugger

#### `components.py`
**Purpose**: Reusable UI components.

**Components**:
- `render_hero_section()`: Hero banner with search form
- `render_sidebar()`: Settings sidebar
- `render_quick_search()`: Quick search form
- `render_suggestions()`: Suggested questions
- `render_voice_input()`: Voice recording interface

#### `styles.py`
**Purpose**: CSS styles for Streamlit.

---

## 🔄 System Flow

### **Voice Input Flow**
```
User speaks → mic_recorder → audio bytes → VoiceService.convert_to_wav() 
→ VoiceService.speech_to_text() → text → ChatEngine.process_message()
```

### **Message Processing Flow**
```
User Input → ChatEngine.process_message()
    ↓
1. Extract city & dates (extractors.py)
    ↓
2. Check semantic cache (ChromaService.hit_answer_cache())
    ↓
3. If not cached:
    a. RAG retrieval (ChromaService.retrieve_context())
    b. Build context with RAG results
    c. Call OpenAI API
    d. Cache answer (ChromaService.push_answer_cache())
    ↓
4. Save conversation (ChromaService.save_conversation())
    ↓
5. Log interaction (LoggerService.log_interaction())
    ↓
6. Display response + enrichments (weather, maps, food, images)
```

### **RAG (Retrieval-Augmented Generation) Flow**
```
User Query → ChromaService.retrieve_context()
    ↓
Embed query → Vector search in travel_kb collection
    ↓
Filter by city (if provided) → Return top-k results
    ↓
Inject context into LLM prompt → Generate response
```

---

## 🚀 Getting Started

### **1. Installation**
```bash
pip install -r requirements.txt
```

### **2. Configuration**
Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your_key"
OPENAI_API_KEY_EMBEDDING = "your_key"
OPENWEATHERMAP_API_KEY = "your_key"
PLACES_API_KEY = "your_key"
PIXABAY_API_KEY = "your_key"
```

Or set environment variables:
```bash
export OPENAI_API_KEY="your_key"
export OPENWEATHERMAP_API_KEY="your_key"
# ... etc
```

### **3. Run Application**
```bash
streamlit run main.py
```

### **4. Seed ChromaDB (Optional)**
In the UI, go to sidebar → click "📥 Seed KB từ CSV" to populate knowledge base.

---

## 🔧 Extending the System

### **Adding a New Service**
1. Create `services/new_service.py`
2. Implement service class with initialization
3. Add to `services/__init__.py`
4. Initialize in `ui/app.py` session state
5. Use in `ChatEngine` or UI components

**Example**:
```python
# services/new_service.py
class NewService:
    def __init__(self):
        self.api_key = get_settings().NEW_API_KEY
    
    def do_something(self):
        # Implementation
        pass
```

### **Adding a New UI Component**
1. Create component function in `ui/components.py`
2. Call from `ui/app.py` where needed

**Example**:
```python
def render_new_component():
    st.write("New component")
```

### **Adding a New Utility Function**
1. Add to appropriate file in `utils/`
2. Export in `utils/__init__.py`
3. Import where needed

---

## 📊 Data Flow Diagram

```
┌─────────────┐
│   User      │
│  (Browser)  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         Streamlit UI (ui/app.py)    │
│  ┌──────────┐  ┌─────────────────┐ │
│  │ Hero     │  │  Chat Interface │ │
│  │ Search   │  │  Voice Input    │ │
│  └──────────┘  └────────┬────────┘ │
└──────────────────────────┼──────────┘
                           │
                           ▼
┌─────────────────────────────────────┐
│      ChatEngine (core/)             │
│  ┌───────────────────────────────┐  │
│  │ 1. Extract Info (utils/)      │  │
│  │ 2. Check Cache (ChromaDB)     │  │
│  │ 3. RAG Retrieval (ChromaDB)   │  │
│  │ 4. LLM Generation (OpenAI)    │  │
│  │ 5. Cache Answer (ChromaDB)    │  │
│  └───────────────────────────────┘  │
└──────┬──────────────────────────────┘
       │
       ├──► ChromaService (RAG, Cache, Memory)
       ├──► WeatherService (OpenWeatherMap)
       ├──► PlacesService (Google Places / CSV)
       ├──► ImageService (Pixabay)
       ├──► VoiceService (Speech-to-Text / TTS)
       └──► LoggerService (SQLite)
```

---

## 🎯 Key Design Principles

1. **Separation of Concerns**: Each module has a single, clear responsibility
2. **Dependency Injection**: Services receive dependencies (e.g., OpenAI client) via setters
3. **Graceful Degradation**: Services handle missing dependencies gracefully
4. **Configuration Centralization**: All config in `config/settings.py`
5. **Modularity**: Easy to add/remove features without affecting others
6. **Production-Ready**: Error handling, logging, caching for scalability

---

## 📝 Notes

- **ChromaDB** is optional - system works without it (no RAG, no cache)
- **Voice features** require `ffmpeg` to be installed
- **All external APIs** have fallbacks (CSV for foods/restaurants, graceful errors)
- **Session state** is used for Streamlit UI state management
- **Database** is SQLite (easy to migrate to PostgreSQL if needed)

---

## 🔮 Future Enhancements

- **FastAPI Backend**: Separate API layer for web app deployment
- **User Profiles**: Per-user conversation history and preferences
- **Multi-language Support**: Full i18n support
- **Real-time Updates**: WebSocket support for live updates
- **Advanced Analytics**: Dashboard with ML insights
- **Testing Suite**: Unit and integration tests in `tests/`

---

**Version**: 2.0 (Modular Architecture)  
**Last Updated**: 2025-11-04

