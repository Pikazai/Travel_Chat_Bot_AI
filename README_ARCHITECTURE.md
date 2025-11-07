# Travel Chat Bot AI - Refactored Architecture

## 🗂️ Directory Structure

```
TRAVEL_CHAT_BOT_AI/
│
├── core/                          # Core chatbot logic & AI processing
│   ├── __init__.py
│   ├── chat_engine.py            # Main conversation orchestration
│   ├── intent_detector.py        # Intent detection using ChromaDB
│   └── entity_extractor.py       # Extract city, dates from text
│
├── services/                      # External services & integrations
│   ├── __init__.py
│   ├── chroma_service.py         # ChromaDB RAG operations
│   ├── langchain_service.py      # LangChain RAG chains & memory
│   ├── voice_service.py          # Speech-to-Text & Text-to-Speech
│   ├── logger_service.py         # SQLite logging
│   ├── weather_service.py        # OpenWeatherMap API
│   ├── geocoding_service.py      # Location lookup & maps
│   ├── image_service.py          # Pixabay image API
│   ├── food_service.py           # Food recommendations
│   └── restaurant_service.py     # Restaurant recommendations
│
├── ui/                            # Streamlit user interface
│   ├── __init__.py
│   └── app.py                    # Main UI application
│
├── config/                        # Configuration management
│   ├── __init__.py
│   └── settings.py               # Environment variables & constants
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── text_processing.py        # Text parsing utilities
│   └── date_utils.py             # Date handling utilities
│
├── data/                          # Data files (CSV, etc.)
│   ├── vietnam_foods.csv
│   ├── restaurants_vn.csv
│   └── vietnam_travel_docs.csv
│
├── chromadb_data/                 # ChromaDB persistent storage
│
├── main.py                        # Application entry point
├── requirements.txt               # Python dependencies
├── travel_chatbot_logs.db        # SQLite database
├── ARCHITECTURE.md                # Architecture documentation
└── README.md                      # Project documentation
```

## 🧩 Module Explanations

### **core/** - Core Chatbot Logic
Contains the main AI processing logic:
- **chat_engine.py**: Orchestrates the entire conversation flow, coordinates between services, prioritizes LangChain RAG if available
- **intent_detector.py**: Detects user intent (weather_query, food_query, etc.) using semantic matching
- **entity_extractor.py**: Extracts structured data (city names, dates) from natural language

### **services/** - External Services
Handles all external integrations:
- **chroma_service.py**: Vector database operations for RAG, memory, and intent matching
- **langchain_service.py**: LangChain integration for enhanced RAG with ConversationalRetrievalChain
- **voice_service.py**: Converts speech ↔ text (STT/TTS)
- **logger_service.py**: Logs interactions to SQLite for analytics
- **weather_service.py**: Fetches weather forecasts from OpenWeatherMap
- **geocoding_service.py**: Converts city names to coordinates and displays maps
- **image_service.py**: Retrieves images from Pixabay API
- **food_service.py** & **restaurant_service.py**: Provides food/restaurant recommendations

### **ui/** - User Interface
Streamlit application:
- **app.py**: Main UI rendering, handles user interactions, displays results

### **config/** - Configuration
Centralized settings:
- **settings.py**: Loads environment variables, provides default values

### **utils/** - Utilities
Helper functions:
- **text_processing.py**: Text parsing (extract days, split foods)
- **date_utils.py**: Date parsing and validation

## 🔄 Complete Flow Diagram

```
USER INPUT (Text/Voice)
    │
    ├─→ [Voice Service] → Speech-to-Text → Text
    │
    ├─→ [Entity Extractor] → Extract: city, dates, validate topic
    │
    ├─→ [Intent Detector] → Query ChromaDB intent collection
    │                        │
    │                        ├─→ Intent Found? → Handle directly (weather/food)
    │                        │
    │                        └─→ No Intent → RAG Query
    │
    ├─→ [LangChain Check] → Available?
    │                        │
    │                        ├─→ YES → [LangChain Service] → ConversationalRetrievalChain
    │                        │                                  → Auto retrieve + generate
    │                        │
    │                        └─→ NO → [Chroma Service] → RAG Query → Retrieve relevant documents
    │                                                      → Memory Recall → Get similar past conversations
    │
    ├─→ [Chat Engine] → Build augmented prompt (RAG + Memory)
    │                  → Call OpenAI LLM (via LangChain or direct)
    │                  → Generate response
    │
    ├─→ [Memory Storage] → Save to LangChain memory (if used)
    │                     → Save to ChromaDB memory collection
    │
    ├─→ [Logger Service] → Log to SQLite database
    │
    ├─→ [Additional Services] → Weather, Maps, Images, Food, Restaurants
    │
    └─→ [UI Rendering] → Display response, sources, maps, images
                        → Text-to-Speech (optional)
```

## 📋 Step-by-Step Process

1. **User Input**: Text or voice input
2. **Voice Processing** (if voice): Convert audio to text
3. **Entity Extraction**: Parse city, dates from text
4. **Topic Validation**: Check if travel-related
5. **Intent Detection**: Try to match intent via ChromaDB
6. **RAG Retrieval & Generation**:
   - **If LangChain available**: Use ConversationalRetrievalChain (auto retrieve + generate)
   - **If LangChain unavailable**: Traditional RAG query + manual LLM call
7. **Memory Recall**: 
   - LangChain ConversationBufferWindowMemory (conversation context)
   - ChromaDB memory recall (similar past conversations)
8. **LLM Generation**: Generate response with augmented context (via LangChain chain or direct API)
9. **Memory Storage**: Save conversation to both LangChain memory and ChromaDB
10. **Logging**: Record interaction in SQLite
11. **Additional Data**: Fetch weather, maps, images, foods
12. **UI Display**: Render response and additional information

## 🚀 Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run main.py
```

## 🔑 Key Benefits

- **Modular**: Each module has a single, clear responsibility
- **Maintainable**: Easy to understand and modify
- **Extensible**: Simple to add new features
- **Testable**: Services can be tested independently
- **Production-Ready**: Clean architecture suitable for deployment

## 📝 Notes

- All data files (`.db`, `.csv`) are preserved
- Environment variables loaded from `.env` or Streamlit secrets
- ChromaDB data persists in `chromadb_data/` directory
- SQLite logs stored in `travel_chatbot_logs.db`

