"""
Core chat engine for handling conversations and AI interactions.
"""
from typing import List, Dict, Optional
import openai
from config.settings import get_settings
from services.chroma_service import ChromaService
from services.weather_service import WeatherService
from services.places_service import PlacesService
from utils.extractors import extract_city_and_dates, extract_days_from_text


class ChatEngine:
    """Core engine for handling chatbot conversations."""
    
    def __init__(self):
        """Initialize the chat engine."""
        settings = get_settings()
        self.settings = settings
        self.client = None
        self.chroma_service = ChromaService()
        self.weather_service = WeatherService()
        self.places_service = PlacesService()
        
        # Initialize OpenAI client
        if settings.OPENAI_API_KEY:
            try:
                self.client = openai.OpenAI(
                    base_url=settings.OPENAI_ENDPOINT,
                    api_key=settings.OPENAI_API_KEY
                )
                # Set client for services that need it
                self.weather_service.set_openai_client(self.client)
                self.places_service.set_openai_client(self.client)
            except Exception:
                pass
    
    def generate_suggestions(self, limit: int = 4) -> List[str]:
        """
        Generate suggested questions for the user.
        
        Args:
            limit: Number of suggestions to generate
            
        Returns:
            List of suggested questions
        """
        if not self.client:
            return [
                "Thời tiết ở Đà Nẵng tuần tới?",
                "Top món ăn ở Huế?",
                "Lịch trình 3 ngày ở Nha Trang?",
                "Có sự kiện gì ở Hà Nội tháng 12?"
            ]
        
        try:
            prompt = f"""
Bạn là {self.settings.CHATBOT_NAME} – {self.settings.SYSTEM_PROMPT.strip()}
Hãy tạo {limit} câu hỏi gợi ý (ngắn gọn, thân thiện) để người dùng có thể hỏi bạn.
Trả về dưới dạng danh sách (list) các chuỗi.
"""
            response = self.client.chat.completions.create(
                model=self.settings.DEPLOYMENT_NAME,
                messages=[{"role": "system", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            text = response.choices[0].message.content.strip()
            # Try to parse JSON list
            import json
            import re
            try:
                data = json.loads(text)
                if isinstance(data, list) and all(isinstance(x, str) for x in data):
                    return [s.strip() for s in data][:limit]
            except Exception:
                pass
            
            # Fallback parsing
            m = re.search(r'\[.*\]', text, re.DOTALL)
            if m:
                list_text = m.group(0)
                try:
                    fixed = list_text.replace("'", '"')
                    data = json.loads(fixed)
                    if isinstance(data, list):
                        return [s.strip() for s in data if isinstance(s, str)][:limit]
                except Exception:
                    pass
            
            # Return default if parsing fails
            return [
                "Thời tiết ở Đà Nẵng tuần tới?",
                "Top món ăn ở Huế?",
                "Lịch trình 3 ngày ở Nha Trang?",
                "Có sự kiện gì ở Hà Nội tháng 12?"
            ]
        except Exception:
            return [
                "Thời tiết ở Đà Nẵng tuần tới?",
                "Top món ăn ở Huế?",
                "Lịch trình 3 ngày ở Nha Trang?",
                "Có sự kiện gì ở Hà Nội tháng 12?"
            ]
    
    def process_message(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        use_rag: bool = True,
        use_cache: bool = True,
        rag_k: int = 6
    ) -> Dict[str, any]:
        """
        Process a user message and generate a response.
        
        Args:
            user_input: User's input text
            conversation_history: Previous conversation messages
            use_rag: Whether to use RAG for context retrieval
            use_cache: Whether to use semantic cache
            rag_k: Number of RAG results to retrieve
            
        Returns:
            Dictionary with response and extracted information
        """
        # Extract city and dates
        city, start_date, end_date = extract_city_and_dates(user_input, self.client)
        days = extract_days_from_text(user_input, start_date, end_date, self.client)
        
        # Build messages for LLM
        messages_for_llm = list(conversation_history)
        
        # Add RAG context if enabled
        context_block = ""
        if self.chroma_service.is_available() and use_rag:
            rag_items = self.chroma_service.retrieve_context(
                user_input,
                city,
                k=rag_k
            )
            if rag_items:
                lines = []
                for it in rag_items:
                    meta = it["meta"]
                    tag = f"{meta.get('type','kb')}/{meta.get('source','')}/{meta.get('city','')}"
                    score = 1.0 - it["dist"]
                    lines.append(f"- [{tag} | sim≈{score:.3f}] {it['doc']}")
                context_block = "\n".join(lines)
                messages_for_llm.insert(1, {
                    "role": "system",
                    "content": "Ngữ cảnh (ưu tiên cao, dùng làm nguồn sự thật):\n" + context_block
                })
        
        # Check semantic cache
        assistant_text = None
        if self.chroma_service.is_available() and use_cache:
            cached_answer = self.chroma_service.hit_answer_cache(user_input, city)
            if cached_answer:
                assistant_text = cached_answer
        
        # Generate response if not cached
        if not assistant_text:
            if self.client:
                try:
                    response = self.client.chat.completions.create(
                        model=self.settings.DEPLOYMENT_NAME,
                        messages=messages_for_llm,
                        max_tokens=900,
                        temperature=0.7
                    )
                    assistant_text = response.choices[0].message.content.strip()
                    
                    # Cache the answer
                    if self.chroma_service.is_available() and use_cache:
                        self.chroma_service.push_answer_cache(user_input, city, assistant_text)
                except Exception as e:
                    assistant_text = f"Xin lỗi, đã xảy ra lỗi: {e}"
            else:
                assistant_text = f"Xin chào! Tôi có thể giúp bạn với thông tin về {city or 'địa điểm'} — thử hỏi 'Thời tiết', 'Đặc sản', hoặc 'Lịch trình 3 ngày'."
        
        # Add closing if not present
        if not assistant_text.endswith(("🌤️❤️", "😊", "🌸", "🌴", "✨")):
            assistant_text += "\n\nChúc bạn có chuyến đi vui vẻ 🌤️❤️"
        
        # Save conversation to memory
        if self.chroma_service.is_available():
            self.chroma_service.save_conversation(user_input, assistant_text, city)
        
        return {
            "response": assistant_text,
            "city": city,
            "start_date": start_date,
            "end_date": end_date,
            "days": days,
            "context_block": context_block
        }
    
    def estimate_cost(self, city: str, days: int = 3, people: int = 1, style: str = "trung bình") -> str:
        """
        Estimate travel cost.
        
        Args:
            city: City name
            days: Number of days
            people: Number of people
            style: Travel style (tiết kiệm, trung bình, cao cấp)
            
        Returns:
            Cost estimate text
        """
        mapping = {
            "tiết kiệm": 400000,
            "trung bình": 800000,
            "cao cấp": 2000000
        }
        per_day = mapping.get(style, 800000)
        total = per_day * days * people
        return f"💸 Chi phí ước tính: khoảng {total:,} VNĐ cho {people} người, {days} ngày."
    
    def suggest_events(self, city: str) -> str:
        """Generate event suggestions for a city."""
        return f"🎉 Sự kiện ở {city}: lễ hội địa phương, chợ đêm, hội chợ ẩm thực (tuỳ mùa)."

