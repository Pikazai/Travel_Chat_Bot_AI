"""
Chat engine module.
Main orchestration logic for chatbot conversations.
"""

from typing import Optional, List, Dict
from datetime import datetime

from config.settings import Settings
from services.chroma_service import ChromaService
from services.logger_service import LoggerService
from services.langchain_service import LangChainService
from core.intent_detector import IntentDetector
from core.entity_extractor import EntityExtractor
from utils.text_processing import extract_days_from_text


class ChatEngine:
    """Main chat engine orchestrating conversation flow."""
    
    def __init__(self, settings: Settings, openai_client, chroma_service: ChromaService,
                 logger_service: LoggerService, langchain_service: Optional[LangChainService] = None):
        self.settings = settings
        self.client = openai_client
        self.chroma_service = chroma_service
        self.logger_service = logger_service
        self.intent_detector = IntentDetector(settings, chroma_service)
        self.entity_extractor = EntityExtractor(settings, openai_client)
        # Initialize LangChain service if not provided
        self.langchain_service = langchain_service or LangChainService(settings, chroma_service)
    
    def process_message(self, user_input: str, conversation_history: List[Dict],
                      **kwargs) -> Dict:
        """
        Process user message and generate response.
        
        Returns:
            Dictionary with response text and metadata
        """
        # Check if travel-related
        if not self.entity_extractor.is_travel_related(user_input):
            return {
                "response": "Xin lỗi 😅, tôi chỉ hỗ trợ các câu hỏi liên quan đến **du lịch Việt Nam**, như thời tiết, địa điểm, món ăn, lịch trình...",
                "intent": None,
                "rag_used": False,
                "sources_count": 0
            }
        
        # Extract entities
        city, start_date, end_date = self.entity_extractor.extract_city_and_dates(user_input)
        
        # Default dates to today if not provided
        today = datetime.now().date()
        if start_date is None:
            start_date = datetime.combine(today, datetime.min.time())
        if end_date is None:
            end_date = datetime.combine(today, datetime.min.time())
        if end_date < start_date:
            end_date = start_date
        
        days = extract_days_from_text(user_input, start_date, end_date)
        
        # Track metadata
        rag_used = False
        sources_count = 0
        intent_used = None
        
        # Try intent detection first
        detected_intent = self.intent_detector.detect_intent(user_input)
        
        if detected_intent:
            intent_used = detected_intent
            # Handle intent-specific responses
            response = self.intent_detector.handle_intent(
                detected_intent, user_input, city, start_date, end_date,
                days=days, **kwargs
            )
            
            if response:
                # Save to memory
                self.chroma_service.add_to_memory(user_input, role="user", city=city)
                self.chroma_service.add_to_memory(response, role="assistant", city=city)
                
                # Log interaction
                self.logger_service.log_interaction(
                    user_input, city, start_date, end_date, intent_used, False, 0
                )
                
                return {
                    "response": response,
                    "intent": intent_used,
                    "intent_used": intent_used,
                    "rag_used": False,
                    "sources_count": 0,
                    "memory_used": False,
                    "city": city,
                    "start_date": start_date,
                    "end_date": end_date
                }
        
        # Fallback to RAG + LLM generation using LangChain
        # Try LangChain RAG first
        langchain_result = None
        if self.langchain_service and self.langchain_service.llm:
            try:
                langchain_result = self.langchain_service.generate_with_rag(user_input, conversation_history)
                if langchain_result and langchain_result.get("response"):
                    assistant_text = langchain_result.get("response")
                    rag_used = langchain_result.get("rag_used", False)
                    sources_count = langchain_result.get("sources_count", 0)
                    langchain_sources = langchain_result.get("sources", [])
                    
                    # Convert LangChain sources to docs format
                    docs = []
                    for src in langchain_sources:
                        docs.append({
                            "text": src.get("content", ""),
                            "meta": src.get("metadata", {}),
                            "id": src.get("metadata", {}).get("id", "")
                        })
                    
                    # Recall memories for metadata
                    recent_mem = self.chroma_service.recall_memories(user_input, k=self.settings.MEMORY_RECALL_K)
                    memory_used = len(recent_mem) > 0
                    
                    # Add to LangChain memory
                    self.langchain_service.add_to_memory(user_input, assistant_text)
                    
                    # Save to ChromaDB memory as well
                    self.chroma_service.add_to_memory(user_input, role="user", city=city)
                    self.chroma_service.add_to_memory(assistant_text, role="assistant", city=city)
                    
                    # Log interaction
                    self.logger_service.log_interaction(
                        user_input, city, start_date, end_date, intent_used, rag_used, sources_count
                    )
                    
                    return {
                        "response": assistant_text,
                        "intent": intent_used,
                        "intent_used": intent_used,
                        "rag_used": rag_used,
                        "sources_count": sources_count,
                        "memory_used": memory_used,
                        "city": city,
                        "start_date": start_date,
                        "end_date": end_date,
                        "rag_docs": docs
                    }
            except Exception as e:
                print(f"[WARN] LangChain RAG failed: {e}, falling back to traditional RAG")
        
        # Fallback to traditional RAG + LLM generation
        docs, rag_context = self.chroma_service.rag_query(user_input, k=self.settings.RAG_TOP_K)
        sources_count = len(docs)
        rag_used = sources_count > 0
        
        # Recall memories
        recent_mem = self.chroma_service.recall_memories(user_input, k=self.settings.MEMORY_RECALL_K)
        memory_used = len(recent_mem) > 0
        
        # Build context augmentation
        augmentation = "\n\n--- Thông tin tham khảo nội bộ (trích dẫn): ---\n"
        if rag_context:
            augmentation += rag_context + "\n\n"
        
        if recent_mem:
            recall_parts = []
            for m in recent_mem:
                ts = m.get("meta", {}).get("timestamp", "")
                role = m.get("meta", {}).get("role", "")
                recall_parts.append(f"[mem:{m.get('id')}] ({role} {ts}) {m.get('text')[:400]}")
            augmentation += "\n--- Nhớ gần đây ---\n" + "\n".join(recall_parts) + "\n\n"
        
        augmentation += "--- Khi trả lời, nếu dùng thông tin từ phần trên hãy đánh dấu nguồn như [src:ID] hoặc [mem:ID]. ---\n"
        
        # Build messages for LLM
        temp_messages = [{"role": "system", "content": self.settings.SYSTEM_PROMPT + "\n\n" + augmentation}]
        temp_messages.extend(conversation_history[-12:])  # Keep last 12 messages
        
        # Generate response
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.DEPLOYMENT_NAME,
                    messages=temp_messages,
                    max_tokens=900,
                    temperature=0.7
                )
                assistant_text = response.choices[0].message.content.strip()
            except Exception as e:
                assistant_text = f"⚠️ Lỗi khi tạo phản hồi: {e}"
        else:
            assistant_text = f"Xin chào! Tôi có thể giúp bạn với thông tin về {city or 'địa điểm'} — thử hỏi 'Thời tiết', 'Đặc sản', hoặc 'Lịch trình 3 ngày'."
        
        # Add closing message
        if not assistant_text.endswith(("🌤️❤️", "😊", "🌸", "🌴", "✨")):
            assistant_text += "\n\nChúc bạn có chuyến đi vui vẻ 🌤️❤️"
        
        # Save to memory
        self.chroma_service.add_to_memory(user_input, role="user", city=city)
        self.chroma_service.add_to_memory(assistant_text, role="assistant", city=city)
        
        # Log interaction
        self.logger_service.log_interaction(
            user_input, city, start_date, end_date, intent_used, rag_used, sources_count
        )
        
        return {
            "response": assistant_text,
            "intent": intent_used,
            "intent_used": intent_used,
            "rag_used": rag_used,
            "sources_count": sources_count,
            "memory_used": memory_used,
            "city": city,
            "start_date": start_date,
            "end_date": end_date,
            "rag_docs": docs
        }

