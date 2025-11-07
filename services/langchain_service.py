"""
LangChain service for managing chains, RAG, and memory.
Integrates LangChain with ChromaDB and OpenAI.
"""

import os
from typing import Optional, List, Dict, Any
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st

from config.settings import Settings


class LangChainService:
    """Service for LangChain operations including RAG chains and memory management."""
    
    def __init__(self, settings: Settings, chroma_service=None):
        self.settings = settings
        self.chroma_service = chroma_service
        self.llm: Optional[ChatOpenAI] = None
        self.vectorstore: Optional[Chroma] = None
        self.memory: Optional[ConversationBufferWindowMemory] = None
        self.rag_chain: Optional[ConversationalRetrievalChain] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self._initialize()
    
    def _initialize(self):
        """Initialize LangChain components."""
        try:
            # Check if OpenAI API key is available
            if not self.settings.OPENAI_API_KEY:
                print("[WARN] LangChain: OpenAI API key not found, skipping initialization")
                return
            
            # Initialize OpenAI LLM
            llm_kwargs = {
                "model": self.settings.DEPLOYMENT_NAME,
                "temperature": 0.7,
                "max_tokens": 900,
            }
            
            # Add API key if available
            if self.settings.OPENAI_API_KEY:
                llm_kwargs["openai_api_key"] = self.settings.OPENAI_API_KEY
            
            # Add base URL if custom endpoint
            if self.settings.OPENAI_ENDPOINT and self.settings.OPENAI_ENDPOINT != "https://api.openai.com/v1":
                llm_kwargs["base_url"] = self.settings.OPENAI_ENDPOINT
            
            self.llm = ChatOpenAI(**llm_kwargs)
            
            # Initialize embeddings (using same model as ChromaService)
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.settings.EMBEDDING_MODEL_NAME,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': False}
                )
            except Exception as e:
                print(f"[WARN] LangChain: Failed to load embeddings: {e}")
                self.embeddings = None
            
            # Initialize memory
            try:
                self.memory = ConversationBufferWindowMemory(
                    k=12,  # Keep last 12 messages
                    return_messages=True,
                    memory_key="chat_history",
                    output_key="answer"
                )
            except Exception as e:
                print(f"[WARN] LangChain: Failed to initialize memory: {e}")
                self.memory = None
            
            # Initialize vectorstore if ChromaDB is available
            if self.chroma_service and self.chroma_service.client:
                self._initialize_vectorstore()
            
        except ImportError as e:
            print(f"[WARN] LangChain dependencies not installed: {e}")
            print("[INFO] Install with: pip install langchain langchain-openai langchain-community")
            self.llm = None
        except Exception as e:
            print(f"[WARN] LangChain initialization failed: {e}")
            self.llm = None
    
    def _initialize_vectorstore(self):
        """Initialize ChromaDB vectorstore for LangChain."""
        try:
            # Use existing ChromaDB client and collection
            if self.chroma_service.travel_col:
                self.vectorstore = Chroma(
                    client=self.chroma_service.client,
                    collection_name="vietnam_travel_v2",
                    embedding_function=self.embeddings
                )
        except Exception as e:
            print(f"[WARN] Failed to initialize LangChain vectorstore: {e}")
    
    def create_rag_chain(self, chat_history: List = None) -> Optional[ConversationalRetrievalChain]:
        """Create a RAG chain with LangChain."""
        if not self.llm or not self.vectorstore:
            return None
        
        try:
            # Create system prompt
            system_prompt = self.settings.SYSTEM_PROMPT + """
            
Khi trả lời, hãy sử dụng thông tin từ các tài liệu được cung cấp. Nếu bạn tham khảo thông tin từ tài liệu, hãy đánh dấu nguồn một cách rõ ràng.
Luôn trả lời bằng tiếng Việt một cách tự nhiên và thân thiện.
"""
            
            # Create prompt template for combine_docs_chain
            # Note: StuffDocumentsChain requires 'context' variable for documents
            combine_docs_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    system_prompt + "\n\nSử dụng các thông tin sau đây để trả lời câu hỏi:\n\n{context}"
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ])
            
            # Create RAG chain without memory (we'll pass chat_history directly)
            # Use get_chat_history to handle chat_history conversion
            def get_chat_history(inputs) -> List:
                """Convert chat_history from input to list of messages."""
                if isinstance(inputs, list):
                    return inputs
                return []
            
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": self.settings.RAG_TOP_K}
                ),
                memory=None,  # Don't use memory, pass chat_history directly
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": combine_docs_prompt},
                get_chat_history=get_chat_history,
                verbose=False
            )
            
            return chain
        except Exception as e:
            print(f"[WARN] Failed to create RAG chain: {e}")
            return None
    
    def generate_with_rag(self, user_input: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """
        Generate response using RAG chain.
        
        Returns:
            Dictionary with response and metadata
        """
        if not self.llm:
            return None  # Return None to indicate LangChain is not available, will fallback to traditional method
        
        # Build chat_history as list of messages from conversation_history
        chat_history_messages = []
        for msg in conversation_history[-12:]:  # Keep last 12 messages
            if msg.get("role") == "user":
                chat_history_messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                chat_history_messages.append(AIMessage(content=msg.get("content", "")))
        
        # Try RAG chain if vectorstore is available
        if self.vectorstore:
            try:
                chain = self.create_rag_chain()
                if chain:
                    # Pass chat_history as list of messages
                    result = chain.invoke({
                        "question": user_input,
                        "chat_history": chat_history_messages
                    })
                    response = result.get("answer", "")
                    source_docs = result.get("source_documents", [])
                    
                    # Format sources
                    sources = []
                    for doc in source_docs:
                        sources.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        })
                    
                    return {
                        "response": response,
                        "sources": sources,
                        "rag_used": True,
                        "sources_count": len(sources)
                    }
            except Exception as e:
                print(f"[WARN] RAG chain failed: {e}, falling back to direct LLM")
        
        # Fallback to direct LLM call
        return self._generate_direct_llm(user_input, conversation_history)
    
    def _generate_direct_llm(self, user_input: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Generate response using direct LLM call without RAG."""
        try:
            # Build messages
            messages = [SystemMessage(content=self.settings.SYSTEM_PROMPT)]
            
            # Add conversation history
            for msg in conversation_history[-12:]:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))
            
            # Add current user input
            messages.append(HumanMessage(content=user_input))
            
            # Generate response
            response = self.llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "response": response_text,
                "sources": [],
                "rag_used": False,
                "sources_count": 0
            }
        except Exception as e:
            return {
                "response": f"⚠️ Lỗi khi tạo phản hồi: {e}",
                "sources": [],
                "rag_used": False,
                "sources_count": 0
            }
    
    def add_to_memory(self, user_input: str, assistant_response: str):
        """Add conversation to memory."""
        if self.memory:
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(assistant_response)
    
    def clear_memory(self):
        """Clear conversation memory."""
        if self.memory:
            self.memory.chat_memory.clear()

