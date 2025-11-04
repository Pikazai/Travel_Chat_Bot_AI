"""
ChromaDB service for vector storage and retrieval.
Handles RAG (Retrieval-Augmented Generation), semantic caching, and conversation memory.
"""
import time
import pandas as pd
import re
from datetime import datetime
from typing import Optional, List, Dict, Any
from chromadb.utils import embedding_functions
from config.settings import get_settings

# ChromaDB import with graceful fallback
CHROMA_AVAILABLE = True
try:
    import chromadb
except Exception:
    CHROMA_AVAILABLE = False
    chromadb = None


class ChromaService:
    """Service for ChromaDB vector storage and retrieval."""
    
    def __init__(self):
        """Initialize the ChromaDB service."""
        settings = get_settings()
        self.chroma_path = settings.CHROMA_PATH
        self.embedding_model = settings.EMBEDDING_MODEL
        self.api_key = settings.OPENAI_API_KEY
        self.api_key_embedding = settings.OPENAI_API_KEY_EMBEDDING
        self.client = None
        self.kb_collection = None
        self.cache_collection = None
        self.conv_collection = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and collections."""
        if not CHROMA_AVAILABLE:
            return
        
        try:
            self.client = chromadb.PersistentClient(path=self.chroma_path)
            
            # Create embedding function
            embedder = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.api_key_embedding or self.api_key,
                model_name=self.embedding_model
            )
            
            # Get or create collections
            self.kb_collection = self.client.get_or_create_collection(
                name="travel_kb",
                embedding_function=embedder,
                metadata={"hnsw:space": "cosine"}
            )
            
            self.cache_collection = self.client.get_or_create_collection(
                name="answer_cache",
                embedding_function=embedder,
                metadata={"hnsw:space": "cosine"}
            )
            
            self.conv_collection = self.client.get_or_create_collection(
                name="conversations",
                embedding_function=embedder,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Warning: Could not initialize ChromaDB: {e}")
            self.client = None
            self.kb_collection = None
            self.cache_collection = None
            self.conv_collection = None
    
    def is_available(self) -> bool:
        """Check if ChromaDB is available."""
        return CHROMA_AVAILABLE and self.client is not None
    
    def seed_kb_from_csvs(self, foods_csv: str, restaurants_csv: str) -> int:
        """
        Seed knowledge base from CSV files.
        
        Args:
            foods_csv: Path to foods CSV file
            restaurants_csv: Path to restaurants CSV file
            
        Returns:
            Number of documents added
        """
        if not self.kb_collection:
            return 0
        
        docs, metas, ids = [], [], []
        
        # Add foods
        try:
            df_food = pd.read_csv(foods_csv, dtype=str)
            for _, r in df_food.iterrows():
                city = (r.get("city") or "").strip()
                foods_cell = (r.get("foods") or "").strip()
                if not city or not foods_cell:
                    continue
                
                foods = [f.strip() for f in re.split(r"[|,;]", foods_cell) if f.strip()]
                if not foods:
                    continue
                
                text = f"Đặc sản ở {city}: " + ", ".join(foods)
                docs.append(text)
                metas.append({"city": city, "type": "food", "source": "csv"})
                ids.append(f"food::{city}::{len(ids)}")
        except Exception:
            pass
        
        # Add restaurants
        try:
            df_res = pd.read_csv(restaurants_csv, dtype=str)
            for _, r in df_res.iterrows():
                city = (r.get("city") or "").strip()
                name = (r.get("place_name") or r.get("name") or "").strip()
                addr = (r.get("address") or "").strip()
                if not city or not name:
                    continue
                
                text = f"Nhà hàng gợi ý ở {city}: {name} - {addr}"
                docs.append(text)
                metas.append({"city": city, "type": "restaurant", "source": "csv"})
                ids.append(f"rest::{city}::{len(ids)}")
        except Exception:
            pass
        
        if docs:
            self.kb_collection.add(documents=docs, metadatas=metas, ids=ids)
        
        return len(docs)
    
    def retrieve_context(self, query: str, city: Optional[str] = None, k: int = 6) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from knowledge base.
        
        Args:
            query: Search query
            city: Optional city filter
            k: Number of results to retrieve
            
        Returns:
            List of retrieved items with metadata
        """
        if not self.kb_collection:
            return []
        
        try:
            where = {"city": city} if city else None
            res = self.kb_collection.query(
                query_texts=[query],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances", "ids"]
            )
            
            if not res or not res.get("documents"):
                return []
            
            items = []
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            dists = res["distances"][0]
            
            for doc, meta, dist in zip(docs, metas, dists):
                items.append({
                    "doc": doc,
                    "meta": meta,
                    "dist": float(dist)
                })
            
            items.sort(key=lambda x: x["dist"])
            return items
        except Exception:
            return []
    
    def hit_answer_cache(self, query: str, city: Optional[str] = None, threshold: float = 0.09) -> Optional[str]:
        """
        Check if a similar query has been cached.
        
        Args:
            query: Search query
            city: Optional city filter
            threshold: Similarity threshold
            
        Returns:
            Cached answer or None
        """
        if not self.cache_collection:
            return None
        
        try:
            where = {"city": city} if city else None
            res = self.cache_collection.query(
                query_texts=[query],
                n_results=1,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            if not res or not res.get("documents") or not res["documents"][0]:
                return None
            
            dist = float(res["distances"][0][0])
            if dist <= threshold:
                return res["documents"][0][0]
            
            return None
        except Exception:
            return None
    
    def push_answer_cache(self, query: str, city: Optional[str], answer: str):
        """
        Cache an answer for future use.
        
        Args:
            query: Original query
            city: City name
            answer: Answer to cache
        """
        if not self.cache_collection:
            return
        
        try:
            self.cache_collection.add(
                documents=[answer],
                metadatas=[{"city": city or ""}],
                ids=[f"cache::{city or 'na'}::{int(time.time()*1000)}"]
            )
        except Exception:
            pass
    
    def save_conversation(self, user_input: str, assistant_text: str, city: Optional[str]):
        """
        Save a conversation to memory.
        
        Args:
            user_input: User's input
            assistant_text: Assistant's response
            city: City name
        """
        if not self.conv_collection:
            return
        
        try:
            self.conv_collection.add(
                documents=[f"Q: {user_input}\nA: {assistant_text}"],
                metadatas=[{"city": city or "", "timestamp": datetime.utcnow().isoformat()}],
                ids=[f"conv::{int(time.time()*1000)}"]
            )
        except Exception:
            pass

