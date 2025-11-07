"""
ChromaDB service for RAG (Retrieval-Augmented Generation).
Handles vector database operations, embeddings, and knowledge retrieval.
"""

import os
import uuid
import time
from typing import Optional, List, Dict, Any
from datetime import datetime
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import streamlit as st

from config.settings import Settings


class ChromaService:
    """Service for ChromaDB operations including RAG, memory, and intent detection."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client: Optional[PersistentClient] = None
        self.travel_col = None
        self.memory_col = None
        self.intent_col = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and collections."""
        try:
            os.makedirs(self.settings.CHROMA_PERSIST_DIR, exist_ok=True)
            
            # Initialize ChromaDB client
            if "chroma_client" not in st.session_state or st.session_state.get("chroma_client") is None:
                st.session_state["chroma_client"] = PersistentClient(path=self.settings.CHROMA_PERSIST_DIR)
            self.client = st.session_state["chroma_client"]
            
            # Load embedding model
            self.embedding_model = self._load_embedding_model()
            
            # Initialize collections
            self.travel_col = self._safe_get_collection("vietnam_travel_v2", self.settings.EMBEDDING_DIMENSION)
            self.memory_col = self._safe_get_collection("chat_memory_v2", self.settings.EMBEDDING_DIMENSION)
            self.intent_col = self._safe_get_collection("intent_bank_v2", self.settings.EMBEDDING_DIMENSION)
            
            # Preload intents
            self._preload_intents()
            
        except Exception as e:
            print(f"[WARN] ChromaDB initialization failed: {e}")
    
    @st.cache_resource
    def _load_embedding_model(_self):
        """Load local embedding model."""
        try:
            model_path = str(_self.settings.DATA_DIR / "all-MiniLM-L6-v2")
            if os.path.exists(model_path) and os.path.isdir(model_path):
                if any(file.endswith('model.safetensors') for file in os.listdir(model_path)):
                    return SentenceTransformer(model_path)
            
            print("📥 Loading embedding model from Hugging Face...")
            model = SentenceTransformer(_self.settings.EMBEDDING_MODEL_NAME)
            os.makedirs(model_path, exist_ok=True)
            model.save(model_path)
            return model
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            return None
    
    def _safe_get_collection(self, name: str, expected_dim: int):
        """Safely get or create a ChromaDB collection."""
        try:
            col = None
            try:
                col = self.client.get_collection(name)
            except Exception:
                try:
                    col = self.client.get_or_create_collection(name=name)
                except Exception:
                    col = self.client.create_collection(name=name)
            
            # Test dimension compatibility
            try:
                test_emb = [0.0] * expected_dim
                col.query(query_embeddings=[test_emb], n_results=1)
            except Exception as qe:
                if "dimension" in str(qe).lower():
                    print(f"🧹 Deleting collection {name} due to dimension mismatch")
                    self.client.delete_collection(name=name)
                    col = self.client.create_collection(name=name)
            
            return col
        except Exception as e:
            print(f"[WARN] Failed to get collection {name}: {e}")
            return None
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text."""
        if not self.embedding_model or not text:
            return None
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            print(f"[WARN] Embedding generation failed: {e}")
            return None
    
    def rag_query(self, user_text: str, k: int = 5) -> tuple[List[Dict], str]:
        """
        Query RAG knowledge base.
        
        Returns:
            Tuple of (documents list, context string)
        """
        if not self.travel_col:
            return [], ""
        
        emb = self.get_embedding(user_text)
        if not emb:
            return [], ""
        
        try:
            res = self.travel_col.query(
                query_embeddings=[emb],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            docs = []
            docs_texts = res.get("documents", [[]])
            if docs_texts and isinstance(docs_texts, list):
                docs_texts = docs_texts[0] if docs_texts and isinstance(docs_texts[0], list) else docs_texts
            
            metadatas = res.get("metadatas", [[]])
            if metadatas and isinstance(metadatas, list):
                metadatas = metadatas[0] if metadatas and isinstance(metadatas[0], list) else metadatas
            
            ids = res.get("ids", [[]])
            if ids and isinstance(ids, list):
                ids = ids[0] if ids and isinstance(ids[0], list) else ids
            
            distances = res.get("distances", [[]])
            if distances and isinstance(distances, list):
                distances = distances[0] if distances and isinstance(distances[0], list) else distances
            
            docs_texts = docs_texts or []
            metadatas = metadatas or [{}] * len(docs_texts)
            ids = ids or [f"doc_{i}" for i in range(len(docs_texts))]
            distances = distances or [None] * len(docs_texts)
            
            for i, txt in enumerate(docs_texts):
                meta = metadatas[i] if i < len(metadatas) else {}
                doc_id = ids[i] if i < len(ids) else f"doc_{i}"
                distance = distances[i] if i < len(distances) else None
                
                docs.append({
                    "id": doc_id,
                    "text": txt,
                    "metadata": meta,
                    "distance": distance
                })
            
            context_parts = []
            for d in docs:
                src = d["metadata"].get("source", "") if isinstance(d.get("metadata"), dict) else ""
                context_parts.append(f"[src:{d['id']}{('|' + src) if src else ''}] {d['text'][:1200]}")
            
            context = "\n\n".join(context_parts)
            return docs, context
            
        except Exception as e:
            print(f"[WARN] RAG query error: {e}")
            return [], ""
    
    def add_to_memory(self, text: str, role: str = "user", city: Optional[str] = None, 
                     extra_meta: Optional[Dict] = None):
        """Add conversation to memory collection."""
        if not self.memory_col:
            return
        
        emb = self.get_embedding(text)
        if not emb:
            return
        
        try:
            doc_id = f"mem_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
            meta = {"role": role, "city": city or "", "timestamp": datetime.utcnow().isoformat()}
            if extra_meta:
                meta.update(extra_meta)
            
            try:
                self.memory_col.add(documents=[text], metadatas=[meta], ids=[doc_id], embeddings=[emb])
            except TypeError:
                self.memory_col.add(documents=[text], metadatas=[meta], ids=[doc_id])
        except Exception as e:
            print(f"[WARN] Failed to add to memory: {e}")
    
    def recall_memories(self, user_text: str, k: int = 3) -> List[Dict]:
        """Recall recent memories similar to user text."""
        if not self.memory_col:
            return []
        
        emb = self.get_embedding(user_text)
        if not emb:
            return []
        
        try:
            res = self.memory_col.query(
                query_embeddings=[emb],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            items = []
            docs_texts = res.get("documents", [[]])
            if docs_texts and isinstance(docs_texts, list):
                docs_texts = docs_texts[0] if docs_texts and isinstance(docs_texts[0], list) else docs_texts
            
            metadatas = res.get("metadatas", [[]])
            if metadatas and isinstance(metadatas, list):
                metadatas = metadatas[0] if metadatas and isinstance(metadatas[0], list) else metadatas
            
            ids = res.get("ids", [[]])
            if ids and isinstance(ids, list):
                ids = ids[0] if ids and isinstance(ids[0], list) else ids
            
            distances = res.get("distances", [[]])
            if distances and isinstance(distances, list):
                distances = distances[0] if distances and isinstance(distances[0], list) else distances
            
            docs_texts = docs_texts or []
            metadatas = metadatas or [{}] * len(docs_texts)
            ids = ids or [f"mem_{i}" for i in range(len(docs_texts))]
            distances = distances or [None] * len(docs_texts)
            
            for i, t in enumerate(docs_texts):
                meta = metadatas[i] if i < len(metadatas) else {}
                item_id = ids[i] if i < len(ids) else f"mem_{i}"
                distance = distances[i] if i < len(distances) else None
                
                items.append({
                    "id": item_id,
                    "text": t,
                    "meta": meta,
                    "distance": distance
                })
            
            return items
        except Exception as e:
            print(f"[WARN] Memory recall error: {e}")
            return []
    
    def get_intent(self, user_text: str, threshold: float = 0.18) -> Optional[str]:
        """Detect intent from user text."""
        if not self.intent_col:
            return None
        
        emb = self.get_embedding(user_text)
        if not emb:
            return None
        
        try:
            res = self.intent_col.query(
                query_embeddings=[emb],
                n_results=1,
                include=["metadatas", "distances"]
            )
            
            distances = res.get("distances", [[]])
            if distances and isinstance(distances, list):
                distances = distances[0] if distances and isinstance(distances[0], list) else distances
            
            metadatas = res.get("metadatas", [[]])
            if metadatas and isinstance(metadatas, list):
                metadatas = metadatas[0] if metadatas and isinstance(metadatas[0], list) else metadatas
            
            if distances and len(distances) > 0 and distances[0] is not None and distances[0] < threshold:
                meta = metadatas[0] if metadatas and len(metadatas) > 0 else {}
                return meta.get("intent") if isinstance(meta, dict) else None
        except Exception as e:
            print(f"[WARN] Intent detection error: {e}")
        
        return None
    
    def recommend_similar_trips(self, city: str, k: int = 3) -> List[Dict]:
        """Find similar trips based on city from memory."""
        if not self.memory_col:
            return []
        
        emb = self.get_embedding(city)
        if not emb:
            return []
        
        try:
            res = self.memory_col.query(
                query_embeddings=[emb],
                n_results=10,
                include=["documents", "metadatas", "distances", "ids"]
            )
            
            docs_texts = res.get("documents", [[]])
            if docs_texts and isinstance(docs_texts, list):
                docs_texts = docs_texts[0] if docs_texts and isinstance(docs_texts[0], list) else docs_texts
            
            metadatas = res.get("metadatas", [[]])
            if metadatas and isinstance(metadatas, list):
                metadatas = metadatas[0] if metadatas and isinstance(metadatas[0], list) else metadatas
            
            ids = res.get("ids", [[]])
            if ids and isinstance(ids, list):
                ids = ids[0] if ids and isinstance(ids[0], list) else ids
            
            recommendations = []
            seen_cities = set()
            
            for i, meta in enumerate(metadatas):
                if i >= len(docs_texts):
                    break
                rec_city = meta.get("city") if isinstance(meta, dict) else None
                if rec_city and rec_city.lower() != city.lower() and rec_city not in seen_cities:
                    doc = docs_texts[i] if i < len(docs_texts) else ""
                    rec_id = ids[i] if i < len(ids) else None
                    recommendations.append({
                        "city": rec_city,
                        "meta": meta,
                        "doc": doc,
                        "id": rec_id
                    })
                    seen_cities.add(rec_city)
                if len(recommendations) >= k:
                    break
            
            return recommendations
        except Exception as e:
            print(f"[WARN] Recommend similar trips error: {e}")
            return []
    
    def _preload_intents(self):
        """Preload common intents into intent collection."""
        if not self.intent_col:
            return
        
        try:
            samples = [
                ("Thời tiết ở {city} tuần tới?", {"intent": "weather_query"}),
                ("Lịch trình 3 ngày ở {city}", {"intent": "itinerary_request"}),
                ("Đặc sản {city}", {"intent": "food_query"}),
                ("Gợi ý nhà hàng ở {city}", {"intent": "restaurant_query"})
            ]
            
            docs = []
            metas = []
            ids = []
            
            for i, (t, meta) in enumerate(samples):
                docs.append(t)
                metas.append(meta)
                ids.append(f"intent_sample_{i}")
            
            self.intent_col.add(documents=docs, metadatas=metas, ids=ids)
        except Exception as e:
            print(f"[WARN] Failed to preload intents: {e}")
    
    def seed_from_csv(self, csv_path: str) -> bool:
        """Seed travel knowledge base from CSV file."""
        if not self.travel_col or not os.path.exists(csv_path):
            return False
        
        try:
            import csv
            docs = []
            metas = []
            ids = []
            
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("text") or row.get("description") or ""
                    if not text:
                        continue
                    docs.append(text)
                    metas.append({
                        "title": row.get("title", ""),
                        "city": row.get("city", ""),
                        "source": row.get("source", "")
                    })
                    ids.append(row.get("id") or f"doc_{uuid.uuid4().hex[:8]}")
            
            if docs:
                self.travel_col.add(documents=docs, metadatas=metas, ids=ids)
                print(f"✅ Seeded {len(docs)} documents to travel knowledge base")
                return True
        except Exception as e:
            print(f"[WARN] Seed error: {e}")
        
        return False

