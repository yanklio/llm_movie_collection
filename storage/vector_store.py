"""
VectorStore - Abstraction over ChromaDB for entity document storage.

Handles document storage and similarity search for the RAG system.
Entities can be movies, books, or other media items.
"""

import os
from typing import Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    """
    Vector database wrapper for entity document storage and retrieval.
    
    Uses ChromaDB with its default embedding function.
    """
    
    COLLECTION_NAME = "entities"
    
    def __init__(self, persist_dir: Optional[str] = None):
        """Initialize the VectorStore."""
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Entity documents for RAG retrieval"}
        )
    
    def add(self, entity_id: str, document: str, metadata: dict, entity_type: str = "movie") -> bool:
        """
        Add an entity document to the vector store.
        
        Args:
            entity_id: Unique identifier for the entity.
            document: Text content to index.
            metadata: Additional metadata.
            entity_type: Type of entity (e.g., "movie", "review").
        
        Returns:
            True if added, False if already exists.
        """
        existing = self.collection.get(ids=[entity_id])
        if existing["ids"]:
            return False
        
        metadata["entity_type"] = entity_type
        
        self.collection.add(
            ids=[entity_id],
            documents=[document],
            metadatas=[metadata]
        )
        return True
    
    def search(self, query: str, top_k: int = 5, entity_type: Optional[str] = None, filters: Optional[dict] = None) -> list[dict]:
        """
        Search for similar entities based on query.
        
        Args:
            query: Search query text.
            top_k: Number of results to return.
            entity_type: Optional filter by entity type.
            filters: Additional metadata filters (ChromaDB where clause).
        
        Returns:
            List of dicts with 'id', 'document', 'metadata', 'distance'.
        """
        conditions = []
        if entity_type:
            conditions.append({"entity_type": entity_type})
        
        if filters:
            for k, v in filters.items():
                if v is None: continue
                
                # Handle simplified year ranges from agents
                if k == "year_min":
                    conditions.append({"year": {"$gte": int(v)}})
                elif k == "year_max":
                    conditions.append({"year": {"$lte": int(v)}})
                elif k == "rating" and isinstance(v, (int, float)):
                    # Handle rating threshold
                    conditions.append({"rating": {"$gte": v}})
                elif k == "year":
                    # Try to cast year to int for exact match
                    try:
                        conditions.append({"year": int(str(v))})
                    except:
                        conditions.append({"year": v})
                # Check if value is a dictionary with multiple operators (e.g. range passed explicitly)
                elif isinstance(v, dict) and len(v) > 1:
                    # ChromaDB req: split {'$gte': 2000, '$lte': 2010} into multiple dicts
                    for op, val in v.items():
                        conditions.append({k: {op: val}})
                else:
                    conditions.append({k: v})
        
        if not conditions:
            where_filter = None
        elif len(conditions) == 1:
            where_filter = conditions[0]
        else:
            where_filter = {"$and": conditions}
            
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        items = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                items.append({
                    "id": doc_id,
                    "document": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                })
        
        return items
    
    def get(self, entity_id: str) -> Optional[dict]:
        """Get a specific entity by ID."""
        result = self.collection.get(
            ids=[entity_id],
            include=["documents", "metadatas"]
        )
        
        if not result["ids"]:
            return None
        
        return {
            "id": result["ids"][0],
            "document": result["documents"][0] if result["documents"] else "",
            "metadata": result["metadatas"][0] if result["metadatas"] else {},
        }
    
    def delete(self, entity_id: str) -> bool:
        """Delete an entity from the vector store."""
        existing = self.collection.get(ids=[entity_id])
        if not existing["ids"]:
            return False
        
        self.collection.delete(ids=[entity_id])
        return True
    
    def get_all(self, entity_type: Optional[str] = None) -> list[dict]:
        """
        Get all entities in the collection.
        
        Args:
            entity_type: Optional filter by entity type.
        """
        where_filter = {"entity_type": entity_type} if entity_type else None
        
        result = self.collection.get(
            where=where_filter,
            include=["documents", "metadatas"]
        )
        
        items = []
        for i, doc_id in enumerate(result["ids"]):
            items.append({
                "id": doc_id,
                "document": result["documents"][i] if result["documents"] else "",
                "metadata": result["metadatas"][i] if result["metadatas"] else {},
            })
        
        return items
    
    def count(self) -> int:
        """Return total number of entities."""
        return self.collection.count()
    
    def clear(self) -> None:
        """Delete all documents from the collection."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Entity documents for RAG retrieval"}
        )
