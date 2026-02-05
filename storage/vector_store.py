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
    
    def add(self, entity_id: str, document: str, metadata: dict) -> bool:
        """
        Add an entity document to the vector store.
        
        Returns:
            True if added, False if already exists.
        """
        existing = self.collection.get(ids=[entity_id])
        if existing["ids"]:
            return False
        
        self.collection.add(
            ids=[entity_id],
            documents=[document],
            metadatas=[metadata]
        )
        return True
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search for similar entities based on query.
        
        Returns:
            List of dicts with 'id', 'document', 'metadata', 'distance'.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
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
    
    def get_all(self) -> list[dict]:
        """Get all entities in the collection."""
        result = self.collection.get(include=["documents", "metadatas"])
        
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
