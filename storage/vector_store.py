"""
VectorStore - Abstraction over ChromaDB for movie document storage.

Handles document storage and similarity search for the movie RAG system.
"""

import os
from typing import Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    """
    Vector database wrapper for movie document storage and retrieval.
    
    Uses ChromaDB with its default embedding function.
    """
    
    COLLECTION_NAME = "movies"
    
    def __init__(self, persist_dir: Optional[str] = None):
        """
        Initialize the VectorStore.
        
        Args:
            persist_dir: Directory for ChromaDB persistence.
        """
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Movie documents for RAG retrieval"}
        )
    
    def add_movie(self, imdb_id: str, document: str, metadata: dict) -> bool:
        """
        Add a movie document to the vector store.
        
        Returns:
            True if added, False if already exists.
        """
        existing = self.collection.get(ids=[imdb_id])
        if existing["ids"]:
            return False
        
        self.collection.add(
            ids=[imdb_id],
            documents=[document],
            metadatas=[metadata]
        )
        return True
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search for similar movies based on query.
        
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
    
    def get_movie(self, imdb_id: str) -> Optional[dict]:
        """Get a specific movie by ID."""
        result = self.collection.get(
            ids=[imdb_id],
            include=["documents", "metadatas"]
        )
        
        if not result["ids"]:
            return None
        
        return {
            "id": result["ids"][0],
            "document": result["documents"][0] if result["documents"] else "",
            "metadata": result["metadatas"][0] if result["metadatas"] else {},
        }
    
    def delete_movie(self, imdb_id: str) -> bool:
        """Delete a movie from the vector store."""
        existing = self.collection.get(ids=[imdb_id])
        if not existing["ids"]:
            return False
        
        self.collection.delete(ids=[imdb_id])
        return True
    
    def get_all_movies(self) -> list[dict]:
        """Get all movies in the collection."""
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
        """Return total number of movies."""
        return self.collection.count()
    
    def clear(self) -> None:
        """Delete all documents from the collection."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Movie documents for RAG retrieval"}
        )
