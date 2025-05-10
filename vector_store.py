"""
Vector store implementation for the RAG-Powered Multi-Agent Q&A Assistant.

This module handles the embedding and retrieval of document chunks.
"""

import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from models import TextChunk

# Constants
EMBEDDING_DIMENSION = 384  # SentenceTransformer default dimension
TOP_K_RESULTS = 3  # Number of chunks to retrieve


class VectorStore:
    """Manages vector embeddings and retrieval."""
    
    def __init__(self, api_key: str = None):
        self.index = None
        self.chunks = []
        # Use SentenceTransformer for embeddings (runs locally)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_chunks(self, chunks: List[TextChunk]):
        """Add chunks to the vector store."""
        if not chunks:
            return
        
        # Store the chunks
        start_idx = len(self.chunks)
        self.chunks.extend(chunks)
        
        # Create embeddings for the chunks
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        # Initialize or update the FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
            
        self.index.add(np.array(embeddings).astype('float32'))
    
    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[TextChunk]:
        """Search for chunks most relevant to the query."""
        if self.index is None or not self.chunks:
            return []
        
        # Get the query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search in the FAISS index
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), 
            min(top_k, len(self.chunks))
        )
        
        # Return the corresponding chunks
        return [self.chunks[i] for i in indices[0]]