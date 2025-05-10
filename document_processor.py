"""
Document processing utilities for the RAG-Powered Multi-Agent Q&A Assistant.

This module handles loading and chunking of documents.
"""

import nltk
from nltk.tokenize import sent_tokenize
from typing import List
from models import Document, TextChunk

# Download necessary nltk data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class DocumentProcessor:
    """Handles document loading and chunking."""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_document(self, file_path: str) -> Document:
        """Load a document from a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        doc_id = file_path.split('/')[-1]  # Use filename as document ID
        return Document(doc_id=doc_id, content=content)
    
    def chunk_document(self, doc: Document) -> List[TextChunk]:
        """Split a document into overlapping chunks."""
        chunks = []
        sentences = sent_tokenize(doc.content)
        
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed chunk size, save the current chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(TextChunk(
                    chunk_id=f"{doc.doc_id}_{chunk_id}",
                    doc_id=doc.doc_id,
                    content=chunk_text,
                    metadata={"source": doc.doc_id}
                ))
                chunk_id += 1
                
                # Handle overlap: keep some sentences for the next chunk
                overlap_sentences = []
                overlap_size = 0
                
                # Work backwards through current_chunk to find overlap sentences
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s) + 1  # +1 for space
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(TextChunk(
                chunk_id=f"{doc.doc_id}_{chunk_id}",
                doc_id=doc.doc_id,
                content=chunk_text,
                metadata={"source": doc.doc_id}
            ))
        
        return chunks