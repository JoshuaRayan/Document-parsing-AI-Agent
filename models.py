"""
Data models for the RAG-Powered Multi-Agent Q&A Assistant.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Document:
    """Represents a document with its content and metadata."""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = None


@dataclass
class TextChunk:
    """Represents a chunk of text from a document."""
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any] = None