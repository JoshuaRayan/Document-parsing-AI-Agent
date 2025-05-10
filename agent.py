"""
Agent implementation for the RAG-Powered Multi-Agent Q&A Assistant.

This module orchestrates the workflow between different components.
"""

import re
import time
from typing import Dict, Any, List
from vector_store import VectorStore
from llm_client import LlamaClient
from tools import Tools


class Agent:
    """Orchestrates the workflow between different components."""
    
    def __init__(self, vector_store: VectorStore, llm_client: LlamaClient):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.tools = Tools()
        self.last_response = None
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return results."""
        start_time = time.time()
        result = {
            "query": query,
            "agent_path": None,
            "retrieved_chunks": [],
            "answer": None,
            "processing_time": None
        }
        
        # Determine which agent path to take
        agent_path = self._determine_agent_path(query)
        result["agent_path"] = agent_path
        
        if agent_path == "calculate":
            # Extract the expression from the query
            expression = self._extract_calculation_expression(query)
            result["answer"] = self.tools.calculate(expression)
        
        elif agent_path == "define":
            # Extract the term to define
            term = self._extract_term_to_define(query)
            result["answer"] = self.tools.define(term)
        
        else:  # Default to RAG
            # Retrieve relevant chunks
            chunks = self.vector_store.search(query)
            result["retrieved_chunks"] = [
                {"id": chunk.chunk_id, "content": chunk.content, "source": chunk.metadata.get("source", "unknown")}
                for chunk in chunks
            ]
            
            # Generate answer using RAG approach
            context = "\n\n".join([chunk.content for chunk in chunks])
            answer = self._generate_rag_answer(query, context)
            result["answer"] = answer
        
        result["processing_time"] = time.time() - start_time
        self.last_response = result
        return result
    
    def _determine_agent_path(self, query: str) -> str:
        """Determine which tool or approach to use based on the query."""
        query_lower = query.lower()
        
        # Check for calculation queries
        calc_keywords = ["calculate", "compute", "solve", "what is", "evaluate"]
        has_calc_pattern = any(kw in query_lower for kw in calc_keywords) and re.search(r'[0-9+\-*/()^]', query)
        
        # Check for definition queries
        define_keywords = ["define", "what is", "meaning of", "definition of","what is the meaning of","describe","meaning+of","what+is+the+meaning+of","what+is+the+definition+of","what+is+the+description+of"]
        has_define_pattern = any(kw in query_lower for kw in define_keywords)
        
        if has_calc_pattern and not has_define_pattern:
            return "calculate"
        elif has_define_pattern and not has_calc_pattern:
            return "define"
        else:
            return "rag"
    
    def _extract_calculation_expression(self, query: str) -> str:
        """Extract the mathematical expression from a query."""
        # Try to find expressions containing numbers and operators
        matches = re.findall(r'([0-9+\-*/().\^ ]+)', query)
        if matches:
            # Take the longest match as it's likely the full expression
            expression = max(matches, key=len).strip()
            return expression
        
        # If no clear expression is found, try to clean up the query
        cleaned = re.sub(r'(calculate|compute|solve|what is|evaluate)', '', query, flags=re.IGNORECASE).strip()
        return cleaned
    
    def _extract_term_to_define(self, query: str) -> str:
        """Extract the term to be defined from a query."""
        # Remove definition keywords
        cleaned = re.sub(r'(define|what is|meaning of|definition of)', '', query, flags=re.IGNORECASE).strip()
        
        # Remove punctuation
        cleaned = re.sub(r'[?.,!]', '', cleaned).strip()
        
        return cleaned
    
    def _generate_rag_answer(self, query: str, context: str) -> str:
        """Generate an answer based on the query and retrieved context."""
        prompt = f"""You are a helpful AI assistant that provides accurate and detailed answers based on the given context.
Your task is to answer the user's question using ONLY the information provided in the context.
If the context doesn't contain enough information to answer the question completely, acknowledge this and provide a partial answer based on what is available.
Do not make up or infer information that is not present in the context.

Context:
{context}

Question: {query}

Instructions:
1. Base your answer ONLY on the provided context
2. If the context is insufficient, say so and provide what information you can
3. Be specific and detailed in your response
4. If relevant, cite which parts of the context you're using
5. Maintain a helpful and professional tone

Answer:"""
        return self.llm_client.generate(prompt, max_tokens=1024, temperature=0.7)