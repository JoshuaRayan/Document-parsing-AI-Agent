"""
LLM client for the RAG-Powered Multi-Agent Q&A Assistant.

This module handles communication with the local Llama 3 API running on Ollama.
"""

import requests

# Constants
LLAMA_API_URL = "http://localhost:11434/api/generate"  # Ollama API endpoint for Llama3


class LlamaClient:
    """Client for the local Llama 3 API running on Ollama."""
    
    def __init__(self, api_url: str = LLAMA_API_URL, model_name: str = "llama3"):
        self.api_url = api_url
        self.model_name = model_name
    
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Generate text using the Llama 3 API via Ollama."""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=60  # Local model might be slower
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                print(f"Error from Ollama API: {response.status_code}")
                print(response.text)
                return f"Error generating response: {response.status_code}"
        except Exception as e:
            print(f"Exception when calling Ollama API: {e}")
            return f"Error generating response: {str(e)}"