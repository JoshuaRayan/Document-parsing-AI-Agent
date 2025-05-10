# RAG-Powered Multi-Agent Q&A Assistant

This project implements a knowledge assistant that combines Retrieval-Augmented Generation (RAG) with a multi-agent framework to answer questions based on document knowledge or specialized tools.
I have used an open-sourced LLM models as I often hit the token limiters with the various other API services like gemini and open AI.
So do install ollama and pull the llama3 model using your powershell and run this code on streamlit. It is also the reason I could not deploy as it uses a local hosted LLM.
## Architecture Overview

The system consists of the following components:

### 1. Document Processing & Vector Store
- **Document Loader**: Reads text documents from a designated directory
- **Document Chunker**: Splits documents into smaller, semantic chunks with configurable overlap
- **Vector Store**: Uses FAISS for efficient similarity search with locally hosted embeddings

### 2. LLM Integration
- Uses a locally hosted **Llama 3** model for text generation
- Communicates with the model via a REST API on localhost:8000

### 3. Multi-Agent Framework
- **Agent Router**: Analyzes queries to determine the appropriate processing path
- **Tool Agents**:
  - **Calculator**: Handles mathematical expressions
  - **Dictionary**: Looks up term definitions using a free dictionary API
  - **RAG Agent**: Retrieves relevant document chunks and generates answers

### 4. User Interface
- Built with **Streamlit** for easy interaction
- Displays each step of the processing pipeline
- Provides document upload functionality

## Key Design Choices

### Local-First Approach
- **Local Embeddings**: Uses SentenceTransformer for document embeddings, avoiding external API calls
- **Local LLM**: Uses Llama 3 hosted locally, to bypass the free tier of other APIs, which improved latency and saved a lot of money.

### Efficient Chunking Strategy
- **Sentence-Based Chunking**: Preserves semantic meaning by chunking at sentence boundaries

## Project Structure

```
rag-agent/
├── main.py                # Main entry point and Streamlit UI
├── models.py              # Data models for documents and chunks
├── document_processor.py  # Document loading and chunking
├── vector_store.py        # Vector embeddings and retrieval
├── llm_client.py          # Interface to local Llama3 API
├── tools.py               # Implementation of calculator and dictionary tools
├── agent.py               # Agent orchestration and routing logic
├── requirements.txt       # Dependencies
└── data/                  # Directory for document storage
```

## Setup and Installation

1. Clone this repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Make sure you have Llama 3 running locally on port 8000
4. Run the application:
   ```
   streamlit run main.py
   ```

## Usage

1. Upload text documents through the UI
2. Enter questions in the input field
3. View the agent path taken, retrieved chunks, and final answer

## Running Llama 3 Locally with Ollama

This project is configured to use Llama 3 running on Ollama at `http://localhost:11434/api/generate`. To set up Ollama:

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the Llama 3 model: 
   ollama pull llama3
   ollama run llama3
3. Ollama will automatically start the API server at the required endpoint
