"""
Main entry point for the RAG-Powered Multi-Agent Q&A Assistant.

This script initializes the Streamlit UI and orchestrates the various components.
"""

import os
import streamlit as st
from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_client import LlamaClient
from agent import Agent

class DocumentCollection:
    """Manages a collection of documents for the knowledge base."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def load_documents(self):
        """Load all documents from the data directory."""
        loaded_docs = []
        all_chunks = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith(('.txt', '.md', '.pdf')):
                file_path = os.path.join(self.data_dir, filename)
                
                # For simplicity, we're only handling text files here
                # For PDFs, you would need to use a PDF parser
                if filename.endswith('.pdf'):
                    print(f"PDF support requires additional libraries. Skipping {filename}")
                    continue
                
                try:
                    # Load and chunk the document
                    doc = self.doc_processor.load_document(file_path)
                    chunks = self.doc_processor.chunk_document(doc)
                    
                    loaded_docs.append(doc)
                    all_chunks.extend(chunks)
                    
                    print(f"Processed {filename}: {len(chunks)} chunks created")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        # Add chunks to the vector store
        if all_chunks:
            self.vector_store.add_chunks(all_chunks)
            print(f"Added {len(all_chunks)} chunks to the vector store")
        else:
            print("No chunks were added to the vector store")
        
        return loaded_docs


def create_ui():
    """Create a Streamlit UI for the RAG assistant."""
    # Custom CSS for better chat interface
    st.markdown("""
    <style>
        .stTextInput > div > div > input {
            position: fixed;
            bottom: 3rem;
            background-color: white;
            z-index: 100;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            background-color: #2b313e;
            color: white;
        }
        .chat-message.assistant {
            background-color: #f0f2f6;
        }
        .chat-message .content {
            display: flex;
            flex-direction: column;
        }
        .chat-message .metadata {
            font-size: 0.8rem;
            color: #666;
            margin-top: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("RAG-Powered Multi-Agent Q&A Assistant")
    
    # Initialize session state for messages if not exists
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Initialize the document collection and agent components
    if 'doc_collection' not in st.session_state:
        st.session_state.doc_collection = DocumentCollection()
        try:
            st.session_state.doc_collection.load_documents()
        except Exception as e:
            st.error(f"Error loading documents: {e}")
    
    if 'agent' not in st.session_state:
        # Initialize LlamaClient with the Ollama API URL and model name
        llm_client = LlamaClient(model_name="llama3")
        st.session_state.agent = Agent(
            st.session_state.doc_collection.vector_store,
            llm_client
        )
    
    # Document uploader in sidebar
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_file = st.file_uploader("Upload a text document", type=["txt", "md"])
        
        if uploaded_file is not None:
            try:
                # Save the uploaded file
                file_path = os.path.join(st.session_state.doc_collection.data_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Reload documents
                st.session_state.doc_collection.load_documents()
                st.success(f"Uploaded and processed: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing upload: {e}")
    
    # Create a container for chat messages
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display context if available
                if "context" in message:
                    with st.expander("View Retrieved Context"):
                        for i, ctx in enumerate(message["context"]):
                            st.markdown(f"**Source:** {ctx['source']}")
                            st.text(ctx['content'])
                            if i < len(message["context"]) - 1:
                                st.markdown("---")
                
                # Display agent path if available
                if "agent_path" in message:
                    st.caption(f"Agent Path: {message['agent_path'].upper()}")
                
                # Display processing time if available
                if "processing_time" in message:
                    st.caption(f"Processing time: {message['processing_time']:.2f} seconds")
    
    # Query input at the bottom
    query = st.chat_input("Ask a question about your documents...")
    
    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Process query
                    result = st.session_state.agent.process_query(query)
                    
                    # Display the response
                    st.markdown(result["answer"])
                    
                    # Show retrieved chunks if using RAG
                    if result.get("agent_path") == "rag" and result.get("retrieved_chunks"):
                        with st.expander("View Retrieved Context"):
                            for i, chunk in enumerate(result["retrieved_chunks"]):
                                st.markdown(f"**Chunk {i+1}** (from {chunk['source']})")
                                st.text(chunk['content'])
                                if i < len(result["retrieved_chunks"]) - 1:
                                    st.markdown("---")
                    
                    # Show agent path
                    if "agent_path" in result:
                        st.caption(f"Agent Path: {result['agent_path'].upper()}")
                    
                    # Show processing time
                    if "processing_time" in result:
                        st.caption(f"Processing time: {result['processing_time']:.2f} seconds")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "context": result.get("retrieved_chunks", []),
                        "agent_path": result.get("agent_path"),
                        "processing_time": result.get("processing_time")
                    })
                    
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})


def main():
    """Main function to run the application."""
    create_ui()


if __name__ == "__main__":
    main()