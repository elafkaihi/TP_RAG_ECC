"""
Configuration for the RAG system.
"""

CONFIG = {
    "paths": {
        "data_dir": "data",
        "vector_store_dir": "vectorstore"
    },
    "document_processing": {
        "chunk_size": 500,  # Adjusted chunk size
        "chunk_overlap": 100, # Adjusted chunk overlap
        "markdown_headings_splitter": True
    },
    "embedding": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu"  # Or "cuda" if you have a GPU
    },
    "vector_store": {
        "type": "chroma",
        "similarity_metric": "cosine"
    },
    "retrieval": {
        "top_k": 5, #increased the k
        "score_threshold": 0.6 # reduced threshold
    },
    "llm": {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",  #  Could be a HuggingFace Hub repo ID.
    "max_tokens": 2000,
    "temperature": 0.7,  
    "context_window": 4096
    },
    "prompts": {
        "qa_template": """You are a helpful AI assistant that answers questions based on the provided context.  
        If the context does not contain the answer, truthfully say "I don't know".

Context:
{context}

Question: {question}

Answer:"""
    },
    "evaluation": {
        "metrics": [
            "faithfulness",
            "relevance",
            "coherence"
        ]
    }
}
