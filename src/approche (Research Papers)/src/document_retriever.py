# document_retriever.py (minor changes for error handling)
"""
Document Retriever module for the RAG system.
Retrieves relevant documents from the vector store based on a query.
"""

from typing import List, Any, Dict, Tuple, Optional
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from src.utils import load_config  # Import load_config


class DocumentRetriever:
    """Class for retrieving relevant documents."""

    def __init__(self, vector_store: Any, config: Optional[Dict[str, Any]] = None):
        """Initializes the document retriever."""
        self.config = config or load_config()
        self.vector_store = vector_store
        self.top_k = self.config["retrieval"]["top_k"]
        self.score_threshold = self.config["retrieval"]["score_threshold"]
        try:
            self.retriever: VectorStoreRetriever = vector_store.as_retriever(search_kwargs={"k": self.top_k})
        except AttributeError as e: # Handles if vectorstore is None
            print(f"Error creating retriever: {e}. Ensure vector store is properly initialized.")
            self.retriever = None



    def retrieve(self, query: str) -> List[Document]:
        """Retrieves relevant documents based on the query."""
        if not query.strip():
            return []
        if self.retriever is None: #Check if retriever exists
            print("Retriever not initialized.")
            return []
        try:
            return self.retriever.invoke(query)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """Retrieves documents with relevance scores."""
        if not query.strip():
            return []
        if self.vector_store is None: #Check if vectorstore exists
            print("Vector store not initialized.")
            return []
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(query=query, k=self.top_k)
            return docs_with_scores
        except Exception as e:
            print(f"Error retrieving documents with scores: {e}")
            return []

    def get_formatted_context(self, documents: List[Document]) -> str:
        """Formats documents into a context string for the LLM prompt."""
        if not documents:
            return "No relevant documents found."

        formatted_docs = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            formatted_doc = f"[Document {i+1}] Source: {source}, Page: {page}\n{doc.page_content}\n"
            formatted_docs.append(formatted_doc)
        return "\n\n".join(formatted_docs)

    def get_formatted_results_with_scores(self, docs_with_scores: List[Tuple[Document, float]]) -> str:
        """Formats documents with scores into a readable string."""
        if not docs_with_scores:
            return "No relevant documents found."

        formatted_results = []
        for i, (doc, score) in enumerate(docs_with_scores):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            formatted_result = (
                f"[Document {i+1}] Relevance Score: {score:.4f}\n"
                f"Source: {source}, Page: {page}\n"
                f"{doc.page_content}\n"
            )
            formatted_results.append(formatted_result)
        return "\n\n".join(formatted_results)