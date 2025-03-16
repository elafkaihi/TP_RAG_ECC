# rag_system.py
"""
Main RAG System module.
Combines all components to provide a complete RAG pipeline.
"""

from typing import Optional, Dict, Any, Tuple, List
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.documents import Document

from src.document_indexer import DocumentIndexer
from src.document_retriever import DocumentRetriever
from src.llm_processor import LLMProcessor
from src.evaluator import RAGEvaluator
from src.utils import load_config


class ScientificPaperRAG:
    """Main class for the scientific paper RAG system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initializes the RAG system."""
        self.config = config or load_config()
        os.makedirs(self.config["paths"]["data_dir"], exist_ok=True)
        os.makedirs(self.config["paths"]["vector_store_dir"], exist_ok=True)

        self.indexer = DocumentIndexer(self.config)
        self._retriever = None  # Initialize to None
        self._evaluator = None
        self._vector_store = None

        if os.path.exists(self.config["paths"]["vector_store_dir"]):
            try:
                self._vector_store = self.indexer.load_vector_store()
                if self._vector_store:  #Check for NoneType
                    self._retriever = DocumentRetriever(self._vector_store, self.config)
            except Exception as e:
                print(f"Error loading existing vector store: {e}")
                print("Re-indexing required.")

        try:
            self.llm_processor = LLMProcessor(self.config)
        except Exception as e:
            print(f"Error initializing LLM processor: {e}")
            raise

    def index_documents(self, directory: Optional[str] = None) -> None:
        """Indexes documents."""
        try:
            self._vector_store = self.indexer.index_documents(directory)
            # Only create the retriever *after* successful indexing
            if self._vector_store:
                self._retriever = DocumentRetriever(self._vector_store, self.config)
        except Exception as e:
            print(f"Error indexing documents: {e}")
            raise


    def get_retriever(self) -> DocumentRetriever:
        """Gets the document retriever."""
        if self._retriever is None:
            if self._vector_store is None:  # Check vector store before creating retriever
                raise ValueError("No vector store. Index documents first.")
            self._retriever = DocumentRetriever(self._vector_store, self.config)
        return self._retriever

    def get_evaluator(self) -> RAGEvaluator:
        """Gets the RAG evaluator."""
        if self._evaluator is None:
            self._evaluator = RAGEvaluator(self.config, self.llm_processor)
        return self._evaluator

    def query_documents(self, query: str) -> List[Tuple[Document, float]]:
        """Queries the documents."""
        try:
            retriever = self.get_retriever()
            return retriever.retrieve_with_scores(query)
        except Exception as e:
            print(f"Error querying documents: {e}")
            return []

    def get_answer(self, query: str) -> Tuple[str, List[Document]]:
        """Gets an answer for the query."""
        try:
            retriever = self.get_retriever()
            documents = retriever.retrieve(query)
            if not documents:
                return "I couldn't find relevant information.", []
            answer = self.llm_processor.answer_from_documents(query, documents)
            return answer, documents
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Error: {e}", []

    def evaluate_query(self, query: str) -> Dict[str, Any]:
        """Evaluates the RAG system on a query."""
        try:
            answer, documents = self.get_answer(query)
            evaluator = self.get_evaluator()
            results = evaluator.evaluate_end_to_end(query, answer, documents)
            return results
        except Exception as e:
            print(f"Error evaluating: {e}")
            return {  # Return a basic structure even on error
                "query": query,
                "error": str(e),
                "retrieval": {"num_docs_retrieved": 0, "avg_relevance_score": 0},
                "response": {"faithfulness": 0, "relevance": 0, "coherence": 0}
            }



def create_rag_system() -> ScientificPaperRAG:
    """Creates and initializes the RAG system."""
    try:
        return ScientificPaperRAG()
    except Exception as e:
        print(f"Error creating RAG system: {e}")
        raise

# Make sure to include necessary parts for execution:
__all__ = [
    'ScientificPaperRAG',
    'create_rag_system',
    'load_config',
    'visualize_retrieval_metrics',
    'visualize_response_metrics'
]