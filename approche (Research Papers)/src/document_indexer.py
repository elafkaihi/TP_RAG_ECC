# document_indexer.py

"""
Document Indexer module for the RAG system.
Handles loading, splitting, embedding, and storing documents.
"""

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Any, Dict, Optional

from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS  # Changed to FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.utils import load_config


class DocumentIndexer:
    """Class for document indexing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initializes the document indexer."""
        self.config = config or load_config()
        self.data_dir = self.config["paths"]["data_dir"]
        self.vector_store_dir = self.config["paths"]["vector_store_dir"]
        self.chunk_size = self.config["document_processing"]["chunk_size"]
        self.chunk_overlap = self.config["document_processing"]["chunk_overlap"]
        self.use_markdown_splitter = self.config["document_processing"]["markdown_headings_splitter"]

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config["embedding"]["model_name"],
            model_kwargs={"device": self.config["embedding"]["device"]}
        )
        self._vector_store = None  # Initialize to None


    def load_documents(self, directory: Optional[str] = None) -> List[Document]:
        """Loads documents from the specified directory."""
        directory = directory or self.data_dir
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
        try:
            documents = loader.load()
            for doc in documents:
                doc.metadata["source_file"] = os.path.basename(doc.metadata.get("source", "unknown"))
            return documents
        except Exception as e:
            print(f"Error loading documents: {e}")
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits documents into chunks."""
        if not documents:
            return []

        try:
            if self.use_markdown_splitter:
                splitter = MarkdownTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            return splitter.split_documents(documents)
        except Exception as e:
            print(f"Error splitting documents: {e}")
            raise

    def create_vector_store(self, chunks: List[Document], persist: bool = True) -> Any:
        """Creates a vector store from document chunks."""
        if not chunks:
            return None

        if persist and not os.path.exists(self.vector_store_dir):
            os.makedirs(self.vector_store_dir)

        try:
            # Use FAISS.from_documents
            vector_store = FAISS.from_documents(chunks, self.embedding_model)
            if persist:
                vector_store.save_local(self.vector_store_dir)  # Save FAISS index
            self._vector_store = vector_store # Store the created vector store
            return vector_store
        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise

    def load_vector_store(self) -> Any:
        """Loads an existing vector store from disk."""
        index_file = os.path.join(self.vector_store_dir, "index.faiss")
        if not os.path.exists(index_file): #check index.faiss
            raise FileNotFoundError(f"Vector store file not found: {index_file}")

        try:
            # Use FAISS.load_local
            vector_store = FAISS.load_local(self.vector_store_dir, self.embedding_model, allow_dangerous_deserialization=True)
            self._vector_store = vector_store  # Store the loaded vector store.
            return vector_store
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise


    def get_vector_store(self) -> Any:
        """Gets the current vector store or loads it from disk."""
        if self._vector_store is None:
            if os.path.exists(self.vector_store_dir):
                return self.load_vector_store()
            else:
                raise ValueError("Vector store not created yet and doesn't exist on disk.")
        return self._vector_store


    def index_documents(self, directory: Optional[str] = None, persist: bool = True) -> Any:
        """Runs the complete indexing pipeline."""
        if persist:  # Only create directory if persisting
            os.makedirs(self.vector_store_dir, exist_ok=True)

        documents = self.load_documents(directory)
        chunks = self.split_documents(documents)
        vector_store = self.create_vector_store(chunks, persist=persist)
        return vector_store