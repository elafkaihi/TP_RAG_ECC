"""
Scientific Paper RAG System - Combined Implementation
Includes all core modules in a single file with Python-based configuration.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import pandas as pd
import torch
from datetime import datetime

# LangChain imports
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain.evaluation import load_evaluator

# Transformers imports
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)

#################################################
# Configuration (Python dict instead of YAML)
#################################################

CONFIG = {
    # Paths
    "paths": {
        "data_dir": "data/",
        "vector_store_dir": "vectorstore/"
    },
    
    # Document Processing
    "document_processing": {
        "chunk_size": 500,  # Reduced chunk size to avoid context length issues
        "chunk_overlap": 100,
        "markdown_headings_splitter": True
    },
    
    # Embedding Model
    "embedding": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu"  # Change to "cuda" for GPU acceleration
    },
    
    # Vector Store
    "vector_store": {
        "type": "chroma",
        "similarity_metric": "cosine"
    },
    
    # Retrieval
    "retrieval": {
        "top_k": 3,  # Reduced to get fewer documents
        "score_threshold": 0.7
    },
    
    # LLM
    "llm": {
        # Smaller, faster model that works well with the constraints
        "model_name": "google/flan-t5-small",
        "max_tokens": 256,
        "temperature": 0.0,  # No randomness for deterministic outputs
        "context_window": 512
    },
    
    # Prompt Templates
    "prompts": {
        "qa_template": """Answer the following question based on the given context:

Context:
{context}

Question: {question}

Answer:"""
    },
    
    # Evaluation
    "evaluation": {
        "metrics": [
            "faithfulness",
            "relevance",
            "coherence",
            "informativeness"
        ]
    }
}


#################################################
# Utils Module
#################################################

def load_config() -> Dict[str, Any]:
    """
    Get configuration.
    
    Returns:
        Configuration dictionary
    """
    return CONFIG


def format_context_for_display(documents: List, max_chars: int = 300) -> str:
    """
    Format retrieved documents for display, with truncation.
    
    Args:
        documents: List of documents
        max_chars: Maximum characters per document
        
    Returns:
        Formatted string
    """
    formatted = []
    
    for i, doc in enumerate(documents):
        # Extract metadata
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        
        # Truncate content if needed
        content = doc.page_content
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
            
        # Format document
        doc_str = f"Document {i+1}: {source} (Page {page})\n{content}\n"
        formatted.append(doc_str)
        
    return "\n".join(formatted)


def visualize_retrieval_metrics(metrics_list: List[Dict[str, Any]], output_path: Optional[str] = None):
    """
    Visualize retrieval metrics across multiple queries.
    
    Args:
        metrics_list: List of retrieval metrics dictionaries
        output_path: Path to save visualization
    """
    # Convert to DataFrame
    df = pd.DataFrame([m["retrieval"] for m in metrics_list])
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot number of documents retrieved
    df.plot(
        x="query", 
        y="num_docs_retrieved", 
        kind="bar", 
        title="Number of Documents Retrieved per Query",
        ax=axes[0]
    )
    axes[0].set_ylabel("Number of Documents")
    axes[0].set_xlabel("Query")
    
    # Plot average relevance score
    df.plot(
        x="query", 
        y="avg_relevance_score", 
        kind="bar", 
        title="Average Relevance Score per Query",
        ax=axes[1]
    )
    axes[1].set_ylabel("Average Score")
    axes[1].set_xlabel("Query")
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
    
    plt.show()


def visualize_response_metrics(metrics_list: List[Dict[str, Any]], output_path: Optional[str] = None):
    """
    Visualize response quality metrics across multiple queries.
    
    Args:
        metrics_list: List of response metrics dictionaries
        output_path: Path to save visualization
    """
    # Extract response metrics
    response_metrics = []
    for metrics in metrics_list:
        metrics_dict = metrics["response"].copy()
        metrics_dict["query"] = metrics["query"]
        response_metrics.append(metrics_dict)
    
    # Convert to DataFrame
    df = pd.DataFrame(response_metrics)
    
    # Create radar chart for each query
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Get quality metrics (excluding query)
    metrics = [col for col in df.columns if col != "query"]
    num_metrics = len(metrics)
    
    # Set angles for radar chart
    angles = [n / num_metrics * 2 * 3.1415 for n in range(num_metrics)]
    angles += angles[:1]  # Close the loop
    
    # Plot each query
    for i, query in enumerate(df["query"]):
        values = df[df["query"] == query][metrics].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, label=f"Query {i+1}")
        ax.fill(angles, values, alpha=0.1)
    
    # Set labels and properties
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title("Response Quality Metrics")
    ax.grid(True)
    
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
    
    plt.show()


#################################################
# Document Indexer Module
#################################################

class DocumentIndexer:
    """Class for document indexing pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the document indexer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_dir = config["paths"]["data_dir"]
        self.vector_store_dir = config["paths"]["vector_store_dir"]
        self.chunk_size = config["document_processing"]["chunk_size"]
        self.chunk_overlap = config["document_processing"]["chunk_overlap"]
        self.use_markdown_splitter = config["document_processing"]["markdown_headings_splitter"]
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config["embedding"]["model_name"],
            model_kwargs={"device": config["embedding"]["device"]}
        )
        
        # Initialize vector store
        self._vector_store = None
    
    def load_documents(self, directory: Optional[str] = None) -> List:
        """
        Load documents from directory.
        
        Args:
            directory: Directory path to load documents from (defaults to data_dir in config)
            
        Returns:
            List of loaded documents
        """
        if directory is None:
            directory = self.data_dir
            
        print(f"Loading documents from {directory}")
        
        # Check if directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Use DirectoryLoader to load all PDFs
        loader = DirectoryLoader(
            directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        
        try:
            documents = loader.load()
            print(f"Loaded {len(documents)} document pages")
            
            # Add source filename to metadata
            for doc in documents:
                doc.metadata["source_file"] = os.path.basename(doc.metadata.get("source", "unknown"))
            
            return documents
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            raise
    
    def split_documents(self, documents: List) -> List:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        print(f"Splitting documents into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        
        if not documents:
            print("Warning: No documents to split")
            return []
        
        try:
            if self.use_markdown_splitter:
                # Use markdown-optimized splitter
                splitter = MarkdownTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            else:
                # Use default recursive splitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                
            chunks = splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks")
            
            return chunks
        except Exception as e:
            print(f"Error splitting documents: {str(e)}")
            raise
    
    def create_vector_store(self, chunks: List, persist: bool = True) -> Any:
        """
        Create vector store from document chunks.
        
        Args:
            chunks: List of document chunks
            persist: Whether to persist the vector store
            
        Returns:
            Vector store instance
        """
        print(f"Creating vector store with {len(chunks)} chunks")
        
        if not chunks:
            print("Warning: No chunks to create vector store")
            return None
        
        # Create vector store directory if it doesn't exist
        if persist and not os.path.exists(self.vector_store_dir):
            os.makedirs(self.vector_store_dir)
        
        try:
            # Create vector store
            vector_store = Chroma.from_documents(
                chunks,
                self.embedding_model,
                persist_directory=self.vector_store_dir if persist else None,
                collection_metadata={"hnsw:space": self.config["vector_store"]["similarity_metric"]}
            )
            
            if persist:
                vector_store.persist()
                print(f"Vector store persisted to {self.vector_store_dir}")
                
            self._vector_store = vector_store
            return vector_store
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            raise
    
    def load_vector_store(self) -> Any:
        """
        Load existing vector store from disk.
        
        Returns:
            Vector store instance
        """
        print(f"Loading vector store from {self.vector_store_dir}")
        
        # Check if vector store directory exists
        if not os.path.exists(self.vector_store_dir):
            raise FileNotFoundError(f"Vector store directory not found: {self.vector_store_dir}")
        
        try:
            vector_store = Chroma(
                persist_directory=self.vector_store_dir,
                embedding_function=self.embedding_model
            )
            
            self._vector_store = vector_store
            return vector_store
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            raise
    
    def get_vector_store(self) -> Any:
        """
        Get current vector store or load from disk.
        
        Returns:
            Vector store instance
        """
        if self._vector_store is None:
            if os.path.exists(self.vector_store_dir):
                return self.load_vector_store()
            else:
                raise ValueError("Vector store not created yet and doesn't exist on disk")
        
        return self._vector_store
    
    def index_documents(self, directory: Optional[str] = None, persist: bool = True) -> Any:
        """
        Run complete indexing pipeline: load, split, and create vector store.
        
        Args:
            directory: Directory to load documents from
            persist: Whether to persist the vector store
            
        Returns:
            Vector store instance
        """
        # Create vector store directory if it doesn't exist
        if persist and not os.path.exists(self.vector_store_dir):
            os.makedirs(self.vector_store_dir)
            
        # Run pipeline
        documents = self.load_documents(directory)
        chunks = self.split_documents(documents)
        vector_store = self.create_vector_store(chunks, persist=persist)
        
        return vector_store


#################################################
# Document Retriever Module
#################################################

class DocumentRetriever:
    """Class for retrieving relevant documents from vector store."""
    
    def __init__(self, vector_store: Any, config: Dict[str, Any]):
        """
        Initialize the document retriever with vector store and configuration.
        
        Args:
            vector_store: Vector store instance
            config: Configuration dictionary
        """
        self.vector_store = vector_store
        self.config = config
        self.top_k = config["retrieval"]["top_k"]
        self.score_threshold = config["retrieval"]["score_threshold"]
        
        # Create the retriever from vector store
        self.retriever = vector_store.as_retriever(
            search_kwargs={
                "k": self.top_k
            }
        )
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents based on query.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved documents
        """
        if not query.strip():
            print("Warning: Empty query")
            return []
            
        print(f"Retrieving documents for query: '{query}'")
        try:
            documents = self.retriever.invoke(query)
            print(f"Retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []
    
    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents with relevance scores.
        
        Args:
            query: Query string
            
        Returns:
            List of tuples containing documents and their relevance scores
        """
        if not query.strip():
            print("Warning: Empty query")
            return []
            
        print(f"Retrieving documents with scores for query: '{query}'")
        
        try:
            # Use similarity_search_with_score method from vector store
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=self.top_k
            )
            
            print(f"Retrieved {len(docs_with_scores)} documents with scores")
            return docs_with_scores
        except Exception as e:
            print(f"Error retrieving documents with scores: {str(e)}")
            return []
    
    def get_formatted_context(self, documents: List[Document]) -> str:
        """
        Format documents into a context string for LLM prompt.
        
        Args:
            documents: List of documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(documents):
            # Extract source information
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            
            # Format document with metadata
            formatted_doc = f"[Document {i+1}] Source: {source}, Page: {page}\n{doc.page_content}\n"
            formatted_docs.append(formatted_doc)
            
        return "\n\n".join(formatted_docs)
    
    def get_formatted_results_with_scores(self, docs_with_scores: List[Tuple[Document, float]]) -> str:
        """
        Format documents with scores into a readable string.
        
        Args:
            docs_with_scores: List of tuples containing documents and scores
            
        Returns:
            Formatted string with documents and scores
        """
        if not docs_with_scores:
            return "No relevant documents found."
        
        formatted_results = []
        for i, (doc, score) in enumerate(docs_with_scores):
            # Extract source information
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            
            # Format document with metadata and score
            # Note: For cosine similarity, higher is better (closer to 1)
            formatted_result = (
                f"[Document {i+1}] Relevance Score: {score:.4f}\n"
                f"Source: {source}, Page: {page}\n"
                f"{doc.page_content}\n"
            )
            formatted_results.append(formatted_result)
            
        return "\n\n".join(formatted_results)


#################################################
# LLM Module
#################################################

class LLMProcessor:
    """Class for processing queries using a LLM."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM processor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config["llm"]["model_name"]
        self.max_tokens = config["llm"]["max_tokens"]
        self.temperature = config["llm"]["temperature"]
        self.qa_template = config["prompts"]["qa_template"]
        
        # Initialize LLM
        self.llm = self._init_llm()
        
        # Initialize QA prompt
        self.qa_prompt = PromptTemplate(
            template=self.qa_template,
            input_variables=["context", "question"]
        )
        
        # Initialize QA chain using modern LangChain syntax
        self.qa_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _init_llm(self) -> Any:
        """
        Initialize the large language model.
        
        Returns:
            LLM instance
        """
        print(f"Initializing LLM: {self.model_name}")
        
        # Check if CUDA is available
        use_cuda = torch.cuda.is_available() and self.config["embedding"]["device"] == "cuda"
        device = 0 if use_cuda else -1
        print(f"Device set to use {'cuda' if use_cuda else 'cpu'}")
        
        try:
            # Determine if model is a T5-based seq2seq model (like Flan-T5)
            is_seq2seq = "t5" in self.model_name.lower()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load appropriate model based on type
            if is_seq2seq:
                # For sequence-to-sequence models like T5
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    device_map="auto" if use_cuda else None,
                    torch_dtype=torch.float16 if use_cuda else torch.float32
                )
                # Create text generation pipeline
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=(self.temperature > 0),  # Only sample if temperature > 0
                    device=device
                )
            else:
                # For causal LM models like GPT, OPT
                # Quantization config for lower memory usage if using GPU
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                ) if use_cuda else None
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto" if use_cuda else None,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16 if use_cuda else torch.float32
                )
                
                # Create text generation pipeline
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=(self.temperature > 0),  # Only sample if temperature > 0
                    device=device
                )
            
            # Create LangChain HuggingFace pipeline
            llm = HuggingFacePipeline(pipeline=pipe)
            
            return llm
            
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            print("Falling back to a simpler model configuration.")
            
            # Fallback to a simpler configuration
            try:
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
                model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
                
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=64,
                    temperature=0.0,
                    device=device
                )
                
                llm = HuggingFacePipeline(pipeline=pipe)
                
                return llm
            except Exception as fallback_error:
                print(f"Fallback also failed: {str(fallback_error)}")
                raise
    
    def answer_question(self, question: str, context: str) -> str:
        """
        Answer a question based on the provided context.
        
        Args:
            question: Question string
            context: Context information
            
        Returns:
            Answer string
        """
        if not question.strip():
            return "No question provided."
            
        if not context.strip():
            return "No context provided to answer the question."
            
        print(f"Answering question: '{question}'")
        
        try:
            # Ensure context length is within limits
            max_context_len = 1500  # Safe limit for most models
            if len(context) > max_context_len:
                print(f"Context too long ({len(context)} chars), truncating to {max_context_len}")
                context = context[:max_context_len] + "..."
                
            # Use the chain
            answer = self.qa_chain.invoke({"context": context, "question": question})
            
            return answer
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return f"Sorry, I encountered an error while generating an answer: {str(e)}"
    
    def answer_from_documents(self, question: str, documents: List[Document]) -> str:
        """
        Answer a question based on retrieved documents.
        
        Args:
            question: Question string
            documents: List of retrieved documents
            
        Returns:
            Answer string
        """
        if not documents:
            return "No relevant documents found to answer the question."
            
        # Format documents into context string
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                               for i, doc in enumerate(documents)])
        
        # Answer question based on context
        answer = self.answer_question(question, context)
        
        return answer


#################################################
# Evaluation Module
#################################################

@dataclass
class RetrievalMetrics:
    """Class for retrieval metrics."""
    query: str
    num_docs_retrieved: int
    avg_relevance_score: float
    
    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ResponseMetrics:
    """Class for response quality metrics."""
    query: str
    faithfulness: float
    relevance: float
    coherence: float
    informativeness: float
    
    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


class RAGEvaluator:
    """Class for evaluating RAG system performance."""
    
    def __init__(self, config: Dict[str, Any], llm_processor: Any):
        """
        Initialize the RAG evaluator with configuration.
        
        Args:
            config: Configuration dictionary
            llm_processor: LLM processor instance for evaluation
        """
        self.config = config
        self.llm_processor = llm_processor
        self.metrics = config["evaluation"]["metrics"]
        
        # Create evaluation directory if it doesn't exist
        self.eval_dir = "evaluation_results"
        os.makedirs(self.eval_dir, exist_ok=True)
        
        # Initialize evaluators for each metric
        self.evaluators = {}
        for metric in self.metrics:
            try:
                self.evaluators[metric] = load_evaluator(
                    "criteria",
                    criteria=metric,
                    llm=self.llm_processor.llm
                )
            except Exception as e:
                print(f"Error initializing evaluator for metric '{metric}': {str(e)}")
                # Provide a default value if evaluator fails
                self.evaluators[metric] = None
    
    def evaluate_retrieval(self, query: str, docs_with_scores: List[Tuple[Document, float]]) -> RetrievalMetrics:
        """
        Evaluate retrieval performance.
        
        Args:
            query: Query string
            docs_with_scores: List of retrieved documents with scores
            
        Returns:
            RetrievalMetrics
        """
        print(f"Evaluating retrieval for query: '{query}'")
        
        # Calculate metrics
        num_docs = len(docs_with_scores)
        avg_score = sum(score for _, score in docs_with_scores) / max(1, num_docs) if docs_with_scores else 0
        
        metrics = RetrievalMetrics(
            query=query,
            num_docs_retrieved=num_docs,
            avg_relevance_score=avg_score
        )
        
        return metrics
    
    def evaluate_response(self, query: str, response: str, context: str) -> ResponseMetrics:
        """
        Evaluate response quality.
        
        Args:
            query: Query string
            response: Generated response
            context: Context used for generation
            
        Returns:
            ResponseMetrics
        """
        print(f"Evaluating response for query: '{query}'")
        
        # Evaluate each metric
        results = {}
        for metric in self.metrics:
            try:
                if self.evaluators[metric] is not None:
                    eval_result = self.evaluators[metric].evaluate_strings(
                        prediction=response,
                        reference=context,
                        input=query
                    )
                    # Extract score from evaluation result
                    score = float(eval_result.get("score", 0))
                    results[metric] = score
                else:
                    # Default value if evaluator is not available
                    results[metric] = 0.5
            except Exception as e:
                print(f"Error evaluating metric '{metric}': {str(e)}")
                # Provide a default score if evaluation fails
                results[metric] = 0.5
        
        metrics = ResponseMetrics(
            query=query,
            faithfulness=results.get("faithfulness", 0.0),
            relevance=results.get("relevance", 0.0),
            coherence=results.get("coherence", 0.0),
            informativeness=results.get("informativeness", 0.0)
        )
        
        return metrics
    
    def evaluate_end_to_end(
        self, 
        query: str, 
        response: str, 
        retrieved_docs: List[Document]
    ) -> Dict[str, Any]:
        """
        Perform end-to-end evaluation of the RAG system.
        
        Args:
            query: Query string
            response: Generated response
            retrieved_docs: Retrieved documents
            
        Returns:
            Dictionary with evaluation results
        """
        # Get document scores - fallback to 0.8 as default score if not available
        docs_with_scores = [(doc, doc.metadata.get("score", 0.8)) for doc in retrieved_docs]
        
        # Get context from documents
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # Evaluate retrieval
        retrieval_metrics = self.evaluate_retrieval(query, docs_with_scores)
        
        # Evaluate response
        response_metrics = self.evaluate_response(query, response, context)
        
        # Combine results
        results = {
            "query": query,
            "answer": response,
            "retrieval": retrieval_metrics.to_dict(),
            "response": response_metrics.to_dict(),
        }
        
        return results
    
    def save_evaluation_results(self, results: Any, filename: str = "evaluation_results.json"):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
            filename: Output filename
        """
        filepath = os.path.join(self.eval_dir, filename)
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Convert to JSON-serializable format if needed
            if not isinstance(results, dict) and not isinstance(results, list):
                results = {"results": results}
                
            # Write to file
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)
                
            print(f"Evaluation results saved to {filepath}")
            
            return filepath
        except Exception as e:
            print(f"Error saving evaluation results: {str(e)}")
            return None


#################################################
# RAG System Main Module
#################################################

class ScientificPaperRAG:
    """Main class for the scientific paper RAG system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAG system with configuration.
        
        Args:
            config: Configuration dictionary (uses default if None)
        """
        self.config = config if config is not None else load_config()
        
        # Create necessary directories
        os.makedirs(self.config["paths"]["data_dir"], exist_ok=True)
        os.makedirs(self.config["paths"]["vector_store_dir"], exist_ok=True)
        
        # Initialize components
        self.indexer = DocumentIndexer(self.config)
        self._retriever = None
        self._evaluator = None
        
        # Check if vector store already exists
        self._vector_store = None
        if os.path.exists(self.config["paths"]["vector_store_dir"]):
            try:
                self._vector_store = self.indexer.load_vector_store()
                self._retriever = DocumentRetriever(self._vector_store, self.config)
            except Exception as e:
                print(f"Error loading existing vector store: {str(e)}")
                print("Will need to re-index documents before querying.")
        
        # Initialize LLM processor
        try:
            self.llm_processor = LLMProcessor(self.config)
        except Exception as e:
            print(f"Error initializing LLM processor: {str(e)}")
            raise
    
    def index_documents(self, directory: Optional[str] = None) -> None:
        """
        Index documents for the RAG system.
        
        Args:
            directory: Directory to load documents from
        """
        try:
            self._vector_store = self.indexer.index_documents(directory)
            self._retriever = DocumentRetriever(self._vector_store, self.config)
        except Exception as e:
            print(f"Error indexing documents: {str(e)}")
            raise
    
    def get_retriever(self) -> DocumentRetriever:
        """
        Get the document retriever.
        
        Returns:
            Document retriever instance
        """
        if self._retriever is None:
            if self._vector_store is None:
                raise ValueError("No vector store available. Index documents first.")
            self._retriever = DocumentRetriever(self._vector_store, self.config)
        
        return self._retriever
    
    def get_evaluator(self) -> RAGEvaluator:
        """
        Get the RAG evaluator.
        
        Returns:
            RAG evaluator instance
        """
        if self._evaluator is None:
            self._evaluator = RAGEvaluator(self.config, self.llm_processor)
        
        return self._evaluator
    
    def query_documents(self, query: str) -> List[Tuple[Document, float]]:
        """
        Query documents in the RAG system.
        
        Args:
            query: Query string
            
        Returns:
            List of documents with relevance scores
        """
        try:
            retriever = self.get_retriever()
            docs_with_scores = retriever.retrieve_with_scores(query)
            return docs_with_scores
        except Exception as e:
            print(f"Error querying documents: {str(e)}")
            return []
    
    def get_answer(self, query: str) -> Tuple[str, List[Document]]:
        """
        Get answer for a query.
        
        Args:
            query: Query string
            
        Returns:
            Tuple of answer string and list of retrieved documents
        """
        try:
            # Get retriever
            retriever = self.get_retriever()
            
            # Retrieve documents
            documents = retriever.retrieve(query)
            
            # If no documents retrieved, return early
            if not documents:
                return "I couldn't find any relevant information to answer your question.", []
            
            # Get answer from LLM
            answer = self.llm_processor.answer_from_documents(query, documents)
            
            return answer, documents
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}", []
    
    def evaluate_query(self, query: str) -> Dict[str, Any]:
        """
        Evaluate RAG system on a query.
        
        Args:
            query: Query string
            
        Returns:
            Evaluation results
        """
        try:
            # Get answer and documents
            answer, documents = self.get_answer(query)
            
            # Get evaluator
            evaluator = self.get_evaluator()
            
            # Evaluate
            results = evaluator.evaluate_end_to_end(query, answer, documents)
            
            return results
        except Exception as e:
            print(f"Error evaluating query: {str(e)}")
            # Return basic structure with error information
            return {
                "query": query,
                "error": str(e),
                "retrieval": {"num_docs_retrieved": 0, "avg_relevance_score": 0},
                "response": {"faithfulness": 0, "relevance": 0, "coherence": 0, "informativeness": 0}
            }


def create_rag_system() -> ScientificPaperRAG:
    """
    Create and initialize a RAG system.
        
    Returns:
        Initialized RAG system
    """
    try:
        # Create RAG system
        rag = ScientificPaperRAG(CONFIG)
        
        return rag
    except Exception as e:
        print(f"Error creating RAG system: {str(e)}")
        raise


# Module initialization
__all__ = [
    'ScientificPaperRAG',
    'create_rag_system',
    'load_config',
    'visualize_retrieval_metrics',
    'visualize_response_metrics'
]


#################################################
# Example Usage
#################################################

def main():
    """Example usage of the RAG system."""
    # Create RAG system with default configuration
    rag = create_rag_system()
    
    # Index documents
    # rag.index_documents()
    
    # Query the system
    query = "What are the key findings in these papers?"
    answer, docs = rag.get_answer(query)
    
    print(f"\nQuestion: {query}")
    print(f"\nAnswer: {answer}")
    print("\nRetrieved Documents:")
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()