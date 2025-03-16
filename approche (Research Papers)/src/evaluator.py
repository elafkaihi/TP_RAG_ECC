
import os 
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from langchain.evaluation import load_evaluator
from langchain_core.documents import Document

from src.utils import load_config
from src.llm_processor import LLMProcessor


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

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


class RAGEvaluator:
    """Class for evaluating RAG system performance."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_processor: Optional[LLMProcessor] = None):
        """Initializes the RAG evaluator."""
        self.config = config or load_config()
        if llm_processor is None:
            raise ValueError("llm_processor must be provided.")
        self.llm_processor = llm_processor
        self.metrics = self.config["evaluation"]["metrics"]
        self.eval_dir = "evaluation_results"
        os.makedirs(self.eval_dir, exist_ok=True)
        self.evaluators = {}
        for metric in self.metrics:
            try:
                self.evaluators[metric] = load_evaluator("criteria", criteria=metric, llm=self.llm_processor.llm)
            except Exception as e:
                print(f"Error initializing evaluator for '{metric}': {e}")
                self.evaluators[metric] = None  # Set to None if it fails

    def evaluate_retrieval(self, query: str, docs_with_scores: List[Tuple[Document, float]]) -> RetrievalMetrics:
        """Evaluates retrieval performance."""
        num_docs = len(docs_with_scores)
        avg_score = sum(score for _, score in docs_with_scores) / max(1, num_docs) if docs_with_scores else 0
        return RetrievalMetrics(query=query, num_docs_retrieved=num_docs, avg_relevance_score=avg_score)

    def evaluate_response(self, query: str, response: str, context: str) -> ResponseMetrics:
        """Evaluates response quality."""
        results = {}
        for metric in self.metrics:
            try:
                if self.evaluators[metric] is not None:  # Check if evaluator exists
                    eval_result = self.evaluators[metric].evaluate_strings(prediction=response, reference=context, input=query)
                    results[metric] = float(eval_result.get("score", 0.0))  # Provide a default
                else:
                    results[metric] = 0.0 # Default value if evaluator is missing
            except Exception as e:
                print(f"Error evaluating metric '{metric}': {e}")
                results[metric] = 0.0  # Default in case of error

        return ResponseMetrics(
            query=query,
            faithfulness=results.get("faithfulness", 0.0),
            relevance=results.get("relevance", 0.0),
            coherence=results.get("coherence", 0.0),
        )


    def evaluate_end_to_end(self, query: str, response: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
        """Performs end-to-end evaluation."""
        docs_with_scores = [(doc, doc.metadata.get("score", 0.7)) for doc in retrieved_docs] #increased default score
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        retrieval_metrics = self.evaluate_retrieval(query, docs_with_scores)
        response_metrics = self.evaluate_response(query, response, context)
        return {
            "query": query,
            "answer": response,
            "retrieval": retrieval_metrics.to_dict(),
            "response": response_metrics.to_dict()
        }

    def save_evaluation_results(self, results: Any, filename: str = "evaluation_results.json"):
        """Saves evaluation results to a file."""
        filepath = os.path.join(self.eval_dir, filename)
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if not isinstance(results, (dict, list)):
                results = {"results": results}
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Evaluation results saved to {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving evaluation results: {e}")
            return None

