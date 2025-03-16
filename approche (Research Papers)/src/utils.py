"""
Utility functions for the RAG system.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.documents import Document
import os 
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config() -> Dict[str, Any]:
    """Loads the configuration."""
    from src.config import CONFIG  # Local import
    return CONFIG


def format_context_for_display(documents: List[Document], max_chars: int = 300) -> str:
    """Formats retrieved documents for display, with truncation."""
    formatted = []
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        content = doc.page_content[:max_chars] + ("..." if len(doc.page_content) > max_chars else "")
        doc_str = f"Document {i+1}: {source} (Page {page})\n{content}\n"
        formatted.append(doc_str)
    return "\n".join(formatted)


def visualize_retrieval_metrics(metrics_list: List[Dict[str, Any]], output_path: Optional[str] = None):
    """Visualizes retrieval metrics across multiple queries."""
    df = pd.DataFrame([m["retrieval"] for m in metrics_list])
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    df.plot(x="query", y="num_docs_retrieved", kind="bar", title="Number of Documents Retrieved", ax=axes[0])
    axes[0].set_ylabel("Number of Documents")
    axes[0].set_xlabel("Query")

    df.plot(x="query", y="avg_relevance_score", kind="bar", title="Average Relevance Score", ax=axes[1])
    axes[1].set_ylabel("Average Score")
    axes[1].set_xlabel("Query")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
    plt.show()


def visualize_response_metrics(metrics_list: List[Dict[str, Any]], output_path: Optional[str] = None):
    """Visualizes response quality metrics."""
    response_metrics = []
    for metrics in metrics_list:
        metrics_dict = metrics["response"].copy()
        metrics_dict["query"] = metrics["query"]
        response_metrics.append(metrics_dict)

    df = pd.DataFrame(response_metrics)
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    metrics = [col for col in df.columns if col != "query"]
    num_metrics = len(metrics)
    angles = [n / num_metrics * 2 * 3.1415 for n in range(num_metrics)]
    angles += angles[:1]

    for i, query in enumerate(df["query"]):
        values = df[df["query"] == query][metrics].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=f"Query {i+1}")
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title("Response Quality Metrics")
    ax.grid(True)

    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
    plt.show()



