#!/usr/bin/env python
"""
Command-line interface for the scientific paper RAG system.
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional, List

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# Import the combined implementation
from model_scientific_paper import (
    create_rag_system, 
    ScientificPaperRAG, 
    load_config, 
    visualize_retrieval_metrics, 
    visualize_response_metrics
)

# Create Typer app
app = typer.Typer(help="Scientific Paper RAG System")
console = Console()


def get_rag_system() -> ScientificPaperRAG:
    """Get RAG system instance."""
    try:
        return create_rag_system()
    except Exception as e:
        console.print(f"[bold red]Error creating RAG system: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


@app.command("index")
def index_documents(
    data_dir: Optional[str] = None
):
    """
    Index documents for the RAG system.
    
    Args:
        data_dir: Directory containing documents to index (overrides config)
    """
    console.print(Panel("Indexing Documents", style="bold blue"))
    
    # Load config
    try:
        config = load_config()
    except Exception as e:
        console.print(f"[bold red]Error loading configuration: {str(e)}[/bold red]")
        raise typer.Exit(code=1)
    
    # Use provided data directory or default from config
    if data_dir:
        console.print(f"Using data directory: [bold]{data_dir}[/bold]")
    else:
        data_dir = config["paths"]["data_dir"]
        console.print(f"Using default data directory: [bold]{data_dir}[/bold]")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        console.print(f"[bold yellow]Directory {data_dir} does not exist. Creating it...[/bold yellow]")
        try:
            os.makedirs(data_dir)
        except Exception as e:
            console.print(f"[bold red]Error creating directory: {str(e)}[/bold red]")
            raise typer.Exit(code=1)
    
    # Create and index
    try:
        rag = get_rag_system()
        rag.index_documents(data_dir)
        console.print("[bold green]Indexing completed successfully![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error during indexing: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


@app.command("query")
def query_documents(
    query: str,
    show_docs: bool = True
):
    """
    Query documents in the RAG system.
    
    Args:
        query: Query string
        show_docs: Whether to show retrieved documents
    """
    console.print(Panel(f"Query: {query}", style="bold blue"))
    
    try:
        # Create RAG system
        rag = get_rag_system()
        
        # Query documents
        docs_with_scores = rag.query_documents(query)
        
        # Display results
        console.print(f"[bold green]Found {len(docs_with_scores)} relevant documents[/bold green]")
        
        if show_docs and docs_with_scores:
            # Create table for results
            table = Table(title="Retrieved Documents")
            table.add_column("Document", style="cyan")
            table.add_column("Source", style="green")
            table.add_column("Page", style="yellow")
            table.add_column("Score", style="magenta")
            
            for i, (doc, score) in enumerate(docs_with_scores):
                # Truncate content if too long
                content = doc.page_content
                if len(content) > 100:
                    content = content[:100] + "..."
                
                source = doc.metadata.get("source_file", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                
                table.add_row(
                    f"Document {i+1}",
                    source,
                    str(page),
                    f"{score:.4f}"
                )
            
            console.print(table)
            
    except Exception as e:
        console.print(f"[bold red]Error during query: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


@app.command("ask")
def ask_question(
    question: str,
    show_docs: bool = False,
    evaluate: bool = False
):
    """
    Ask a question to the RAG system.
    
    Args:
        question: Question string
        show_docs: Whether to show retrieved documents
        evaluate: Whether to evaluate the response
    """
    console.print(Panel(f"Question: {question}", style="bold blue"))
    
    try:
        # Create RAG system
        rag = get_rag_system()
        
        if evaluate:
            # Get answer with evaluation
            results = rag.evaluate_query(question)
            answer = results.get("answer", "No answer generated")
            
            # Display answer
            console.print(Panel(Markdown(answer), title="Answer", border_style="green"))
            
            # Display evaluation results
            eval_table = Table(title="Evaluation Results")
            eval_table.add_column("Metric", style="cyan")
            eval_table.add_column("Score", style="magenta")
            
            # Retrieval metrics
            retrieval = results.get("retrieval", {})
            eval_table.add_row("Documents Retrieved", str(retrieval.get("num_docs_retrieved", 0)))
            eval_table.add_row("Avg. Relevance Score", f"{retrieval.get('avg_relevance_score', 0):.4f}")
            
            # Response metrics
            response = results.get("response", {})
            for metric, score in response.items():
                if metric != "query":
                    eval_table.add_row(metric.capitalize(), f"{score:.4f}")
            
            console.print(eval_table)
            
            # Save evaluation results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_{timestamp}.json"
            rag.get_evaluator().save_evaluation_results(results, filename)
            
        else:
            # Get answer without evaluation
            answer, documents = rag.get_answer(question)
            
            # Display answer
            console.print(Panel(Markdown(answer), title="Answer", border_style="green"))
            
            # Display documents if requested
            if show_docs and documents:
                console.print("[bold]Retrieved Documents:[/bold]")
                
                for i, doc in enumerate(documents):
                    source = doc.metadata.get("source_file", "Unknown")
                    page = doc.metadata.get("page", "Unknown")
                    
                    # Truncate content if too long
                    content = doc.page_content
                    if len(content) > 200:
                        content = content[:200] + "..."
                    
                    doc_panel = Panel(
                        content,
                        title=f"Document {i+1}: {source} (Page {page})",
                        border_style="cyan"
                    )
                    console.print(doc_panel)
        
    except Exception as e:
        console.print(f"[bold red]Error during question answering: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


@app.command("evaluate")
def evaluate_system(
    questions_file: str,
    output_file: Optional[str] = None,
    visualize: bool = True
):
    """
    Evaluate the RAG system on a set of questions.
    
    Args:
        questions_file: Path to JSON file with questions
        output_file: Path to output file for evaluation results
        visualize: Whether to visualize evaluation results
    """
    console.print(Panel("Evaluating RAG System", style="bold blue"))
    
    try:
        # Check if questions file exists
        if not os.path.exists(questions_file):
            console.print(f"[bold red]Questions file not found: {questions_file}[/bold red]")
            raise typer.Exit(code=1)
            
        # Load questions
        with open(questions_file, "r") as f:
            questions = json.load(f)
        
        # Validate questions format
        if not isinstance(questions, list):
            console.print("[bold red]Questions file must contain a list of strings[/bold red]")
            raise typer.Exit(code=1)
        
        # Create RAG system
        rag = get_rag_system()
        
        # Evaluate each question
        console.print(f"Evaluating {len(questions)} questions...")
        results = []
        
        with console.status("[bold green]Evaluating...[/bold green]"):
            for i, question in enumerate(questions):
                console.print(f"Question {i+1}: {question}")
                result = rag.evaluate_query(question)
                results.append(result)
        
        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation_{timestamp}.json"
        
        evaluator = rag.get_evaluator()
        evaluator.save_evaluation_results(results, output_file)
        
        # Display summary
        console.print(f"[bold green]Evaluation completed and saved to {output_file}[/bold green]")
        
        # Visualize if requested
        if visualize:
            console.print("[bold]Generating visualizations...[/bold]")
            
            # Visualize retrieval metrics
            vis_file = output_file.replace(".json", "_retrieval.png")
            visualize_retrieval_metrics(results, vis_file)
            
            # Visualize response metrics
            vis_file = output_file.replace(".json", "_response.png")
            visualize_response_metrics(results, vis_file)
        
    except Exception as e:
        console.print(f"[bold red]Error during evaluation: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


@app.command("info")
def system_info():
    """
    Display information about the RAG system.
    """
    console.print(Panel("RAG System Information", style="bold blue"))
    
    try:
        # Load config
        config = load_config()
        
        # Create info table
        table = Table(title="System Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        # Add config entries
        table.add_row("Data Directory", config["paths"]["data_dir"])
        table.add_row("Vector Store Directory", config["paths"]["vector_store_dir"])
        table.add_row("Embedding Model", config["embedding"]["model_name"])
        table.add_row("LLM Model", config["llm"]["model_name"])
        table.add_row("Chunk Size", str(config["document_processing"]["chunk_size"]))
        table.add_row("Chunk Overlap", str(config["document_processing"]["chunk_overlap"]))
        table.add_row("Top K Documents", str(config["retrieval"]["top_k"]))
        
        console.print(table)
        
        # Check if vector store exists
        vector_store_path = config["paths"]["vector_store_dir"]
        if os.path.exists(vector_store_path):
            console.print(f"[bold green]Vector store exists at {vector_store_path}[/bold green]")
        else:
            console.print("[bold yellow]Vector store has not been created yet[/bold yellow]")
        
        # Check data directory
        data_dir = config["paths"]["data_dir"]
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
            console.print(f"[bold]Found {len(files)} PDF files in data directory:[/bold]")
            for file in files:
                console.print(f"  - {file}")
        else:
            console.print(f"[bold yellow]Data directory {data_dir} does not exist[/bold yellow]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()