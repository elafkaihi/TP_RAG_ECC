# RAG System

A powerful Retrieval-Augmented Generation (RAG) system for processing and querying document collections. This system enables semantic search and contextual question answering, leveraging the power of modern language models to provide accurate responses based on your document repository.

## Features

* **Document Processing**: Automatically processes various document formats including PDFs, Word documents, and text files
* **Semantic Search**: Uses advanced embeddings to find the most relevant content for your queries
* **Intelligent Chunking**: Splits documents into optimal segments while preserving context
* **Vector Database Integration**: Stores and retrieves document embeddings efficiently
* **LLM Integration**: Connects with state-of-the-art language models to generate accurate responses
* **Customizable Prompting**: Tailor system prompts to optimize response quality
* **Evaluation Framework**: Built-in metrics to measure retrieval quality and answer accuracy
* **Multi-platform Support**: Works on Windows, macOS, and Linux
* **Simple CLI**: Easy-to-use command line interface for all operations

## Requirements

* Python 3.8+
* Required libraries (install via `pip install -r requirements.txt`):
  


## Project Structure

```
rag-system/
├── src/
│   ├── config.py           # Configuration settings
│   ├── cli.py              # Command-line interface
│   ├── document_processor.py # Document loading and chunking
│   ├── embeddings.py       # Embedding generation
│   ├── vector_store.py     # Vector database interactions
│   ├── retriever.py        # Document retrieval logic
│   ├── llm.py              # LLM integration
│   ├── rag_engine.py       # Core RAG system
│   ├── evaluator.py        # Evaluation framework
│   └── utils.py            # Utility functions
├── data/                   # Place your documents here
├── vectorstore/            # Vector database storage
├── evaluation/             # Evaluation results
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Getting Started

1. **Setup the Environment:**

   ```bash
   git clone 
   cd rag-system
   pip install -r requirements.txt
   ```

2. **Add Documents:**
   
   Place your documents in the `data/` directory.

3. **Index Your Documents:**

   ```bash
   python src/cli.py index --dir data
   ```

4. **Ask Questions:**

   ```bash
   python src/cli.py query "What is the main topic discussed in these documents?"
   ```

5. **Run Evaluations:**

   ```bash
   # Single query evaluation
   python src/cli.py evaluate --query "Your question here" --output eval_results.json
   
   # Batch evaluation
   python src/cli.py evaluate-batch --file queries.txt --output batch_results.json
   ```

## Configuration

Edit `src/config.py` to customize system behavior:

### Document Processing
```python
CHUNK_SIZE = 1000           # Size of text chunks
CHUNK_OVERLAP = 200         # Overlap between chunks
USE_STRUCTURE = True        # Use document structure for chunking
```

### Embeddings
```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cuda"   # Use "cpu" if no GPU available
```

### Retrieval
```python
TOP_K = 5                   # Number of documents to retrieve
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score
```

### LLM
```python
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_TOKENS = 1024           # Maximum response length
TEMPERATURE = 0.1           # LLM randomness (0.0 = deterministic)
```

## Advanced Usage

### Custom Prompts

Create custom prompts by editing the templates in `config.py`:

```python
QA_PROMPT_TEMPLATE = """
Answer the following question based solely on the provided context.

Context:
{context}

Question: {question}

Answer:
"""
```

### Evaluation Metrics

The system provides several evaluation metrics:

* **Retrieval Accuracy**: Measures how relevant the retrieved documents are
* **Answer Faithfulness**: Determines if the answer is supported by the context
* **Answer Completeness**: Evaluates if the answer addresses all aspects of the question
* **Answer Coherence**: Assesses the clarity and structure of the response

## Troubleshooting

* **Memory Issues**: Reduce `CHUNK_SIZE` or process fewer documents at once
* **Slow Performance**: Enable GPU acceleration or use a smaller embedding model
* **Low-Quality Answers**: Adjust prompts, increase `TOP_K`, or try a different LLM
* **Indexing Errors**: Ensure documents are in supported formats and accessible

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
