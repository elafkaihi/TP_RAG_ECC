"""
LLM Processor module for the RAG system.
Handles interactions with the Large Language Model.
"""

from typing import List, Any, Dict, Optional
import os 
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from src.utils import load_config


class LLMProcessor:
    """Class for processing queries using an LLM."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initializes the LLM processor."""
        self.config = config or load_config()
        self.model_name = self.config["llm"]["model_name"]
        self.max_tokens = self.config["llm"]["max_tokens"]
        self.temperature = self.config["llm"]["temperature"]
        self.qa_template = self.config["prompts"]["qa_template"]
        self.llm = self._init_llm()
        self.qa_prompt = PromptTemplate(template=self.qa_template, input_variables=["context", "question"])
        self.qa_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )

    def _init_llm(self) -> Any:
        """Initializes the LLM."""
        print(f"Initializing LLM: {self.model_name}")
        self.llm = HuggingFaceEndpoint(
                repo_id=self.model_name,
                task="text-generation",
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        return self.llm

    def answer_question(self, question: str, context: str) -> str:
        """Answers a question based on the provided context."""
        if not question.strip() or not context.strip():
            return "No question or context provided." if not question.strip() else "No context provided."

        try:
            max_context_len = 3500  # Increased context length
            if len(context) > max_context_len:
                print(f"Context too long ({len(context)} chars), truncating to {max_context_len}")
                context = context[:max_context_len] + "..."
            answer = self.qa_chain.invoke({"context": context, "question": question})
            return answer
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Sorry, I encountered an error: {e}"

    def answer_from_documents(self, question: str, documents: List[Document]) -> str:
        """Answers a question based on retrieved documents."""
        if not documents:
            return "No relevant documents found."
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)])
        return self.answer_question(question, context)

