from typing import Dict, Any, List, Tuple

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema import Document

from src.retriever import DocumentRetriever
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Récupérer la clé API
API_KEY = os.getenv("OPENAI_API_KEY")

class QASystem:
    """Classe pour le système de question-réponse basé sur un LLM."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le système de question-réponse.
        
        Args:
            config: Configuration du projet.
        """
        self.config = config
        
        # Initialiser le retriever pour la recherche de documents
        self.retriever = DocumentRetriever(config)
        
        # Initialiser le LLM avec OpenAI
        print("Initialisation du modèle OpenAI...")
        
        # Récupérer les paramètres de configuration pour OpenAI
        openai_config = config["openai"]
        model_name = openai_config.get("model_name", "gpt-4o-mini")
        temperature = openai_config.get("temperature", 0.1)
        max_tokens = openai_config.get("max_tokens", 256)
        
        # Initialiser le modèle OpenAI
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=openai_config.get("api_key", None), 
        )
        print(f"Modèle {model_name} initialisé avec succès.")
        
        # Estimer la taille maximale de contexte selon le modèle
        model_context_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
        }
        self.max_context_tokens = model_context_limits.get(model_name, 4096) - max_tokens - 100  
        
        # Template de prompt pour OpenAI
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="Tu es un assistant d'aide à la recherche documentaire. Utilise uniquement les informations du contexte fourni pour répondre à la question. Si tu ne peux pas répondre à partir du contexte, dis-le clairement.\n\nContexte: {context}\n\nQuestion: {question}\n\nRéponse:"
        )
        
        # Créer la chaîne LLM
        self.qa_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )
    
    def answer_question(self, question: str) -> Tuple[str, List[Document]]:
        """
        Répond à une question en utilisant les documents pertinents et le LLM.
        
        Args:
            question: Question à laquelle répondre.
            
        Returns:
            Un tuple contenant la réponse générée et les documents sources utilisés.
        """
        # Rechercher les documents pertinents 
        docs = self.retriever.search(question, k=5)  
        
        if not docs:
            return "Je n'ai pas trouvé d'information pertinente pour répondre à cette question.", []
        
        # Construire le contexte en respectant la limite de tokens
        context_parts = []
        token_count = 0
        
        chars_per_token = 4
        prompt_chars = len("Tu es un assistant d'aide à la recherche documentaire. Utilise uniquement les informations du contexte fourni pour répondre à la question. Si tu ne peux pas répondre à partir du contexte, dis-le clairement.\n\nContexte: \n\nQuestion: " + question + "\n\nRéponse:")
        prompt_tokens = prompt_chars / chars_per_token
        remaining_tokens = self.max_context_tokens - int(prompt_tokens)
        
        # Construire le contexte en respectant la limite de tokens
        for i, doc in enumerate(docs):
            doc_snippet = f"Doc{i+1} p{doc.metadata.get('page', 'N/A')}: {doc.page_content}"
            doc_tokens = len(doc_snippet) / chars_per_token
            
            if remaining_tokens - doc_tokens > 0:
                context_parts.append(doc_snippet)
                remaining_tokens -= doc_tokens
            else:
                # Si pas assez d'espace, tronquer le dernier document
                if remaining_tokens > 50:  
                    safe_chars = int(remaining_tokens * chars_per_token * 0.9)  
                    truncated = doc_snippet[:safe_chars] + "..."
                    context_parts.append(truncated)
                break
        
        context = "\n\n".join(context_parts)
        
        try:
            # Générer la réponse avec OpenAI
            response = self.qa_chain.run(context=context, question=question)
            return response.strip(), docs
        except Exception as e:
            print(f"Erreur lors de la génération avec OpenAI: {e}")
            return "Je n'ai pas pu générer une réponse en raison d'une erreur. Voici les informations pertinentes:" + \
                   "\n\n" + "\n".join([doc.page_content[:200] + "..." for doc in docs[:2]]), docs