from typing import Dict, Any, List
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


from langchain.schema import Document

class DocumentRetriever:
    """Classe pour la recherche de documents pertinents dans la base vectorielle."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le système de recherche.
        
        Args:
            config: Configuration du projet.
        """
        self.config = config
        self.db_dir = config["vector_store"]["db_dir"]
        
        # Configuration des embeddings (même modèle que pour l'indexation)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config["embeddings"]["model_name"],
            model_kwargs={'device': config["embeddings"]["device"]}
        )
        
        # Type de recherche et paramètres
        self.search_type = config["retriever"]["search_type"]
        self.search_kwargs = config["retriever"]["search_kwargs"]
        
        # Charger la base vectorielle
        if os.path.exists(self.db_dir):
            self.vectorstore = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings
            )
        else:
            raise FileNotFoundError(f"La base vectorielle n'existe pas: {self.db_dir}. Veuillez indexer des documents d'abord.")
        
        print(f"Embedding model utilisé : {self.embeddings.model_name}")
        

    
    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Recherche les documents les plus pertinents pour une requête.
        
        Args:
            query: Requête utilisateur.
            k: Nombre de résultats à retourner (remplace la valeur de config si spécifiée).
            
        Returns:
            Liste des documents les plus pertinents avec leurs scores de similarité.
        """
        # Paramètres de recherche
        search_kwargs = self.search_kwargs.copy()
        if k is not None:
            search_kwargs["k"] = k
        
        # Effectuer la recherche
        if self.search_type == "similarity":
            search_kwargs.pop("score_threshold", None) 
            results = self.vectorstore.similarity_search_with_score(query, **search_kwargs)
            # Transformer les résultats en Documents avec score dans les métadonnées
            documents = []
            for doc, score in results:
                doc.metadata["score"] = score
                documents.append(doc)
            return documents
        else:
            # Pour d'autres types de recherche (mmr, etc.)
            return self.vectorstore.similarity_search(query, **search_kwargs)