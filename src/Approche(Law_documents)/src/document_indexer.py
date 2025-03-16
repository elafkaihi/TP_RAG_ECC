import os
from typing import List, Dict, Any
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter




class DocumentIndexer:
    """Classe pour l'indexation des documents juridiques."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise l'indexeur de documents.
        
        Args:
            config: Configuration du projet.
        """
        self.config = config
        self.data_dir = config["document_loader"]["data_dir"]
        self.supported_extensions = config["document_loader"]["supported_extensions"]
        self.db_dir = config["vector_store"]["db_dir"]
        
        # Configuration du text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunking"]["chunk_size"],
            chunk_overlap=config["chunking"]["chunk_overlap"],
            separators=config["chunking"]["separators"]
        )
        
        # Configuration des embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config["embeddings"]["model_name"],
            model_kwargs={'device': config["embeddings"]["device"]}
        )
    
    def _get_document_files(self) -> List[str]:
        """
        Récupère tous les fichiers de document à indexer.
        
        Returns:
            Liste des chemins vers les fichiers à indexer.
        """
        files = []
        for root, _, filenames in os.walk(self.data_dir):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.supported_extensions:
                    files.append(os.path.join(root, filename))
        return files
    
    def _load_documents(self, file_paths: List[str]) -> List:
        """
        Charge les documents depuis les fichiers spécifiés.
        
        Args:
            file_paths: Liste des chemins de fichiers à charger.
            
        Returns:
            Liste des documents chargés.
        """
        documents = []
        for file_path in file_paths:
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # Ajouter des métadonnées supplémentaires
                for doc in docs:
                    doc.metadata["file_path"] = file_path
                    doc.metadata["file_name"] = os.path.basename(file_path)
                
                documents.extend(docs)
                print(f"Chargé: {file_path} ({len(docs)} pages)")
            except Exception as e:
                print(f"Erreur lors du chargement de {file_path}: {e}")
        
        return documents
    
    def _split_documents(self, documents: List) -> List:
        """
        Divise les documents en chunks plus petits.
        
        Args:
            documents: Liste des documents à diviser.
            
        Returns:
            Liste des chunks de documents.
        """
        chunks = self.text_splitter.split_documents(documents)
        print(f"Documents divisés en {len(chunks)} chunks")
        return chunks
    
    def _store_embeddings(self, chunks: List) -> Chroma:
        """
        Calcule les embeddings des chunks et les stocke dans un vector store.
        
        Args:
            chunks: Liste des chunks de documents.
            
        Returns:
            Instance du vector store Chroma.
        """
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
        vectorstore.persist()
        print(f"Embeddings calculés et stockés dans {self.db_dir}")
        return vectorstore
    
    def index_documents(self, rebuild: bool = False) -> None:
        """
        Indexe les documents dans le vector store.
        
        Args:
            rebuild: Si True, reconstruit l'index à partir de zéro.
        """
        # Si rebuild est True, supprimer la base existante
        if rebuild and os.path.exists(self.db_dir):
            shutil.rmtree(self.db_dir)
            print(f"Base vectorielle existante supprimée: {self.db_dir}")
        
        # Obtenir les fichiers à indexer
        file_paths = self._get_document_files()
        
        if not file_paths:
            print(f"Aucun document trouvé dans {self.data_dir} avec les extensions suivantes: {self.supported_extensions}")
            return
        
        print(f"Indexation de {len(file_paths)} documents...")
        
        # Charger les documents
        documents = self._load_documents(file_paths)
        
        # Diviser les documents
        chunks = self._split_documents(documents)
        
        # Calculer les embeddings et les stocker
        self._store_embeddings(chunks)
        
        print("Indexation terminée avec succès!")