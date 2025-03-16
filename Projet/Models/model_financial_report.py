import os
import json
import logging
import argparse
import gc
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastRAG:
    """
    Système RAG optimisé pour la vitesse et les performances.
    """
    def __init__(self, base_dir="rag_data", embedding_model_name="BAAI/bge-small-en-v1.5"):
        # Répertoires
        self.base_dir = base_dir
        self.index_dir = os.path.join(base_dir, "index")
        self.chunks_dir = os.path.join(base_dir, "chunks")
        
        # Créer les répertoires
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)
        
        # Modèle d'embedding
        self.embedding_model_name = embedding_model_name
        logger.info(f"Chargement du modèle d'embedding: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Paramètres de récupération
        self.retriever_top_k = 5
        self.retriever_similarity_threshold = 0.6

    def process_pdf(self, pdf_path):
        """Traiter un document PDF page par page."""
        if not os.path.exists(pdf_path):
            logger.error(f"Le fichier PDF n'existe pas: {pdf_path}")
            return False
        
        doc_id = os.path.basename(pdf_path)
        
        # Vérifier si le document est déjà traité
        doc_info = self._load_doc_info()
        if doc_id in doc_info:
            logger.info(f"Document {doc_id} déjà traité - Utilisation de l'index existant")
            return True
        
        # Charger ou créer l'index FAISS
        index = self._get_faiss_index()
        
        # Traiter avec PyPDF2
        try:
            import PyPDF2
            logger.info(f"Traitement du PDF {pdf_path} avec PyPDF2")
            
            chunk_ids = []
            chunk_mapping = {}
            index_position = index.ntotal
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                logger.info(f"PDF contient {total_pages} pages")
                
                for page_num in tqdm(range(total_pages), desc="Traitement des pages"):
                    try:
                        # Extraire le texte
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text() or ""
                        
                        if not page_text.strip():
                            continue
                        
                        # Formater le contenu
                        page_content = f"[Page {page_num+1}]\n{page_text}"
                        
                        # Identifiant du chunk
                        chunk_id = f"{doc_id}_page_{page_num+1}"
                        
                        # Métadonnées
                        metadata = {
                            "page": page_num + 1,
                            "doc_id": doc_id,
                            "source": pdf_path,
                            "type": "pdf"
                        }
                        
                        # Sauvegarder le chunk
                        chunk_file = os.path.join(self.chunks_dir, f"{chunk_id}.json")
                        with open(chunk_file, 'w', encoding='utf-8') as f:
                            json.dump({"text": page_content, "metadata": metadata}, f, ensure_ascii=False)
                        
                        # Créer l'embedding
                        with torch.no_grad():
                            embedding = self.embedding_model.encode([page_content], convert_to_tensor=True)
                            embedding_np = embedding.cpu().numpy()
                        
                        # Normaliser
                        faiss.normalize_L2(embedding_np)
                        
                        # Ajouter à l'index
                        index.add(embedding_np)
                        
                        # Mettre à jour le mappage
                        chunk_ids.append(chunk_id)
                        chunk_mapping[str(index_position)] = chunk_id
                        index_position += 1
                        
                        # Libérer mémoire
                        del embedding
                        del embedding_np
                        del page
                        del page_text
                    except Exception as e:
                        logger.error(f"Erreur sur page {page_num+1}: {str(e)}")
                
                # Sauvegarder l'index
                self._save_faiss_index(index)
                self._save_chunk_mapping(chunk_mapping)
                
                # Sauvegarder les informations du document
                doc_info = self._load_doc_info()
                doc_info[doc_id] = {
                    "path": pdf_path,
                    "type": "pdf",
                    "total_pages": total_pages,
                    "processed_pages": len(chunk_ids),
                    "chunk_ids": chunk_ids,
                    "processed_date": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                self._save_doc_info(doc_info)
                
                logger.info(f"Traitement PDF terminé: {len(chunk_ids)} pages")
                return True
                
        except ImportError:
            logger.warning("PyPDF2 non installé, essai avec pdfplumber...")
            
            # Implémentation similaire avec pdfplumber si nécessaire
            pass
        
        except Exception as e:
            logger.error(f"Erreur de traitement PDF: {str(e)}")
            logger.error(traceback.format_exc())
        
        return False

    def retrieve(self, query):
        """Récupérer les chunks pertinents pour une requête."""
        # Obtenir l'index FAISS
        index = self._get_faiss_index(create_if_missing=False)
        
        if index is None or index.ntotal == 0:
            logger.warning("Pas de documents dans l'index")
            return []
        
        # Charger le mappage des chunks
        chunk_mapping = self._load_chunk_mapping()
        
        # Créer l'embedding de la requête
        with torch.no_grad():
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
            query_embedding_np = query_embedding.cpu().numpy()
        
        # Normaliser
        faiss.normalize_L2(query_embedding_np)
        
        # Rechercher
        k = min(self.retriever_top_k, index.ntotal)
        scores, indices = index.search(query_embedding_np, k)
        
        # Convertir les résultats
        retrieved_docs = []
        
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            
            score = float(scores[0][i])
            
            if score < self.retriever_similarity_threshold:
                continue
            
            # Obtenir le chunk
            chunk_id = chunk_mapping.get(str(idx))
            if not chunk_id:
                continue
            
            # Charger le chunk
            chunk_data = self._load_chunk(chunk_id)
            if not chunk_data:
                continue
            
            # Ajouter aux résultats
            retrieved_docs.append({
                "text": chunk_data["text"],
                "metadata": chunk_data["metadata"],
                "score": score
            })
        
        # Libérer mémoire
        del index
        del query_embedding
        del query_embedding_np
        
        return retrieved_docs

    def _get_faiss_index(self, create_if_missing=True):
        """Obtenir l'index FAISS existant ou en créer un nouveau."""
        index_file = os.path.join(self.index_dir, "index.faiss")
        
        if os.path.exists(index_file):
            return faiss.read_index(index_file)
        
        if create_if_missing:
            return faiss.IndexFlatIP(self.embedding_dim)
        
        return None

    def _save_faiss_index(self, index):
        """Sauvegarder l'index FAISS."""
        index_file = os.path.join(self.index_dir, "index.faiss")
        faiss.write_index(index, index_file)

    def _save_chunk_mapping(self, mapping):
        """Sauvegarder le mappage des chunks."""
        mapping_file = os.path.join(self.index_dir, "chunk_mapping.json")
        
        # Charger le mappage existant s'il existe
        existing_mapping = {}
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                existing_mapping = json.load(f)
        
        # Mettre à jour avec le nouveau mappage
        existing_mapping.update(mapping)
        
        # Sauvegarder
        with open(mapping_file, 'w') as f:
            json.dump(existing_mapping, f)

    def _load_chunk_mapping(self):
        """Charger le mappage des chunks."""
        mapping_file = os.path.join(self.index_dir, "chunk_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                return json.load(f)
        return {}

    def _load_chunk(self, chunk_id):
        """Charger un chunk depuis le disque."""
        chunk_file = os.path.join(self.chunks_dir, f"{chunk_id}.json")
        if os.path.exists(chunk_file):
            with open(chunk_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _save_doc_info(self, doc_info):
        """Sauvegarder les informations du document."""
        info_file = os.path.join(self.base_dir, "doc_info.json")
        with open(info_file, 'w') as f:
            json.dump(doc_info, f)

    def _load_doc_info(self):
        """Charger les informations du document."""
        info_file = os.path.join(self.base_dir, "doc_info.json")
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                return json.load(f)
        return {}


class FastLLM:
    """
    Composant LLM optimisé pour la vitesse.
    """
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initialisation du LLM: {model_name} sur {self.device}")
        
        # Chargement du tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Chargement du modèle avec optimisation
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Prompt simple
        self.prompt_template = """Based on the following information, please answer the question in a concise and informative way, in one or two sentences preferably.Do not ask any follow-up questions in your response.

Information:
{context}

Question: {query}

Answer:"""

    def generate(self, query, context_docs, max_tokens=512):
        """Générer une réponse rapide à partir des documents de contexte."""
        start_time = time.time()
        
        # Préparer le contexte (limité à 3000 caractères)
        context_text = ""
        for i, doc in enumerate(context_docs):
            doc_text = f"[Document {i+1}] {doc['text']}\n\n"
            if len(context_text) + len(doc_text) > 3000:
                # Tronquer pour éviter un contexte trop grand
                if not context_text:
                    # Si le premier document est déjà trop grand
                    context_text = doc_text[:3000]
                break
            context_text += doc_text
        
        # Créer le prompt
        prompt = self.prompt_template.format(context=context_text, query=query)
        
        # Tokeniser
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Nettoyer la mémoire
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Générer la réponse
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Décoder la réponse
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extraire la réponse (après "Answer:")
        try:
            answer = generated_text.split("Answer:")[1].strip()
        except IndexError:
            # Si le format n'est pas comme prévu, retourner tout le texte généré
            answer = generated_text[len(prompt):].strip()
        
        # Nettoyer la mémoire
        del inputs
        del output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Métrique de temps
        generation_time = time.time() - start_time
        
        return answer, {"generation_time": generation_time}


def process_doc(doc_path, base_dir="rag_data", force_reprocess=False):
    """Traiter un document et créer l'index."""
    rag = FastRAG(base_dir=base_dir)
    
    # Déterminer le type de document
    ext = os.path.splitext(doc_path)[1].lower()
    
    if ext == '.pdf':
        return rag.process_pdf(doc_path)
    else:
        logger.error(f"Type de document non pris en charge: {ext}")
        return False


def answer_question(doc_path, query, base_dir="rag_data", 
                   model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                   max_tokens=512):
    """Répondre à une question sur un document."""
    start_time = time.time()
    
    # 1. Charger le RAG
    rag = FastRAG(base_dir=base_dir)
    
    # 2. Vérifier si le document est traité, sinon le traiter
    doc_id = os.path.basename(doc_path)
    doc_info = rag._load_doc_info()
    
    if doc_id not in doc_info:
        logger.info(f"Document {doc_id} non traité. Traitement...")
        process_doc(doc_path, base_dir)
    
    # 3. Rechercher les chunks pertinents
    retrieval_start = time.time()
    retrieved_docs = rag.retrieve(query)
    retrieval_time = time.time() - retrieval_start
    
    logger.info(f"Récupération de {len(retrieved_docs)} chunks en {retrieval_time:.2f} secondes")
    
    if not retrieved_docs:
        return "Je n'ai pas trouvé d'informations pertinentes pour répondre à cette question.", {}
    
    # 4. Générer la réponse
    llm = FastLLM(model_name=model_name)
    answer, metrics = llm.generate(query, retrieved_docs, max_tokens)
    
    # 5. Métriques de performance
    metrics["retrieval_time"] = retrieval_time
    metrics["total_time"] = time.time() - start_time
    
    return answer, metrics, retrieved_docs


def main(doc_path, query=None, process_only=False, base_dir="rag_data", 
         model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_tokens=512):
    """
    Fonction principale pouvant être appelée comme une fonction.
    
    Args:
        doc_path: Chemin du document à traiter
        query: Question à poser (None pour simplement traiter le document)
        process_only: Si True, traite seulement le document sans répondre à une question
        base_dir: Répertoire des données RAG
        model: Modèle LLM à utiliser
        max_tokens: Nombre max de tokens générés
    
    Returns:
        Si process_only=True: True si traitement réussi, False sinon
        Si query est fourni: Tuple (réponse, métriques, chunks récupérés)
    """
    # Traiter le document si demandé
    if process_only:
        success = process_doc(doc_path, base_dir)
        if success:
            print(f"Document traité avec succès: {doc_path}")
        else:
            print(f"Erreur lors du traitement du document: {doc_path}")
        return success
    
    # Répondre à une question
    if query:
        answer, metrics, retrieved_docs = answer_question(
            doc_path, query, base_dir, model, max_tokens
        )
        
        # Afficher les résultats
        print("\n" + "="*50)
        print(f"QUESTION: {query}")
        print("="*50)
        '''print("\nCHUNKS PERTINENTS:")
        print("-"*50)
        
        for i, doc in enumerate(retrieved_docs):
            print(f"Chunk {i+1} (Score: {doc['score']:.4f}):")
            source_info = ""
            if 'metadata' in doc:
                if 'doc_id' in doc['metadata']:
                    source_info += f"Document: {doc['metadata']['doc_id']}"
                if 'page' in doc['metadata']:
                    source_info += f", Page: {doc['metadata']['page']}"
            
            print(f"Source: {source_info}")
            print(f"{doc['text'][:300]}...")
            print()
        
        print("="*50)'''
        print("RÉPONSE:")
        print("-"*50)
        print(answer)
        print("="*50)
        
        print("\nPERFORMANCE:")
        print("-"*50)
        print(f"Temps de récupération: {metrics['retrieval_time']:.2f} secondes")
        print(f"Temps de génération: {metrics['generation_time']:.2f} secondes")
        print(f"Temps total: {metrics['total_time']:.2f} secondes")
        print("="*50)
        
        return answer, metrics , retrieved_docs