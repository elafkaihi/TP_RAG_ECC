import json
from typing import Dict, Any, List
import os

from src.qa_system import QASystem
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class RAGEvaluator:
    """Classe pour évaluer les performances du système RAG."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise l'évaluateur.
        
        Args:
            config: Configuration du projet.
        """
        self.config = config
        self.qa_system = QASystem(config)
        
    def evaluate_answer(self, question: str, expected_answer: str) -> Dict[str, float]:
        """
        Évalue une réponse générée par rapport à la réponse attendue.
        
        Args:
            question: Question posée.
            expected_answer: Réponse attendue.
            
        Returns:
            Dictionnaire des scores d'évaluation.
        """
        # Générer la réponse avec le système QA
        answer, sources = self.qa_system.answer_question(question)
        
        # Calculer les métriques d'évaluation
        scores = {}
        
        # Score simple de longueur (ratio par rapport à la réponse attendue)
        len_ratio = len(answer) / max(1, len(expected_answer))
        scores["length_ratio"] = min(1.0, len_ratio) if len_ratio <= 1.0 else 1.0 / len_ratio
        
        # Score de présence des mots-clés
        expected_words = set(expected_answer.lower().split())
        answer_words = set(answer.lower().split())
        common_words = expected_words.intersection(answer_words)
        
        if expected_words:
            scores["keyword_overlap"] = len(common_words) / len(expected_words)
        else:
            scores["keyword_overlap"] = 0.0
        
        # Score moyen
        scores["average_score"] = sum(scores.values()) / len(scores)
        
        return scores
    
    def evaluate_from_file(self, questions_file: str) -> Dict[str, float]:
        """
        Évalue le système à partir d'un fichier de questions-réponses.
        
        Args:
            questions_file: Chemin vers un fichier JSON contenant des paires question-réponse.
            
        Returns:
            Dictionnaire des scores d'évaluation moyens.
        """
        try:
            with open(questions_file, "r", encoding="utf-8") as f:
                qa_pairs = json.load(f)
        except Exception as e:
            raise Exception(f"Erreur lors du chargement du fichier de questions: {e}")
        
        # Initialiser les scores totaux
        total_scores = {}
        
        # Évaluer chaque paire question-réponse
        for i, qa_pair in enumerate(qa_pairs):
            question = qa_pair.get("question")
            expected_answer = qa_pair.get("answer")
            
            if not question or not expected_answer:
                print(f"Paire question-réponse {i+1} incomplète. Ignorée.")
                continue
            
            print(f"Évaluation de la question {i+1}: {question[:50]}...")
            scores = self.evaluate_answer(question, expected_answer)
            
            # Ajouter les scores à la somme totale
            for metric, score in scores.items():
                total_scores[metric] = total_scores.get(metric, 0) + score
        
        # Calculer les moyennes
        num_questions = len(qa_pairs)
        avg_scores = {metric: score / num_questions for metric, score in total_scores.items()}
        
        return avg_scores