import argparse
import os
import sys


import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Ajouter le dossier parent au chemin pour pouvoir importer src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_indexer import DocumentIndexer
from src.retriever import DocumentRetriever
from src.qa_system import QASystem
from src.evaluator import RAGEvaluator
from src.config_loader import ConfigLoader

def main():
    parser = argparse.ArgumentParser(description="Système de recherche et question-réponse pour documents juridiques")
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")

    # Commande d'indexation
    index_parser = subparsers.add_parser("index", help="Indexer des documents")
    index_parser.add_argument("--rebuild", action="store_true", help="Reconstruire l'index à partir de zéro")

    # Commande de recherche
    search_parser = subparsers.add_parser("search", help="Rechercher dans les documents")
    search_parser.add_argument("query", type=str, help="Requête de recherche")
    search_parser.add_argument("--k", type=int, default=5, help="Nombre de résultats à retourner")

    # Commande de question-réponse
    qa_parser = subparsers.add_parser("qa", help="Poser une question au système")
    qa_parser.add_argument("question", type=str, help="Question à poser")

    # Commande d'évaluation
    eval_parser = subparsers.add_parser("evaluate", help="Évaluer le système RAG")
    eval_parser.add_argument("--questions", type=str, required=True, 
                            help="Chemin vers un fichier JSON contenant des paires question-réponse pour l'évaluation")

    args = parser.parse_args()
    
    # Charger la configuration
    config = ConfigLoader("./config.yaml").load_config()
    
    if args.command == "index":
        indexer = DocumentIndexer(config)
        indexer.index_documents(rebuild=args.rebuild)
        print("Indexation terminée avec succès!")
        
    elif args.command == "search":
        retriever = DocumentRetriever(config)
        results = retriever.search(args.query, k=args.k)
        
        print(f"\nRésultats pour la requête: '{args.query}'")
        print("-" * 50)
        
        for i, doc in enumerate(results):
            print(f"Document {i+1} (Score: {doc.metadata['score']:.4f}):")
            print(f"Source: {doc.metadata.get('source', 'Inconnue')}")
            if 'page' in doc.metadata:
                print(f"Page: {doc.metadata['page']}")
            print("-" * 30)
            print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
            print("\n" + "-" * 50)
            
    elif args.command == "qa":
        qa_system = QASystem(config)
        answer, sources = qa_system.answer_question(args.question)
        
        print(f"\nQuestion: {args.question}")
        print("-" * 50)
        print(f"Réponse:\n{answer}")
        
        print("\nSources:")
        for i, source in enumerate(sources):
            print(f"{i+1}. {source.metadata.get('source', 'Inconnue')}, " 
                 f"Page: {source.metadata.get('page', 'N/A')}")
            
    elif args.command == "evaluate":
        evaluator = RAGEvaluator(config)
        results = evaluator.evaluate_from_file(args.questions)
        
        print("\nRésultats de l'évaluation:")
        print("-" * 50)
        for metric, score in results.items():
            print(f"{metric}: {score}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()