# cli.py
import argparse
import os 
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_system import create_rag_system, ScientificPaperRAG
from src.utils import format_context_for_display, visualize_retrieval_metrics, visualize_response_metrics


def main():
    parser = argparse.ArgumentParser(description="Scientific Paper RAG System CLI")
    parser.add_argument("command", type=str, choices=["index", "query", "evaluate", "evaluate_batch"], help="Command to execute")
    parser.add_argument("--directory", type=str, help="Directory for indexing", default=None)
    parser.add_argument("--query", type=str, help="Query string")
    parser.add_argument("--eval_file", type=str, help="File containing queries for batch evaluation")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Output file for evaluation results")

    args = parser.parse_args()

    rag_system = create_rag_system()

    if args.command == "index":
        rag_system.index_documents(args.directory)
        print("Indexing complete.")

    elif args.command == "query":
        if not args.query:
            print("Error: --query argument is required for 'query' command.")
            return

        answer, docs = rag_system.get_answer(args.query)
        print(f"\nQuestion: {args.query}")
        print(f"\nAnswer: {answer}")
        print("\nRetrieved Documents:")
        print(format_context_for_display(docs))


    elif args.command == "evaluate":
        if not args.query:
            print("Error: --query argument is required for 'evaluate' command.")
            return

        results = rag_system.evaluate_query(args.query)
        print("\nEvaluation Results:")
        for key, value in results.items():
            print(f"{key}: {value}")

        if args.output_file:
            rag_system.get_evaluator().save_evaluation_results(results, args.output_file)

    elif args.command == "evaluate_batch":
        if not args.eval_file:
            print("Error: --eval_file argument is required for 'evaluate_batch' command")
            return

        if not os.path.exists(args.eval_file):
            print(f"Error: Evaluation file not found: {args.eval_file}")
            return
            
        try:
            with open(args.eval_file, "r") as f:
                queries = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print("An error occurred:", e) #improved error handling
            return


        all_results = []
        for query in queries:
            print(f"Evaluating query: {query}")
            try:
                results = rag_system.evaluate_query(query)
                all_results.append(results)
            except Exception as e:
                print("An error occurred:", e) #improved error handling
                continue #continue with the next evaluation

        print("\nBatch Evaluation Complete.")

        if args.output_file:
            rag_system.get_evaluator().save_evaluation_results(all_results, args.output_file)

        # Visualization
        if all_results:
            visualize_retrieval_metrics([r for r in all_results if "retrieval" in r])
            visualize_response_metrics([r for r in all_results if "response" in r])


if __name__ == "__main__":
    main()