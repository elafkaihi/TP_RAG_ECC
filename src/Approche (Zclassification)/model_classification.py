import os
import chromadb
import pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.optim as optim

class DocumentIndexer:
    def __init__(self, data_dir="data/"):
        self.data_dir = data_dir
        self.vector_store = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.vector_store.get_or_create_collection(name="documents")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = []

    def load_and_index_documents(self):
        """Charge et indexe tous les documents PDF du dossier spécifié avec gestion des erreurs."""
        with ThreadPoolExecutor() as executor:
            futures = []
            for file in os.listdir(self.data_dir):
                if file.endswith(".pdf"):
                    file_path = os.path.join(self.data_dir, file)
                    futures.append(executor.submit(self.process_pdf, file_path, file))
            
            # Attente de la fin de tous les threads
            for future in futures:
                future.result()

    def process_pdf(self, file_path, filename):
        """Traite chaque PDF de manière sécurisée."""
        try:
            text = self.extract_text_from_pdf(file_path)
            embedding = self.model.encode(text)
            self.collection.add(embeddings=[embedding.tolist()], metadatas=[{"filename": filename}], ids=[filename])
            self.documents.append((filename, text))
        except Exception as e:
            print(f"Erreur lors de l'extraction du fichier {filename}: {e}")

    @staticmethod
    def extract_text_from_pdf(file_path):
        """Extrait le texte brut d'un fichier PDF avec gestion des erreurs."""
        try:
            reader = PdfReader(file_path)
            return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except Exception as e:
            print(f"Erreur lors de l'extraction du texte du fichier {file_path}: {e}")
            return ""

class DocumentClassifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        self.categories = ["financial_report", "law_articles", "scientific_paper"]
        self.load_training_data()

    def load_training_data(self):
        """Charge les documents indexés et prépare les données pour l'entraînement."""
        indexer = DocumentIndexer()
        indexer.load_and_index_documents()
        
        texts, labels = [], []
        for doc, text in indexer.documents:
            category = self.detect_category(doc)
            texts.append(text)
            labels.append(self.categories.index(category))

        if texts:
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            labels = torch.tensor(labels)
            train_dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)

            # Entraînement du modèle
            train_args = TrainingArguments(
                output_dir='./results',          
                num_train_epochs=3,             
                per_device_train_batch_size=8,  
                logging_dir='./logs',            
            )

            trainer = Trainer(
                model=self.model,              
                args=train_args,              
                train_dataset=train_dataset     
            )
            trainer.train()
            self.save_model()

    def detect_category(self, filename):
        """Déduit la catégorie du document selon son nom."""
        if "finance" in filename.lower():
            return "financial_report"
        elif "law" in filename.lower():
            return "law_articles"
        else:
            return "scientific_paper"

    def classify_document(self, text):
        """Classifie un texte donné dans une des catégories définies avec BERT."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        return self.categories[predicted_class]
    
    def save_model(self):
        """Sauvegarde le modèle BERT après l'entraînement."""
        self.model.save_pretrained("bert_classifier_model")
        self.tokenizer.save_pretrained("bert_classifier_model")

    def load_model(self):
        """Charge le modèle et le tokenizer préalablement sauvegardés."""
        self.model = BertForSequenceClassification.from_pretrained("bert_classifier_model")
        self.tokenizer = BertTokenizer.from_pretrained("bert_classifier_model")

class QueryProcessor:
    def __init__(self):
        self.indexer = DocumentIndexer()
        self.classifier = DocumentClassifier()

    def search_documents(self, query):
        """Recherche les documents les plus pertinents pour une requête."""
        embedding = self.indexer.model.encode(query).tolist()
        results = self.indexer.collection.query(query_embeddings=[embedding], n_results=3)
        return [(res["filename"], res["distance"]) for res in results["metadatas"][0]]

    def classify_query(self, query):
        """Classifie une requête utilisateur dans une des catégories définies."""
        return self.classifier.classify_document(query)

if __name__ == "__main__":
    classifier = DocumentClassifier()
    test_text = "This document talks about financial markets and investments."
    print(f"Catégorie prédite : {classifier.classify_document(test_text)}")
