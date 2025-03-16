import streamlit as st
import os
import shutil
import tempfile
from pathlib import Path
import time

# Importer le module RAG (assurez-vous que le fichier Python est dans le même dossier)
# Nous importons directement les fonctions depuis le module optimisé
from model_financial_report import process_doc, answer_question

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Assistant de document PDF", 
    page_icon="📑",
    layout="wide"
)

# Fonction pour obtenir le chemin du dossier RAG
def get_rag_dir():
    # Créer un dossier relatif à l'emplacement actuel
    current_dir = Path(__file__).parent
    rag_dir = current_dir / "rag_data"
    return str(rag_dir)

# Fonction pour nettoyer le dossier RAG
def clean_rag_data():
    rag_dir = get_rag_dir()
    if os.path.exists(rag_dir):
        shutil.rmtree(rag_dir)
        st.session_state["processed_file"] = None
        st.session_state["file_processed"] = False
        return True
    return False

# Initialiser les variables de session
if "processed_file" not in st.session_state:
    st.session_state["processed_file"] = None
    
if "file_processed" not in st.session_state:
    st.session_state["file_processed"] = False

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "doc_path" not in st.session_state:
    st.session_state["doc_path"] = ""

# Titre principal
st.title("Assistant de document PDF 📑")

# Sidebar pour le téléchargement et les options
with st.sidebar:
    st.header("Configuration")
    
    uploaded_file = st.file_uploader("Télécharger un document PDF", type=['pdf'])
    
    if uploaded_file is not None:
        # Si un nouveau fichier est téléchargé et différent du fichier précédent, nettoyer les données
        if st.session_state["processed_file"] != uploaded_file.name:
            if st.session_state["processed_file"] is not None:
                st.info("Nouveau fichier détecté. Réinitialisation du système...")
                clean_rag_data()
            
            st.session_state["processed_file"] = uploaded_file.name
            st.session_state["file_processed"] = False
        
        # Traiter le fichier s'il n'est pas encore traité
        if not st.session_state["file_processed"]:
            with st.spinner("Traitement du document en cours..."):
                # Créer un fichier temporaire pour sauvegarder le contenu téléchargé
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                
                # Sauvegarder le fichier téléchargé
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Traiter le document
                rag_dir = get_rag_dir()
                success = process_doc(temp_path, base_dir=rag_dir)
                
                if success:
                    st.success(f"Document '{uploaded_file.name}' traité avec succès!")
                    st.session_state["file_processed"] = True
                    st.session_state["doc_path"] = temp_path
                else:
                    st.error("Erreur lors du traitement du document.")
    
    # Options du modèle (visibles seulement si un fichier est traité)
    if st.session_state["file_processed"]:
        st.subheader("Options")
        
        max_tokens = st.slider(
            "Longueur maximum de la réponse",
            min_value=100,
            max_value=1000,
            value=512,
            step=50,
            help="Nombre maximum de tokens dans la réponse"
        )
        
        # Bouton pour effacer l'historique
        if st.button("Effacer l'historique"):
            st.session_state["chat_history"] = []
            st.success("Historique effacé")
        
        # Bouton pour réinitialiser complètement
        if st.button("Réinitialiser le système"):
            clean_rag_data()
            st.session_state["chat_history"] = []
            st.success("Système réinitialisé")

# Zone principale
if st.session_state["file_processed"]:
    st.subheader(f"Document actuel: {st.session_state['processed_file']}")
    
    # Zone de requête
    query = st.text_input("Posez une question sur le document:")
    
    if query:
        with st.spinner("Recherche de la réponse..."):
            start_time = time.time()
            
            # Obtenir la réponse - en utilisant uniquement les paramètres compatibles
            rag_dir = get_rag_dir()
            
            # Vérifier la signature de la fonction answer_question pour déterminer quels arguments passer
            import inspect
            sig = inspect.signature(answer_question)
            params = {}
            
            # Ajouter les paramètres obligatoires
            params["doc_path"] = st.session_state["doc_path"]
            params["query"] = query
            
            # Ajouter les paramètres optionnels si la fonction les accepte
            if "base_dir" in sig.parameters:
                params["base_dir"] = rag_dir
                
            if "max_tokens" in sig.parameters and 'max_tokens' in locals():
                params["max_tokens"] = max_tokens
            
            # Appeler la fonction avec les paramètres compatibles
            try:
                result = answer_question(**params)
                
                # Déterminer le format du résultat (tuple ou juste la réponse)
                if isinstance(result, tuple) and len(result) >= 3:
                    answer, metrics, retrieved_docs = result
                elif isinstance(result, tuple) and len(result) == 2:
                    answer, metrics = result
                    retrieved_docs = []
                else:
                    answer = result
                    metrics = {"total_time": time.time() - start_time}
                    retrieved_docs = []
                
                total_time = time.time() - start_time
                
                # Ajouter à l'historique
                st.session_state["chat_history"].append({
                    "query": query,
                    "answer": answer,
                    "time": total_time,
                    "chunks": retrieved_docs[:3] if retrieved_docs else []  # Limiter à 3 chunks pour l'affichage
                })
                
            except Exception as e:
                st.error(f"Erreur lors de la génération de la réponse: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    # Afficher l'historique des questions/réponses
    if st.session_state["chat_history"]:
        st.subheader("Historique des questions")
        
        for i, item in enumerate(reversed(st.session_state["chat_history"])):
            with st.expander(f"Q: {item['query']} ({item['time']:.2f}s)"):
                st.markdown("### Réponse:")
                st.write(item['answer'])
                
                if item.get('chunks'):
                    st.markdown("### Sources:")
                    for j, chunk in enumerate(item['chunks']):
                        if not isinstance(chunk, dict) or 'metadata' not in chunk:
                            continue
                            
                        source_info = f"Document: {chunk['metadata'].get('doc_id', 'Inconnu')}"
                        if 'page' in chunk['metadata']:
                            source_info += f", Page: {chunk['metadata']['page']}"
                        
                        st.markdown(f"**Source {j+1}** ({source_info}):")
                        if 'text' in chunk:
                            st.text(chunk['text'][:300] + "...")
else:
    st.info("👈 Veuillez télécharger un document PDF dans le panneau de gauche pour commencer.")
    st.markdown("""
    ### Comment utiliser cet assistant:
    1. Téléchargez un document PDF dans le panneau latéral
    2. Attendez que le traitement du document soit terminé
    3. Posez des questions sur le contenu du document
    4. Consultez l'historique des questions/réponses ci-dessous
    
    Pour changer de document, téléchargez simplement un nouveau fichier.
    Le système sera automatiquement réinitialisé.
    """)