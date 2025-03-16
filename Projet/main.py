import os
import subprocess
import sys

def run_streamlit_app(app_path):
    """
    Exécute une application Streamlit spécifiée par son chemin.
    
    Args:
        app_path (str): Chemin vers le fichier de l'application Streamlit à exécuter
    """
    try:
        # Construction de la commande pour exécuter l'application Streamlit
        command = f"streamlit run {app_path}"
        
        # Affichage de l'application qui va être lancée
        print(f"Lancement de l'application: {app_path}")
        
        # Exécution de la commande
        subprocess.run(command, shell=True, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de l'application Streamlit: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Une erreur s'est produite: {e}")
        sys.exit(1)

def main():
    # Obtenir le chemin du répertoire actuel où se trouve ce script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Définir le chemin vers le dossier Models
    models_dir = os.path.join(current_dir, "Models")
    
    # Définir les chemins vers les applications Streamlit
    app_fin_path = os.path.join(models_dir, "app_finance.py")
    #app_law_path = os.path.join(models_dir, "app2.py") etc
    
    
    if 1 == 1: #A changer
        run_streamlit_app(app_fin_path)


if __name__ == "__main__":
    main()