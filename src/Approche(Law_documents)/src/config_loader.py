import yaml
from typing import Dict, Any

class ConfigLoader:
    """Classe pour charger la configuration depuis un fichier YAML."""
    
    def __init__(self, config_path: str):
        """
        Initialise le chargeur de configuration.
        
        Args:
            config_path: Chemin vers le fichier de configuration YAML.
        """
        self.config_path = config_path
        
    def load_config(self) -> Dict[str, Any]:
        """
        Charge la configuration depuis le fichier YAML.
        
        Returns:
            Dictionnaire contenant la configuration.
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise Exception(f"Erreur lors du chargement de la configuration: {e}")