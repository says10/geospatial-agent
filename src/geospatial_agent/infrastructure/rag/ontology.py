import yaml
from typing import Dict, Any

def load_ontology(filepath: str) -> Dict[str, Any]:
    """Loads the tool ontology from a YAML file.

    Args:
        filepath: The path to the YAML file.

    Returns:
        A dictionary representing the ontology.
    """
    try:
        with open(filepath, "r") as f:
            ontology = yaml.safe_load(f)
        return ontology
    except FileNotFoundError:
        raise FileNotFoundError(f"Ontology file not found at: {filepath}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing ontology YAML file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading ontology: {e}")

if __name__ == '__main__':
    ontology = load_ontology("data/ontology.yaml")
    print(ontology)