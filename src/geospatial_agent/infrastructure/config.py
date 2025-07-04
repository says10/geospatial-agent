import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the application."""

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")

    ONTOLOGY_FILE_PATH = os.getenv("ONTOLOGY_FILE_PATH", "data/ontology.yaml")
    # Add other configuration variables here

if __name__ == '__main__':
    print(f"OpenAI API Key (first 4 chars): {Config.OPENAI_API_KEY[:4]}")
    print(f"Ontology file path: {Config.ONTOLOGY_FILE_PATH}")