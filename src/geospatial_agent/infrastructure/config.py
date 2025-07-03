
import os

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


class Settings:
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    # Add other settings here as needed


settings = Settings()