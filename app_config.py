import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# RAG Configuration
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.3
MAX_HISTORY_MESSAGES = 10

# Paths - these are relative to the app directory
DATA_DIR = "data_transformations"
CHROMA_DB_PATH = "chroma_db"