import os
from dotenv import load_dotenv

load_dotenv()

# =====================================================
# Application
# =====================================================

APP_NAME = "Enterprise RAG Assistant"
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# =====================================================
# LLM Configuration
# =====================================================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv(
    "ANTHROPIC_MODEL",
    "claude-3-5-sonnet-latest"
)

# =====================================================
# Embeddings
# =====================================================

EMBEDDING_PROVIDER = os.getenv(
    "EMBEDDING_PROVIDER",
    "huggingface"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =====================================================
# Vector Database
# =====================================================

VECTOR_DB = os.getenv("VECTOR_DB", "chroma")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")

# =====================================================
# Retrieval
# =====================================================

TOP_K = int(os.getenv("TOP_K", 4))

# =====================================================
# Data
# =====================================================

DATA_DIRECTORY = os.getenv("DATA_DIRECTORY", "./data")