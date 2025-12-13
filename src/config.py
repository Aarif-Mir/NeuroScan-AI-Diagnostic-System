"""
Configuration module for path management and settings.

All paths are relative to the project root directory for portability.
"""

from pathlib import Path
import os

# Get project root (parent of src/ directory)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Paths relative to project root (defaults)
_DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db"
_DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "rag_pdfs"
_DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "tf_cnn"
_DEFAULT_MODEL_PATH = _DEFAULT_MODELS_DIR / "resnet_model3.keras"
_DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# Allow override via environment variables, otherwise use defaults
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(_DEFAULT_CHROMA_DIR)))
PDF_DIR = Path(os.getenv("PDF_DIR", str(_DEFAULT_PDF_DIR)))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(_DEFAULT_MODEL_PATH)))
DATA_DIR = Path(os.getenv("DATA_DIR", str(_DEFAULT_DATA_DIR)))

# Other configuration
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

LABEL_MAP = {
    0: "Aneurysm",
    1: "Cancer",
    2: "Tumor",
}

# RAG Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
RAG_COLLECTION_NAME = "rag_collection"
RAG_K_RETRIEVE = int(os.getenv("RAG_K_RETRIEVE", "20"))

# LLM Configuration
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")

