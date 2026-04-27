from __future__ import annotations

import os
from pathlib import Path

# Force GPU usage
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


# Feature flags
USE_EMBEDDINGS: bool = _env_bool("USE_EMBEDDINGS", False)
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "phi3")
MAX_AI_INTERNSHIPS: int = int(os.getenv("MAX_AI_INTERNSHIPS", "50"))

# Paths
INTERN_CSV_PATH: str = os.getenv(
    "INTERN_CSV_PATH",
    str(BASE_DIR / "data" / "internships_10000_realorgs.csv"),
)
EMBEDDINGS_PATH: str = os.getenv(
    "EMBEDDINGS_PATH",
    str(BASE_DIR / "data"),
)
CACHE_DIR: str = os.getenv(
    "CACHE_DIR",
    str(BASE_DIR / "uploads" / "embeddings"),
)

# Provider settings
REMOTIVE_ENABLED: bool = _env_bool("REMOTIVE_ENABLED", True)
THEIRSTACK_ENABLED: bool = _env_bool("THEIRSTACK_ENABLED", True)
THEIRSTACK_API_KEY: str = os.getenv("THEIRSTACK_API_KEY", "")
RAPIDAPI_INTERNSHIPS_ENABLED: bool = _env_bool("RAPIDAPI_INTERNSHIPS_ENABLED", True)
ARBEITNOW_ENABLED: bool = _env_bool("ARBEITNOW_ENABLED", True)
GREENHOUSE_ENABLED: bool = _env_bool("GREENHOUSE_ENABLED", False)
LEVER_ENABLED: bool = _env_bool("LEVER_ENABLED", False)
INTERNSHALA_ENABLED: bool = _env_bool("INTERNSHALA_ENABLED", False)
