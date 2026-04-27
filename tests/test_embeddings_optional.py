from __future__ import annotations

import os

import importlib
import pytest

from backend import config


@pytest.mark.skipif(os.getenv("USE_EMBEDDINGS", "0").lower() not in {"1", "true", "yes", "on"}, reason="Embeddings disabled")
def test_embeddings_flag_present() -> None:
    importlib.reload(config)
    assert config.USE_EMBEDDINGS is True
