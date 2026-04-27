from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence
from urllib.parse import quote_plus

try:
    from .config import OLLAMA_MODEL, OLLAMA_URL
except ImportError:  # pragma: no cover - direct script execution
    from config import OLLAMA_MODEL, OLLAMA_URL

@lru_cache(maxsize=1)
def _load_learning_resources() -> dict:
    path = Path(__file__).resolve().parent / "learning_resources.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def get_learning_resources_for_skill(skill: str) -> List[str]:
    """
    Returns clean matching resources strictly from our curated taxonomy.
    Falls back to a Coursera search URL if no exact match found.
    """
    data = _load_learning_resources()
    # Exact match
    if skill in data:
        return data[skill]
    # Case-insensitive match
    skill_lower = skill.lower()
    for key, urls in data.items():
        if key.lower() == skill_lower:
            return urls
    # Partial match (e.g. "machine learning" matches "Machine Learning")
    for key, urls in data.items():
        if skill_lower in key.lower() or key.lower() in skill_lower:
            return urls
    # Fallback: Coursera search URL (real link, not placeholder)
    return [f"https://www.coursera.org/search?query={quote_plus(skill)}"]


def get_learning_resources_with_ai(
    skill: str,
    matched_skills: Sequence[str] | None = None,
    missing_skills: Sequence[str] | None = None,
    job_title: str = "",
) -> Dict[str, object]:
    """
    Ollama-backed learning guidance helper with local link fallbacks.
    """
    matched = [str(item).strip() for item in (matched_skills or []) if str(item).strip()]
    missing = [str(item).strip() for item in (missing_skills or []) if str(item).strip()]
    try:
        from .ollama_helper import LEARNING_SCHEMA, _call_ollama, _extract_json_payload
    except ImportError:  # pragma: no cover - direct script execution
        from ollama_helper import LEARNING_SCHEMA, _call_ollama, _extract_json_payload

    prompt = (
        "You are an AI career assistant.\n"
        f"Target skill: {skill}\n"
        f"Matched skills: {json.dumps(matched, ensure_ascii=True)}\n"
        f"Missing skills: {json.dumps(missing or [skill], ensure_ascii=True)}\n"
        f"Job title: {job_title or skill}\n"
        "Return ONLY valid JSON with one key: summary."
    )
    raw = _call_ollama(prompt, OLLAMA_URL, OLLAMA_MODEL, LEARNING_SCHEMA)
    summary = ""
    if raw:
        try:
            parsed = _extract_json_payload(raw)
            if isinstance(parsed, dict):
                summary = str(parsed.get("summary", "")).strip()
        except Exception:
            summary = ""

    return {
        "source": "ollama" if summary else "fallback",
        "skill": skill,
        "summary": summary or f"Focus on {skill} next and build one project that demonstrates it.",
        "resources": get_learning_resources_for_skill(skill),
    }
