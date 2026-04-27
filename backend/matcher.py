from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

try:
    from .config import MAX_AI_INTERNSHIPS
    from .ollama_helper import OllamaCareerEngine
except ImportError:
    from config import MAX_AI_INTERNSHIPS
    from ollama_helper import OllamaCareerEngine

logger = logging.getLogger("internmatch.matcher")


def match_internships_with_ai(
    cv_text: str,
    departments: Sequence[str],
    internships: List[Dict[str, Any]],
    manual_skills: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """
    Ollama-first matcher entry point.
    """
    shortlist = list(internships)
    # Note: providers already filter by keywords — skip exact department re-filter
    # to avoid excluding valid results that use different dept labels (e.g. TheirStack)

    # Ensure skills_required_list exists
    for item in shortlist:
        if "skills_required_list" not in item:
            skills_str = item.get("skills_required", "") or ""
            item["skills_required_list"] = [s.strip() for s in skills_str.split(';') if s.strip()]

    # Sort and limit
    if shortlist and "_sort_date" in shortlist[0]:
        shortlist.sort(key=lambda x: x.get("_sort_date"), reverse=True)

    shortlist = shortlist[:MAX_AI_INTERNSHIPS]  # type: ignore[index]

    engine = OllamaCareerEngine()

    return engine.analyze_cv_and_match(
        cv_text=cv_text,
        selected_departments=departments,
        internships_subset=shortlist,
        manual_skills=manual_skills,
    )


def _build_explanation(title: object, matched: List[str], missing: List[str]) -> str:
    matched_str = ", ".join(matched) if matched else "no direct skill matches"
    missing_str = ", ".join(missing) if missing else "no obvious missing skills from the listing"
    return (
        f"For '{title}', you have {matched_str}; "
        f"you may want to strengthen {missing_str}."
    )
