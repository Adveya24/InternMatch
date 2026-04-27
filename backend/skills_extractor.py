from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from rapidfuzz import fuzz, process

logger = logging.getLogger("internmatch.skills_extractor")

BASE_DIR = Path(__file__).resolve().parent
TAXONOMY_PATH = BASE_DIR / "skills_taxonomy.json"
SYNONYMS_PATH = BASE_DIR / "synonyms.json"


@dataclass
class ExtractedSkill:
    name: str
    confidence: float


def load_taxonomy() -> Dict[str, List[str]]:
    """Load the skills taxonomy from JSON."""
    with TAXONOMY_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def load_synonyms() -> Dict[str, str]:
    """Load the synonyms mapping from JSON."""
    with SYNONYMS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _all_canonical_skills(taxonomy: Dict[str, List[str]]) -> List[str]:
    skills: List[str] = []
    for group in taxonomy.values():
        skills.extend(group)
    return skills


def _normalize_token(token: str, synonyms: Dict[str, str]) -> str | None:
    """Map a raw token to a canonical skill via synonyms dict (case-insensitive)."""
    key = token.strip().lower()
    for syn, canonical in synonyms.items():
        if key == syn.lower():
            return canonical
    return None


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.strip().lower()).replace(r"\ ", r"\s+")
    pattern = rf"(?<!\w){escaped}(?!\w)"
    return re.search(pattern, text) is not None


def _build_ngrams(tokens: List[str], max_words: int) -> List[str]:
    ngrams: List[str] = []
    for size in range(1, max_words + 1):
        for idx in range(0, max(len(tokens) - size + 1, 0)):
            ngrams.append(" ".join(tokens[idx : idx + size]))
    return ngrams


def _fuzzy_match_token(
    token: str,
    candidates: List[str],
    short_threshold: int = 82,
    long_threshold: int = 72,
) -> Tuple[str | None, float]:
    """
    Fuzzy match a token against candidate skills.

    Returns:
        (best_match, confidence) where confidence is 0..1
    """
    token_clean = token.strip()
    if not token_clean:
        return None, 0.0

    scorer = fuzz.WRatio
    best = process.extractOne(token_clean, candidates, scorer=scorer)
    if not best:
        return None, 0.0

    match, score, _ = best
    threshold = short_threshold if len(token_clean) <= 10 else long_threshold
    if score < threshold:
        return None, score / 100.0
    return match, score / 100.0


def extract_skills_from_text(
    text: str,
    taxonomy: Dict[str, List[str]] | None = None,
    synonyms: Dict[str, str] | None = None,
) -> List[ExtractedSkill]:
    """
    Extract and normalize skills from raw CV text using:

    - Exact match against taxonomy.
    - Synonym lookup.
    - Fuzzy matching via rapidfuzz.

    Semantic embedding fallback will be added later; for now, we
    focus on robust keyword + fuzzy extraction.
    """
    if taxonomy is None:
        taxonomy = load_taxonomy()
    if synonyms is None:
        synonyms = load_synonyms()

    text_lower = re.sub(r"\s+", " ", text.lower())
    candidates = _all_canonical_skills(taxonomy)
    candidate_lookup = {skill.lower(): skill for skill in candidates}
    candidate_word_buckets: Dict[int, List[str]] = {}
    for skill in candidates:
        words = len(skill.split())
        candidate_word_buckets.setdefault(words, []).append(skill)

    synonym_lookup = {syn.lower(): canonical for syn, canonical in synonyms.items()}
    max_words = max(
        1,
        max((len(skill.split()) for skill in candidates), default=1),
        max((len(syn.split()) for syn in synonyms), default=1),
    )

    found: Dict[str, float] = {}

    # Step A: exact phrase presence in text.
    for skill in candidates:
        if _contains_phrase(text_lower, skill):
            found[skill] = max(found.get(skill, 0.0), 0.95)

    for syn, canonical in synonyms.items():
        if _contains_phrase(text_lower, syn):
            found[canonical] = max(found.get(canonical, 0.0), 0.92)

    # Step B & C: ngram-based synonym and fuzzy detection.
    raw_tokens = re.findall(r"[a-z0-9][a-z0-9\+\#\.\-/]*", text_lower)
    for phrase in _build_ngrams(raw_tokens, max_words):
        if not phrase or len(phrase) < 4:
            continue

        if phrase in candidate_lookup:
            canonical = candidate_lookup[phrase]
            found[canonical] = max(found.get(canonical, 0.0), 0.95)
            continue

        canonical = synonym_lookup.get(phrase)
        if canonical:
            found[canonical] = max(found.get(canonical, 0.0), 0.9)
            continue

        word_count = len(phrase.split())
        candidates_for_phrase = candidate_word_buckets.get(word_count, [])
        if not candidates_for_phrase:
            continue

        # Conservative fuzzy matching against skills with the same word count.
        threshold = 91 if word_count == 1 else 87
        if len(phrase) <= 6:
            threshold += 3
        match, conf = _fuzzy_match_token(
            phrase,
            candidates_for_phrase,
            short_threshold=threshold,
            long_threshold=threshold,
        )
        if match and conf > 0:
            found[match] = max(found.get(match, 0.0), min(conf, 0.88))

    extracted = [ExtractedSkill(name=skill, confidence=float(conf)) for skill, conf in sorted(found.items())]
    logger.info("Extracted %d skills from CV text", len(extracted))
    return extracted


def normalize_skills(
    raw_skills: List[str],
    taxonomy: Dict[str, List[str]] | None = None,
    synonyms: Dict[str, str] | None = None,
) -> List[str]:
    """
    Normalize a list of raw skill strings to canonical taxonomy names.

    Uses the same synonym and fuzzy pipeline as extract_skills_from_text,
    but operates on a list of tokens rather than free-form CV text.
    """
    if taxonomy is None:
        taxonomy = load_taxonomy()
    if synonyms is None:
        synonyms = load_synonyms()

    combined = "\n".join(raw_skills)
    return sorted({skill.name for skill in extract_skills_from_text(combined, taxonomy, synonyms)})

