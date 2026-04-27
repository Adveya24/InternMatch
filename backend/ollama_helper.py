from __future__ import annotations

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import logging
import re
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import requests

# Import config/helpers in a way that works both when executed as a package
# (`python -m backend.app`) and when tools execute modules with altered sys.path.
try:
    from .config import MAX_AI_INTERNSHIPS, OLLAMA_MODEL, OLLAMA_URL
except ModuleNotFoundError:  # pragma: no cover
    from backend.config import MAX_AI_INTERNSHIPS, OLLAMA_MODEL, OLLAMA_URL

try:
    from .skills_extractor import extract_skills_from_text, load_synonyms, load_taxonomy
except ModuleNotFoundError:  # pragma: no cover
    from backend.skills_extractor import extract_skills_from_text, load_synonyms, load_taxonomy

try:
    from .providers import department_relevance_score
except ModuleNotFoundError:  # pragma: no cover
    from backend.providers import department_relevance_score

logger = logging.getLogger("internmatch.ollama")

MAX_AI_MATCHES = 6
# Only used to filter out pure noise matches when job requirements are known.
# Set to 0 so we never end up with empty results due to provider schema gaps.
MIN_SKILL_MATCH_RATIO = 0.0
AI_PROFILE_SUMMARY_UNAVAILABLE = "AI profile summary unavailable right now."
AI_EXPLANATION_UNAVAILABLE = "AI explanation unavailable for this role right now."
AI_LEARNING_UNAVAILABLE = "AI learning suggestions unavailable right now."
AI_ALIGNMENT_UNAVAILABLE = "AI fit summary unavailable for this role right now."

PROFILE_ANALYSIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "detected_skills": {"type": "array", "items": {"type": "string"}},
        "interests": {"type": "array", "items": {"type": "string"}},
        "cv_rating": {"type": "integer"},
        "recommended_search_keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "detected_skills",
        "interests",
        "cv_rating",
        "recommended_search_keywords",
    ],
}
PROFILE_SUMMARY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "experience_summary": {"type": "string"},
        "education_summary": {"type": "string"},
        "projects_summary": {"type": "string"}
    },
    "required": ["summary", "experience_summary", "education_summary", "projects_summary"],
}
MATCHES_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "company": {"type": "string"},
            "match_score": {"type": "integer"},
            "matched_skills": {"type": "array", "items": {"type": "string"}},
            "missing_skills": {"type": "array", "items": {"type": "string"}},
            "competitiveness": {"type": "string"},
        },
        "required": [
            "title",
            "company",
            "match_score",
            "matched_skills",
            "missing_skills",
            "competitiveness",
        ],
    },
}
MATCH_REWRITE_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "company": {"type": "string"},
            "explanation": {"type": "string"},
            "suggestions": {"type": "array", "items": {"type": "string"}},
            "experience_alignment": {"type": "string"}
        },
        "required": ["title", "company", "explanation", "suggestions", "experience_alignment"]
    }
}
LEARNING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


def call_ollama(prompt: str) -> str:
    return _call_ollama(prompt, OLLAMA_URL, OLLAMA_MODEL)


def _call_ollama(prompt: str, url: str, model: str, response_format: Any | None = None) -> str:
    logger.info("Calling Ollama: %s [%s]", url, model)
    try:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_gpu": -1
            },
        }
        if response_format is not None:
            # Phi3 doesn't fully support structural JSON schema formats, so strictly use "json" mode
            payload["format"] = "json"

        response = requests.post(
            f"{url.rstrip('/')}/api/generate",
            json=payload,
            timeout=120,
            proxies={'http': None, 'https': None}
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:  # pragma: no cover - network/process variability
        print(f"\n[!!!] OLLAMA CONNECTION ERROR: {e}")
        logger.error("Ollama error: %s", e)
        return ""


def _strip_code_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()

def _extract_json_payload(text: str) -> Any:
    cleaned = _strip_code_fence(text)
    if not cleaned:
        raise ValueError("Empty Ollama response")

    # Highly robust JSON extractor for smaller models like Phi3
    # Find the outermost curly braces or brackets
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    
    start_arr = cleaned.find("[")
    end_arr = cleaned.rfind("]")
    
    # Check if it looks more like an object or an array
    if start_idx != -1 and end_idx != -1 and (start_arr == -1 or start_idx < start_arr):
        candidate = cleaned[start_idx:end_idx+1]
    elif start_arr != -1 and end_arr != -1:
        candidate = cleaned[start_arr:end_arr+1]
    else:
        candidate = cleaned

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to decode JSON from Ollama output: {exc}")


def _dedupe_keep_order(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    output: List[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _extract_local_cv_sections(cv_text: str) -> Dict[str, str]:
    text = re.sub(r"\s+", " ", cv_text or "").strip()
    lower = text.casefold()

    def snippet(keywords: Sequence[str], fallback_words: int = 80) -> str:
        for keyword in keywords:
            position = lower.find(keyword)
            if position >= 0:
                start = max(0, position - 20)
                end = min(len(text), position + 350)
                return text[start:end].strip() + "..."
        return " ".join(text.split()[:fallback_words]).strip() + "..."

    return {
        "skills_section": snippet(["skills", "technical skills", "tools"], 60),
        "experience_section": snippet(["experience", "internship", "employment", "work experience"], 90),
        "projects_section": snippet(["projects", "project", "portfolio", "built"], 90),
        "education_section": snippet(["education", "university", "college", "bachelor", "master", "cgpa", "gpa"], 70),
    }


def _coerce_skill_payload(skills: Sequence[str], confidence: float = 0.9) -> List[Dict[str, Any]]:
    return [{"name": skill, "confidence": confidence} for skill in _dedupe_keep_order(skills)]


def _ai_profile_summary_payload() -> Dict[str, str]:
    return {
        "summary": AI_PROFILE_SUMMARY_UNAVAILABLE,
        "experience_summary": AI_PROFILE_SUMMARY_UNAVAILABLE,
        "education_summary": AI_PROFILE_SUMMARY_UNAVAILABLE,
        "projects_summary": AI_PROFILE_SUMMARY_UNAVAILABLE,
    }


def _ai_match_text_payload() -> Dict[str, Any]:
    return {
        "explanation": AI_EXPLANATION_UNAVAILABLE,
        "suggestions": [AI_LEARNING_UNAVAILABLE],
        "experience_alignment": AI_ALIGNMENT_UNAVAILABLE,
    }


def _clean_prose_line(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = cleaned.strip("-* \t")
    cleaned = re.sub(r"^[A-Z][A-Z _-]+:\s*", "", cleaned)
    return cleaned


def _is_unavailable_text(text: str, unavailable: str) -> bool:
    normalized = _clean_prose_line(text).casefold()
    return not normalized or normalized == unavailable.casefold()


def _looks_like_heuristic_explanation(text: str) -> bool:
    normalized = str(text or "").strip().casefold()
    if not normalized:
        return True
    if "%" in normalized:
        return True
    return bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:weak|medium|strong)\s+match\b", normalized))


def _looks_like_match_template(text: str) -> bool:
    normalized = str(text or "").strip().casefold()
    if not normalized:
        return True
    return bool(re.search(r"\b(?:weak|medium|strong)\s+match\b", normalized))


def _lookup_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _json_safe_internship_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in record.items():
        if key == "_sort_date":
            continue
        if value is None:
            cleaned[key] = ""
        elif hasattr(value, "isoformat"):
            cleaned[key] = value.isoformat()
        else:
            cleaned[key] = value
    return cleaned


def _find_internship_record(shortlist: List[Dict[str, Any]], role: str, company: str) -> Mapping[str, Any] | None:
    role_key = _lookup_key(role.strip())
    company_key = _lookup_key(company.strip())
    role_only_matches: List[Dict[str, Any]] = []

    for row in shortlist:
        row_role = _lookup_key(str(row.get("title", "")).strip())
        row_company = _lookup_key(str(row.get("company", "")).strip())
        row_dict = _json_safe_internship_record(row)
        if row_role == role_key and row_company == company_key:
            return row_dict
        if row_role == role_key:
            role_only_matches.append(row_dict)

    if len(role_only_matches) == 1:
        return role_only_matches[0]
    return None


def _make_shortlist_records(internships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in internships[:MAX_AI_INTERNSHIPS]:
        records.append(
            {
                "title": str(row.get("title", "")),
                "company": str(row.get("company", "")),
                "location": str(row.get("location", "")),
                "source": str(row.get("source", "")),
                "requirements": list(row.get("skills_required_list", []))
                or [x.strip() for x in str(row.get("requirements", row.get("skills_required", ""))).split(";") if x.strip()],
                "description": str(row.get("description", ""))[:320],
                "apply_url": str(row.get("apply_url", "")),
            }
        )
    return records


def _build_learning_resources(skills: Sequence[str]) -> Dict[str, List[str]]:
    try:
        from .learning_resources import get_learning_resources_for_skill
    except ImportError:  # pragma: no cover - direct script execution
        from learning_resources import get_learning_resources_for_skill

    resources: Dict[str, List[str]] = {}
    for skill in _dedupe_keep_order(skills):
        urls = get_learning_resources_for_skill(skill)
        if urls:
            resources[skill] = urls
    return resources


def _extract_labeled_fields(raw: str, labels: Sequence[str]) -> Dict[str, str]:
    cleaned = _strip_code_fence(raw)
    output: Dict[str, str] = {}
    upper_cleaned = cleaned.upper()
    for index, label in enumerate(labels):
        start_marker = f"{label.upper()}:"
        start = upper_cleaned.find(start_marker)
        if start < 0:
            continue
        start += len(start_marker)
        end = len(cleaned)
        for next_label in labels[index + 1:]:
            next_marker = f"{next_label.upper()}:"
            next_pos = upper_cleaned.find(next_marker, start)
            if next_pos >= 0:
                end = next_pos
                break
        output[label.casefold()] = _clean_prose_line(cleaned[start:end])
    return output


def _coerce_suggestions_list(raw_suggestions: Any) -> List[str]:
    if isinstance(raw_suggestions, Sequence) and not isinstance(raw_suggestions, (str, bytes)):
        suggestions = _dedupe_keep_order([_clean_prose_line(str(item)) for item in raw_suggestions])
        return [item for item in suggestions if item]

    text = _clean_prose_line(str(raw_suggestions or ""))
    if not text:
        return []

    parts = re.split(r"[;\n]|(?:\s{2,})", text)
    suggestions = _dedupe_keep_order([_clean_prose_line(part) for part in parts if _clean_prose_line(part)])
    if suggestions:
        return suggestions[:3]
    return [text]


def _summarize_cv_section(cv_text: str, section_name: str, section_text: str, url: str, model: str) -> str:
    if not _clean_prose_line(section_text):
        return AI_PROFILE_SUMMARY_UNAVAILABLE

    prompt = (
        f"Summarize the candidate's {section_name} in 1-2 natural sentences.\n"
        "Rules:\n"
        "- Write clean human prose.\n"
        "- No bullets, no JSON, no labels.\n"
        "- Do not repeat the candidate's contact information.\n"
        "- Do not mention percentage match language.\n"
        f"CV excerpt:\n{cv_text[:1200]}\n"
        f"{section_name.title()} details:\n{section_text[:1200]}\n"
        f"{section_name.title()} summary:"
    )
    raw = _call_ollama(prompt, url, model)
    summary = _clean_prose_line(raw)
    if not summary or _looks_like_match_template(summary):
        return AI_PROFILE_SUMMARY_UNAVAILABLE
    return summary


def _rewrite_profile_summaries(cv_text: str, url: str, model: str) -> Dict[str, str]:
    prompt = (
        "Rewrite this CV into clean, human profile summaries. Return ONLY valid JSON matching the schema.\n"
        "Rules:\n"
        "- summary: 1 sentence about the candidate overall.\n"
        "- experience_summary: 1-2 natural sentences about work or internship experience.\n"
        "- education_summary: 1-2 natural sentences about academic background.\n"
        "- projects_summary: 1-2 natural sentences about practical projects.\n"
        "- Do not copy raw CV fragments. Do not output bullets. Do not mention percentages unless they are clearly academic scores.\n"
        f"Schema: {json.dumps(PROFILE_SUMMARY_SCHEMA, ensure_ascii=True)}\n"
        f"CV:\n{cv_text[:4000]}"
    )

    raw = _call_ollama(prompt, url, model, PROFILE_SUMMARY_SCHEMA)
    if raw:
        try:
            parsed = _extract_json_payload(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Expected JSON object from Ollama profile summary response")
            summary = _clean_prose_line(parsed.get("summary", ""))
            experience = _clean_prose_line(parsed.get("experience_summary", ""))
            education = _clean_prose_line(parsed.get("education_summary", ""))
            projects = _clean_prose_line(parsed.get("projects_summary", ""))
            if all([summary, experience, education, projects]):
                return {
                    "summary": summary,
                    "experience_summary": experience,
                    "education_summary": education,
                    "projects_summary": projects,
                }
            raise ValueError("Incomplete profile summary response")
        except Exception as exc:
            logger.warning("Failed to parse Ollama profile summary response: %s", exc)

    labeled_prompt = (
        "Summarize this CV into exactly four labeled lines.\n"
        "Format:\n"
        "SUMMARY: ...\n"
        "EXPERIENCE: ...\n"
        "EDUCATION: ...\n"
        "PROJECTS: ...\n"
        "Rules:\n"
        "- Each line must be 1-2 natural sentences.\n"
        "- No bullets. No JSON. No markdown.\n"
        "- Do not copy raw CV fragments.\n"
        f"CV:\n{cv_text[:3500]}"
    )
    labeled_raw = _call_ollama(labeled_prompt, url, model)
    if labeled_raw:
        labeled = _extract_labeled_fields(labeled_raw, ["SUMMARY", "EXPERIENCE", "EDUCATION", "PROJECTS"])
        summary = labeled.get("summary", "")
        experience = labeled.get("experience", "")
        education = labeled.get("education", "")
        projects = labeled.get("projects", "")
        if all([summary, experience, education, projects]):
            return {
                "summary": summary,
                "experience_summary": experience,
                "education_summary": education,
                "projects_summary": projects,
            }

    sections = _extract_local_cv_sections(cv_text)
    experience = _summarize_cv_section(cv_text, "experience", sections.get("experience_section", ""), url, model)
    education = _summarize_cv_section(cv_text, "education", sections.get("education_section", ""), url, model)
    projects = _summarize_cv_section(cv_text, "projects", sections.get("projects_section", ""), url, model)
    summary = _summarize_cv_section(cv_text, "overall profile", cv_text[:1800], url, model)
    return {
        "summary": summary,
        "experience_summary": experience,
        "education_summary": education,
        "projects_summary": projects,
    }


def _rewrite_single_match_text(
    cv_text: str,
    detected_skills: Sequence[str],
    match_item: Mapping[str, Any],
    internship: Mapping[str, Any],
    url: str,
    model: str,
) -> Dict[str, Any]:
    prompt = (
        "Write a natural internship fit summary using exactly three labeled lines.\n"
        "Format:\n"
        "EXPLANATION: ...\n"
        "SUGGESTIONS: ...\n"
        "ALIGNMENT: ...\n"
        "Rules:\n"
        "- explanation: 1-2 human sentences about why the role fits.\n"
        "- suggestions: 1-3 short learning next steps, separated by semicolons.\n"
        "- alignment: 1 sentence about how the candidate's background aligns.\n"
        "- Do not mention weak match / medium match / strong match.\n"
        "- Do not repeat the numeric score.\n"
        f"Candidate skills: {json.dumps(list(detected_skills), ensure_ascii=True)}\n"
        f"CV excerpt: {cv_text[:1800]}\n"
        f"Internship title: {internship.get('title', '')}\n"
        f"Company: {internship.get('company', '')}\n"
        f"Description: {str(internship.get('description', ''))[:500]}\n"
        f"Requirements: {json.dumps(internship.get('requirements', internship.get('skills_required_list', [])), ensure_ascii=True)}\n"
        f"Matched skills: {json.dumps(list(match_item.get('matched_skills', [])), ensure_ascii=True)}\n"
        f"Missing skills: {json.dumps(list(match_item.get('missing_skills', [])), ensure_ascii=True)}"
    )
    raw = _call_ollama(prompt, url, model)
    if not raw:
        return _ai_match_text_payload()

    labeled = _extract_labeled_fields(raw, ["EXPLANATION", "SUGGESTIONS", "ALIGNMENT"])
    explanation = labeled.get("explanation", "")
    if _looks_like_heuristic_explanation(explanation):
        explanation = AI_EXPLANATION_UNAVAILABLE

    suggestions = _coerce_suggestions_list(labeled.get("suggestions", ""))
    if not suggestions:
        suggestions = [AI_LEARNING_UNAVAILABLE]

    experience_alignment = labeled.get("alignment", "") or AI_ALIGNMENT_UNAVAILABLE
    return {
        "explanation": explanation,
        "suggestions": suggestions,
        "experience_alignment": experience_alignment,
    }


def _rewrite_match_texts(
    cv_text: str,
    detected_skills: Sequence[str],
    ranked_matches: Sequence[Mapping[str, Any]],
    shortlist: List[Dict[str, Any]],
    url: str,
    model: str,
) -> Dict[tuple[str, str], Dict[str, Any]]:
    shortlist_payload: List[Dict[str, Any]] = []
    for item in ranked_matches:
        title = str(item.get("title", item.get("role", ""))).strip()
        company = str(item.get("company", "")).strip()
        internship = _find_internship_record(shortlist, title, company)
        if internship is None:
            continue
        shortlist_payload.append(
            {
                "title": title,
                "company": company,
                "match_score": item.get("match_score", 0),
                "matched_skills": list(item.get("matched_skills", [])),
                "missing_skills": list(item.get("missing_skills", [])),
                "description": str(internship.get("description", ""))[:500],
                "requirements": internship.get("requirements", internship.get("skills_required_list", [])),
            }
        )

    if not shortlist_payload:
        return {}

    prompt = (
        "Rewrite these internship results into natural human explanations. Return ONLY valid JSON matching the schema.\n"
        "Rules:\n"
        "- explanation: 1-2 natural sentences about why the role fits this candidate.\n"
        "- suggestions: 1-3 short learning next steps in plain language.\n"
        "- experience_alignment: 1 natural sentence about how the candidate's background aligns.\n"
        "- Do not mention 'weak match', 'medium match', 'strong match', or repeat the numeric score.\n"
        "- Do not use bullet points inside strings.\n"
        f"Schema: {json.dumps(MATCH_REWRITE_SCHEMA, ensure_ascii=True)}\n"
        f"Candidate skills: {json.dumps(list(detected_skills), ensure_ascii=True)}\n"
        f"CV excerpt: {cv_text[:2500]}\n"
        f"Ranked internships: {json.dumps(shortlist_payload, ensure_ascii=True)}"
    )

    rewritten: Dict[tuple[str, str], Dict[str, Any]] = {}
    raw = _call_ollama(prompt, url, model, MATCH_REWRITE_SCHEMA)
    if raw:
        try:
            parsed = _extract_json_payload(raw)
            if not isinstance(parsed, list):
                raise ValueError("Expected JSON array from Ollama match rewrite response")
            for item in parsed:
                if not isinstance(item, MutableMapping):
                    continue
                title = str(item.get("title", "")).strip()
                company = str(item.get("company", "")).strip()
                if not title or not company:
                    continue
                explanation = _clean_prose_line(item.get("explanation", ""))
                if _looks_like_heuristic_explanation(explanation):
                    explanation = AI_EXPLANATION_UNAVAILABLE
                suggestions = _coerce_suggestions_list(item.get("suggestions", [])) or [AI_LEARNING_UNAVAILABLE]
                experience_alignment = _clean_prose_line(item.get("experience_alignment", "")) or AI_ALIGNMENT_UNAVAILABLE
                rewritten[(_lookup_key(title), _lookup_key(company))] = {
                    "explanation": explanation,
                    "suggestions": suggestions,
                    "experience_alignment": experience_alignment,
                }
        except Exception as exc:
            logger.warning("Failed to parse Ollama match rewrite response: %s", exc)

    for item in ranked_matches:
        title = str(item.get("title", item.get("role", ""))).strip()
        company = str(item.get("company", "")).strip()
        if not title or not company:
            continue
        key = (_lookup_key(title), _lookup_key(company))
        existing = rewritten.get(key, {})
        has_good_explanation = not _is_unavailable_text(existing.get("explanation", ""), AI_EXPLANATION_UNAVAILABLE)
        has_good_alignment = not _is_unavailable_text(existing.get("experience_alignment", ""), AI_ALIGNMENT_UNAVAILABLE)
        has_good_suggestions = existing.get("suggestions") and existing.get("suggestions") != [AI_LEARNING_UNAVAILABLE]
        if has_good_explanation and has_good_alignment and has_good_suggestions:
            continue

        internship = _find_internship_record(shortlist, title, company)
        if internship is None:
            continue
        single = _rewrite_single_match_text(cv_text, detected_skills, item, internship, url, model)
        if single:
            merged = {
                "explanation": existing.get("explanation", ""),
                "suggestions": existing.get("suggestions", []),
                "experience_alignment": existing.get("experience_alignment", ""),
            }
            if not has_good_explanation and not _is_unavailable_text(single.get("explanation", ""), AI_EXPLANATION_UNAVAILABLE):
                merged["explanation"] = single["explanation"]
            elif not merged["explanation"]:
                merged["explanation"] = single["explanation"]

            if not has_good_alignment and not _is_unavailable_text(single.get("experience_alignment", ""), AI_ALIGNMENT_UNAVAILABLE):
                merged["experience_alignment"] = single["experience_alignment"]
            elif not merged["experience_alignment"]:
                merged["experience_alignment"] = single["experience_alignment"]

            if not has_good_suggestions and single.get("suggestions"):
                merged["suggestions"] = single["suggestions"]
            elif not merged["suggestions"]:
                merged["suggestions"] = single["suggestions"]
            rewritten[key] = merged

    return rewritten


def _holistic_match_score(
    internship_record: Mapping[str, Any],
    required_skills: Sequence[str],
    matched_skills: Sequence[str],
    selected_departments: Sequence[str] | None
) -> float:
    """Compute a holistic score bounded to 5..100 based on skills, dept relevance, and internship fit."""
    skill_ratio = len(list(matched_skills)) / max(len(list(required_skills)), 1)
    
    title_l = str(internship_record.get("title", "")).casefold()
    
    dept_match = 1.0
    if selected_departments:
        dept_match = department_relevance_score(dict(internship_record), [str(dep) for dep in selected_departments if str(dep).strip()])
                
    is_true_intern = "intern" in title_l or "student" in title_l or "trainee" in title_l or "stage" in title_l
    
    # Favor explicit department relevance more heavily so selected lanes stay coherent.
    raw_score = (skill_ratio * 55.0) + (dept_match * 30.0) + (15.0 if is_true_intern else 0.0)

    # If a role barely matches the chosen department, keep it from floating to the top.
    if selected_departments and dept_match < 0.18:
        raw_score *= 0.45
    
    # Cap at 90.0% to account for soft skills / cultural match
    score = max(5.0, min(90.0, 5.0 + raw_score * (85.0 / 100.0)))
    return round(score, 1)


def _infer_required_skills_from_title(title: str) -> List[str]:
    title_lower = str(title or "").casefold()
    required: List[str] = []
    if "software" in title_lower or "developer" in title_lower or "engineer" in title_lower:
        required.extend(["Software Engineering", "Python", "Algorithms"])
    if "design" in title_lower or "graphic" in title_lower or "ui" in title_lower:
        required.extend(["Graphic Design", "UI/UX", "Figma"])
    if "data" in title_lower or "analyst" in title_lower:
        required.extend(["Data Analysis", "SQL", "Python"])
    if "product" in title_lower or "pm" in title_lower.split():
        required.extend(["Product Management", "Agile", "Communication"])
    if "marketing" in title_lower or "seo" in title_lower:
        required.extend(["Marketing", "SEO", "Communication"])
    return _dedupe_keep_order(required)


def _derive_required_skills(internship_record: Mapping[str, Any], selected_departments: Sequence[str] | None) -> List[str]:
    """
    Ensure we always have a non-empty required skill list.
    Sources (in priority order):
    1) provider tags (`skills_required_list`)
    2) raw requirements strings (`requirements` / `skills_required`)
    3) taxonomy extraction from title+description
    4) title heuristics
    5) selected departments (last resort)
    """
    required_skills = [
        str(skill).strip()
        for skill in (internship_record.get("skills_required_list", []) or [])
        if str(skill).strip()
    ]

    if not required_skills:
        raw = str(internship_record.get("requirements", "") or internship_record.get("skills_required", "") or "")
        required_skills = [x.strip() for x in re.split(r"[;,/|]+", raw) if x.strip()]

    if not required_skills:
        job_text = f"{internship_record.get('title', '')} {internship_record.get('description', '')}"
        extracted_job_skills = extract_skills_from_text(job_text, load_taxonomy(), load_synonyms())
        required_skills = _dedupe_keep_order([s.name for s in extracted_job_skills])

    if not required_skills:
        required_skills = _infer_required_skills_from_title(str(internship_record.get("title", "")))

    if not required_skills and selected_departments:
        required_skills = _dedupe_keep_order([str(d).strip() for d in selected_departments if str(d).strip()])

    return _dedupe_keep_order(required_skills)


def _local_profile(cv_text: str, manual_skills: Sequence[str] | None = None) -> Dict[str, Any]:
    extracted = extract_skills_from_text(cv_text, load_taxonomy(), load_synonyms())
    skills = _dedupe_keep_order([skill.name for skill in extracted] + list(manual_skills or []))
    payload = _coerce_skill_payload(skills)

    # Prefer manual hints, otherwise use the strongest extracted skills as search seeds.
    recommended = _dedupe_keep_order(list(manual_skills or []))[:2]
    if not recommended:
        recommended = skills[:2]

    return {
        "detected_skills": skills,
        "skills_detected": payload,
        **_ai_profile_summary_payload(),
        "cv_rating": min(8, max(1, len(skills) // 2)),
        "recommended_search_keywords": recommended,
        "source": "fallback",
        "mode": "fallback"
    }


def _local_match_payload(
    cv_text: str,
    shortlist: List[Dict[str, Any]],
    manual_skills: Sequence[str] | None = None,
    selected_departments: Sequence[str] | None = None,
) -> Dict[str, Any]:
    profile = _local_profile(cv_text, manual_skills)
    detected_skills = profile.get("detected_skills", [])
    detected_set = {skill.casefold() for skill in detected_skills}

    matches: List[Dict[str, Any]] = []
    for internship in shortlist:
        internship_record = _json_safe_internship_record(internship)
        required_skills = _derive_required_skills(internship_record, selected_departments)


        matched_skills = [skill for skill in required_skills if skill.casefold() in detected_set]
        missing_skills = [skill for skill in required_skills if skill.casefold() not in detected_set]

        skill_ratio = len(matched_skills) / max(len(required_skills), 1)
        if required_skills and skill_ratio < MIN_SKILL_MATCH_RATIO:
            # Keep this only when we have explicit/inferred requirements; it prevents pure noise.
            continue

        score = _holistic_match_score(internship_record, required_skills, matched_skills, selected_departments)

        unavailable_text = _ai_match_text_payload()
        suggestions = list(unavailable_text["suggestions"])

        matches.append(
            {
                "internship": internship_record,
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "experience_alignment": unavailable_text["experience_alignment"],
                "skill_match": round(len(matched_skills) / max(len(required_skills), 1), 3) if required_skills else 0.0,
                "embedding_similarity": 0.0,
                "final_score": score,
                "match_score": score,
                "explanation": unavailable_text["explanation"],
                "learning_suggestion": suggestions[0],
                "learning_suggestions": suggestions,
                "learning_resources": _build_learning_resources(missing_skills),
                "competitiveness": "Medium",
            }
        )

    matches.sort(key=lambda item: item.get("final_score", 0), reverse=True)
    return {
        **profile,
        "best_matches": matches[:MAX_AI_MATCHES],
        "matches": matches[:MAX_AI_MATCHES],
        "message": "Local AI unavailable, showing basic matches",
    }


class OllamaCareerEngine:
    def __init__(self, url: str = OLLAMA_URL, model: str = OLLAMA_MODEL) -> None:
        self.url = url.rstrip("/")
        self.model = model.strip() or "llama3"

    @property
    def enabled(self) -> bool:
        """Actually ping Ollama to check if it's running."""
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=3, proxies={'http': None, 'https': None})
            return resp.status_code == 200
        except Exception:
            return False

    def extract_cv_profile(self, cv_text: str, manual_skills: Sequence[str] | None = None) -> Dict[str, Any]:
        manual = _dedupe_keep_order(manual_skills or [])
        taxonomy = load_taxonomy()
        allowed_skills = {s.casefold() for group in taxonomy.values() for s in group}
        local_extracted = extract_skills_from_text(cv_text, taxonomy, load_synonyms())
        local_skill_names = _dedupe_keep_order([s.name for s in local_extracted])
        prompt = (
            "Extract a career profile from this CV. Return ONLY valid JSON matching the schema.\n"
            "cv_rating is 1-10: rate the CV quality for someone at their career stage "
            "(education year, experience level). 1=very weak, 5=average, 10=exceptional.\n"
            "recommended_search_keywords: output exactly 2 broad keywords best representing this CV (e.g. ['python', 'backend'], ['design', 'UI']).\n"
            "Only include concrete canonical skills; do not include project names.\n"
            f"Schema: {json.dumps(PROFILE_ANALYSIS_SCHEMA, ensure_ascii=True)}\n"
            f"Manual skill hints: {json.dumps(manual, ensure_ascii=True)}\n"
            "IMPORTANT: Output NOTHING except the raw JSON object. Do not use Markdown, do not explain.\n"
            f"CV:\n{cv_text[:4000]}"
        )

        raw = _call_ollama(prompt, self.url, self.model, PROFILE_ANALYSIS_SCHEMA)
        if not raw:
            return _local_profile(cv_text, manual)

        try:
            parsed = _extract_json_payload(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Expected JSON object from Ollama profile response")
            # Ollama can return project/task phrases. Anchor to taxonomy skills using local extraction.
            ollama_skills_raw = _dedupe_keep_order([str(x) for x in parsed.get("detected_skills", [])])
            combined = _dedupe_keep_order(manual + local_skill_names + ollama_skills_raw)
            skills = [s for s in combined if s.casefold() in allowed_skills]
            if not skills:
                skills = _dedupe_keep_order(manual + local_skill_names)[:20]

            recommended_keywords_raw = _dedupe_keep_order([str(x) for x in parsed.get("recommended_search_keywords", [])])
            recommended_keywords = [k for k in recommended_keywords_raw if k.casefold() in allowed_skills][:2]
            if not recommended_keywords:
                recommended_keywords = skills[:2]
            try:
                cv_rating = max(1, min(10, int(parsed.get("cv_rating", 5))))
            except (TypeError, ValueError):
                cv_rating = 5
            summary_payload = _rewrite_profile_summaries(cv_text, self.url, self.model)
            return {
                "skills_detected": _coerce_skill_payload(skills),
                "detected_skills": skills,
                "experience_summary": summary_payload["experience_summary"],
                "education_summary": summary_payload["education_summary"],
                "projects_summary": summary_payload["projects_summary"],
                "interests": _dedupe_keep_order(parsed.get("interests", []))[:5],
                "summary": summary_payload["summary"],
                "cv_rating": cv_rating,
                "recommended_search_keywords": recommended_keywords,
                "mode": "ai",
                "source": "ollama",
            }
        except Exception as exc:
            logger.warning("Failed to parse Ollama CV profile: %s", exc)
            return _local_profile(cv_text, manual)

    def analyze_cv_and_match(
        self,
        cv_text: str,
        selected_departments: Sequence[str],
        internships_subset: List[Dict[str, Any]],
        manual_skills: Sequence[str] | None = None,
        cv_profile: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        shortlist = internships_subset[:MAX_AI_INTERNSHIPS]
        if not shortlist:
            return {
                "skills_detected": [],
                "detected_skills": [],
                "best_matches": [],
                "matches": [],
                "message": "No internships were available for the selected departments.",
                "mode": "fallback",
                "source": "fallback",
            }

        manual = _dedupe_keep_order(manual_skills or [])
        profile = dict(cv_profile or self.extract_cv_profile(cv_text, manual))
        profile_skills = _dedupe_keep_order(manual + list(profile.get("detected_skills", [])))
        shortlist_records = _make_shortlist_records(shortlist)

        prompt = (
            "Rank these internships for this candidate. Return ONLY valid JSON matching the schema.\n"
            "Rules:\n"
            "- matched_skills: ONLY short technical skill names (e.g. 'React', 'Python'). No sentences.\n"
            "- missing_skills: ONLY short technical skill names. No descriptive phrases.\n"
            "- matched_skills must only include skills the candidate EXPLICITLY HAS in their CV.\n"
            "- competitiveness: 'Low', 'Medium', or 'High'.\n"
            "- Do not generate explanations or learning advice in this pass.\n"
            f"Schema: {json.dumps(MATCHES_SCHEMA, ensure_ascii=True)}\n"
            f"Internships: {json.dumps(shortlist_records, ensure_ascii=True)}\n"
            f"CV (truncated):\n{cv_text[:4000]}"
        )

        raw = _call_ollama(prompt, self.url, self.model, MATCHES_SCHEMA)
        if not raw:
            return _local_match_payload(cv_text, shortlist, manual, selected_departments)

        try:
            parsed = _extract_json_payload(raw)
            # Aggressive conversion: Find the first list in a dictionary response
            if isinstance(parsed, dict):
                # Check known keys first
                found = False
                for key in ("matches", "rankings", "results", "data", "internships", "items"):
                    if isinstance(parsed.get(key), list):
                        parsed = parsed[key]
                        found = True
                        break
                # If no known key, just take the first list we find
                if not found:
                    for val in parsed.values():
                        if isinstance(val, list):
                            parsed = val
                            found = True
                            break
            if not isinstance(parsed, list):
                raise ValueError(f"Ollama returned an object ({type(parsed).__name__}), but we need a list.")
            rewrites = _rewrite_match_texts(cv_text, profile_skills, parsed, shortlist, self.url, self.model)
            payload_matches = self._shape_ai_matches(profile_skills, parsed, shortlist, selected_departments, rewrites)
            if not payload_matches:
                raise ValueError("No usable matches returned by Ollama")
            return {
                "skills_detected": _coerce_skill_payload(profile_skills),
                "detected_skills": profile_skills,
                "experience_summary": str(profile.get("experience_summary", "")).strip(),
                "education_summary": str(profile.get("education_summary", "")).strip(),
                "projects_summary": str(profile.get("projects_summary", "")).strip(),
                "summary": str(profile.get("summary", "")).strip(),
                "cv_rating": profile.get("cv_rating", 5),
                "best_matches": payload_matches,
                "matches": payload_matches,
                "message": "",
                "mode": "ai",
                "source": "ollama",
            }
        except Exception as exc:
            logger.warning("Failed to parse Ollama ranking response: %s", exc)
            return _local_match_payload(cv_text, shortlist, manual, selected_departments)

    def _shape_ai_matches(
        self,
        detected_skills: Sequence[str],
        raw_matches: Sequence[Any],
        shortlist: List[Dict[str, Any]],
        selected_departments: Sequence[str],
        rewritten_text: Mapping[tuple[str, str], Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        detected_set = {skill.casefold() for skill in _dedupe_keep_order(detected_skills)}
        rewritten_text = rewritten_text or {}

        for item in raw_matches:
            if not isinstance(item, MutableMapping):
                continue

            title = str(item.get("title", item.get("role", ""))).strip()
            company = str(item.get("company", "")).strip()
            internship = _find_internship_record(shortlist, title, company)
            if internship is None:
                continue

            required_skills = _derive_required_skills(internship, selected_departments)

            matched_skills = _dedupe_keep_order(item.get("matched_skills", []))
            missing_skills = _dedupe_keep_order(item.get("missing_skills", []))
            if not matched_skills:
                matched_skills = [skill for skill in required_skills if skill.casefold() in detected_set]
            if not missing_skills:
                missing_skills = [skill for skill in required_skills if skill.casefold() not in detected_set]
            # Ensure no overlap: remove matched from missing
            matched_set = {s.casefold() for s in matched_skills}
            missing_skills = [s for s in missing_skills if s.casefold() not in matched_set]

            skill_ratio = len(matched_skills) / max(len(required_skills), 1)
            if required_skills and skill_ratio < MIN_SKILL_MATCH_RATIO:
                continue
            final_score = _holistic_match_score(internship, required_skills, matched_skills, selected_departments)

            rewrite_key = (_lookup_key(title), _lookup_key(company))
            rewrite = dict(rewritten_text.get(rewrite_key, _ai_match_text_payload()))
            suggestions = _dedupe_keep_order(rewrite.get("suggestions", [])) or [AI_LEARNING_UNAVAILABLE]
            explanation = str(rewrite.get("explanation", "")).strip()
            if _looks_like_heuristic_explanation(explanation):
                explanation = AI_EXPLANATION_UNAVAILABLE

            experience_alignment = str(rewrite.get("experience_alignment", "")).strip()
            competitiveness = str(item.get("competitiveness", "")).strip()
            # Derive competitiveness from required skill count if Ollama didn't supply it
            if competitiveness not in ("Low", "Medium", "High"):
                if len(required_skills) >= 6:
                    competitiveness = "High"
                elif len(required_skills) >= 3:
                    competitiveness = "Medium"
                else:
                    competitiveness = "Low"
            # Derive experience_alignment from score if Ollama didn't supply it
            if not experience_alignment:
                experience_alignment = AI_ALIGNMENT_UNAVAILABLE

            payload.append(
                {
                    "internship": dict(internship),
                    "matched_skills": matched_skills,
                    "missing_skills": missing_skills,
                    "experience_alignment": experience_alignment,
                    "competitiveness": competitiveness,
                    "skill_match": round(len(matched_skills) / max(len(required_skills), 1), 3) if required_skills else 0.0,
                    "embedding_similarity": 0.0,
                    "final_score": final_score,
                    "match_score": final_score,
                    "explanation": explanation or AI_EXPLANATION_UNAVAILABLE,
                    "learning_suggestion": " ".join(suggestions[:2]).strip()
                    or AI_LEARNING_UNAVAILABLE,
                    "learning_suggestions": suggestions[:3] or [AI_LEARNING_UNAVAILABLE],
                    "learning_resources": _build_learning_resources(missing_skills),
                }
            )

        payload.sort(key=lambda item: item.get("final_score", 0), reverse=True)
        return payload[:MAX_AI_MATCHES]

    def ask_job_question(self, cv_text: str, job: Dict[str, Any], question: str) -> str:
        """Answer a single question about a specific job listing in the context of the CV."""
        job_summary = f"{job.get('title', '')} at {job.get('company', '')} ({job.get('location', '')})"
        prompt = (
            f"You are a professional Career Advisor. Your goal is to answer the candidate's question about the following internship directly and concisely.\n\n"
            f"**STRICT RULES:**\n"
            f"1. DIRECT ANSWER: Start immediately with the answer. Do NOT say 'It seems there might be confusion', 'Here is the answer', or make speculative explanations.\n"
            f"2. NO FLUFF: Do not include background stories, filler, or robotic pleasantries.\n"
            f"3. UNRELATED QUESTIONS: If the question is entirely unrelated to the job or CV, just say 'This question does not appear related to the internship listing.'\n"
            f"4. CONTEXT ONLY: Base your answer strictly on the job details and CV provided. Do not guess.\n"
            f"5. LENGTH: Keep it between 2 to 4 short, professional sentences.\n\n"
            f"Job Summary: {job_summary}\n"
            f"Job Description: {str(job.get('description', ''))[:800]}\n\n"
            f"Candidate CV (truncated): {cv_text[:2000]}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        return _call_ollama(prompt, self.url, self.model) or "I couldn't generate an answer right now. Please try again."
