from __future__ import annotations

from pathlib import Path

from backend.learning_resources import get_learning_resources_with_ai
from backend.matcher import match_internships_with_ai
from backend.ollama_helper import OllamaCareerEngine
from backend.parser import parse_cv_with_ai


def _sample_internships() -> list[dict]:
    return [
        {
            "id": 1,
            "title": "AI Intern",
            "company": "DemoCo",
            "department": "Computer Science / AI",
            "skills_required": "Python;SQL;TensorFlow",
            "skills_required_list": ["Python", "SQL", "TensorFlow"],
            "description": "Build ML prototypes.",
            "location": "Remote",
            "source": "fallback",
        }
    ]


def test_ollama_career_engine_falls_back_to_basic_matches_when_empty(monkeypatch) -> None:
    monkeypatch.setattr("backend.ollama_helper._call_ollama", lambda prompt, url, model, response_format=None: "")
    engine = OllamaCareerEngine()

    result = engine.analyze_cv_and_match(
        cv_text="Experienced with Python and SQL projects.",
        selected_departments=["Computer Science / AI"],
        internships_subset=_sample_internships(),
    )

    assert result["mode"] == "fallback"
    assert result["source"] == "fallback"
    assert result["message"] == "Local AI unavailable, showing basic matches"
    assert result["matches"]
    assert result["matches"][0]["internship"]["title"] == "AI Intern"
    assert result["matches"][0]["explanation"] == "AI explanation unavailable for this role right now."
    assert result["matches"][0]["experience_alignment"] == "AI fit summary unavailable for this role right now."


def test_parser_ai_entrypoint_uses_ollama_or_fallback(monkeypatch) -> None:
    monkeypatch.setattr("backend.ollama_helper._call_ollama", lambda prompt, url, model, response_format=None: "")
    profile = parse_cv_with_ai(Path("data/sample_cvs/cv_tech.txt"))

    assert "cv_text" in profile
    assert profile["source"] in {"fallback", "ollama"}
    assert "detected_skills" in profile
    assert profile["experience_summary"] == "AI profile summary unavailable right now."


def test_matcher_ai_entrypoint_returns_basic_results_when_ollama_empty(monkeypatch) -> None:
    monkeypatch.setattr("backend.ollama_helper._call_ollama", lambda prompt, url, model, response_format=None: "")
    result = match_internships_with_ai(
        cv_text="Experienced with Python and SQL projects.",
        departments=["Computer Science / AI"],
        internships=_sample_internships(),
    )

    assert result["source"] == "fallback"
    assert result["best_matches"]


def test_learning_resources_ai_entrypoint_falls_back_when_ollama_empty(monkeypatch) -> None:
    monkeypatch.setattr("backend.ollama_helper._call_ollama", lambda prompt, url, model, response_format=None: "")
    result = get_learning_resources_with_ai("TensorFlow")

    assert result["source"] == "fallback"
    assert result["resources"]


def test_profile_summaries_use_ai_rewrite_output(monkeypatch) -> None:
    responses = iter(
        [
            """
            {
              "detected_skills": ["Python", "React"],
              "interests": ["web development"],
              "cv_rating": 7,
              "recommended_search_keywords": ["frontend", "react"]
            }
            """,
            """
            {
              "summary": "The candidate is an early-career web developer with hands-on project experience.",
              "experience_summary": "The candidate has practical full-stack experience from internships and student work.",
              "education_summary": "The candidate is studying computer science and building a strong technical base.",
              "projects_summary": "The candidate has built web applications that show frontend and backend experience."
            }
            """,
        ]
    )
    monkeypatch.setattr(
        "backend.ollama_helper._call_ollama",
        lambda prompt, url, model, response_format=None: next(responses),
    )

    profile = parse_cv_with_ai(Path("data/sample_cvs/cv_tech.txt"))

    assert profile["source"] == "ollama"
    assert profile["summary"].startswith("The candidate is an early-career web developer")
    assert profile["experience_summary"].startswith("The candidate has practical full-stack experience")


def test_ai_ranking_uses_humanized_rewrite_text(monkeypatch) -> None:
    responses = iter(
        [
            """
            {
              "detected_skills": ["Python", "SQL"],
              "interests": ["data"],
              "cv_rating": 7,
              "recommended_search_keywords": ["python", "analytics"]
            }
            """,
            """
            {
              "summary": "The candidate is an early-career data student with practical project work.",
              "experience_summary": "The candidate has hands-on project experience with Python and SQL.",
              "education_summary": "The candidate is building a data-focused academic foundation.",
              "projects_summary": "The candidate has completed projects that demonstrate analytical and technical skills."
            }
            """,
            """
            [
              {
                "title": "AI Intern",
                "company": "DemoCo",
                "match_score": 61,
                "matched_skills": ["Python", "SQL"],
                "missing_skills": ["TensorFlow"],
                "competitiveness": "Medium"
              }
            ]
            """,
            """
            [
              {
                "title": "AI Intern",
                "company": "DemoCo",
                "explanation": "This role fits because the candidate already shows strong Python and SQL fundamentals relevant to the internship.",
                "suggestions": ["Learn TensorFlow for practical model-building."],
                "experience_alignment": "The candidate's current project work aligns well with an entry-level AI internship."
              }
            ]
            """,
        ]
    )
    monkeypatch.setattr(
        "backend.ollama_helper._call_ollama",
        lambda prompt, url, model, response_format=None: next(responses),
    )
    engine = OllamaCareerEngine()

    result = engine.analyze_cv_and_match(
        cv_text="Experienced with Python and SQL projects.",
        selected_departments=["Computer Science / AI"],
        internships_subset=_sample_internships(),
    )

    top_match = result["matches"][0]
    assert "Weak Match" not in top_match["explanation"]
    assert "%" not in top_match["explanation"]
    assert top_match["experience_alignment"].startswith("The candidate's current project work aligns well")


def test_visible_match_text_falls_back_to_unavailable_when_rewrite_fails(monkeypatch) -> None:
    responses = iter(
        [
            """
            {
              "detected_skills": ["Python", "SQL"],
              "interests": ["data"],
              "cv_rating": 7,
              "recommended_search_keywords": ["python", "analytics"]
            }
            """,
            """
            {
              "summary": "The candidate is an early-career data student.",
              "experience_summary": "The candidate has hands-on project experience.",
              "education_summary": "The candidate is building a data-focused academic foundation.",
              "projects_summary": "The candidate has completed relevant technical projects."
            }
            """,
            """
            [
              {
                "title": "AI Intern",
                "company": "DemoCo",
                "match_score": 61,
                "matched_skills": ["Python", "SQL"],
                "missing_skills": ["TensorFlow"],
                "competitiveness": "Medium"
              }
            ]
            """,
            "",
        ]
    )
    monkeypatch.setattr(
        "backend.ollama_helper._call_ollama",
        lambda prompt, url, model, response_format=None: next(responses),
    )
    engine = OllamaCareerEngine()

    result = engine.analyze_cv_and_match(
        cv_text="Experienced with Python and SQL projects.",
        selected_departments=["Computer Science / AI"],
        internships_subset=_sample_internships(),
    )

    top_match = result["matches"][0]
    assert top_match["explanation"] == "AI explanation unavailable for this role right now."
    assert top_match["learning_suggestion"] == "AI learning suggestions unavailable right now."


def test_profile_summary_uses_labeled_fallback_when_json_summary_fails(monkeypatch) -> None:
    responses = iter(
        [
            """
            {
              "detected_skills": ["Python", "React"],
              "interests": ["web development"],
              "cv_rating": 7,
              "recommended_search_keywords": ["frontend", "react"]
            }
            """,
            "not valid json at all",
            """
            SUMMARY: The candidate is an early-career developer with practical web application experience.
            EXPERIENCE: The candidate has hands-on full-stack internship and project exposure.
            EDUCATION: The candidate is studying computer science and building a solid software foundation.
            PROJECTS: The candidate has built web applications that show frontend and backend skills.
            """,
        ]
    )
    monkeypatch.setattr(
        "backend.ollama_helper._call_ollama",
        lambda prompt, url, model, response_format=None: next(responses),
    )

    profile = parse_cv_with_ai(Path("data/sample_cvs/cv_tech.txt"))

    assert profile["summary"].startswith("The candidate is an early-career developer")
    assert profile["experience_summary"].startswith("The candidate has hands-on full-stack")


def test_match_rewrite_retries_per_role_when_batch_rewrite_fails(monkeypatch) -> None:
    responses = iter(
        [
            """
            {
              "detected_skills": ["Python", "SQL"],
              "interests": ["data"],
              "cv_rating": 7,
              "recommended_search_keywords": ["python", "analytics"]
            }
            """,
            """
            {
              "summary": "The candidate is an early-career data student.",
              "experience_summary": "The candidate has hands-on project experience.",
              "education_summary": "The candidate is building a data-focused academic foundation.",
              "projects_summary": "The candidate has completed relevant technical projects."
            }
            """,
            """
            [
              {
                "title": "AI Intern",
                "company": "DemoCo",
                "match_score": 61,
                "matched_skills": ["Python", "SQL"],
                "missing_skills": ["TensorFlow"],
                "competitiveness": "Medium"
              }
            ]
            """,
            "not valid json at all",
            """
            EXPLANATION: This internship fits because the candidate already demonstrates Python and SQL skills that translate well into the role.
            SUGGESTIONS: Learn TensorFlow for model-building practice; build one end-to-end AI project
            ALIGNMENT: The candidate's current technical projects line up well with an entry-level AI internship.
            """,
        ]
    )
    monkeypatch.setattr(
        "backend.ollama_helper._call_ollama",
        lambda prompt, url, model, response_format=None: next(responses),
    )
    engine = OllamaCareerEngine()

    result = engine.analyze_cv_and_match(
        cv_text="Experienced with Python and SQL projects.",
        selected_departments=["Computer Science / AI"],
        internships_subset=_sample_internships(),
    )

    top_match = result["matches"][0]
    assert top_match["explanation"].startswith("This internship fits because the candidate already demonstrates Python and SQL")
    assert top_match["learning_suggestions"][0] == "Learn TensorFlow for model-building practice"
    assert top_match["experience_alignment"].startswith("The candidate's current technical projects line up well")
