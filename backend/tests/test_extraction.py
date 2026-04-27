from __future__ import annotations

from pathlib import Path

from backend.parser import _clean_text
from backend.skills_extractor import extract_skills_from_text, load_synonyms, load_taxonomy


BASE_DIR = Path(__file__).resolve().parents[2]
SAMPLES_DIR = BASE_DIR / "data" / "sample_cvs"


def test_clean_text_normalizes_whitespace_and_case() -> None:
    raw = " Hello \nWORLD\tTest "
    cleaned = _clean_text(raw)
    assert cleaned == "hello world test"


def test_extract_skills_handles_typos_and_synonyms_tech_cv() -> None:
    tech_cv = (SAMPLES_DIR / "cv_tech.txt").read_text(encoding="utf-8")
    skills = extract_skills_from_text(tech_cv, load_taxonomy(), load_synonyms())
    names = {s.name for s in skills}
    # From prompt: pyhton, machine learnng, SQL, powerbi
    assert "Python" in names
    assert "Machine Learning" in names
    assert "SQL" in names
    assert "Power BI" in names


def test_extract_skills_handles_nontech_cv_synonyms() -> None:
    nontech_cv = (SAMPLES_DIR / "cv_nontech.txt").read_text(encoding="utf-8")
    skills = extract_skills_from_text(nontech_cv, load_taxonomy(), load_synonyms())
    names = {s.name for s in skills}
    assert "Classroom Management" in names
    assert "Lesson Planning" in names
