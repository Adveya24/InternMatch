from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List
import logging

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import (
    REMOTIVE_ENABLED,
    THEIRSTACK_ENABLED,
    OLLAMA_MODEL,
    OLLAMA_URL,
)
from backend.data_loader import load_internships, shortlist_internships
from backend.ollama_helper import OllamaCareerEngine
from backend.parser import extract_text_from_file
from backend.skills_extractor import load_taxonomy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run InternMatch from CLI")
    parser.add_argument("--cv", required=True, help="Path to CV file (pdf/docx/txt)")
    parser.add_argument(
        "--departments",
        required=True,
        help='Comma-separated departments, e.g. "Computer Science / AI,Data Science / Analytics"',
    )
    parser.add_argument(
        "--manual-skills",
        default="",
        help='Optional comma-separated skill hints, e.g. "Python,SQL"',
    )
    args = parser.parse_args()

    cv_path = Path(args.cv)
    departments = [item.strip() for item in args.departments.split(",") if item.strip()]
    manual_skills = [item.strip() for item in args.manual_skills.split(",") if item.strip()]

    text = extract_text_from_file(cv_path)

    internships: List[dict] = []
    providers = []

    if REMOTIVE_ENABLED:
        from backend.providers import RemotiveProvider
        providers.append(RemotiveProvider())
    if THEIRSTACK_ENABLED:
        from backend.providers import TheirStackProvider
        providers.append(TheirStackProvider())

    used_providers = []
    if providers:
        print(f"Fetching internships from providers: {[p.__class__.__name__ for p in providers]}")
        for provider in providers:
            try:
                fetched = provider.fetch_internships(keywords=departments)
                internships.extend(fetched)
                if fetched:
                    used_providers.append(provider.__class__.__name__)
            except Exception as exc:
                print(f"Failed to fetch internships from {provider.__class__.__name__}: {exc}")

    source = ", ".join(used_providers) if used_providers else "fallback"

    if not internships:
        print("No internships from providers, falling back to CSV")
        source = "fallback"
        all_internships = load_internships()
        internships = shortlist_internships(all_internships, departments)

    if not internships:
        print("No internships found for the selected departments.")
        return

    engine = OllamaCareerEngine(url=OLLAMA_URL, model=OLLAMA_MODEL)
    result = engine.analyze_cv_and_match(
        cv_text=text,
        selected_departments=departments,
        internships_subset=internships,
        manual_skills=manual_skills,
    )

    skills = result.get("skills_detected", [])
    matches = result.get("matches", [])

    if not matches and not skills:
        taxonomy = load_taxonomy()
        suggested = sorted({skill for values in taxonomy.values() for skill in values})
        print("No skills detected in the CV. Try manual skills or a clearer file.")
        print("Suggested skills:")
        for skill in suggested[:25]:
            print(f"  - {skill}")
        return

    mode = result.get("mode", "unavailable")
    print(f"Mode: {mode}")
    print(f"Internship source: {source}")
    print("AI Source: Local (Ollama)")
    if result.get("message"):
        print(result["message"])
    print(f"Detected skills: {', '.join(item['name'] for item in skills) or 'None'}")
    print(f"Experience summary: {result.get('experience_summary', '')}")
    print(f"Education summary: {result.get('education_summary', '')}")
    print(f"Projects summary: {result.get('projects_summary', '')}")
    print(f"Top {len(matches)} matches for CV: {cv_path}")

    for idx, match in enumerate(matches, start=1):
        internship = match.get("internship", {})
        print(
            f"\n#{idx}: {internship.get('title', 'Unknown')} - "
            f"{internship.get('company', 'Unknown')} [{internship.get('department', '')}]"
        )
        print(f"  Final score: {match.get('final_score', 0)}%")
        print(f"  Source: {internship.get('source', 'N/A')}")
        print(f"  Matched skills: {', '.join(match.get('matched_skills', [])) or 'None'}")
        print(f"  Missing skills: {', '.join(match.get('missing_skills', [])) or 'None'}")
        print(f"  Explanation: {match.get('explanation', '')}")
        print(f"  Learn next: {match.get('learning_suggestion', '')}")
        resources = match.get("learning_resources", {})
        if resources:
            print("  Learning resources:")
            for skill, urls in resources.items():
                for url in urls:
                    print(f"    - {skill}: {url}")


if __name__ == "__main__":
    main()
