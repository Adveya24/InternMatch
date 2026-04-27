from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

try:
    from .config import INTERN_CSV_PATH, MAX_AI_INTERNSHIPS
except ImportError:
    from config import INTERN_CSV_PATH, MAX_AI_INTERNSHIPS


def load_internships() -> List[Dict[str, Any]]:
    """Loads internship data from the CSV file."""
    csv_path = Path(INTERN_CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"Internships CSV not found at {csv_path}")

    internships = []
    with open(csv_path, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            # Date conversion
            try:
                row["_sort_date"] = datetime.strptime(row.get("date_posted"), '%Y-%m-%d')
            except (ValueError, TypeError):
                row["_sort_date"] = datetime.min

            # Skills list creation
            skills_str = row.get("skills_required", "") or row.get("requirements", "") or ""
            row["skills_required_list"] = [s.strip() for s in skills_str.split(';') if s.strip()]

            # Normalize to match API schema
            row["requirements"] = skills_str
            row["source"] = row.get("source", "")
            row["apply_url"] = row.get("link", row.get("apply_url", ""))
            row["rating"] = row.get("rating", "")
            row["reviews"] = row.get("reviews", "")

            internships.append(row)
    return internships


def shortlist_internships(internships: List[Dict[str, Any]], departments: List[str]) -> List[Dict[str, Any]]:
    """Shortlists internships based on selected departments."""
    dep_set = {department.strip() for department in departments if department.strip()}
    
    if dep_set:
        internships = [i for i in internships if i.get("department") in dep_set]

    # Sort by date and limit
    internships.sort(key=lambda x: x["_sort_date"], reverse=True)
    
    return internships[:MAX_AI_INTERNSHIPS]
