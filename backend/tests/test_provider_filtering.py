from __future__ import annotations

from backend.providers import department_relevance_score, filter_internships_for_departments


def _job(title: str, description: str, requirements: str = "") -> dict:
    return {
        "title": title,
        "company": "DemoCo",
        "description": description,
        "requirements": requirements,
        "skills_required_list": [item.strip() for item in requirements.split(",") if item.strip()],
        "source": "provider",
    }


def test_web_department_filters_out_machine_learning_role() -> None:
    internships = [
        _job(
            "PhD Machine Learning Engineer, Intern",
            "Build machine learning systems with Python, NumPy, and Spark.",
            "Python, NumPy, Spark",
        ),
        _job(
            "Frontend Web Developer Intern",
            "Build React interfaces, ship HTML/CSS components, and work on frontend web experiences.",
            "React, HTML, CSS, JavaScript",
        ),
    ]

    filtered = filter_internships_for_departments(internships, ["Web Development"])

    assert len(filtered) == 1
    assert filtered[0]["title"] == "Frontend Web Developer Intern"


def test_finance_department_requires_explicit_finance_signal() -> None:
    ml_role = _job(
        "PhD Machine Learning Engineer, Intern",
        "Build machine learning systems with Python, NumPy, and Spark.",
        "Python, NumPy, Spark",
    )
    finance_role = _job(
        "Finance Intern",
        "Support budgeting, accounting, and financial reporting for the team.",
        "Accounting, Financial Analysis, Excel",
    )

    ml_score = department_relevance_score(ml_role, ["Finance / Accounting"])
    finance_score = department_relevance_score(finance_role, ["Finance / Accounting"])

    assert ml_score < 0.18
    assert finance_score >= 0.18


def test_aerospace_department_prefers_real_aerospace_roles() -> None:
    internships = [
        _job(
            "Backend Engineering Intern",
            "Build backend APIs and internal tools with Python and SQL.",
            "Python, SQL, REST APIs",
        ),
        _job(
            "Aerospace Engineering Intern",
            "Work on aerodynamics, CFD analysis, and aircraft performance modeling.",
            "CFD, Aerodynamics, MATLAB",
        ),
    ]

    filtered = filter_internships_for_departments(internships, ["Aviation / Aerospace"])

    assert len(filtered) == 1
    assert filtered[0]["title"] == "Aerospace Engineering Intern"


def test_mechanical_department_prefers_real_mechanical_roles() -> None:
    internships = [
        _job(
            "Business Operations Intern",
            "Support reporting, excel tracking, and general operations.",
            "Excel, Communication, CRM",
        ),
        _job(
            "Mechanical Design Intern",
            "Design parts in SolidWorks and support manufacturing and testing workflows.",
            "SolidWorks, CAD, Manufacturing",
        ),
    ]

    filtered = filter_internships_for_departments(internships, ["Manufacturing / Mechanical"])

    assert len(filtered) == 1
    assert filtered[0]["title"] == "Mechanical Design Intern"
