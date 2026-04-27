from __future__ import annotations

import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, request, send_from_directory

# Keep imports robust across launch styles (package vs adjusted sys.path).
try:
    from .config import (
        ARBEITNOW_ENABLED,
        GREENHOUSE_ENABLED,
        INTERNSHALA_ENABLED,
        LEVER_ENABLED,
        RAPIDAPI_INTERNSHIPS_ENABLED,
        REMOTIVE_ENABLED,
        THEIRSTACK_ENABLED,
        OLLAMA_MODEL,
    )
    from .ollama_helper import OllamaCareerEngine
    from .parser import extract_text_from_file
    from .providers import (
        filter_internships_for_departments,
        RemotiveProvider,
        TheirStackProvider,
        RapidAPIInternshipsProvider,
        ArbeitnowProvider,
        GreenhouseProvider,
        LeverProvider,
        InternshalaProvider,
    )
    from .skills_extractor import load_taxonomy
    from .data_loader import load_internships, shortlist_internships
except ModuleNotFoundError:  # pragma: no cover
    from backend.config import (
        ARBEITNOW_ENABLED,
        GREENHOUSE_ENABLED,
        INTERNSHALA_ENABLED,
        LEVER_ENABLED,
        RAPIDAPI_INTERNSHIPS_ENABLED,
        REMOTIVE_ENABLED,
        THEIRSTACK_ENABLED,
    )
    from backend.ollama_helper import OllamaCareerEngine
    from backend.parser import extract_text_from_file
    from backend.providers import (
        filter_internships_for_departments,
        RemotiveProvider,
        TheirStackProvider,
        RapidAPIInternshipsProvider,
        ArbeitnowProvider,
        GreenhouseProvider,
        LeverProvider,
        InternshalaProvider,
    )
    from backend.skills_extractor import load_taxonomy
    from backend.data_loader import load_internships, shortlist_internships


def create_app() -> Flask:
    base_dir = Path(__file__).resolve().parent.parent
    frontend_dir = base_dir / "frontend"

    app = Flask(
        __name__,
        static_folder=str(frontend_dir / "static"),
        template_folder=str(frontend_dir),
    )

    app.config["UPLOAD_FOLDER"] = str(base_dir / "uploads")
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    app.config["DEBUG"] = os.getenv("INTERNMATCH_DEBUG", "0") == "1"

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if app.config["DEBUG"] else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger = logging.getLogger("internmatch.app")

    engine = OllamaCareerEngine()

    def manual_skill_suggestions() -> List[str]:
        taxonomy = load_taxonomy()
        return sorted({skill for skills in taxonomy.values() for skill in skills})

    @app.get("/")
    def serve_index() -> Any:
        return send_from_directory(frontend_dir, "index.html")

    @app.get("/results")
    def serve_results() -> Any:
        return send_from_directory(frontend_dir, "results.html")

    @app.get("/health")
    def health_check() -> Any:
        logger.debug("Health check requested")
        return jsonify(
            {
                "status": "ok",
                "gemini_enabled": False,
                "ollama_enabled": engine.enabled,
                "embeddings_ready": False,
                "mode": "ai" if engine.enabled else "unavailable",
                "source": "ollama",
            }
        ), 200

    @app.post("/upload-cv")
    def upload_cv() -> Any:
        if "file" not in request.files:
            logger.warning("Upload attempted without file part")
            return jsonify({"error": "No file part in request"}), 400

        file = request.files["file"]
        if file.filename == "":
            logger.warning("Upload attempted with empty filename")
            return jsonify({"error": "No selected file"}), 400

        allowed_ext = {".pdf", ".docx"}
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_ext:
            logger.warning("Upload attempted with disallowed extension: %s", ext)
            return jsonify({"error": "Unsupported file type. Use PDF or DOCX."}), 400

        upload_dir = Path(app.config["UPLOAD_FOLDER"])
        upload_dir.mkdir(parents=True, exist_ok=True)

        safe_name = Path(file.filename).name
        save_path = upload_dir / safe_name
        file.save(save_path)
        logger.info("Saved CV to %s", save_path)

        try:
            text = extract_text_from_file(save_path)
        except Exception as exc:
            logger.exception("Failed to parse or extract profile from CV %s", save_path)
            return (
                jsonify(
                    {
                        "message": "File uploaded but parsing failed. Try a clearer PDF/DOCX.",
                        "cv_id": safe_name,
                        "path": str(save_path),
                        "error": str(exc),
                    }
                ),
                500,
            )

        response: Dict[str, Any] = {
            "message": "File uploaded and parsed successfully. AI ranking runs on the next step.",
            "cv_id": safe_name,
            "path": str(save_path),
            "skills": [],
            "detected_skills": [],
            "experience_summary": "",
            "education_summary": "",
            "projects_summary": "",
            "summary": "",
            "mode": "pending" if engine.enabled else "unavailable",
            "source": "local-parse",
        }
        return jsonify(response), 201

    @app.get("/skills")
    def list_skills() -> Any:
        taxonomy = load_taxonomy()
        # The taxonomy is the source of truth for the exact department lanes shown in the UI.
        return jsonify(taxonomy), 200

    @app.post("/matches")
    def get_matches() -> Any:
        data = request.get_json(silent=True) or {}
        cv_path = data.get("cv_path")
        departments: List[str] = (data.get("departments") or [])[:4]  # type: ignore[index]
        manual_skills: List[str] = data.get("manual_skills") or []

        if not cv_path:
            return jsonify({"error": "cv_path is required"}), 400

        try:
            text = extract_text_from_file(cv_path)
        except Exception as exc:
            logger.exception("Failed to parse CV for matches: %s", cv_path)
            return jsonify({"error": f"Failed to parse CV: {exc}"}), 500
        
        # 1. AI extracts profile and generates dynamic search keywords FIRST
        try:
            cv_profile = engine.extract_cv_profile(text, manual_skills)
            ai_keywords = cv_profile.get("recommended_search_keywords", [])
            
            # Combine UI explicit selections with AI + skill-derived keywords.
            # Global Match (no departments): use detected skills as primary provider keywords.
            search_keywords: List[str] = []
            if departments:
                search_keywords = departments.copy()
            else:
                detected = cv_profile.get("detected_skills", []) if isinstance(cv_profile, dict) else []
                if isinstance(detected, list):
                    for skill in detected[:6]:
                        if isinstance(skill, str) and skill.strip():
                            search_keywords.append(skill.strip().title())
                if isinstance(ai_keywords, list):
                    for kw in ai_keywords:
                        if isinstance(kw, str) and kw.strip():
                            search_keywords.append(kw.strip().title())
            
            # Dedupe preserve order
            seen = set()
            search_keywords = [k for k in search_keywords if not (k.casefold() in seen or seen.add(k.casefold()))]
            logger.info("Provider search keywords: %s", search_keywords)
        except Exception as e:
            logger.error("Failed to extract CV profile for API search: %s", e)
            cv_profile = None
            search_keywords = departments

        # 2. Add Providers
        internships = []
        providers = []
        if REMOTIVE_ENABLED:
            providers.append(RemotiveProvider())
        if THEIRSTACK_ENABLED:
            providers.append(TheirStackProvider())
        if ARBEITNOW_ENABLED:
            providers.append(ArbeitnowProvider())
        if GREENHOUSE_ENABLED:
            providers.append(GreenhouseProvider())
        if LEVER_ENABLED:
            providers.append(LeverProvider())
        if INTERNSHALA_ENABLED:
            providers.append(InternshalaProvider())
        if RAPIDAPI_INTERNSHIPS_ENABLED:
            providers.append(RapidAPIInternshipsProvider())
        
        # 3. Fetch from APIs using AI-enhanced keywords
        # TODO (Cloud Scaling): Expand inference pipelines toward Vertex AI / Azure Open AI. Local Ollama used strictly as a localized showcase MVP.
        used_providers = []
        if providers:
            logger.info(f"Fetching internships from providers: {[p.__class__.__name__ for p in providers]}")
            for provider in providers:
                try:
                    fetched = provider.fetch_internships(keywords=search_keywords)
                    if fetched:
                        internships.extend(fetched)
                        used_providers.append(provider.__class__.__name__)
                except Exception as exc:
                    logger.error(f"Failed to fetch internships from {provider.__class__.__name__}: {exc}")

        # Deduplication based on title + company + source
        unique_internships = []
        seen_keys = set()
        for intern in internships:
            # We lowercase title and company. Any 100% duplicate is squashed.
            key = (str(intern.get("title", "")).strip().casefold(), str(intern.get("company", "")).strip().casefold())
            if key not in seen_keys:
                seen_keys.add(key)
                unique_internships.append(intern)
            
        internships = unique_internships

        if departments and internships:
            before_filter = len(internships)
            internships = filter_internships_for_departments(internships, departments)
            logger.info(
                "Department relevance filter kept %d/%d internships for %s",
                len(internships),
                before_filter,
                departments,
            )

        source = ", ".join(used_providers) if used_providers else "none"

        if not internships:
            logger.warning("No internships found from live providers.")
            return jsonify({
                "matches": [],
                "best_matches": [],
                "message": "No live results found. The system is set to Live-API-Only mode.",
                "mode": "ai",
                "source": "none",
                "ai_source": "Ollama (Bypassed)",
                "ai_model": OLLAMA_MODEL,
            }), 200



        result = engine.analyze_cv_and_match(
            cv_text=text,
            selected_departments=departments,
            internships_subset=internships,
            manual_skills=manual_skills,
            cv_profile=cv_profile,
        )

        matches = result.get("matches", [])
        skills_detected = result.get("skills_detected", [])
        if not skills_detected and manual_skills:
            skills_detected = [{"name": skill, "confidence": 0.85} for skill in manual_skills]

        _provider_labels = {
            "RemotiveProvider": "Remotive",
            "TheirStackProvider": "TheirStack",
            "RapidAPIInternshipsProvider": "RapidAPI",
        }
        readable_providers = [str(_provider_labels.get(p, p)) for p in used_providers]
        source = ", ".join(readable_providers) if readable_providers else "none"

        result_source = result.get("source", "unavailable")
        if result_source == "ollama":
            result_source = f"ollama (from {source})"
        else:
            result_source = source

        if not matches and result.get("source") == "unavailable":
            return jsonify(
                {
                    "matches": [],
                    "best_matches": [],
                    "best_match": None,
                    "skills_detected": [],
                    "detected_skills": [],
                    "experience_summary": result.get("experience_summary", ""),
                    "education_summary": result.get("education_summary", ""),
                    "projects_summary": result.get("projects_summary", ""),
                    "summary": result.get("summary", ""),
                    "parsed_experience_section": result.get("parsed_experience_section", ""),
                    "parsed_education_section": result.get("parsed_education_section", ""),
                    "parsed_projects_section": result.get("parsed_projects_section", ""),
                    "cv_rating": result.get("cv_rating", 5),
                    "message": result.get("message", "Ollama is unavailable. AI-only mode cannot rank internships."),
                    "mode": result.get("mode", "unavailable"),
                    "source": result_source,
                    "ai_source": "Ollama" if result.get("source") == "ollama" else "Local (Fallback)",
                    "ai_model": OLLAMA_MODEL,
                }
            ), 200
            
        if not matches and not skills_detected:
            return jsonify(
                {
                    "matches": [],
                    "best_matches": [],
                    "best_match": None,
                    "skills_detected": [],
                    "detected_skills": [],
                    "experience_summary": result.get("experience_summary", ""),
                    "education_summary": result.get("education_summary", ""),
                    "projects_summary": result.get("projects_summary", ""),
                    "summary": result.get("summary", ""),
                    "suggested_skills": manual_skill_suggestions(),
                    "message": "No skills detected. Choose skills manually and retry.",
                    "mode": result.get("mode", "unavailable"),
                    "source": result_source,
                    "ai_source": "Ollama" if result.get("source") == "ollama" else "Local (Fallback)",
                    "ai_model": OLLAMA_MODEL,
                }
            ), 200
            
        return jsonify(
            {
                "matches": matches,
                "best_matches": result.get("best_matches", matches),
                "best_match": matches[0] if matches else None,
                "mode": result.get("mode", "ai"),
                "source": result_source,
                "ai_source": "Ollama" if result.get("source") == "ollama" else "Local (Fallback)",
                "ai_model": OLLAMA_MODEL,
                "message": result.get("message", ""),
                "skills_detected": skills_detected,
                "detected_skills": result.get("detected_skills", [item.get("name") for item in skills_detected]),
                "experience_summary": result.get("experience_summary", ""),
                "education_summary": result.get("education_summary", ""),
                "projects_summary": result.get("projects_summary", ""),
                "summary": result.get("summary", ""),
                "cv_rating": result.get("cv_rating", 5),
            }
        ), 200

    @app.post("/delete-cv")
    def delete_cv() -> Any:
        data = request.get_json(silent=True) or {}
        cv_path = data.get("cv_path")
        if not cv_path:
            return jsonify({"error": "cv_path is required"}), 400

        try:
            path = Path(cv_path)
            if path.exists():
                path.unlink()
                return jsonify({"deleted": True}), 200
            return jsonify({"deleted": False, "message": "File not found"}), 404
        except Exception as exc:
            logger.exception("Failed to delete CV %s", cv_path)
            return jsonify({"error": str(exc)}), 500

    @app.post("/ask-job-question")
    def ask_job_question() -> Any:
        data = request.get_json(silent=True) or {}
        cv_path = data.get("cv_path")
        job = data.get("job") or {}
        question = str(data.get("question") or "").strip()

        if not cv_path:
            return jsonify({"error": "cv_path is required"}), 400
        if not question:
            return jsonify({"error": "question is required"}), 400

        try:
            cv_text = extract_text_from_file(cv_path)
        except Exception as exc:
            logger.exception("Failed to parse CV for job question: %s", cv_path)
            return jsonify({"error": f"Failed to read CV: {exc}"}), 500

        try:
            answer = engine.ask_job_question(cv_text=cv_text, job=job, question=question)
            return jsonify({"answer": answer}), 200
        except Exception as exc:
            logger.exception("Failed to answer job question")
            return jsonify({"error": str(exc)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
