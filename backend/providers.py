import os
import re
import logging
import requests
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger("internmatch.providers")

# Expand common abbreviations to full terms that APIs understand
_ABBREV_EXPANSIONS: Dict[str, List[str]] = {
    "cs": ["software", "computer"],
    "ai": ["machine learning", "artificial intelligence"],
    "ml": ["machine learning"],
    "ds": ["data science"],
    "ui": ["design"],
    "ux": ["design"],
    "hr": ["human resources"],
    "it": ["information technology"],
    "pm": ["product management"],
}

_DEPARTMENT_ALIASES: Dict[str, str] = {
    "Software Development / Engineering": "Computer Science / AI",
    "Data Science / AI / Machine Learning": "Data Science / Analytics",
    "Cloud / DevOps": "Computer Science / AI",
    "Mobile App Development": "Web Development",
    "Product Management": "Marketing / Business",
    "UI/UX Design": "Design / UX",
    "Digital Marketing": "Marketing / Business",
    "Finance / FinTech": "Finance / Accounting",
    "Business / Operations": "Marketing / Business",
    "Sales / Growth": "Retail / Sales",
    "Law / Legal": "Law / Legal",
    "Aviation / Aerospace": "Aviation / Aerospace",
    "Manufacturing / Mechanical": "Manufacturing / Mechanical",
}

_CRITICAL_KEYWORDS: Dict[str, List[str]] = {
    "Architecture / Civil": ["architecture", "architectural", "civil", "construction", "site engineer", "structural", "revit", "autocad", "surveying"],
    "Arts / Media": ["media", "content creator", "creative", "video", "editing", "photography", "branding", "illustration"],
    "Aviation / Aerospace": ["aerospace", "aviation", "aerodynamics", "aircraft", "flight", "propulsion", "avionics", "cfd", "ansys fluent"],
    "Biotech / Medical": ["biotech", "biotechnology", "medical", "clinical", "laboratory", "pcr", "pharma", "healthcare", "biology"],
    "Computer Science / AI": ["software engineer", "software developer", "computer science", "machine learning", "artificial intelligence", "python", "algorithms", "backend", "developer"],
    "Culinary / Food": ["culinary", "food", "kitchen", "chef", "recipe", "plating", "food safety", "restaurant"],
    "Data Science / Analytics": ["data science", "analytics", "data analyst", "data scientist", "business intelligence", "sql", "power bi", "tableau", "machine learning"],
    "Design / UX": ["ux", "ui", "product design", "figma", "wireframe", "prototype", "design system", "visual design", "user research"],
    "Education / Teaching": ["teaching", "teacher", "education", "curriculum", "classroom", "lesson plan", "tutoring", "student"],
    "Fashion / Textile": ["fashion", "textile", "apparel", "garment", "merchandising", "pattern making", "styling"],
    "Finance / Accounting": ["finance", "financial", "accounting", "bookkeeping", "audit", "valuation", "tax", "corporate finance", "investment"],
    "Hospitality / Tourism": ["hospitality", "tourism", "hotel", "travel", "guest relations", "front desk", "event", "reservation"],
    "Humanities / Journalism": ["journalism", "journalist", "editorial", "reporting", "writing", "editing", "interview", "news", "media writer"],
    "Law / Legal": ["legal", "law", "contract", "compliance", "litigation", "paralegal", "legal research", "policy"],
    "Logistics / Supply Chain": ["logistics", "supply chain", "warehouse", "procurement", "inventory", "operations", "shipment", "fulfillment"],
    "Manufacturing / Mechanical": ["mechanical", "manufacturing", "cad", "solidworks", "product design", "thermodynamics", "ansys", "machining", "production"],
    "Marketing / Business": ["marketing", "business", "brand", "campaign", "seo", "sem", "growth", "market research", "business strategy"],
    "NGO / Social Work": ["ngo", "nonprofit", "social work", "community outreach", "fundraising", "grant", "case management", "volunteer"],
    "Public Health / Epidemiology": ["public health", "epidemiology", "surveillance", "biostatistics", "health promotion", "community health", "research"],
    "Research / Policy": ["research", "policy", "policy analysis", "think tank", "literature review", "report writing", "survey", "analyst"],
    "Retail / Sales": ["sales", "retail", "store", "merchandising", "customer success", "lead generation", "account executive", "business development"],
    "Sports / Coaching": ["sports", "coaching", "fitness", "trainer", "athlete", "performance", "team coach"],
    "Veterinary / Animal Care": ["veterinary", "animal care", "vet", "pet care", "livestock", "clinic assistant", "animal health"],
    "Web Development": ["web developer", "frontend", "backend", "fullstack", "react", "html", "css", "javascript", "node.js"]
}

def _department_keywords(departments: List[str]) -> List[str]:
    """Map explicit departments securely to tight technical keywords."""
    words: List[str] = []
    for dep in departments:
        dep = _DEPARTMENT_ALIASES.get(dep, dep)
        if dep in _CRITICAL_KEYWORDS:
            words.extend(_CRITICAL_KEYWORDS[dep])
        else:
            stop_words = {"and", "or", "the", "of", "in", "for", "a", "an", "/", "-", "sciences", "technology", "development", "management", "science"}
            for token in dep.replace("/", " ").replace("-", " ").split():
                token_lower = token.strip().lower()
                if not token_lower or token_lower in stop_words:
                    continue
                if token_lower in _ABBREV_EXPANSIONS:
                    words.extend(_ABBREV_EXPANSIONS[token_lower])
                else:
                    words.append(token_lower)
    return list(dict.fromkeys(words))  # dedupe, preserve order


def department_keywords(departments: List[str]) -> List[str]:
    """Public wrapper for department keyword expansion."""
    return _department_keywords(departments)


def _contains_search_term(text: str, term: str) -> bool:
    normalized = term.strip().casefold()
    if not normalized:
        return False
    escaped = re.escape(normalized).replace(r"\ ", r"\s+")
    return re.search(rf"(?<!\w){escaped}(?!\w)", text) is not None


def department_relevance_score(job: Dict[str, Any], departments: List[str]) -> float:
    """
    Score how well a job aligns with the selected departments.
    Title hits are weighted heavily (0.50 per hit).
    Body hits are weighted lightly (0.05 per hit).
    Negative keywords (Sales/Marketing in a Tech lane) zero out the score.
    """
    if not departments:
        return 1.0

    title_text = str(job.get("title", "")).casefold()
    full_text = " ".join([
        str(job.get("title", "")),
        str(job.get("department", "")),
        str(job.get("company", "")),
        str(job.get("requirements", "")),
        str(job.get("skills_required", "")),
        " ".join(str(x) for x in (job.get("skills_required_list", []) or [])),
        str(job.get("description", ""))[:1200],
    ]).casefold()

    # Exclusion logic: If title says "Sales" but we want "Web Dev", it's a 0.
    excluded_terms = {
        "Web Development": ["sales", "marketing", "business development", "account executive", "recruiter", "hr", "human resources"],
        "Software Development / Engineering": ["sales", "marketing", "business development", "account executive", "recruiter", "hr", "human resources"],
        "Data Science / AI / Machine Learning": ["sales", "marketing", "business development", "account executive", "recruiter", "hr", "human resources"],
        "Cybersecurity": ["sales", "marketing", "business development", "account executive"],
    }

    best_score = 0.0
    for department in departments:
        # Check cross-department exclusion
        if department in excluded_terms:
            if any(term in title_text for term in excluded_terms[department]):
                continue

        terms = _department_keywords([department])
        if not terms:
            continue

        title_hits = sum(1 for term in terms if _contains_search_term(title_text, term))
        body_hits = sum(1 for term in terms if _contains_search_term(full_text, term))

        # Title match is critical. Body match is secondary.
        score = 0.0
        score += title_hits * 0.50
        score += body_hits * 0.05
        
        # High confidence signal if title matches
        if title_hits > 0:
            score += 0.20
            
        best_score = max(best_score, min(score, 1.0))

    return round(best_score, 3)


def filter_internships_for_departments(
    internships: List[Dict[str, Any]],
    departments: List[str],
    minimum_score: float = 0.18,
) -> List[Dict[str, Any]]:
    """Keep only internships with a meaningful explicit department signal."""
    if not departments:
        return internships

    filtered: List[Dict[str, Any]] = []
    for internship in internships:
        score = department_relevance_score(internship, departments)
        if score >= minimum_score:
            enriched = dict(internship)
            enriched["department_relevance"] = score
            filtered.append(enriched)

    filtered.sort(
        key=lambda item: (
            float(item.get("department_relevance", 0.0) or 0.0),
            str(item.get("_sort_date", "")),
        ),
        reverse=True,
    )
    return filtered

def _compile_search_regex(search_words: List[str]) -> re.Pattern[str] | None:
    if not search_words:
        return None
    escaped = [re.escape(w) for w in search_words]
    return re.compile(r'\b(?:' + '|'.join(escaped) + r')\b', re.IGNORECASE)


_SENIOR_TERMS = {"senior", "lead", "principal", "staff", "head", "director",
                  "manager", "vp", "president", "chief", "architect", "freelance"}

def is_valid_internship(title: str, text: str) -> bool:
    title_lower = title.lower()
    if any(t in title_lower.split() for t in _SENIOR_TERMS):
        return False
    
    # 1. First Priority: Explicit Internships
    if re.search(r'\b(?:intern(?:ships?|s)?|student|stage|trainee)\b', title_lower):
        return True
        
    # 2. Fallback Priority: Junior, Entry-Level, Freshers
    if re.search(r'\b(?:junior|entry(?:-level)?|associate|graduate|fresher|new grad)\b', title_lower):
        return True
        
    return False


class RemotiveProvider:
    def fetch_internships(self, keywords: List[str]) -> List[Dict[str, Any]]:
        logger.info("Using Remotive provider")
        search_words = _department_keywords(keywords)
        search_regex = _compile_search_regex(search_words)
        results = []

        # Always search "intern" first to get actual internship postings
        # Then add domain keyword as secondary query for more coverage
        search_queries = ["intern"]
        if search_words and search_words[0] not in ("intern",):
            search_queries.append(search_words[0])

        seen_ids: set = set()
        for query in search_queries:
            if len(results) >= 20:
                break
            url = f"https://remotive.com/api/remote-jobs?search={requests.utils.quote(query)}"
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                jobs = data.get("jobs", [])
                logger.info(f"Remotive query='{query}' returned {len(jobs)} jobs")

                for job in jobs:
                    job_id = job.get("id", job.get("url", ""))
                    if job_id in seen_ids:
                        continue

                    title_lower = str(job.get("title", "")).lower()
                    department = keywords[0] if keywords else "Internship"
                    tags = job.get("tags", [])
                    if isinstance(tags, str):
                        tags = [tags]
                        
                    job_text = " ".join([
                        str(job.get("title", "")),
                        department,
                        " ".join(tags),
                        (job.get("description", "") or "")[:600],
                    ]).lower()

                    if not is_valid_internship(job.get("title", ""), job_text):
                        continue

                    # For domain queries: filter by domain keyword
                    if query != "intern" and search_regex and not search_regex.search(job_text):
                        continue

                    seen_ids.add(job_id)
                    results.append({
                        "title": job.get("title", ""),
                        "company": job.get("company_name", ""),
                        "location": job.get("candidate_required_location", "Remote"),
                        "department": department,
                        "description": (job.get("description", "") or "")[:1000],
                        "requirements": ", ".join(job.get("tags", [])) or department,
                        "skills_required_list": job.get("tags", []),
                        "source": "Remotive",
                        "apply_url": job.get("url", ""),
                        "rating": "",
                        "reviews": "",
                        "_sort_date": job.get("publication_date", datetime.now().isoformat())
                    })

                    if len(results) >= 20:
                        break

            except Exception as e:
                logger.error(f"Remotive API error (query='{query}'): {e}")

        logger.info(f"Remotive total results: {len(results)}")
        return results


class TheirStackProvider:
    API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ2ZXIiOjEsImp0aSI6IjhlMDgxYzU1LWM3M2ItNDhlYS1iNTQ5LWMxNDgyMGVmYzAwYiIsImNyZWF0ZWRfYnkiOjE1MDM3NSwicGVybWlzc2lvbnMiOltdLCJhdWQiOiJhcGkiLCJpYXQiOjE3NzQxMDExMTEsInN1YiI6IjE0OTc0NSIsIm5hbWUiOiJJbnRlcm5NYXRjaCIsImVtYWlsIjoiYWR2ZXlhMjRAZ21haWwuY29tIn0.-30ML5TDgh2oJh_B0FClNLEptRBnY20GiLwh-h8P-90"

    def __init__(self):
        self.api_key = os.getenv("THEIRSTACK_API_KEY", self.API_KEY)

    def fetch_internships(self, keywords: List[str]) -> List[Dict[str, Any]]:
        if not self.api_key:
            logger.warning("THEIRSTACK_API_KEY not provided. TheirStack disabled.")
            return []

        logger.info("Using TheirStack provider")
        url = "https://api.theirstack.com/v1/jobs/search"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        search_words = _department_keywords(keywords)
        search_regex = _compile_search_regex(search_words)

        payload = {
            "job_title_or": ["intern", "internship"],
            "limit": 20,
            "posted_at_max_age_days": 30,
        }

        results = []
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            jobs = data.get("data", [])
            logger.info(f"TheirStack returned {len(jobs)} jobs")

            for job in jobs:
                title = str(job.get("job_title", "Internship"))
                company = str(job.get("company", "Unknown Company"))
                desc = str(job.get("description", "") or "")
                
                job_text = " ".join([title, company, desc[:600]]).lower()
                
                if not is_valid_internship(title, job_text):
                    continue
                    
                if search_regex and not search_regex.search(job_text):
                    continue

                techs = job.get("keyword_slugs", job.get("technology_slugs", []))
                loc = job.get("short_location") or job.get("long_location") or job.get("country") or "Remote"
                results.append({
                    "title": title,
                    "company": company,
                    "location": loc,
                    "department": keywords[0] if keywords else "Internship",
                    "description": desc[:1000],
                    "requirements": ", ".join(techs),
                    "skills_required_list": techs,
                    "source": "TheirStack",
                    "apply_url": job.get("url", ""),
                    "rating": "",
                    "reviews": "",
                    "_sort_date": str(job.get("date_posted", datetime.now().isoformat()))
                })
        except Exception as e:
            logger.error(f"TheirStack API error: {e}")

        logger.info(f"TheirStack total results: {len(results)}")
        return results


class RapidAPIInternshipsProvider:
    API_KEY = "df6a9c9dafmshc5982067ddd6d55p1ded13jsncc4fc24f447c"
    API_HOST = "internships-api.p.rapidapi.com"
    BASE_URL = "https://internships-api.p.rapidapi.com/active-jb-7d"

    def __init__(self):
        self.api_key = os.getenv("RAPIDAPI_INTERNSHIPS_KEY", self.API_KEY)

    def fetch_internships(self, keywords: List[str]) -> List[Dict[str, Any]]:
        logger.info("Using RapidAPI Internships provider")
        headers = {
            "x-rapidapi-host": self.API_HOST,
            "x-rapidapi-key": self.api_key,
            "Content-Type": "application/json",
        }

        search_words = _department_keywords(keywords)
        search_regex = _compile_search_regex(search_words)
        results = []
        try:
            # Use title_filter to get only internship postings
            params: Dict[str, str] = {"title_filter": "intern OR internship"}
            # Add domain keywords as description filter if provided
            if search_words:
                params["description_filter"] = " OR ".join(search_words[:3])  # type: ignore[index]

            resp = requests.get(self.BASE_URL, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                data = data.get("data", data.get("jobs", []))
            jobs = data if isinstance(data, list) else []
            logger.info(f"RapidAPI Internships returned {len(jobs)} jobs")

            for job in jobs:
                # Filter: must explicitly be a valid internship and not a senior isolated role
                title_l = str(job.get("title", ""))
                desc_l = str(job.get("description", "") or "")
                if not is_valid_internship(title_l, desc_l):
                    continue

                title = str(job.get("title", "Internship"))
                company = str(job.get("organization", job.get("company", "Unknown")))
                # Extract location from locations_derived array
                locs: list = job.get("locations_derived") or []
                location = next(iter(locs), "Remote" if job.get("remote_derived") else "Unknown")
                description = str(job.get("description", "") or "")
                skills = job.get("skills", job.get("tags", []))
                if isinstance(skills, str):
                    skills = [s.strip() for s in skills.split(",") if s.strip()]
                apply_url = str(job.get("url", ""))

                desc_preview: str = description[:600]
                job_text = " ".join([title, company, location, desc_preview]).lower()

                if search_regex and not search_regex.search(job_text):
                    continue

                results.append({
                    "title": title,
                    "company": company,
                    "location": location,
                    "department": keywords[0] if keywords else "Internship",
                    "description": description[:1000],  # type: ignore[index]
                    "requirements": ", ".join(skills) if skills else "",
                    "skills_required_list": skills,
                    "source": "RapidAPI",
                    "apply_url": apply_url,
                    "rating": "",
                    "reviews": "",
                    "_sort_date": str(job.get("date_posted", datetime.now().isoformat()))
                })

                if len(results) >= 20:
                    break

        except Exception as e:
            logger.error(f"RapidAPI Internships error: {e}")

        logger.info(f"RapidAPI Internships total results: {len(results)}")
        return results


class ArbeitnowProvider:
    def fetch_internships(self, keywords: List[str]) -> List[Dict[str, Any]]:
        logger.info("Using Arbeitnow provider")
        search_words = _department_keywords(keywords)
        search_regex = _compile_search_regex(search_words)
        results = []
        try:
            url = "https://www.arbeitnow.com/api/job-board-api"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            logger.info(f"Arbeitnow returned {len(data)} jobs")
            for job in data:
                title = str(job.get("title", ""))
                company = str(job.get("company_name", ""))
                loc = str(job.get("location", ""))
                desc = str(job.get("description", ""))
                tags = job.get("tags", [])
                
                desc_clean = re.sub(r'<[^>]*>', ' ', desc).strip()
                job_text = " ".join([title, company, loc, desc_clean[:600]]).lower()
                
                if not is_valid_internship(title, job_text):
                    continue
                if search_regex and not search_regex.search(job_text):
                    continue
                    
                results.append({
                    "title": title,
                    "company": company,
                    "location": loc,
                    "department": keywords[0] if keywords else "Internship",
                    "description": desc_clean[:1000],
                    "requirements": ", ".join(tags),
                    "skills_required_list": tags,
                    "source": "Arbeitnow",
                    "apply_url": str(job.get("url", "")),
                    "rating": "",
                    "reviews": "",
                    "_sort_date": str(job.get("created_at", datetime.now().isoformat()))
                })
                if len(results) >= 20: break
        except Exception as e:
            logger.error(f"Arbeitnow API error: {e}")
        return results

class GreenhouseProvider:
    def fetch_internships(self, keywords: List[str]) -> List[Dict[str, Any]]:
        logger.info("Using Greenhouse provider")
        search_words = _department_keywords(keywords)
        search_regex = _compile_search_regex(search_words)
        results = []
        companies = [
            "figma", "canonical", "discord", "reddit", "stripe", "airbnb", 
            "pinterest", "dropbox", "lyft", "asana", "robinhood", "door_dash", "snowflake"
        ]
        
        for company in companies:
            if len(results) >= 25: break
            try:
                url = f"https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true"
                resp = requests.get(url, timeout=10)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for job in data.get("jobs", []):
                    title = str(job.get("title", ""))
                    loc = str(job.get("location", {}).get("name", "Remote"))
                    desc_html = str(job.get("content", ""))
                    
                    desc = re.sub(r'<[^>]*>', ' ', desc_html).strip() or "Please visit the application page for full internship details."
                    job_text = " ".join([title, loc, desc[:600]]).lower()
                    
                    if not is_valid_internship(title, job_text):
                        continue
                    if search_regex and not search_regex.search(job_text):
                        continue

                    results.append({
                        "title": title,
                        "company": company.title(),
                        "location": loc,
                        "department": keywords[0] if keywords else "Internship",
                        "description": desc[:3000],
                        "requirements": "",
                        "skills_required_list": [],
                        "source": "Greenhouse",
                        "apply_url": str(job.get("absolute_url", "")),
                        "rating": "",
                        "reviews": "",
                        "_sort_date": str(job.get("updated_at", datetime.now().isoformat()))
                    })
            except Exception as e:
                logger.error(f"Greenhouse API error for {company}: {e}")
        return results

class LeverProvider:
    def fetch_internships(self, keywords: List[str]) -> List[Dict[str, Any]]:
        logger.info("Using Lever provider")
        search_words = _department_keywords(keywords)
        search_regex = _compile_search_regex(search_words)
        results = []
        companies = [
            "notion", "anthropic", "scaleapi", "canva", "spotify", 
            "databricks", "plaid", "affirm", "ramp", "vanta", "rippling", "verkada", "palantir"
        ]
        
        for company in companies:
            if len(results) >= 25: break
            try:
                url = f"https://api.lever.co/v0/postings/{company}"
                resp = requests.get(url, timeout=10)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for job in data:
                    title = str(job.get("text", ""))
                    core_desc = str(job.get("descriptionPlain", ""))
                    bullet_points = []
                    for l in job.get("lists", []):
                        bullet_points.append(str(l.get("text", "")))
                        clean_content = re.sub(r'<[^>]*>', ' ', str(l.get("content", ""))).strip()
                        bullet_points.append(clean_content)
                        
                    desc = core_desc + " " + " ".join(bullet_points)
                    categories = job.get("categories") or {}
                    loc = str(categories.get("location", "Remote"))
                    dep = str(categories.get("department", "Internship"))
                    
                    job_text = " ".join([title, dep, desc[:600]]).lower()
                    if not is_valid_internship(title, job_text):
                        continue
                    if search_regex and not search_regex.search(job_text):
                        continue
                        
                    results.append({
                        "title": title,
                        "company": company.title(),
                        "location": loc,
                        "department": dep,
                        "description": desc[:3000],
                        "requirements": dep,
                        "skills_required_list": [],
                        "source": "Lever",
                        "apply_url": str(job.get("hostedUrl", "")),
                        "rating": "",
                        "reviews": "",
                        "_sort_date": datetime.now().isoformat()
                    })
            except Exception as e:
                logger.error(f"Lever API error for {company}: {e}")
        return results

class InternshalaProvider:
    def fetch_internships(self, keywords: List[str]) -> List[Dict[str, Any]]:
        search_words = _department_keywords(keywords)
        search_regex = _compile_search_regex(search_words)
        results = []
        try:
            from bs4 import BeautifulSoup
            url = "https://internshala.com/internships/work-from-home-internships/"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            cards = soup.find_all("div", class_="individual_internship")
            
            for card in cards:
                title_elem = card.find("h3")
                company_elem = card.find("h4")
                if not title_elem or not company_elem:
                    continue
                    
                title = title_elem.text.strip()
                company = company_elem.text.strip()
                loc_elem = card.find("a", class_="location_link")
                loc = loc_elem.text.strip() if loc_elem else "Work From Home"
                
                apply_link = card.find("a", href=True)
                apply_url = "https://internshala.com" + apply_link['href'] if apply_link and apply_link['href'].startswith("/") else (apply_link['href'] if apply_link else "https://internshala.com")
                
                job_text = " ".join([title, company, loc]).lower()
                
                if not is_valid_internship(title, job_text):
                    continue
                if search_regex and not search_regex.search(job_text):
                    continue

                results.append({
                    "title": title,
                    "company": company,
                    "location": loc,
                    "department": keywords[0] if keywords else "Internship",
                    "description": f"Internshala opportunity: {title} at {company}. Apply directly on Internshala to view the full stipend and requirement breakdown.",
                    "requirements": "",
                    "skills_required_list": [],
                    "source": "Internshala",
                    "apply_url": apply_url,
                    "rating": "",
                    "reviews": "",
                    "_sort_date": datetime.now().isoformat()
                })
                if len(results) >= 20: break
        except ImportError:
            logger.error("beautifulsoup4 not installed. Run: pip install beautifulsoup4")
        except Exception as e:
            logger.error(f"Internshala Scraping error: {e}")
            
        return results
