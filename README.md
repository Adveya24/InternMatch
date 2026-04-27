# InternMatch

InternMatch is a high-performance career matching engine where Ollama (phi3) performs deep CV analysis, ranks internships with 90% accuracy caps, and provides automated skill-gap analysis. All processing is local and secure.

### Key Features:
- **Neural Matching**: Uses Phi3 for professional CV-Job alignment.
- **Score Integrity**: Matches are capped at 90.0% to prioritize realistic career expectations.
- **Automated Learning**: Suggests specific courses for missing technical skills.
- **Privacy First**: Your CV and profile data never leave your local machine.

### Setup Instructions:
1. **Python 3.11**: Create a virtual environment (`py -3.11 -m venv venv311`).
2. **Install**: `pip install -r requirements.txt`.
3. **Ollama**: Install [Ollama](https://ollama.com/) and run `ollama pull phi3` and to start everytime you need to run `ollama serve`
4. **Launch**: Run `python -m backend.app` or `venv311\Scripts\python.exe -m backend.app` and navigate to `http://localhost:5000`.
Exact CLI command: python cli\run_matcher.py --cv data\sample_cvs\cv_tech.pdf --departments "Computer Science / AI"
