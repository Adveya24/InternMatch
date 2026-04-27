from __future__ import annotations

from pathlib import Path

from docx import Document
from fpdf import FPDF


BASE_DIR = Path(__file__).resolve().parent.parent
TXT_DIR = BASE_DIR / "data" / "sample_cvs"


def txt_to_docx(txt_path: Path, out_path: Path) -> None:
    text = txt_path.read_text(encoding="utf-8")
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)


def txt_to_pdf(txt_path: Path, out_path: Path) -> None:
    text = txt_path.read_text(encoding="utf-8")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in text.splitlines():
        pdf.multi_cell(190, 8, line or " ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(out_path)


def main() -> None:
    samples = [
        "cv_tech",
        "cv_nontech",
        "cv_mixed",
    ]
    for name in samples:
        txt = TXT_DIR / f"{name}.txt"
        if not txt.exists():
            continue
        txt_to_docx(txt, TXT_DIR / f"{name}.docx")
        txt_to_pdf(txt, TXT_DIR / f"{name}.pdf")


if __name__ == "__main__":
    main()
