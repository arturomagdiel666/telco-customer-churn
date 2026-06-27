"""Convierte reports/report.md -> report.html -> report.pdf usando markdown + Chromium (Playwright).

Uso:  python reports/build_pdf.py
Requiere:  pip install markdown playwright  &&  python -m playwright install chromium
"""
from __future__ import annotations
from pathlib import Path
import markdown

REPORTS = Path(__file__).resolve().parent
MD = REPORTS / "report.md"
HTML = REPORTS / "report.html"
PDF = REPORTS / "report.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
       font-size: 11pt; line-height: 1.5; color: #1b2a32; max-width: 100%; }
h1 { color: #264653; font-size: 22pt; border-bottom: 3px solid #2a9d8f; padding-bottom: 6px; }
h2 { color: #264653; font-size: 16pt; border-bottom: 1px solid #d8e2e0; padding-bottom: 4px;
     margin-top: 26px; page-break-after: avoid; }
h3 { color: #2a9d8f; font-size: 12.5pt; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #c9d6d3; padding: 5px 8px; text-align: left; }
th { background: #2a9d8f; color: #fff; }
tr:nth-child(even) td { background: #f3f8f6; }
img { max-width: 100%; height: auto; display: block; margin: 10px auto;
      border: 1px solid #e0e6e5; border-radius: 4px; page-break-inside: avoid; }
em { color: #5a6f76; }
blockquote { border-left: 4px solid #e9c46a; background: #fdf6e3; margin: 12px 0;
             padding: 8px 14px; color: #6b5a1f; page-break-inside: avoid; }
code { background: #eef3f2; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
pre { background: #1b2a32; color: #e8eef0; padding: 12px; border-radius: 6px;
      overflow-x: auto; font-size: 8.5pt; page-break-inside: avoid; }
pre code { background: transparent; color: inherit; }
hr { border: none; border-top: 1px solid #d8e2e0; margin: 20px 0; }
"""

def main() -> None:
    md_text = MD.read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<title>Reporte Customer Churn</title><style>{CSS}</style></head>
<body>{body}</body></html>"""
    HTML.write_text(html, encoding="utf-8")
    print(f"HTML escrito: {HTML} ({HTML.stat().st_size/1024:.1f} KB)")

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(HTML.as_uri(), wait_until="networkidle")
        page.pdf(path=str(PDF), format="A4", print_background=True,
                 margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"})
        browser.close()
    print(f"PDF escrito: {PDF} ({PDF.stat().st_size/1024:.1f} KB)")

if __name__ == "__main__":
    main()
