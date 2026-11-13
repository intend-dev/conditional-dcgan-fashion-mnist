#!/usr/bin/env python3
r"""Build Generative_AI_Analysis_Report.pdf from Generative_AI_Analysis_Report.md
(Chrome headless) and verify it.

Run from the repo root (or anywhere; paths resolve from this file):

    .venv\Scripts\python.exe build\build_report.py

Exit 0 only when every check passes; exit 1 prints each failing check.
The script never writes Generative_AI_Analysis_Report.md.

Rendering: a centered cover page split off before the first h2, per-page page
numbers via the @page counter, each image wrapped with its caption in one
<figure> block so a page break can never separate the two, reference URLs
wrapped at a smaller size, and the References section starting on its own
page. The caption regex matches this project's plain (non-bold) 'Figure N:'
captions.

Verification: all eight section headings present with matching PDF bookmarks,
every pinned value from the executed notebook run present in the PDF text,
exactly seven embedded images, every 'Figure N:' caption on a page that
carries an image (fig0-real-samples is Figure 1), the cover fields on page 1,
and the page count inside a 10-30 sanity band. The report carries no date
line, so none is checked.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "Generative_AI_Analysis_Report.md"
PDF = ROOT / "Generative_AI_Analysis_Report.pdf"
OUT_DIR = ROOT / "build" / "out"
HTML = OUT_DIR / "report.html"
PREV_PDF = ROOT / "test-runs" / "Generative_AI_Analysis_Report.previous.pdf"
CHROME = Path(os.environ.get("CHROME_BIN") or r"C:\Program Files\Google\Chrome\Application\chrome.exe")

HEADINGS = (
    "Overview",
    "Dataset or Prompt Description",
    "Model Design and Training Approach",
    "Output Evaluation and Interpretation",
    "Interpretation for a Non-Technical Audience",
    "Ethical Considerations and Responsible Use",
    "Limitations and Future Improvements",
    "References",
)

COVER_FIELDS = ("Author:", "Dataset:")
TITLE = "Generative AI Applications: Analysis Report"
AUTHOR = "Francisco Mateus"

# Pinned values from the executed notebook run, as they appear in the report.
PINNED = (
    ("70,000", "Fashion-MNIST images"),
    ("10,000", "validation / test split size"),
    ("50,000", "classifier training pool"),
    ("2,149,409", "generator parameters"),
    ("1,467,648", "generator dense projection parameters"),
    ("421,834", "classifier parameters"),
    ("2e-4", "Adam learning rate"),
    ("(0.5, 0.999)", "Adam betas"),
    ("50", "GAN training epochs"),
    ("128", "batch size"),
    ("42", "fixed seed"),
    ("444", "training seconds"),
    ("8.9", "seconds per epoch"),
    ("RTX 4080", "training GPU"),
    ("0.9019", "classifier epoch-1 validation accuracy"),
    ("0.9199", "classifier final validation accuracy"),
    ("0.9074", "classifier real test accuracy (evaluated once)"),
    ("0.9008", "generated-tile macro agreement"),
    ("0.0066", "generated-vs-real agreement gap"),
    ("0.75", "T-shirt/top generated agreement"),
    ("0.81", "T-shirt/top real-test agreement"),
    ("187", "real T-shirt/top tiles mislabelled"),
    ("135", "real T-shirt/top tiles read as Shirt"),
    ("167", "generated T-shirt/top tiles read as Shirt"),
    ("0.8830", "train-mode mixed-batch agreement"),
    ("0.1070", "train-mode single-class agreement"),
    ("0.8980", "eval-mode macro agreement"),
    ("1.9997", "train-mode cross-composition pixel difference"),
    ("0.0000", "eval-mode cross-composition pixel difference"),
    ("0.0022", "batch-shape floating-point control"),
    ("20.9619", "Fréchet distance generated vs real test"),
    ("1.3127", "real-vs-real Fréchet reference"),
    ("19.7376", "per-class Fréchet minimum (Trouser)"),
    ("77.4557", "per-class Fréchet maximum (Bag)"),
    ("0.87", "intra-class diversity macro ratio"),
    ("0.77", "diversity ratio minimum (T-shirt/top and Dress)"),
    ("0.98", "diversity ratio maximum (Sandal)"),
    ("0.9020", "seed-43 macro agreement"),
    ("0.8843", "seed-44 macro agreement"),
    ("0.9100", "one-sided smoothing agreement"),
    ("16.1506", "one-sided smoothing feature distance"),
    ("12", "flagged tiles in the failure strip"),
    ("row 8, column 1", "named failure tile"),
)

IMAGE_COUNT = 7
# Sanity band only: an accident alarm for silent content loss or runaway
# duplication. Never a length target - page count is an output of the
# content, not a constraint to justify against.
MIN_PAGES = 10
MAX_PAGES = 30

MD_EXTENSIONS = ["tables", "fenced_code", "toc", "attr_list", "md_in_html"]

# Print CSS for the letter-size render; .ref-url wraps reference URLs so they
# cannot overflow the margin.
CSS = """
@page {
  size: letter;
  margin: 0.9in;
  @bottom-right {
    content: counter(page);
    font-family: "Times New Roman", Times, serif;
    font-size: 10px;
    color: #000;
  }
}
body {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 16px;
  line-height: 1.5;
  color: #000;
  margin: 0;
  overflow-wrap: break-word;
}
.cover {
  page-break-after: always;
  break-after: page;
  text-align: center;
  padding-top: 1.5in;
}
.cover h1 {
  font-size: 26.7px;
  line-height: 1.3;
  margin: 0 0 1.6em;
}
.cover p {
  margin: 0.15em 0;
  font-size: 16px;
}
h2 {
  font-size: 21.3px;
  margin: 1.5em 0 0.6em;
  page-break-after: avoid;
  break-after: avoid;
}
h3 {
  font-size: 18px;
  margin: 1.2em 0 0.4em;
  page-break-after: avoid;
  break-after: avoid;
}
h2.new-page {
  page-break-before: always;
  break-before: page;
}
p { margin: 0 0 0.75em; orphans: 3; widows: 3; }
p:has(+ ul), p:has(+ ol), p:has(+ table) {
  page-break-after: avoid;
  break-after: avoid;
}
ul, ol { margin: 0.4em 0 0.9em 1.5em; padding: 0; page-break-inside: avoid; break-inside: avoid; }
li { margin-bottom: 0.25em; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.5em 0 1em;
  font-size: 14.7px;
  page-break-inside: avoid;
  break-inside: avoid;
}
th, td {
  border: 1px solid #555;
  padding: 0.25em 0.5em;
  text-align: left;
}
th { background: #eeeeee; }
figure {
  margin: 1em 0;
  page-break-inside: avoid;
  break-inside: avoid;
  text-align: center;
}
figure img {
  max-width: 100%;
  height: auto;
}
figcaption {
  font-size: 14.7px;
  margin-top: 0.4em;
  text-align: left;
}
.ref-url {
  font-size: 12px;
  overflow-wrap: anywhere;
  word-break: break-all;
}
"""

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Generative AI Applications: Analysis Report</title>
<style>
{css}
</style>
</head>
<body>
{body}
</body>
</html>
"""

# An image paragraph followed by its 'Figure N:' caption becomes one <figure>
# block, so a page break can never separate the two. This project's captions
# are plain (non-bold) paragraphs.
FIGURE_RE = re.compile(
    r"<p>(<img\b[^>]*>)</p>\s*<p>(Figure \d+:.*?)</p>",
    re.S,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(lines: list[str]) -> int:
    for line in lines:
        print(f"FAIL {line}")
    return 1


def absolutize_images(html_text: str) -> str:
    """Point every <img src> at an absolute file:/// URI; a missing figure fails."""
    def repl(match: re.Match[str]) -> str:
        src = match.group(1)
        target = (ROOT / src).resolve()
        if not target.is_file():
            raise SystemExit(f"FAIL figure not found: {src} -> {target}")
        return f'src="{target.as_uri()}"'

    return re.sub(r'src="([^"]+)"', repl, html_text)


def wrap_figures(html_text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return f"<figure>{match.group(1)}<figcaption>{match.group(2)}</figcaption></figure>"

    return FIGURE_RE.sub(repl, html_text)


def split_cover(html_text: str) -> str:
    """Everything before the first h2 is the cover page; the rest is the body."""
    index = html_text.find("<h2")
    if index == -1:
        raise SystemExit("FAIL no <h2> section heading found")
    cover, body = html_text[:index], html_text[index:]
    return f'<section class="cover">{cover}</section>\n{body}'


def has_page_number(page, number: int) -> bool:
    """True when the page's bottom margin carries its own page number."""
    content_height = float(page.mediabox.height) * 96 / 72
    found = False

    def visit(text, cm, tm, font_dict, font_size):
        nonlocal found
        if text.strip() == str(number) and tm[5] > 0.9 * content_height:
            found = True

    page.extract_text(visitor_text=visit)
    return found


def heading_pages(reader: PdfReader) -> dict[str, int]:
    """Page index for each section heading, scanned in document order."""
    page_for: dict[str, int] = {}
    start = 0
    for heading in HEADINGS:
        for index in range(start, len(reader.pages)):
            flat = re.sub(r"\s+", " ", reader.pages[index].extract_text() or "")
            if heading in flat:
                page_for[heading] = index
                start = index
                break
    return page_for


def add_outline_and_metadata() -> None:
    """Add the eight section bookmarks and the title/author metadata."""
    reader = PdfReader(str(PDF))
    page_for = heading_pages(reader)
    missing = [h for h in HEADINGS if h not in page_for]
    if missing:
        raise SystemExit(f"FAIL bookmark target not found: {missing}")

    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata({"/Title": TITLE, "/Author": AUTHOR})
    for heading in HEADINGS:
        writer.add_outline_item(heading, page_for[heading])

    temp = OUT_DIR / "report.outline.pdf"
    with open(temp, "wb") as handle:
        writer.write(handle)
    temp.replace(PDF)


def main() -> int:
    if not MD.is_file():
        return fail([f"source missing: {MD}"])
    if not CHROME.is_file():
        return fail([f"chrome missing: {CHROME}"])

    md_hash_before = sha256(MD)
    html_text = markdown.markdown(
        MD.read_text(encoding="utf-8"), extensions=MD_EXTENSIONS
    )
    html_text = absolutize_images(html_text)
    html_text = wrap_figures(html_text)
    html_text = split_cover(html_text)
    # Reference URLs at a smaller size that wraps instead of running into the
    # margin.
    html_text = re.sub(
        r"(https?://[^\s<\"]+)", r'<span class="ref-url">\1</span>', html_text
    )
    # References start on their own page.
    html_text = html_text.replace("<h2>References</h2>", '<h2 class="new-page">References</h2>', 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML.write_text(TEMPLATE.format(css=CSS, body=html_text), encoding="utf-8")

    if PDF.exists():
        PREV_PDF.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PDF, PREV_PDF)
        PDF.unlink()

    cmd = [
        str(CHROME),
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF}",
        HTML.as_uri(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return fail([f"chrome exit {proc.returncode}", (proc.stderr or "").strip()[:500]])
    if not PDF.is_file():
        return fail(["chrome reported success but wrote no PDF"])
    if PDF.stat().st_size == 0:
        return fail(["PDF is empty"])
    if PDF.read_bytes()[:4] != b"%PDF":
        return fail(["PDF does not start with %PDF"])

    add_outline_and_metadata()

    reader = PdfReader(str(PDF))
    problems: list[str] = []
    texts: list[str] = []
    image_count = 0
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        texts.append(text)
        page_images = len(page.images)
        image_count += page_images
        if not text.strip() and page_images == 0:
            problems.append(f"blank page {number}")
        if not has_page_number(page, number):
            problems.append(f"page number missing on page {number}")

    flat = re.sub(r"\s+", " ", "\n".join(texts))

    cover_text = texts[0] if texts else ""
    for field in COVER_FIELDS:
        if field not in cover_text:
            problems.append(f"cover field missing on page 1: {field}")
    if "Overview" in cover_text:
        problems.append("cover page contains body content (Overview)")

    for heading in HEADINGS:
        if heading not in flat:
            problems.append(f"heading missing: {heading}")
    for value, label in PINNED:
        if value not in flat:
            problems.append(f"pinned value missing: {value} ({label})")
    if image_count != IMAGE_COUNT:
        problems.append(f"image count {image_count} != {IMAGE_COUNT}")

    # Each report figure caption must sit on a page that carries an image.
    for number, page in enumerate(reader.pages, 1):
        text = texts[number - 1]
        for figure in ("Figure 1:", "Figure 2:", "Figure 3:", "Figure 4:",
                       "Figure 5:", "Figure 6:", "Figure 7:"):
            if figure in text and len(page.images) == 0:
                problems.append(f"{figure} caption on page {number} without an image")

    if not (MIN_PAGES <= len(reader.pages) <= MAX_PAGES):
        problems.append(f"page count {len(reader.pages)} out of {MIN_PAGES}-{MAX_PAGES}")

    outline = [item for item in reader.outline if not isinstance(item, list)]
    titles = [str(item.title) for item in outline]
    if titles != list(HEADINGS):
        problems.append(f"bookmarks mismatch: {titles}")
    else:
        expected = heading_pages(reader)
        for item in outline:
            if reader.get_destination_page_number(item) != expected.get(str(item.title)):
                problems.append(f"bookmark target wrong: {item.title}")

    metadata = reader.metadata or {}
    if (metadata.get("/Author") or "") != AUTHOR:
        problems.append(f"PDF author metadata missing or wrong: {metadata.get('/Author')!r}")
    if (metadata.get("/Title") or "") != TITLE:
        problems.append(f"PDF title metadata missing or wrong: {metadata.get('/Title')!r}")

    text = "\n".join(texts)
    if "`" in text:
        problems.append("literal backtick in extracted text")
    if "$" in text:
        problems.append("literal dollar sign in extracted text")
    if "\ufffd" in text:
        problems.append("U+FFFD replacement character in extracted text")

    md_hash_after = sha256(MD)
    if md_hash_after != md_hash_before:
        problems.append("source Markdown changed during the build")

    if problems:
        return fail(problems)

    print(f"PASS pages={len(reader.pages)} images={image_count} bookmarks={len(outline)} size={PDF.stat().st_size} bytes")
    print(f"pdf sha256={sha256(PDF)}")
    print(f"md  sha256={md_hash_after}  source md unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())

