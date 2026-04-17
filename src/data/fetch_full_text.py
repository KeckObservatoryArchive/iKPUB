"""Download PDFs from ADS and extract full text for publications.

Text extraction is imperfect (broken equations, column merges, odd hyphenations)
but good enough for classification. ~10,000 papers takes ~5-15 min.

Usage:
    python src/data/fetch_full_text.py --year 2024
    python src/data/fetch_full_text.py --collection test_articles --start-year 2020 --end-year 2025
"""

import os
import json
import argparse
from pathlib import Path
import requests
import fitz  # pymupdf

from data.db_mongo_conn import from_env

PROJECT_ROOT = Path(__file__).parents[2]
PDF_BASE = PROJECT_ROOT / "data" / "pubs" / "pdf"
TEXT_BASE = PROJECT_ROOT / "data" / "pubs" / "full_text"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/pdf,application/octet-stream,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://ui.adsabs.harvard.edu/",
    "Connection": "keep-alive"
}


# --- PDF helpers ---

def valid_pdf(path):
    """Check whether a file is a valid readable PDF."""
    if not os.path.exists(path):
        return False

    if os.path.getsize(path) < 5000:
        return False

    try:
        doc = fitz.open(path)
        doc.close()
        return True
    except Exception:
        return False


def download_pdf(url, outpath, retries=3):
    """Download a PDF from a given URL and validate it."""
    outpath = Path(outpath)
    if outpath.exists() and valid_pdf(outpath):
        return True

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)

            if resp.status_code != 200:
                continue

            ctype = resp.headers.get("content-type", "").lower()

            if "text/html" in ctype:
                return False

            if "pdf" not in ctype and not resp.content.startswith(b"%PDF"):
                return False

            with open(outpath, "wb") as f:
                f.write(resp.content)

            if not valid_pdf(outpath):
                if os.path.exists(outpath):
                    os.remove(outpath)
                return False

            return True

        except Exception:
            pass

    return False


def extract_block_text(pdf_file):
    """Extract text from a PDF using block-based layout ordering."""
    doc = fitz.open(pdf_file)
    pages = []

    for page in doc:
        blocks = page.get_text("blocks")
        blocks = [b for b in blocks if b[4].strip()]
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
        text = "\n".join(b[4] for b in blocks)
        pages.append(text)

    doc.close()
    return "\n".join(pages)


# --- Pipeline logic ---

def parse_pdf_urls(links_data):
    """Extract PDF URLs from links_data (list of JSON strings or dicts)."""
    if not links_data:
        return []

    links = [json.loads(item) if isinstance(item, str) else item for item in links_data]

    urls = []
    for link in links:
        url = link.get("url", "")

        if url and "arxiv.org" in url:
            arxiv_id = url.split("arxiv.org/")[-1]
            if arxiv_id.startswith("abs/"):
                arxiv_id = arxiv_id[4:]
            urls.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")
        elif link.get("type") in ["pdf", "PUB_PDF", "EPRINT_PDF"]:
            urls.append(url)

    urls.sort(key=lambda x: "arxiv" not in x)
    return urls


def run(year, collection_name="test_articles"):
    pdf_dir = PDF_BASE / str(year)
    text_dir = TEXT_BASE / str(year)
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    conn = from_env("kpub", collection_name)
    rows = list(conn.collection.find(
        {"year": year},
        {"bibcode": 1, "links_data": 1},
    ))

    total = len(rows)
    print(f"Found {total} papers for {year} in {collection_name}")

    extracted = 0
    already_done = 0
    skipped = 0
    failed = 0

    for i, doc in enumerate(rows, 1):
        bibcode = doc["bibcode"]
        outfile = pdf_dir / f"{bibcode}.pdf"
        text_outfile = text_dir / f"{bibcode}.txt"

        if os.path.exists(text_outfile):
            already_done += 1
            continue

        pdf_urls = parse_pdf_urls(doc.get("links_data"))

        if not pdf_urls:
            skipped += 1
            print(f"[{i}/{total}] {bibcode} — SKIP: no PDF URLs")
            continue

        success = False
        for pdf_url in pdf_urls:
            if download_pdf(pdf_url, outfile):
                success = True
                break

        if not success:
            failed += 1
            print(f"[{i}/{total}] {bibcode} — FAIL: download failed")
            continue

        if not os.path.exists(outfile):
            failed += 1
            print(f"[{i}/{total}] {bibcode} — FAIL: PDF missing after download")
            continue

        text = extract_block_text(outfile)

        with open(text_outfile, "w") as tf:
            tf.write(text)

        extracted += 1
        print(f"[{i}/{total}] {bibcode} — ok")

    print(f"\nDone: {extracted} extracted, {already_done} already done, {skipped} skipped, {failed} failed (of {total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract text from ADS papers")
    parser.add_argument("--collection", default="test_articles",
                        help="MongoDB collection (default: test_articles)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year", type=int, help="single year to process")
    group.add_argument("--start-year", type=int, help="first year in range")
    parser.add_argument("--end-year", type=int, help="last year in range (required with --start-year)")
    args = parser.parse_args()

    years = [args.year] if args.year else range(args.start_year, args.end_year + 1)
    if not args.year and not args.end_year:
        parser.error("--end-year is required with --start-year")

    for y in years:
        run(y, args.collection)
