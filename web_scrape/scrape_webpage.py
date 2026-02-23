#!/usr/bin/env python3
"""Scrape a webpage and save the extracted text to a .txt file."""

import argparse
import re
import sys
from typing import List, Tuple
from pathlib import Path
from urllib.parse import urlparse, urljoin

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Required packages: requests, beautifulsoup4")
    print("Install with: pip install requests beautifulsoup4")
    sys.exit(1)


def extract_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Extract all http/https links from the page."""
    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        if parsed.scheme in ("http", "https") and full_url not in seen:
            seen.add(full_url)
            links.append(full_url)
    return sorted(links)


def scrape_url(url: str) -> Tuple[str, List[str]]:
    """Fetch a webpage and return (text content, list of links)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    links = extract_links(soup, url)

    # Remove script, style, and other non-content tags
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse multiple newlines and spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip(), links


def filename_from_url(url: str) -> str:
    """Generate a safe filename from the URL."""
    parsed = urlparse(url)
    name = parsed.netloc or "page"
    name = re.sub(r"[^\w\-.]", "_", name)
    return f"{name}.txt"


def main():
    parser = argparse.ArgumentParser(
        description="Scrape a webpage and save text to a file")
    parser.add_argument("url", help="URL of the webpage to scrape")
    parser.add_argument(
        "-o",
        "--output",
        help="Output .txt file name or path (default: derived from URL)",
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        default="scraped_pages",
        help="Directory to store output file (default: scraped_pages)",
    )
    args = parser.parse_args()

    url = args.url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(filename_from_url(url))

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_path.is_absolute():
            output_path = output_dir / output_path.name

    if output_path.suffix != ".txt":
        output_path = output_path.with_suffix(".txt")

    print(f"Fetching: {url}")
    text, links = scrape_url(url)

    output_path.write_text(text, encoding="utf-8")
    print(f"Saved {len(text)} characters to {output_path}")

    if links:
        print(f"\nLinks found ({len(links)}):")
        for link in links:
            print(f"  {link}")


if __name__ == "__main__":
    main()
