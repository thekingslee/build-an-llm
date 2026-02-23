#!/usr/bin/env python3
"""Scrape a webpage and save the extracted text to a .txt file."""

import argparse
import re
import sys
from typing import List, Optional, Set, Tuple
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
        # Drop fragments to avoid duplicates like /page#section
        full_url = parsed._replace(fragment="").geturl()
        parsed = urlparse(full_url)
        if parsed.scheme in ("http", "https") and full_url not in seen:
            seen.add(full_url)
            links.append(full_url)
    return sorted(links)


def find_main_content_node(soup: BeautifulSoup):
    """
    Heuristically find the main content node for a generic webpage.

    Strategy (generic, not site-specific):
      1. Prefer <main>, [role=main], or <article> if present.
      2. Otherwise, pick the <div>/<section> with the most text and <p> tags.
      3. Fallback to the whole document if nothing stands out.
    """
    # 1. Prefer semantic containers
    candidates = []
    for selector in ("main", "[role=main]", "article"):
        for node in soup.select(selector):
            text = node.get_text(" ", strip=True)
            if not text:
                continue
            p_count = len(node.find_all("p"))
            score = len(text) + p_count * 100
            candidates.append((score, node))

    # 2. Fallback: densest div/section with paragraphs
    if not candidates:
        for node in soup.find_all(["div", "section"]):
            p_count = len(node.find_all("p"))
            if p_count == 0:
                continue
            text = node.get_text(" ", strip=True)
            if not text:
                continue
            score = len(text) + p_count * 100
            candidates.append((score, node))

    if not candidates:
        return soup

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def scrape_url(url: str) -> Tuple[str, List[str]]:
    """Fetch a webpage and return (main text content, list of links)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    links = extract_links(soup, url)

    # Remove obviously non-content tags everywhere first
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Heuristically pick the main content node
    main_node = find_main_content_node(soup)

    # Strip nav-like elements inside the chosen main node
    for tag in main_node.find_all(["nav", "header", "footer", "aside"]):
        tag.decompose()

    text = main_node.get_text(separator="\n")
    # Collapse multiple newlines and spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip(), links


def filename_from_url(url: str) -> str:
    """Generate a safe filename from the URL."""
    parsed = urlparse(url)
    parts: List[str] = []

    netloc = parsed.netloc or "page"
    parts.append(re.sub(r"[^\w\-.]", "_", netloc))

    if parsed.path and parsed.path != "/":
        path_part = parsed.path.strip("/").replace("/", "_")
        parts.append(re.sub(r"[^\w\-.]", "_", path_part))

    if parsed.query:
        query_part = re.sub(r"[^\w\-.]", "_", parsed.query[:50])
        parts.append(f"q_{query_part}")

    name = "_".join(parts)
    return f"{name}.txt"


def scope_from_start_url(start_url: str) -> Tuple[str, str]:
    """
    Return (netloc, path_prefix) that defines crawl scope.

    The scope is restricted to the *domain* AND the *path prefix* derived from the start URL.
    Example:
      start: https://example.com/blog/post1  -> netloc=example.com, path_prefix=/blog/
      start: https://example.com/           -> netloc=example.com, path_prefix=/
    """
    parsed = urlparse(start_url)
    netloc = parsed.netloc

    # Use the full path from the start URL as the prefix.
    # Example: start https://www.bbc.com/yoruba -> prefix "/yoruba"
    path = parsed.path or "/"
    if not path.startswith("/"):
        path = "/" + path

    return netloc, path


def url_in_scope(candidate_url: str, scope_netloc: str, scope_path_prefix: str) -> bool:
    """Return True if candidate_url matches scope netloc and path prefix."""
    parsed = urlparse(candidate_url)
    if parsed.netloc != scope_netloc:
        return False
    candidate_path = parsed.path or "/"
    return candidate_path.startswith(scope_path_prefix)


def make_output_path(
    start_url: str,
    current_url: str,
    output: Optional[str],
    output_dir: Optional[Path],
) -> Path:
    """Decide where to save the text for a given URL."""
    if current_url == start_url and output:
        path = Path(output)
    else:
        path = Path(filename_from_url(current_url))

    if output_dir:
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        if not path.is_absolute():
            path = output_dir / path.name

    if path.suffix != ".txt":
        path = path.with_suffix(".txt")

    return path


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
    scope_netloc, scope_path_prefix = scope_from_start_url(url)

    visited: Set[str] = set()
    frontier: List[str] = [url]
    pages_scraped = 0

    while frontier:
        current_url = frontier.pop(0)
        if current_url in visited:
            continue

        if not url_in_scope(current_url, scope_netloc, scope_path_prefix):
            continue

        print(f"\nFetching [{pages_scraped + 1}]: {current_url}")
        try:
            text, links = scrape_url(current_url)
        except Exception as exc:  # noqa: BLE001
            print(f"Error fetching {current_url}: {exc}", file=sys.stderr)
            visited.add(current_url)
            continue

        output_path = make_output_path(
            url, current_url, args.output, output_dir)
        output_path.write_text(text, encoding="utf-8")
        print(f"Saved {len(text)} characters to {output_path}")

        scoped_links = [
            link for link in links if url_in_scope(link, scope_netloc, scope_path_prefix)
        ]

        visited.add(current_url)
        pages_scraped += 1

        for link in scoped_links:
            if link not in visited and link not in frontier:
                frontier.append(link)

        if frontier:
            print(f"Frontier size: {len(frontier)} URLs")
        else:
            print("Frontier empty.")


if __name__ == "__main__":
    main()
