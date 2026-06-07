"""Content ingestion from URLs, bookmarks, and Chrome history."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import trafilatura
from bs4 import BeautifulSoup

from learnlens.models.types import ContentItem
from learnlens.storage.database import Database
from learnlens.storage.queries import insert_content
from learnlens.utils.text import clean_text, extract_domain, truncate_text

logger = logging.getLogger(__name__)


def _extract_title_from_metadata(metadata: str | None, fallback: str) -> str:
    """Extract title from trafilatura metadata XML or fallback to first line."""
    if metadata:
        import re

        match = re.search(r'<title[^>]*>(.*?)</title>', metadata, re.S)
        if match:
            return clean_text(match.group(1))
    first_line = fallback.strip().split("\n")[0]
    return clean_text(first_line)[:200]


def ingest_url(url: str, db: Database) -> ContentItem | None:
    """Fetch URL content and store in database.

    Returns None if URL is unreachable or content extraction fails.
    Deduplicates by URL (updates if already exists).
    """
    logger.info("Ingesting URL: %s", url)

    if not url.startswith(("http://", "https://")):
        logger.warning("Invalid URL: %s", url)
        return None

    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        logger.warning("Failed to fetch URL: %s", url)
        return None

    result = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        output_format="txt",
    )
    if result is None:
        logger.warning("Failed to extract content from URL: %s", url)
        return None

    metadata = trafilatura.extract(
        downloaded,
        output_format="xmltei",
        include_comments=False,
    )

    body_text = clean_text(truncate_text(result, max_length=50000))
    title = _extract_title_from_metadata(metadata, body_text)
    domain = extract_domain(url)
    word_count = len(body_text.split())

    row_id = insert_content(
        conn=db.connection,
        url=url,
        title=title,
        body_text=body_text,
        source_type="url",
        word_count=word_count,
        domain=domain,
    )

    logger.info("Ingested URL as content id %d: %s", row_id, url)
    return ContentItem(
        id=row_id,
        url=url,
        title=title,
        body_text=body_text,
        source_type="url",
        word_count=word_count,
        ingested_at=datetime.now(),
    )


def ingest_bookmarks(file_path: Path, db: Database) -> list[ContentItem]:
    """Parse bookmarks HTML export and ingest all URLs.

    Returns list of successfully ingested items.
    Fetches content for each bookmark URL via trafilatura.
    """
    logger.info("Parsing bookmarks from %s", file_path)

    if not file_path.suffix.lower() == ".html":
        logger.error("Bookmarks file must be .html, got %s", file_path.suffix)
        return []

    html = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    bookmarks = []
    for link in soup.find_all("a"):
        url = link.get("href", "")
        title = link.get_text(strip=True)
        if url.startswith("http"):
            bookmarks.append({"url": url, "title": title})

    logger.info("Found %d bookmarks in HTML", len(bookmarks))

    ingested: list[ContentItem] = []
    for item in bookmarks:
        result = ingest_url(item["url"], db)
        if result is not None:
            ingested.append(result)

    logger.info("Successfully ingested %d/%d bookmarks", len(ingested), len(bookmarks))
    return ingested


def ingest_chrome_history(
    file_path: Path,
    db: Database,
    limit: int = 500,
) -> list[ContentItem]:
    """Read Chrome history SQLite and ingest top URLs by visit count.

    User uploads a COPY of their Chrome history file.
    Only processes http/https URLs.
    """
    logger.info("Parsing Chrome history from %s", file_path)

    if file_path.suffix.lower() not in (".sqlite", ".db", ""):
        logger.warning("Unexpected Chrome history file extension: %s", file_path.suffix)

    conn = sqlite3.connect(str(file_path))
    try:
        rows = conn.execute(
            """
            SELECT url, title, visit_count, last_visit_time
            FROM urls
            WHERE url LIKE 'http%'
            ORDER BY last_visit_time DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    except sqlite3.OperationalError as e:
        logger.error("Failed to read Chrome history: %s", e)
        conn.close()
        return []
    finally:
        conn.close()

    logger.info("Found %d history entries", len(rows))

    ingested: list[ContentItem] = []
    for row in rows:
        url = row[0]
        title = row[1] or ""
        result = ingest_url(url, db)
        if result is not None:
            ingested.append(result)

    logger.info("Successfully ingested %d/%d history items", len(ingested), len(rows))
    return ingested
