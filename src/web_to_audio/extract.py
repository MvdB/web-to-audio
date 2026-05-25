"""Extract clean body text from web pages.

The primary target is vatican.va documents (encyclicals, apostolic letters, ...),
which all share a consistent HTML layout:

  <div class="documento">
    <div class="testo">
      <div class="abstract text parbase vaticanrichtext"> ... title ... </div>
      <div class="text parbase vaticanrichtext">
         ... TOC (paragraphs whose only links are href="#anchor") ...
         ... body, beginning at the first <a name="..."> anchor ...
         <hr align="left" .../>            <-- separator
         ... footnotes (each <p> begins with an <a name="_ftnN"> link) ...
      </div>
    </div>
  </div>

A generic fallback strategy (``extract_from_url`` without explicit hint) tries
the vatican-specific path first, then falls back to a heuristic main-text
extraction based on the largest text block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

USER_AGENT = (
    "web-to-audio/0.1 (+https://github.com/your-username/web-to-audio) "
    "python-requests"
)

# Footnote markers like "[1]", "[12]" — used to strip references inline.
_FOOTNOTE_MARK_RE = re.compile(r"\[\d{1,4}\]")
# Sequences of whitespace
_WS_RE = re.compile(r"[ \t ]+")


@dataclass
class ExtractedDocument:
    """A document parsed out of a web page.

    Attributes:
        url: Origin URL.
        title: Best-effort document title (HTML ``<title>`` or first headline).
        language: Two-letter language code if known.
        paragraphs: Ordered list of clean body paragraphs (no TOC, no footnotes,
            footnote markers like ``[1]`` stripped).
    """

    url: str
    title: str = ""
    language: str = ""
    paragraphs: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Single string with paragraphs joined by blank lines."""
        return "\n\n".join(self.paragraphs)

    def __len__(self) -> int:
        return sum(len(p) for p in self.paragraphs)


def fetch_html(url: str, *, timeout: float = 30.0) -> str:
    """Download a URL and return the decoded HTML."""
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "de,en;q=0.5"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    # vatican.va declares ISO-8859-1 in headers but actual bytes are UTF-8;
    # let requests detect from content where it can.
    if resp.encoding and resp.encoding.lower() in {"iso-8859-1", "latin-1"}:
        resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def extract_from_url(url: str, *, html: str | None = None) -> ExtractedDocument:
    """Dispatch to a site-specific extractor based on the URL's host."""
    if html is None:
        html = fetch_html(url)
    host = urlparse(url).netloc.lower()
    if host.endswith("vatican.va"):
        return extract_vatican(url, html)
    raise ValueError(
        f"No extractor registered for host {host!r}. "
        "Currently supported: vatican.va. Pass --html or extend extract.py."
    )


# ---------------------------------------------------------------------------
# vatican.va
# ---------------------------------------------------------------------------

def extract_vatican(url: str, html: str) -> ExtractedDocument:
    """Extract clean body text from a vatican.va document page."""
    soup = BeautifulSoup(html, "lxml")

    title = _document_title(soup)
    language = _document_language(soup, url)

    rich = soup.select("div.documento div.testo div.vaticanrichtext")
    if not rich:
        raise ValueError("Page does not look like a vatican.va document (no .vaticanrichtext).")

    # The last vaticanrichtext block contains TOC + body + footnotes.
    body_block = rich[-1]

    # Convert children to a flat ordered list, splitting the block at:
    #   1) The first <a name="..."> anchor (body start) — drop everything before.
    #   2) The first <hr> directly inside the block (footnote separator) — drop after.
    paragraphs = list(_iter_body_paragraphs(body_block))

    return ExtractedDocument(
        url=url,
        title=title,
        language=language,
        paragraphs=paragraphs,
    )


def _document_title(soup: BeautifulSoup) -> str:
    # Prefer the abstract block (centered, contains the document title).
    abstract = soup.select_one("div.documento div.abstract")
    if abstract:
        # Pull strong/title-color span if available, else use plain text.
        node = abstract.select_one(".title-1-color") or abstract.select_one("b")
        if node:
            t = _clean(node.get_text(" ", strip=True))
            if t:
                return t
    if soup.title and soup.title.string:
        return _clean(soup.title.string)
    return ""


def _document_language(soup: BeautifulSoup, url: str) -> str:
    html_tag = soup.find("html")
    if html_tag and html_tag.get("lang"):
        return html_tag["lang"].split("-")[0].lower()
    # vatican.va URLs encode the language as /content/<pope>/<lang>/...
    parts = urlparse(url).path.split("/")
    if len(parts) >= 4 and len(parts[3]) == 2:
        return parts[3].lower()
    return ""


def _iter_body_paragraphs(block: Tag) -> Iterable[str]:
    """Yield clean body paragraphs from a ``.vaticanrichtext`` block.

    Body = content from the first ``<a name="...">`` (an anchor target, not a
    TOC link) up to the first horizontal rule that separates the body from
    the footnotes.
    """
    body_started = False

    for child in block.children:
        if isinstance(child, NavigableString):
            continue
        if not isinstance(child, Tag):
            continue

        # End of body: <hr> separates body and footnotes.
        if child.name == "hr":
            break

        # Detect first body anchor: a <p> that contains <a name="...">
        if not body_started:
            if child.find("a", attrs={"name": True}):
                body_started = True
            else:
                continue

        if child.name not in {"p", "blockquote", "ul", "ol", "div"}:
            continue

        text = _paragraph_text(child)
        if not text:
            continue
        # Skip the trailing "_____" line and standalone "[Multimedia]" boxes.
        if _is_decorative(text):
            continue
        yield text


def _paragraph_text(node: Tag) -> str:
    """Get clean text from a paragraph-like node.

    Strips footnote reference superscripts like ``[1]`` and normalizes
    whitespace. Blockquotes and lists are flattened to plain text.
    """
    # Remove footnote-reference anchors entirely (they render as "[N]").
    for a in node.find_all("a", attrs={"name": re.compile(r"^_ftnref")}):
        a.decompose()
    # Also strip <sup>[1]</sup>-style markers if any remain.
    for sup in node.find_all("sup"):
        if _FOOTNOTE_MARK_RE.search(sup.get_text(strip=True)):
            sup.decompose()

    text = node.get_text(" ", strip=True)
    text = _FOOTNOTE_MARK_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _is_decorative(text: str) -> bool:
    if not text:
        return True
    # The Vatican templates use a long underscore separator and bracketed
    # "[Multimedia]" / "[Cum bonum]" boxes.
    stripped = text.strip()
    if set(stripped) <= {"_", " "}:
        return True
    if stripped.startswith("[") and stripped.endswith("]") and len(stripped) < 80:
        return True
    return False


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()
