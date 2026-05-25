"""Split an extracted document into numbered chapters.

Vatican encyclicals are organised as a sequence of numbered sections
("1.", "2.", … up to ~245 in *Magnifica humanitas*), interleaved with
unnumbered headings such as ``EINLEITUNG``, ``ERSTES KAPITEL`` or the
italic sub-headings that introduce a group of sections.

For per-chapter audio we treat **each numbered section as one chapter**.
Any unnumbered heading lines that appear *before* a section are carried
along and become that chapter's heading (and are spoken at the start of
its audio). Trailing unnumbered matter after the last section (the place
and date of issue, the signature) is appended to the final chapter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .extract import ExtractedDocument

# A section starts a paragraph with "<n>." — note that the source is not
# always consistent about the space after the dot (e.g. "102.Der Einsatz").
_SECTION_RE = re.compile(r"^(\d{1,4})\.\s*(.*)$", re.DOTALL)


@dataclass
class Chapter:
    """One numbered section plus any heading lines that introduce it."""

    number: int
    headings: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        """Human-readable title for playlists, e.g. ``17 – Erstes Kapitel``."""
        if self.headings:
            return f"{self.number} – {' · '.join(self.headings)}"
        return f"Abschnitt {self.number}"

    @property
    def text(self) -> str:
        """Full spoken text: heading lines first, then the section body."""
        return "\n\n".join([*self.headings, *self.paragraphs])

    def __len__(self) -> int:
        return len(self.text)


def split_into_chapters(doc: ExtractedDocument) -> list[Chapter]:
    """Group ``doc.paragraphs`` into a list of :class:`Chapter`.

    A paragraph is recognised as the start of a new chapter when it begins
    with a number-and-dot whose value is greater than the current chapter's
    number (monotonic increase). This guards against body text that happens
    to start with a small number being mistaken for a section marker.
    """
    chapters: list[Chapter] = []
    pending_headings: list[str] = []
    current: Chapter | None = None

    for para in doc.paragraphs:
        m = _SECTION_RE.match(para)
        number = int(m.group(1)) if m else None
        is_section = m is not None and (current is None or number > current.number)

        if is_section:
            assert m is not None and number is not None
            # Keep the full paragraph (including the "17." marker) as the body
            # so the section number is spoken — an audible anchor per chapter.
            current = Chapter(number=number, headings=pending_headings, paragraphs=[para])
            chapters.append(current)
            pending_headings = []
        elif current is None:
            # Heading material that appears before the very first section.
            pending_headings.append(para)
        else:
            # Unnumbered paragraph after a section: until we see the next
            # numbered section we cannot tell whether it heads the next
            # chapter or closes the document, so park it.
            pending_headings.append(para)

    # Whatever headings are left over never introduced a section (e.g. the
    # place, date and signature at the very end): fold them into the last
    # chapter as closing matter.
    if pending_headings and chapters:
        chapters[-1].paragraphs.extend(pending_headings)

    return chapters
