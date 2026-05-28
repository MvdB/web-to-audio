"""Split an extracted document into chapter-sized heading groups.

Vatican encyclicals layer their text into three kinds of paragraphs:

* numbered sections ("1.", "2.", … up to ~245 in *Magnifica humanitas*),
* top-level chapter markers in ALL CAPS ("EINLEITUNG", "ERSTES KAPITEL",
  often followed by an ALL-CAPS subtitle), and
* mixed-case sub-headings that introduce a thematic group of sections.

For per-chapter audio we treat **each heading group as one chapter**: every
heading paragraph (or run of consecutive heading paragraphs) starts a new
chapter, which then absorbs all subsequent numbered sections until the next
heading. Trailing unnumbered matter at the very end of the document (place
and date of issue, signature) is appended to the final chapter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .extract import ExtractedDocument
from .normalize import normalize_for_speech

# A section starts a paragraph with "<n>." — note that the source is not
# always consistent about the space after the dot (e.g. "102.Der Einsatz").
_SECTION_RE = re.compile(r"^(\d{1,4})\.\s*(.*)$", re.DOTALL)

# Characters that are unsafe (or just ugly) in file names across FAT32/NTFS/HFS.
_UNSAFE_FILENAME_RE = re.compile(r'[\\/:*?"<>|\n\r\t]')


@dataclass
class Chapter:
    """A heading-bounded group of one or more numbered sections."""

    headings: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    first_number: int | None = None
    last_number: int | None = None

    @property
    def title(self) -> str:
        """Joined heading text used for playlists and filenames."""
        if self.headings:
            return " – ".join(self.headings)
        if self.first_number is not None and self.last_number is not None:
            if self.first_number == self.last_number:
                return f"Abschnitt {self.first_number}"
            return f"Abschnitte {self.first_number}–{self.last_number}"
        return "Ohne Titel"

    @property
    def text(self) -> str:
        """Full spoken text: heading lines first, then every section body.

        ALL-CAPS heading lines are reformatted to sentence case so the TTS
        reads them as words (see :mod:`web_to_audio.normalize`).
        """
        return normalize_for_speech("\n\n".join([*self.headings, *self.paragraphs]))

    def __len__(self) -> int:
        return len(self.text)


def split_into_chapters(doc: ExtractedDocument) -> list[Chapter]:
    """Group ``doc.paragraphs`` into a list of :class:`Chapter`.

    A paragraph starting with ``<n>.`` is treated as a numbered section when
    ``n`` is greater than the largest section number we've seen so far — that
    monotonic guard prevents body text that happens to start with a small
    number from being mistaken for a section marker.

    Any other paragraph is heading material. Heading paragraphs start a new
    chapter and consecutive headings get bundled into the same chapter's
    ``headings`` list, so a "ERSTES KAPITEL" line plus its ALL-CAPS subtitle
    travel together. Trailing headings after the final numbered section
    (signature, place, date) fold into the last chapter as closing matter.
    """
    chapters: list[Chapter] = []
    current: Chapter | None = None
    pending_headings: list[str] = []
    last_number = 0

    for para in doc.paragraphs:
        m = _SECTION_RE.match(para)
        number = int(m.group(1)) if m else None
        is_section = number is not None and number > last_number

        if is_section:
            assert number is not None
            last_number = number
            if pending_headings or current is None:
                # A heading block (or the very first section, headingless)
                # starts a new chapter.
                current = Chapter(
                    headings=pending_headings,
                    paragraphs=[para],
                    first_number=number,
                    last_number=number,
                )
                chapters.append(current)
                pending_headings = []
            else:
                current.paragraphs.append(para)
                current.last_number = number
        else:
            pending_headings.append(para)

    # Trailing unnumbered material never introduced a section — it's closing
    # matter (signature/date). Glue it onto the last chapter.
    if pending_headings and chapters:
        chapters[-1].paragraphs.extend(pending_headings)

    return chapters


def chapter_filename(pos: int, chapter: Chapter, *, suffix: str = ".mp3",
                     max_title_chars: int = 100) -> str:
    """Build a filesystem-safe ``NN - Title.ext`` filename for a chapter.

    ``pos`` is the chapter's 1-based position in the chapter list. The title
    is sanitised (path separators dropped, whitespace collapsed) and
    truncated to ``max_title_chars`` so even very long heading runs produce
    sensible filenames on every common filesystem.
    """
    safe = _UNSAFE_FILENAME_RE.sub("", chapter.title)
    safe = re.sub(r"\s+", " ", safe).strip(" .")
    if len(safe) > max_title_chars:
        safe = safe[:max_title_chars].rstrip(" .")
    return f"{pos:02d} - {safe}{suffix}" if safe else f"{pos:02d}{suffix}"
