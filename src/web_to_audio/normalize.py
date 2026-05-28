"""Normalise extracted text for natural-sounding TTS.

Voxtral (and many other TTS models) mispronounce paragraphs written
entirely in ALL CAPS: they spell letters out, shout, or insert unnatural
pauses. Vatican encyclicals use ALL CAPS for chapter markers
(``EINLEITUNG``, ``ERSTES KAPITEL``, the ALL-CAPS subtitle below it) and
for the final signature block (``LEO PP. XIV``). This module rewrites such
lines as sentence case so the model reads them as words. Strict Roman
numerals stay uppercase, otherwise a signature like ``LEO PP. XIV`` would
become an unreadable ``leo pp. xiv``.
"""

from __future__ import annotations

import re

# A line is treated as "predominantly uppercase" when ≥85% of its letters
# are uppercase and there are at least 4 letters total. Single-token shouts
# like "KI" don't qualify, so embedded abbreviations in normal body text
# are left alone.
_MIN_ALPHA = 4
_UPPER_RATIO = 0.85

# Strict Roman numeral matcher — accepts II … MMMCMXCIX but not arbitrary
# letter sequences that happen to use only those characters (e.g. "EIN").
_ROMAN_RE = re.compile(
    r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)


def normalize_for_speech(text: str) -> str:
    """Return ``text`` with predominantly-uppercase lines reformatted.

    Each paragraph (separated by blank lines) is processed line by line.
    Mixed-case lines pass through unchanged. A line whose letters are
    mostly uppercase gets lowercased, the first letter is re-capitalised,
    and tokens that look like real Roman numerals stay uppercase.
    """
    return "\n\n".join(
        "\n".join(_normalise_line(line) for line in para.split("\n"))
        for para in text.split("\n\n")
    )


def _normalise_line(line: str) -> str:
    if not _is_predominantly_uppercase(line):
        return line
    tokens = re.split(r"(\s+)", line)
    lowered = "".join(
        t if (t.strip() and _looks_like_roman(t.strip())) else t.lower()
        for t in tokens
    )
    return lowered[:1].upper() + lowered[1:] if lowered else lowered


def _is_predominantly_uppercase(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if len(letters) < _MIN_ALPHA:
        return False
    return sum(1 for c in letters if c.isupper()) / len(letters) >= _UPPER_RATIO


def _looks_like_roman(token: str) -> bool:
    # _ROMAN_RE matches the empty string too (all groups are optional), so
    # require at least two characters before we trust it.
    return len(token) >= 2 and bool(_ROMAN_RE.match(token))
