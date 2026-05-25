"""Tests for splitting an extracted document into numbered chapters."""

from __future__ import annotations

from web_to_audio.chapters import split_into_chapters
from web_to_audio.extract import ExtractedDocument


def _doc(paragraphs: list[str]) -> ExtractedDocument:
    return ExtractedDocument(url="http://x", title="T", language="de", paragraphs=paragraphs)


def test_splits_on_numbered_sections():
    doc = _doc([
        "EINLEITUNG",
        "1. Erster Abschnitt.",
        "2. Zweiter Abschnitt.",
        "ERSTES KAPITEL",
        "Eine Überschrift",
        "3. Dritter Abschnitt.",
    ])
    chapters = split_into_chapters(doc)

    assert [c.number for c in chapters] == [1, 2, 3]
    # Heading before section 1 attaches to chapter 1.
    assert chapters[0].headings == ["EINLEITUNG"]
    assert chapters[0].paragraphs == ["1. Erster Abschnitt."]
    # Section 2 has no preceding heading.
    assert chapters[1].headings == []
    # Both heading lines before section 3 attach to chapter 3.
    assert chapters[2].headings == ["ERSTES KAPITEL", "Eine Überschrift"]


def test_handles_missing_space_after_dot():
    # The source sometimes omits the space, e.g. "102.Der Einsatz...".
    doc = _doc(["1. A.", "2.Bündig ohne Leerzeichen."])
    chapters = split_into_chapters(doc)
    assert chapters[1].number == 2
    assert chapters[1].paragraphs == ["2.Bündig ohne Leerzeichen."]


def test_trailing_matter_folds_into_last_chapter():
    doc = _doc([
        "1. Inhalt.",
        "Gegeben zu Rom, am 15. Mai 2026.",
        "LEO PP. XIV",
    ])
    chapters = split_into_chapters(doc)
    assert len(chapters) == 1
    assert "LEO PP. XIV" in chapters[0].paragraphs
    assert "Gegeben zu Rom, am 15. Mai 2026." in chapters[0].paragraphs


def test_body_number_does_not_start_new_chapter():
    # A non-monotonic leading number inside body text must not split.
    doc = _doc(["5. Im Jahr 1891 schrieb er.", "1. Das wäre ein Rückschritt der Nummerierung."])
    chapters = split_into_chapters(doc)
    assert [c.number for c in chapters] == [5]
    assert "Rückschritt" in chapters[0].paragraphs[-1]


def test_title_and_text():
    doc = _doc(["ERSTES KAPITEL", "17. Inhalt hier."])
    ch = split_into_chapters(doc)[0]
    assert ch.title == "17 – ERSTES KAPITEL"
    assert ch.text == "ERSTES KAPITEL\n\n17. Inhalt hier."
    assert ch.number == 17
