"""Tests for splitting an extracted document into heading-group chapters."""

from __future__ import annotations

from web_to_audio.chapters import (
    Chapter,
    chapter_filename,
    split_into_chapters,
)
from web_to_audio.extract import ExtractedDocument


def _doc(paragraphs: list[str]) -> ExtractedDocument:
    return ExtractedDocument(url="http://x", title="T", language="de", paragraphs=paragraphs)


def test_each_heading_starts_a_new_chapter():
    doc = _doc([
        "EINLEITUNG",
        "1. Erster Abschnitt.",
        "2. Zweiter Abschnitt.",
        "3. Dritter Abschnitt.",
        "Die res novae unserer Zeit",
        "4. Vierter Abschnitt.",
        "5. Fünfter Abschnitt.",
    ])
    chapters = split_into_chapters(doc)

    assert len(chapters) == 2
    assert chapters[0].headings == ["EINLEITUNG"]
    assert chapters[0].paragraphs == [
        "1. Erster Abschnitt.",
        "2. Zweiter Abschnitt.",
        "3. Dritter Abschnitt.",
    ]
    assert chapters[0].first_number == 1
    assert chapters[0].last_number == 3
    assert chapters[1].headings == ["Die res novae unserer Zeit"]
    assert chapters[1].first_number == 4
    assert chapters[1].last_number == 5


def test_consecutive_headings_bundle_into_one_chapter():
    # "ERSTES KAPITEL" + ALL-CAPS subtitle precede §17, §18 — both headings
    # travel together as the chapter's heading list.
    doc = _doc([
        "ERSTES KAPITEL",
        "EIN DYNAMISCHES DENKEN IM GEISTE DES EVANGELIUMS",
        "17. Inhalt.",
        "18. Mehr Inhalt.",
    ])
    chapters = split_into_chapters(doc)
    assert len(chapters) == 1
    assert chapters[0].headings == [
        "ERSTES KAPITEL",
        "EIN DYNAMISCHES DENKEN IM GEISTE DES EVANGELIUMS",
    ]
    assert chapters[0].first_number == 17
    assert chapters[0].last_number == 18


def test_handles_missing_space_after_dot():
    # The source sometimes omits the space, e.g. "102.Der Einsatz...".
    doc = _doc(["Heading", "1. A.", "2.Bündig ohne Leerzeichen."])
    chapters = split_into_chapters(doc)
    assert len(chapters) == 1
    assert chapters[0].paragraphs == ["1. A.", "2.Bündig ohne Leerzeichen."]
    assert chapters[0].last_number == 2


def test_trailing_matter_folds_into_last_chapter():
    doc = _doc([
        "Heading",
        "1. Inhalt.",
        "Gegeben zu Rom, am 15. Mai 2026.",
        "LEO PP. XIV",
    ])
    chapters = split_into_chapters(doc)
    assert len(chapters) == 1
    assert "LEO PP. XIV" in chapters[0].paragraphs
    assert "Gegeben zu Rom, am 15. Mai 2026." in chapters[0].paragraphs


def test_body_number_does_not_start_new_section():
    doc = _doc([
        "Heading",
        "5. Im Jahr 1891 schrieb er.",
        "1. Das wäre ein Rückschritt der Nummerierung.",
    ])
    chapters = split_into_chapters(doc)
    assert len(chapters) == 1
    assert chapters[0].last_number == 5
    assert "Rückschritt" in chapters[0].paragraphs[-1]


def test_document_starting_with_a_section_creates_headingless_chapter():
    doc = _doc(["1. Inhalt.", "Heading", "2. Mehr."])
    chapters = split_into_chapters(doc)
    assert len(chapters) == 2
    assert chapters[0].headings == []
    assert chapters[0].paragraphs == ["1. Inhalt."]
    assert chapters[1].headings == ["Heading"]


def test_title_and_text():
    # title stays raw (used for filenames / display); spoken text is
    # normalised so the all-caps heading gets read as a word.
    doc = _doc(["EINLEITUNG", "1. Inhalt."])
    ch = split_into_chapters(doc)[0]
    assert ch.title == "EINLEITUNG"
    assert ch.text == "Einleitung\n\n1. Inhalt."


def test_title_for_multi_heading_chapter():
    ch = Chapter(headings=["ERSTES KAPITEL", "EIN DYNAMISCHES DENKEN"],
                 paragraphs=["17. X."], first_number=17, last_number=17)
    assert ch.title == "ERSTES KAPITEL – EIN DYNAMISCHES DENKEN"


def test_title_for_headingless_chapter():
    ch = Chapter(headings=[], paragraphs=["1. X."], first_number=1, last_number=3)
    assert ch.title == "Abschnitte 1–3"


def test_chapter_filename_basic():
    ch = Chapter(headings=["EINLEITUNG"], paragraphs=["1. X."],
                 first_number=1, last_number=3)
    assert chapter_filename(1, ch) == "01 - EINLEITUNG.mp3"


def test_chapter_filename_sanitises_unsafe_chars():
    ch = Chapter(headings=["Was/ist:das?"], paragraphs=["1."],
                 first_number=1, last_number=1)
    assert chapter_filename(7, ch) == "07 - Wasistdas.mp3"


def test_chapter_filename_truncates_long_titles():
    long = "A" * 200
    ch = Chapter(headings=[long], paragraphs=["1."], first_number=1, last_number=1)
    name = chapter_filename(2, ch, max_title_chars=40)
    assert name == "02 - " + "A" * 40 + ".mp3"


def test_chapter_filename_custom_suffix():
    ch = Chapter(headings=["X"], paragraphs=["1."], first_number=1, last_number=1)
    assert chapter_filename(1, ch, suffix=".wav") == "01 - X.wav"
