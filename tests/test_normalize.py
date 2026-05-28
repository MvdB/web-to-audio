"""Tests for the TTS text normaliser."""

from __future__ import annotations

from web_to_audio.normalize import normalize_for_speech


def test_lowercases_all_caps_heading():
    assert normalize_for_speech("EINLEITUNG") == "Einleitung"


def test_lowercases_multi_word_all_caps_heading():
    src = "EIN DYNAMISCHES DENKEN IM GEISTE DES EVANGELIUMS"
    assert normalize_for_speech(src) == "Ein dynamisches denken im geiste des evangeliums"


def test_preserves_roman_numeral_in_signature():
    # XIV is a real Roman numeral and stays uppercase; the rest is lowercased.
    assert normalize_for_speech("LEO PP. XIV") == "Leo pp. XIV"


def test_leaves_mixed_case_alone():
    src = "Die Wahrheit als Gemeingut."
    assert normalize_for_speech(src) == src


def test_leaves_short_or_acronym_lines_alone():
    # Short tokens like "KI" or "EU" inside body sentences are still part of
    # a mixed-case sentence, so the whole line is left alone.
    src = "Er sprach über KI und die EU."
    assert normalize_for_speech(src) == src


def test_normalises_each_line_of_multi_line_paragraph():
    src = "ERSTES KAPITEL\nEIN DYNAMISCHES DENKEN IM GEISTE DES EVANGELIUMS"
    expected = "Erstes kapitel\nEin dynamisches denken im geiste des evangeliums"
    assert normalize_for_speech(src) == expected


def test_normalises_only_all_caps_paragraphs_in_a_block():
    src = "EINLEITUNG\n\n1. Die geschaffene Menschheit steht heute…"
    expected = "Einleitung\n\n1. Die geschaffene Menschheit steht heute…"
    assert normalize_for_speech(src) == expected


def test_preserves_embedded_roman_numeral_in_normalised_line():
    # "EIN" / "KAPITEL" contain characters outside the Roman set so they
    # get lowercased; "DI" is a valid Roman numeral (501) and is preserved.
    assert normalize_for_speech("EIN DI KAPITEL") == "Ein DI kapitel"


def test_does_not_treat_random_letter_sequences_as_roman():
    # "EIN" / "DEN" share letters with Roman numerals but are not valid
    # Roman numerals — they must be lowercased.
    out = normalize_for_speech("EIN DEN KAPITEL")
    assert "EIN" not in out and "DEN" not in out


def test_empty_string():
    assert normalize_for_speech("") == ""


def test_chapter_text_runs_normalisation():
    # End-to-end: Chapter.text should already deliver normalised text so
    # that downstream chunkers/TTS never see the ALL-CAPS line.
    from web_to_audio.chapters import Chapter
    ch = Chapter(
        headings=["ERSTES KAPITEL", "EIN DYNAMISCHES DENKEN IM GEISTE DES EVANGELIUMS"],
        paragraphs=["17. Inhalt."],
        first_number=17,
        last_number=17,
    )
    assert ch.text == (
        "Erstes kapitel\n\n"
        "Ein dynamisches denken im geiste des evangeliums\n\n"
        "17. Inhalt."
    )
    # Display title stays raw (used for filenames / playlist).
    assert "ERSTES KAPITEL" in ch.title
