"""Tests for the Vatican extractor.

These tests parse a small synthetic HTML fixture that mimics the structure
of vatican.va encyclicals; they do not require network access.
"""

from __future__ import annotations

from web_to_audio.extract import extract_vatican


FIXTURE_HTML = """\
<html lang="de"><head><title>Test</title></head><body>
<div class="documento">
  <div class="testo">
    <div class="abstract text parbase vaticanrichtext">
      <p><span class="title-1-color">TEST DOC</span></p>
    </div>
    <div class="text parbase vaticanrichtext">
      <p><b><a href="#intro">Einleitung</a></b></p>
      <p><a href="#k1">Erstes Kapitel</a></p>

      <p style="text-align:center;"><b><a name="intro"></a>EINLEITUNG</b></p>
      <p>1. Das ist der erste Absatz.<a name="_ftnref1" href="#_ftn1" class=" cleaner">[1]</a> Hier weitere Sätze.</p>
      <p>2. Zweiter Absatz, ohne Fußnote.</p>
      <p style="text-align:center;"><b><a name="k1"></a>ERSTES KAPITEL</b></p>
      <p>3. Inhalt im ersten Kapitel.<a name="_ftnref2" href="#_ftn2" class=" cleaner">[2]</a></p>

      <hr align="left" size="1" width="33%" />
      <p><a name="_ftn1" href="#_ftnref1" class=" cleaner">[1]</a>&nbsp;Fußnote 1, sollte nicht erscheinen.</p>
      <p><a name="_ftn2" href="#_ftnref2" class=" cleaner">[2]</a>&nbsp;Fußnote 2, sollte nicht erscheinen.</p>
    </div>
  </div>
</div>
</body></html>
"""


def test_extracts_title_and_language():
    doc = extract_vatican("https://www.vatican.va/content/leo-xiv/de/foo.html", FIXTURE_HTML)
    assert doc.title == "TEST DOC"
    assert doc.language == "de"


def test_skips_toc_and_footnotes():
    doc = extract_vatican("https://www.vatican.va/content/leo-xiv/de/foo.html", FIXTURE_HTML)
    text = doc.text

    # TOC removed
    assert "Einleitung" not in doc.paragraphs[0] or doc.paragraphs[0] == "EINLEITUNG"
    # body included
    assert "1. Das ist der erste Absatz." in text
    assert "2. Zweiter Absatz" in text
    assert "3. Inhalt im ersten Kapitel." in text
    # footnotes excluded
    assert "Fußnote 1, sollte nicht erscheinen." not in text
    assert "Fußnote 2, sollte nicht erscheinen." not in text
    # footnote markers stripped
    assert "[1]" not in text
    assert "[2]" not in text


def test_paragraph_count():
    doc = extract_vatican("https://www.vatican.va/content/leo-xiv/de/foo.html", FIXTURE_HTML)
    # EINLEITUNG, 1., 2., ERSTES KAPITEL, 3.  → 5 paragraphs
    assert len(doc.paragraphs) == 5
