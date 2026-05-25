from web_to_audio.chunk import chunk_text


def test_short_paragraphs_pass_through():
    text = "Erster Absatz.\n\nZweiter Absatz."
    chunks = chunk_text(text, max_chars=100)
    assert chunks == ["Erster Absatz.", "Zweiter Absatz."]


def test_long_paragraph_split_at_sentence_boundary():
    sentence = "Dies ist ein Satz. "
    paragraph = sentence * 20  # ~380 chars
    chunks = chunk_text(paragraph, max_chars=80)
    assert all(len(c) <= 80 for c in chunks)
    assert "".join(chunks).count("Dies ist ein Satz.") == 20


def test_hard_split_when_sentence_too_long():
    paragraph = "A" * 500
    chunks = chunk_text(paragraph, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == paragraph
