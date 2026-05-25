"""Chunk long text into TTS-friendly pieces.

Both Voxtral and Qwen3-TTS perform best on inputs of a few hundred characters
to a couple of sentences. We split first at paragraph boundaries, then by
sentence (regex on .!? followed by whitespace), and finally by a hard cap.
"""

from __future__ import annotations

import re

_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+(?=[A-ZÄÖÜ\"„»])")


def chunk_text(text: str, *, max_chars: int = 800) -> list[str]:
    """Return a list of chunks each at most ``max_chars`` long.

    The function preserves paragraph structure when paragraphs are short
    enough, and splits longer paragraphs at sentence boundaries.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue

        # Sentence-level split for long paragraphs.
        sentences = _SENT_SPLIT_RE.split(paragraph)
        buffer = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if not buffer:
                buffer = sentence
            elif len(buffer) + 1 + len(sentence) <= max_chars:
                buffer = f"{buffer} {sentence}"
            else:
                chunks.append(buffer)
                buffer = sentence
            # If a single sentence is itself too long, force-split.
            while len(buffer) > max_chars:
                cut = buffer.rfind(" ", 0, max_chars)
                if cut < max_chars // 2:
                    cut = max_chars
                chunks.append(buffer[:cut].strip())
                buffer = buffer[cut:].strip()
        if buffer:
            chunks.append(buffer)

    return chunks
