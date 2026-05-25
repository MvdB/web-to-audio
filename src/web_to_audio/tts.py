"""Text-to-speech backends and a small dispatcher.

Two backends are supported:

- ``voxtral`` — Mistral's open-weights TTS, served via vLLM-Omni at an
  OpenAI-compatible HTTP endpoint.
- ``qwen3``   — Alibaba's Qwen3-TTS Custom Voice model, loaded in-process
  via the ``qwen-tts`` Python package.

Both backends produce 24 kHz (Voxtral) or up-to 24 kHz (Qwen3-TTS) audio.
The MP3 conversion is delegated to :mod:`web_to_audio.audio`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

TTSBackend = Literal["voxtral", "qwen3"]


@dataclass
class TTSOptions:
    """Common synthesis options that map to each backend's parameter names."""

    text: str
    language: str = "German"
    voice: str = ""  # backend-specific; sensible defaults are filled in
    instruct: str | None = None  # only honoured by Qwen3-TTS
    max_chunk_chars: int = 800  # split long inputs to stay within model limits


class _BackendProtocol(Protocol):
    def synthesize(self, opts: TTSOptions) -> tuple[np.ndarray, int]:  # pragma: no cover
        ...


def get_backend(name: TTSBackend, **kwargs) -> _BackendProtocol:
    """Construct a backend by name.

    Backends are imported lazily so that users don't need to install
    every dependency (e.g. ``qwen-tts`` only matters for the Qwen3 backend).
    """
    if name == "voxtral":
        from .backends.voxtral import VoxtralBackend

        return VoxtralBackend(**kwargs)
    if name == "qwen3":
        from .backends.qwen3 import Qwen3Backend

        return Qwen3Backend(**kwargs)
    raise ValueError(f"Unknown TTS backend: {name!r}")


def synthesize(
    text: str,
    output_path: str | Path,
    *,
    backend: TTSBackend = "qwen3",
    language: str = "German",
    voice: str = "",
    instruct: str | None = None,
    max_chunk_chars: int = 800,
    backend_kwargs: dict | None = None,
    mp3_bitrate: str = "128k",
) -> Path:
    """Synthesize ``text`` to an MP3 file at ``output_path``.

    Long inputs are split into chunks at paragraph / sentence boundaries.
    The chunks are synthesized independently and concatenated.
    """
    from .audio import save_as_mp3
    from .chunk import chunk_text

    bk = get_backend(backend, **(backend_kwargs or {}))

    chunks = chunk_text(text, max_chars=max_chunk_chars)
    if not chunks:
        raise ValueError("Input text is empty after chunking.")

    audio_pieces: list[np.ndarray] = []
    sample_rate: int | None = None

    for i, chunk in enumerate(chunks, start=1):
        opts = TTSOptions(
            text=chunk,
            language=language,
            voice=voice,
            instruct=instruct,
            max_chunk_chars=max_chunk_chars,
        )
        wav, sr = bk.synthesize(opts)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise RuntimeError(f"Inconsistent sample rates from backend: {sr} vs {sample_rate}")
        audio_pieces.append(wav)

    assert sample_rate is not None
    full = np.concatenate(audio_pieces)
    return save_as_mp3(full, sample_rate, Path(output_path), bitrate=mp3_bitrate)
