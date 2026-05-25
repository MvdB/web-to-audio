"""Voxtral 4B TTS backend (Mistral, served via vLLM-Omni).

The model is *not* loaded in-process. Voxtral is designed to run as a server:

    vllm serve mistralai/Voxtral-4B-TTS-2603 --omni

This backend hits the OpenAI-compatible ``/v1/audio/speech`` endpoint that the
server exposes. The default base URL can be overridden with the
``WEB_TO_AUDIO_VOXTRAL_URL`` environment variable.

Voxtral ships reference voice embeddings per language; for German the
relevant ones are ``de_male`` and ``de_female`` (also: ``casual_male``,
``casual_female``, ``cheerful_female``, ``neutral_male``, ``neutral_female``
and a handful of other locale-specific ones — see the model card / repo
listing for the full set).

Note: the Voxtral HTTP API does *not* take a ``language`` parameter —
language is inferred from the input text. Choose a voice whose locale
matches your text for best prosody.
"""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

if TYPE_CHECKING:
    from ..tts import TTSOptions


DEFAULT_BASE_URL = os.environ.get("WEB_TO_AUDIO_VOXTRAL_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = "mistralai/Voxtral-4B-TTS-2603"
DEFAULT_VOICE = "de_male"


class VoxtralBackend:
    """HTTP client for a Voxtral TTS server.

    The backend exposes both a blocking ``synthesize`` for single-shot calls
    and an async ``asynthesize`` for high-throughput batch use — vLLM-Omni
    services up to ~32 concurrent TTS requests per GPU, so issuing chunks
    in parallel is much faster than serial calls.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 180.0,
    ) -> None:
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The Voxtral backend requires httpx. Install with: pip install httpx"
            ) from e

        self._httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _payload(self, opts: "TTSOptions") -> dict:
        return {
            "model": self.model,
            "input": opts.text,
            "voice": opts.voice or DEFAULT_VOICE,
            "response_format": "wav",
        }

    def synthesize(self, opts: "TTSOptions") -> tuple[np.ndarray, int]:
        response = self._httpx.post(
            f"{self.base_url}/audio/speech",
            json=self._payload(opts),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._decode(response.content)

    async def asynthesize(
        self,
        opts: "TTSOptions",
        client,  # httpx.AsyncClient supplied by caller for connection reuse
    ) -> tuple[np.ndarray, int]:
        response = await client.post(
            f"{self.base_url}/audio/speech",
            json=self._payload(opts),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._decode(response.content)

    @staticmethod
    def _decode(content: bytes) -> tuple[np.ndarray, int]:
        audio, sr = sf.read(io.BytesIO(content), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, int(sr)
