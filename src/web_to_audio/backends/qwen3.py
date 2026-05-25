"""Qwen3-TTS Custom Voice backend.

Uses the ``qwen-tts`` Python package, which loads weights via Hugging Face
Transformers. The default model is ``Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice``;
its supported voices ship with the model. For German text we default to
``Aiden`` (a clear midrange male English voice that handles German cleanly);
any speaker from ``model.get_supported_speakers()`` can be chosen.

The first call after process start downloads the model from Hugging Face
(roughly 3–4 GB) and loads it onto a CUDA device. Subsequent calls reuse the
loaded model in memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..tts import TTSOptions


DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_SPEAKER = "aiden"


class Qwen3Backend:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL,
        device_map: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
    ) -> None:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The Qwen3 backend requires the 'qwen-tts' package and torch.\n"
                "  pip install -U qwen-tts\n"
                "Optional speed-up:\n"
                "  pip install -U flash-attn --no-build-isolation"
            ) from e

        dtype_obj = getattr(torch, dtype)

        load_kwargs: dict = {
            "device_map": device_map,
            "dtype": dtype_obj,
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        try:
            self._model = Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)
        except (ImportError, ValueError, RuntimeError):
            load_kwargs.pop("attn_implementation", None)
            self._model = Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)

        # Both speaker and language names are matched case-insensitively below.
        self._supported_speakers = set(self._model.get_supported_speakers())
        self._supported_languages = set(self._model.get_supported_languages())

    def synthesize(self, opts: "TTSOptions") -> tuple[np.ndarray, int]:
        speaker = (opts.voice or DEFAULT_SPEAKER).lower()
        if speaker not in self._supported_speakers:
            raise ValueError(
                f"Speaker {speaker!r} is not supported by the loaded Qwen3-TTS model. "
                f"Available: {sorted(self._supported_speakers)}"
            )
        language = (opts.language or "german").lower()
        if language not in self._supported_languages:
            language = "auto"

        kwargs = {
            "text": opts.text,
            "language": language,
            "speaker": speaker,
        }
        if opts.instruct:
            kwargs["instruct"] = opts.instruct

        wavs, sr = self._model.generate_custom_voice(**kwargs)
        wav = np.asarray(wavs[0], dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.reshape(-1)
        return wav, int(sr)
