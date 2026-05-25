"""Extract web page body text and synthesize it to audio."""

__version__ = "0.1.0"

from .extract import ExtractedDocument, extract_from_url, extract_vatican
from .tts import TTSBackend, synthesize

__all__ = [
    "ExtractedDocument",
    "extract_from_url",
    "extract_vatican",
    "TTSBackend",
    "synthesize",
    "__version__",
]
