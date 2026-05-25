"""Extract web page body text and synthesize it to audio."""

__version__ = "0.1.0"

from .chapters import Chapter, split_into_chapters
from .extract import ExtractedDocument, extract_from_url, extract_vatican
from .tts import TTSBackend, synthesize, synthesize_chapters

__all__ = [
    "ExtractedDocument",
    "extract_from_url",
    "extract_vatican",
    "Chapter",
    "split_into_chapters",
    "TTSBackend",
    "synthesize",
    "synthesize_chapters",
    "__version__",
]
