"""WAV/MP3 audio helpers."""

from __future__ import annotations

import io
import os
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


def _prepend_imageio_ffmpeg_to_path() -> str | None:
    """If no system ffmpeg is on $PATH, expose imageio-ffmpeg's binary there.

    This is done at *import time*, before pydub is imported anywhere, so
    pydub's own ``_which("ffmpeg")`` check in ``pydub.utils`` does not emit
    its noisy "Couldn't find ffmpeg" warning.
    """
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    bin_dir = os.path.dirname(exe)
    target = os.path.join(bin_dir, "ffmpeg")
    if not os.path.exists(target):
        try:
            os.symlink(exe, target)
        except OSError:
            target = exe
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    return exe


_FFMPEG_PATH = _prepend_imageio_ffmpeg_to_path()


def _ensure_ffmpeg() -> None:
    """Point pydub at the same ffmpeg binary we exposed at module import."""
    from pydub import AudioSegment

    if getattr(AudioSegment, "_w2a_ffmpeg_configured", False):
        return
    if _FFMPEG_PATH:
        AudioSegment.converter = _FFMPEG_PATH
        AudioSegment.ffmpeg = _FFMPEG_PATH
        AudioSegment.ffprobe = _FFMPEG_PATH
    AudioSegment._w2a_ffmpeg_configured = True


def save_as_mp3(audio: np.ndarray, sample_rate: int, path: Path, *, bitrate: str = "128k") -> Path:
    """Write a numpy waveform to an MP3 file.

    ``audio`` is expected to be a 1-D float32 array in [-1.0, 1.0]. Conversion
    goes via an in-memory WAV buffer + ``pydub.AudioSegment``, which uses
    ffmpeg/libmp3lame under the hood.
    """
    _ensure_ffmpeg()
    from pydub import AudioSegment

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if audio.ndim != 1:
        audio = audio.reshape(-1)
    # Clip and convert to 16-bit PCM.
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, pcm, sample_rate, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)

    segment = AudioSegment.from_file(wav_buffer, format="wav")
    segment.export(path, format="mp3", bitrate=bitrate)
    return path
