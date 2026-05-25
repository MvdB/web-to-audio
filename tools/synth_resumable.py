"""Resumable full-document TTS synthesis.

Usage:
    python tools/synth_resumable.py URL OUTPUT.mp3 \
        [--backend voxtral|qwen3] [--concurrency N] [--limit N]

Each chunk's WAV is written to ``OUTPUT.mp3.chunks/NNNN.wav`` so that the
script can resume after a crash / kill / Ctrl-C: re-running with the same
arguments skips chunks that already have a WAV. After all chunks are
synthesized, they are concatenated and encoded to a single MP3.

The Voxtral backend supports parallel HTTP requests (``--concurrency``),
which lets vLLM batch them on the GPU for much higher throughput. The
Qwen3 backend is in-process and always serial.

Designed to be watched with ``Monitor`` — every chunk completion prints
one line of progress to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from web_to_audio.audio import save_as_mp3
from web_to_audio.chunk import chunk_text
from web_to_audio.extract import extract_from_url
from web_to_audio.tts import TTSOptions, get_backend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--backend", default="voxtral")
    parser.add_argument("--voice", default="de_male")
    parser.add_argument("--language", default="German")
    parser.add_argument("--max-chunk-chars", type=int, default=800)
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Parallel in-flight requests (voxtral only).")
    parser.add_argument("--limit", type=int, default=None, help="Only synthesize the first N chunks.")
    parser.add_argument("--mp3-bitrate", default="128k")
    args = parser.parse_args()

    output_path = Path(args.output)
    chunk_dir = output_path.with_suffix(output_path.suffix + ".chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] {args.url}", flush=True)
    doc = extract_from_url(args.url)
    print(f"[extract] {doc.title} – {len(doc.paragraphs)} paragraphs, {len(doc):,} chars", flush=True)

    chunks = chunk_text(doc.text, max_chars=args.max_chunk_chars)
    if args.limit:
        chunks = chunks[: args.limit]
    total = len(chunks)
    print(f"[chunk] {total} chunks of max {args.max_chunk_chars} chars", flush=True)

    print(f"[tts] loading backend={args.backend}", flush=True)
    t0 = time.time()
    backend = get_backend(args.backend)
    print(f"[tts] backend ready in {time.time() - t0:.1f}s", flush=True)

    sample_rate: int | None = None
    audio_seconds_total = 0.0
    inference_seconds_total = 0.0

    # Build the list of (idx, chunk) that still need synthesis; pre-account
    # for the audio length of cached chunks for the running total.
    todo: list[tuple[int, str]] = []
    for idx, chunk in enumerate(chunks, start=1):
        chunk_path = chunk_dir / f"{idx:04d}.wav"
        if chunk_path.exists():
            info = sf.info(chunk_path)
            sample_rate = sample_rate or int(info.samplerate)
            audio_seconds_total += info.duration
            print(f"[skip] chunk {idx}/{total} ({chunk_path.name}, {info.duration:.1f}s, cached)", flush=True)
        else:
            todo.append((idx, chunk))

    if todo:
        if args.backend == "voxtral" and args.concurrency > 1:
            sample_rate, ai, ii = asyncio.run(
                _synth_parallel(backend, args, todo, total, chunk_dir)
            )
            audio_seconds_total += ai
            inference_seconds_total += ii
        else:
            for idx, chunk in todo:
                chunk_path = chunk_dir / f"{idx:04d}.wav"
                opts = TTSOptions(
                    text=chunk,
                    language=args.language,
                    voice=args.voice,
                    max_chunk_chars=args.max_chunk_chars,
                )
                t_chunk = time.time()
                wav, sr = backend.synthesize(opts)
                dt = time.time() - t_chunk
                inference_seconds_total += dt
                audio_seconds_total += len(wav) / sr
                sample_rate = sr
                sf.write(chunk_path, wav, sr, subtype="PCM_16")
                rtf = dt / (len(wav) / sr) if len(wav) else 0
                print(
                    f"[tts ] chunk {idx}/{total} → {chunk_path.name} "
                    f"({len(chunk)} chars, {len(wav)/sr:.1f}s audio, {dt:.1f}s wall, RTF={rtf:.2f})",
                    flush=True,
                )

    print(
        f"[summary] {audio_seconds_total/60:.1f} min of audio, "
        f"{inference_seconds_total/60:.1f} min of inference",
        flush=True,
    )

    print(f"[concat] reading {total} chunks back …", flush=True)
    pieces = []
    for idx in range(1, total + 1):
        wav, sr = sf.read(chunk_dir / f"{idx:04d}.wav", dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        pieces.append(wav)
        if sample_rate is None:
            sample_rate = int(sr)
    full = np.concatenate(pieces)
    assert sample_rate is not None
    print(f"[mp3] encoding {len(full)/sample_rate/60:.1f} min to {output_path}", flush=True)
    save_as_mp3(full, sample_rate, output_path, bitrate=args.mp3_bitrate)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"[done] {output_path}  ({size_mb:.1f} MB)", flush=True)
    return 0


async def _synth_parallel(
    backend,
    args,
    todo: list[tuple[int, str]],
    total: int,
    chunk_dir: Path,
) -> tuple[int, float, float]:
    """Issue Voxtral chunks via vLLM-Omni concurrently and write WAVs as they arrive.

    Returns (sample_rate, audio_seconds_added, inference_seconds_added).
    """
    import httpx

    sample_rate = 0
    audio_seconds = 0.0
    inference_seconds = 0.0
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=backend.timeout) as client:
        async def one(idx: int, chunk: str) -> None:
            nonlocal sample_rate, audio_seconds, inference_seconds
            chunk_path = chunk_dir / f"{idx:04d}.wav"
            opts = TTSOptions(
                text=chunk,
                language=args.language,
                voice=args.voice,
                max_chunk_chars=args.max_chunk_chars,
            )
            async with sem:
                t = time.time()
                wav, sr = await backend.asynthesize(opts, client)
                dt = time.time() - t
            audio_dur = len(wav) / sr
            audio_seconds += audio_dur
            inference_seconds += dt
            sample_rate = sr
            sf.write(chunk_path, wav, sr, subtype="PCM_16")
            print(
                f"[tts ] chunk {idx}/{total} → {chunk_path.name} "
                f"({len(chunk)} chars, {audio_dur:.1f}s audio, {dt:.1f}s wall, RTF={dt/audio_dur:.2f})",
                flush=True,
            )

        await asyncio.gather(*(one(idx, chunk) for idx, chunk in todo))

    return sample_rate, audio_seconds, inference_seconds


if __name__ == "__main__":
    sys.exit(main())
