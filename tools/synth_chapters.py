"""Resumable per-chapter TTS synthesis with a playlist.

Usage:
    python tools/synth_chapters.py URL OUTPUT_DIR \
        [--backend voxtral|qwen3] [--concurrency N] [--limit N]

Splits the document into numbered chapters (see ``web_to_audio.chapters``)
and renders **one MP3 per chapter** into ``OUTPUT_DIR/NNN.mp3`` plus a
``playlist.m3u`` / ``index.json`` over all of them.

Resumable: every chapter's chunk WAVs live in ``OUTPUT_DIR/.chunks/NNN/`` and
the per-chapter MP3 is skipped if it already exists, so a crashed / killed run
can be restarted with the same arguments and picks up where it stopped.

The Voxtral backend issues chunks over HTTP, so ``--concurrency`` lets vLLM
batch them on the GPU for much higher throughput. Designed to be watched with
``Monitor`` — each chapter and chunk completion prints one progress line.
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
from web_to_audio.chapters import Chapter, chapter_filename, split_into_chapters
from web_to_audio.chunk import chunk_text
from web_to_audio.extract import extract_from_url
from web_to_audio.playlist import PlaylistEntry, write_json_index, write_m3u
from web_to_audio.tts import TTSOptions, get_backend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output_dir", help="Directory to write NNN.mp3 + playlist into.")
    parser.add_argument("--backend", default="voxtral")
    parser.add_argument("--voice", default="de_male")
    parser.add_argument("--language", default="German")
    parser.add_argument("--max-chunk-chars", type=int, default=800)
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Parallel in-flight requests (voxtral only).")
    parser.add_argument("--limit", type=int, default=None, help="Only render the first N chapters.")
    parser.add_argument("--mp3-bitrate", default="128k")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    chunk_root = out_dir / ".chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] {args.url}", flush=True)
    doc = extract_from_url(args.url)
    chapters = split_into_chapters(doc)
    if args.limit:
        chapters = chapters[: args.limit]
    print(f"[extract] {doc.title} – {len(chapters)} chapters, {len(doc):,} chars", flush=True)

    # Plan every chunk up front: a flat list of (chapter_idx, chunk_idx, text).
    # chapter_idx is the chapter's position in the list (1-based) so filenames
    # sort correctly even if section numbers ever skip.
    plan: list[tuple[int, Chapter, list[str]]] = []
    total_chunks = 0
    for pos, ch in enumerate(chapters, start=1):
        ch_chunks = chunk_text(ch.text, max_chars=args.max_chunk_chars)
        plan.append((pos, ch, ch_chunks))
        total_chunks += len(ch_chunks)
    print(f"[chunk] {total_chunks} chunks across {len(chapters)} chapters", flush=True)

    # Collect the chunks still needing synthesis (skip cached WAVs).
    todo: list[tuple[str, str]] = []  # (wav_path, text)
    for pos, ch, ch_chunks in plan:
        ch_dir = chunk_root / f"{pos:03d}"
        for ci, chunk in enumerate(ch_chunks, start=1):
            wav_path = ch_dir / f"{ci:04d}.wav"
            if not wav_path.exists():
                todo.append((str(wav_path), chunk))

    print(f"[tts] backend={args.backend}, {len(todo)}/{total_chunks} chunks to synthesize", flush=True)
    t0 = time.time()
    backend = get_backend(args.backend)
    print(f"[tts] backend ready in {time.time() - t0:.1f}s", flush=True)

    if todo:
        chunk_root.mkdir(parents=True, exist_ok=True)
        for wav_path, _ in todo:
            Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
        if args.backend == "voxtral" and args.concurrency > 1:
            asyncio.run(_synth_parallel(backend, args, todo))
        else:
            for n, (wav_path, chunk) in enumerate(todo, start=1):
                opts = TTSOptions(text=chunk, language=args.language, voice=args.voice,
                                  max_chunk_chars=args.max_chunk_chars)
                t = time.time()
                wav, sr = backend.synthesize(opts)
                sf.write(wav_path, wav, sr, subtype="PCM_16")
                dur = len(wav) / sr
                print(f"[tts ] chunk {n}/{len(todo)} → {wav_path} "
                      f"({dur:.1f}s, {time.time()-t:.1f}s wall)", flush=True)

    # Assemble one MP3 per chapter and build the playlist.
    print(f"[mp3] assembling {len(chapters)} chapter files", flush=True)
    entries: list[PlaylistEntry] = []
    for pos, ch, ch_chunks in plan:
        mp3_path = out_dir / chapter_filename(pos, ch)
        ch_dir = chunk_root / f"{pos:03d}"
        pieces = []
        sample_rate = 24000
        for ci in range(1, len(ch_chunks) + 1):
            wav, sr = sf.read(ch_dir / f"{ci:04d}.wav", dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            pieces.append(wav)
            sample_rate = int(sr)
        full = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        duration = len(full) / sample_rate if sample_rate else 0.0
        if not mp3_path.exists():
            save_as_mp3(full, sample_rate, mp3_path, bitrate=args.mp3_bitrate)
        entries.append(PlaylistEntry(path=mp3_path.name, title=ch.title, duration_seconds=duration))
        print(f"[mp3 ] {mp3_path.name}  ({duration/60:.1f} min)  {ch.title[:70]}", flush=True)

    m3u = write_m3u(out_dir / "playlist.m3u", entries)
    write_json_index(out_dir / "index.json", entries, title=doc.title)
    total_min = sum(e.duration_seconds for e in entries) / 60
    print(f"[done] {len(entries)} chapters, {total_min:.1f} min total → {m3u}", flush=True)
    return 0


async def _synth_parallel(backend, args, todo: list[tuple[str, str]]) -> None:
    """Synthesize Voxtral chunks concurrently, writing each WAV as it arrives."""
    import httpx

    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    total = len(todo)

    async with httpx.AsyncClient(timeout=backend.timeout) as client:
        async def one(wav_path: str, chunk: str) -> None:
            nonlocal done
            opts = TTSOptions(text=chunk, language=args.language, voice=args.voice,
                              max_chunk_chars=args.max_chunk_chars)
            async with sem:
                t = time.time()
                wav, sr = await backend.asynthesize(opts, client)
                dt = time.time() - t
            sf.write(wav_path, wav, sr, subtype="PCM_16")
            done += 1
            dur = len(wav) / sr
            print(f"[tts ] chunk {done}/{total} → {wav_path} "
                  f"({dur:.1f}s audio, {dt:.1f}s wall, RTF={dt/dur:.2f})", flush=True)

        await asyncio.gather(*(one(p, c) for p, c in todo))


if __name__ == "__main__":
    sys.exit(main())
