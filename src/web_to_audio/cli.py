"""Command-line entry point: web URL → MP3 audio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .extract import extract_from_url, fetch_html


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="web-to-audio",
        description=(
            "Extract clean body text from a web page (currently: vatican.va) "
            "and synthesize it to MP3 with an open-weights TTS model."
        ),
    )
    parser.add_argument("url", help="URL of the page to convert.")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output MP3 path. Default: derived from the URL slug, in ./out/.",
    )
    parser.add_argument(
        "--text-only", action="store_true",
        help="Just print the extracted text and exit (no TTS).",
    )
    parser.add_argument(
        "--save-text", default=None,
        help="Also save the extracted plain text to this path.",
    )

    backend = parser.add_argument_group("TTS")
    backend.add_argument(
        "--backend", choices=["voxtral", "qwen3"], default="qwen3",
        help="Which TTS model to use. Default: qwen3 (in-process).",
    )
    backend.add_argument(
        "--voice", default="",
        help="Backend-specific voice name. Defaults: qwen3='Aiden', voxtral='narration_male'.",
    )
    backend.add_argument(
        "--language", default="German",
        help="Spoken language (qwen3 only — voxtral infers from text). Default: German.",
    )
    backend.add_argument(
        "--instruct", default=None,
        help="Optional natural-language style instruction (qwen3 only).",
    )
    backend.add_argument(
        "--max-chunk-chars", type=int, default=800,
        help="Maximum characters per TTS chunk. Default: 800.",
    )
    backend.add_argument(
        "--mp3-bitrate", default="128k",
        help="Output MP3 bitrate (e.g. 96k, 128k, 192k). Default: 128k.",
    )

    voxtral = parser.add_argument_group("Voxtral backend options")
    voxtral.add_argument(
        "--voxtral-url", default=None,
        help="Base URL of the running vLLM-Omni Voxtral server (default: http://localhost:8000/v1).",
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _default_output_path(url: str) -> Path:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    if slug.endswith(".html"):
        slug = slug[:-5]
    if not slug:
        slug = "output"
    return Path("out") / f"{slug}.mp3"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    html = fetch_html(args.url)
    doc = extract_from_url(args.url, html=html)

    print(f"[extract] title    : {doc.title}", file=sys.stderr)
    print(f"[extract] language : {doc.language}", file=sys.stderr)
    print(f"[extract] paragraphs: {len(doc.paragraphs)}", file=sys.stderr)
    print(f"[extract] characters: {len(doc):,}", file=sys.stderr)

    if args.save_text:
        Path(args.save_text).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_text).write_text(doc.text + "\n", encoding="utf-8")
        print(f"[extract] saved text → {args.save_text}", file=sys.stderr)

    if args.text_only:
        print(doc.text)
        return 0

    output = Path(args.output) if args.output else _default_output_path(args.url)
    print(f"[tts] backend  : {args.backend}", file=sys.stderr)
    print(f"[tts] output   : {output}", file=sys.stderr)

    backend_kwargs: dict = {}
    if args.backend == "voxtral" and args.voxtral_url:
        backend_kwargs["base_url"] = args.voxtral_url

    from .tts import synthesize

    out_path = synthesize(
        doc.text,
        output_path=output,
        backend=args.backend,
        language=args.language,
        voice=args.voice,
        instruct=args.instruct,
        max_chunk_chars=args.max_chunk_chars,
        backend_kwargs=backend_kwargs,
        mp3_bitrate=args.mp3_bitrate,
    )
    print(f"[tts] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
