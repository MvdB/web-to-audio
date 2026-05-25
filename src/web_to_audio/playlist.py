"""Write playlists that tie per-chapter audio files together.

Produces an extended M3U (``#EXTM3U`` / ``#EXTINF``) — the de-facto standard
understood by VLC, foobar2000, mpv, most car stereos and podcast apps — and,
alongside it, a small JSON index for programmatic use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PlaylistEntry:
    """One track in a playlist."""

    path: str  # path to the audio file, relative to the playlist location
    title: str
    duration_seconds: float = 0.0


def write_m3u(playlist_path: Path, entries: list[PlaylistEntry]) -> Path:
    """Write an extended-M3U playlist."""
    playlist_path = Path(playlist_path)
    playlist_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["#EXTM3U"]
    for e in entries:
        lines.append(f"#EXTINF:{round(e.duration_seconds)},{e.title}")
        lines.append(e.path)

    playlist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return playlist_path


def write_json_index(index_path: Path, entries: list[PlaylistEntry], *, title: str = "") -> Path:
    """Write a JSON index describing the chapters and their audio files."""
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "title": title,
        "chapters": [
            {
                "title": e.title,
                "file": e.path,
                "duration_seconds": round(e.duration_seconds, 2),
            }
            for e in entries
        ],
    }
    index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index_path
