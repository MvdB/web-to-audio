"""Tests for playlist writing."""

from __future__ import annotations

import json

from web_to_audio.playlist import PlaylistEntry, write_json_index, write_m3u


def test_write_m3u(tmp_path):
    entries = [
        PlaylistEntry(path="001.mp3", title="1 – Einleitung", duration_seconds=62.4),
        PlaylistEntry(path="002.mp3", title="2 – Zweiter", duration_seconds=30.0),
    ]
    p = write_m3u(tmp_path / "playlist.m3u", entries)
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "#EXTM3U"
    assert lines[1] == "#EXTINF:62,1 – Einleitung"
    assert lines[2] == "001.mp3"
    assert lines[3] == "#EXTINF:30,2 – Zweiter"
    assert lines[4] == "002.mp3"


def test_write_json_index(tmp_path):
    entries = [PlaylistEntry(path="001.mp3", title="1 – Einleitung", duration_seconds=62.45)]
    p = write_json_index(tmp_path / "index.json", entries, title="Doc")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["title"] == "Doc"
    assert data["chapters"][0] == {
        "title": "1 – Einleitung",
        "file": "001.mp3",
        "duration_seconds": 62.45,
    }
