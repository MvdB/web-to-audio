#!/usr/bin/env bash
# Beispiel: eine MP3 pro Kapitel + Playlist, vertont mit Voxtral (Mistral)
# über einen lokalen vLLM-Omni-Server.
#
# In einem zweiten Terminal vorab starten:
#   vllm serve mistralai/Voxtral-4B-TTS-2603 --omni \
#       --gpu-memory-utilization 0.35 --max-model-len 4096
#
set -euo pipefail

URL="https://www.vatican.va/content/leo-xiv/de/encyclicals/documents/20260515-magnifica-humanitas.html"
OUT_DIR="out/magnifica-humanitas-chapters"

# Wiederaufnehmbarer, paralleler Lauf (eine Datei NNN.mp3 je Abschnitt +
# playlist.m3u + index.json). Bei Abbruch einfach erneut starten.
python tools/synth_chapters.py "$URL" "$OUT_DIR" \
    --backend voxtral \
    --voice de_male \
    --concurrency 8 \
    --mp3-bitrate 128k
