#!/usr/bin/env bash
# Beispiel: Enzyklika "Magnifica humanitas" (Leo XIV., 2026) zu MP3.
#
# Voraussetzungen:
#   pip install -e .
#   pip install -U qwen-tts     # für das Qwen3-Backend
#   apt-get install ffmpeg      # für die MP3-Kodierung
set -euo pipefail

URL="https://www.vatican.va/content/leo-xiv/de/encyclicals/documents/20260515-magnifica-humanitas.html"

mkdir -p out

# 1. Text extrahieren + speichern (kein TTS) — gut zum Verifizieren der Extraktion.
web-to-audio "$URL" \
    --text-only \
    --save-text out/magnifica-humanitas.txt > /dev/null

echo "Text-Extraktion fertig: $(wc -c < out/magnifica-humanitas.txt) Zeichen."

# 2. MP3 erzeugen mit Qwen3-TTS (Standardstimme "Aiden", Sprache Deutsch).
web-to-audio "$URL" \
    --backend qwen3 \
    --voice Aiden \
    --language German \
    --mp3-bitrate 128k \
    -o out/magnifica-humanitas.mp3
