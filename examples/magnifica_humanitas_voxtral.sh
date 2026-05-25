#!/usr/bin/env bash
# Beispiel: Vertonung mit Voxtral (Mistral) über einen lokalen vLLM-Omni-Server.
#
# In einem zweiten Terminal vorab starten:
#   vllm serve mistralai/Voxtral-4B-TTS-2603 --omni
#
set -euo pipefail

URL="https://www.vatican.va/content/leo-xiv/de/encyclicals/documents/20260515-magnifica-humanitas.html"

mkdir -p out

web-to-audio "$URL" \
    --backend voxtral \
    --voxtral-url "${WEB_TO_AUDIO_VOXTRAL_URL:-http://localhost:8000/v1}" \
    --voice narration_male \
    --mp3-bitrate 128k \
    -o out/magnifica-humanitas-voxtral.mp3
