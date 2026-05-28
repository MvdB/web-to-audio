# web-to-audio

Extrahiert sauberen Fließtext aus Web-Seiten (zurzeit: vatican.va-Enzykliken
und vergleichbare Dokumente) und vertont ihn mit modernen Open-Weights-TTS-
Modellen zu einer MP3-Datei.

Unterstützte TTS-Backends:

| Backend  | Modell                                    | Wie es läuft                                 |
| -------- | ----------------------------------------- | -------------------------------------------- |
| `voxtral`| `mistralai/Voxtral-4B-TTS-2603`           | externer vLLM-Omni-Server (`vllm serve --omni`) |
| `qwen3`  | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`    | im Prozess via `qwen-tts` (lädt von HF Hub)  |

Beide Modelle beherrschen Deutsch.

## Installation

```bash
git clone https://github.com/your-username/web-to-audio.git
cd web-to-audio
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Für die MP3-Kodierung wird `ffmpeg` benötigt (von `pydub` aufgerufen):

```bash
# Debian/Ubuntu
sudo apt install ffmpeg
# macOS
brew install ffmpeg
```

### Backend-spezifische Abhängigkeiten

**Qwen3-TTS (in-process, einfacher)**

```bash
pip install -U qwen-tts
# Optional, deutlich schneller auf NVIDIA-GPUs:
pip install -U flash-attn --no-build-isolation
```

Erste Synthese lädt ~3–4 GB Gewichte von Hugging Face nach. Eine CUDA-GPU
mit ≥ 8 GB VRAM wird empfohlen; mit `dtype="float32"` und CPU funktioniert
es technisch, ist aber langsam.

**Voxtral (HTTP-Client zu vLLM-Omni)**

```bash
# Im Server-Container:
uv pip install -U vllm vllm-omni
vllm serve mistralai/Voxtral-4B-TTS-2603 --omni
# Im Client (dieses Projekt): nur httpx wird benötigt — schon enthalten.
```

Das CLI verbindet sich standardmäßig auf `http://localhost:8000/v1`.
Anderer Host: `--voxtral-url https://example.invalid/v1` oder via
Umgebungsvariable `WEB_TO_AUDIO_VOXTRAL_URL`.

## Schnellstart

```bash
# Nur Text extrahieren (kein TTS, gut zum Vorab-Prüfen):
web-to-audio "https://www.vatican.va/content/leo-xiv/de/encyclicals/documents/20260515-magnifica-humanitas.html" \
    --text-only --save-text out/magnifica-humanitas.txt

# Vollständige Vertonung mit Qwen3-TTS, Standardstimme "Aiden":
web-to-audio "https://www.vatican.va/content/leo-xiv/de/encyclicals/documents/20260515-magnifica-humanitas.html" \
    --backend qwen3 \
    --voice Aiden \
    --language German \
    -o out/magnifica-humanitas.mp3

# Vertonung mit Voxtral (Server muss laufen):
web-to-audio "https://www.vatican.va/content/leo-xiv/de/encyclicals/documents/20260515-magnifica-humanitas.html" \
    --backend voxtral \
    --voice de_male \
    --voxtral-url http://localhost:8000/v1 \
    -o out/magnifica-humanitas.mp3
```

### Eine MP3 pro Kapitel + Playlist

Mit `--split-chapters` wird nicht eine große Datei erzeugt, sondern **eine
MP3 pro Überschriften-Gruppe** (bei *Magnifica humanitas* gut 60 statt 245
einzelne Abschnitte). Jede unnummerierte Überschrift im Dokument startet
ein neues Kapitel und nimmt alle darauf folgenden nummerierten Abschnitte
in sich auf. Direkt aufeinanderfolgende Überschriften (z. B. `ERSTES
KAPITEL` + ALL-CAPS-Untertitel) gehören zur selben Gruppe.

Die Dateien werden als `NN - Titel.mp3` benannt, dazu kommen `playlist.m3u`
(von VLC, mpv, Autoradios, Podcast-Apps lesbar) und `index.json`:

```bash
web-to-audio "https://www.vatican.va/content/leo-xiv/de/encyclicals/documents/20260515-magnifica-humanitas.html" \
    --backend voxtral --voice de_male \
    --split-chapters \
    -o out/magnifica-humanitas-chapters
```

Beispiel-Dateinamen:

```
01 - EINLEITUNG.mp3
02 - Die res novae unserer Zeit.mp3
06 - ERSTES KAPITEL – EIN DYNAMISCHES DENKEN IM GEISTE DES EVANGELIUMS.mp3
07 - Eine Kirche unterwegs in der Geschichte der Menschheit.mp3
…
```

Die Schlussformel mit Ort, Datum und Unterschrift fällt in das letzte
Kapitel.

Für lange Dokumente bietet `tools/synth_chapters.py` dieselbe Funktion
**wiederaufnehmbar und parallel** (Voxtral): jeder Chunk wird unter
`.chunks/NNN/` zwischengespeichert, ein Abbruch kann mit identischen
Argumenten fortgesetzt werden, und `--concurrency` lässt vLLM die Anfragen
auf der GPU batchen:

```bash
python tools/synth_chapters.py "$URL" out/magnifica-humanitas-chapters \
    --backend voxtral --voice de_male --concurrency 8
```

## Wie es funktioniert

1. **HTML laden** (`extract.fetch_html`) – `User-Agent` gesetzt, Encoding ggf.
   gegen `apparent_encoding` korrigiert (vatican.va deklariert ISO-8859-1,
   liefert aber UTF-8).
2. **Vatikan-spezifischer Parser** (`extract.extract_vatican`) findet den
   inneren `<div class="vaticanrichtext">`-Block und folgt zwei Markern:
   - Body beginnt beim ersten `<a name="…">`-Anker (alles davor ist TOC).
   - Body endet beim ersten `<hr align="left">` (danach kommen Fußnoten).
3. **Fußnoten-Markierungen** wie `[1]`, `[12]` werden inline entfernt
   (sowohl die `<a name="_ftnref…">`-Anker als auch reine Textreste).
4. **Chunking** (`chunk.chunk_text`): Aufteilung an Absatzgrenzen, dann an
   Satzgrenzen, mit hartem Cap (Standard 800 Zeichen) — das hält Eingaben
   im Sweet-Spot der TTS-Modelle.
5. **Text-Normalisierung** (`normalize.normalize_for_speech`): Zeilen, die
   überwiegend aus Großbuchstaben bestehen (z. B. `EINLEITUNG`,
   `ERSTES KAPITEL`, `LEO PP. XIV`), werden zur Vertonung in Satzschreibung
   gewandelt — Voxtral & Co. lesen sonst Buchstabe für Buchstabe oder mit
   unnatürlicher Betonung. Echte römische Ziffern (`XIV`, …) bleiben groß.
6. **TTS** (`tts.synthesize`): Backend wird über `--backend` ausgewählt.
   Audio jedes Chunks wird konkatieniert.
7. **MP3-Export** (`audio.save_as_mp3`) — via `soundfile` + `pydub`/ffmpeg.

Mit `--split-chapters` tritt zwischen Schritt 3 und 4 die
**Kapitel-Aufteilung** (`chapters.split_into_chapters`): jede
unnummerierte Überschrift startet ein neues Kapitel und sammelt die
nachfolgenden nummerierten Abschnitte ein, dann wird jedes Kapitel einzeln
vertont und über `playlist.write_m3u` eine Playlist geschrieben.

## Projektstruktur

```
src/web_to_audio/
├── __init__.py
├── extract.py         # Text-Extraktion (vatican.va)
├── chunk.py           # Aufteilung in TTS-Chunks
├── chapters.py        # Gruppierung in nummerierte Kapitel
├── normalize.py       # Großschreibung für TTS in Satzschreibung wandeln
├── playlist.py        # M3U- und JSON-Playlist schreiben
├── tts.py             # Backend-Dispatcher + synthesize[_chapters]()
├── audio.py           # WAV → MP3
├── backends/
│   ├── voxtral.py     # HTTP-Client für vLLM-Omni Voxtral
│   └── qwen3.py       # In-Process via qwen-tts
└── cli.py             # web-to-audio Entry-Point
tools/                 # synth_resumable.py, synth_chapters.py (wiederaufnehmbar)
tests/                 # Pytest, läuft ohne GPU/Netz
examples/              # Beispielaufrufe
```

## Weitere Sites unterstützen

`extract_from_url` dispatcht auf Host-Basis. Um z. B. eine andere Domain zu
unterstützen, eine Funktion `extract_meine_seite(url, html) -> ExtractedDocument`
ergänzen und in `extract_from_url` registrieren. Tests in `tests/test_extract.py`
nutzen ein Fixture-HTML und brauchen kein Netz.

## Programmatische Nutzung

```python
from web_to_audio import extract_from_url, synthesize

doc = extract_from_url(
    "https://www.vatican.va/content/leo-xiv/de/encyclicals/documents/20260515-magnifica-humanitas.html"
)
print(doc.title, len(doc.paragraphs), "paragraphs")

synthesize(
    doc.text,
    output_path="out/magnifica-humanitas.mp3",
    backend="qwen3",
    language="German",
    voice="Aiden",
)

# Oder: eine MP3 pro Kapitel + Playlist
from web_to_audio import synthesize_chapters

synthesize_chapters(
    doc,
    output_dir="out/magnifica-humanitas-chapters",
    backend="voxtral",
    voice="de_male",
)
```

## Lizenz

- Code in diesem Repository: MIT (siehe `LICENSE`).
- Voxtral-Modell und Referenzstimmen: CC BY-NC 4.0 (nur nicht-kommerzielle
  Nutzung – siehe Mistral-Modellkarte).
- Qwen3-TTS: Apache 2.0.
- Texte von vatican.va: © Libreria Editrice Vaticana – diese Pipeline lädt
  und verarbeitet sie nur; eigene Veröffentlichung der erzeugten Audiodateien
  erfordert ggf. eine eigene Lizenzklärung.
