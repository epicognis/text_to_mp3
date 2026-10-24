# tts_to_mp3 — local text-to-speech with Kokoro-82M

A small Python CLI that turns text (string, file, or stdin) into an MP3 using
the open-weight [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS
model, running locally on your machine.

## Why Kokoro

- **Tiny**: 82M parameters (~330 MB), so model load + VRAM use is trivial on
  a 16 GB card.
- **Fast**: comfortably real-time on CPU, multiple-x realtime on a modern GPU.
- **Open-weight (Apache 2.0)**: free to run offline.
- **Decent voices**: 50+ across 8 languages.

## Hardware notes — RTX 5060 Ti 16 GB / Xeon / 64 GB

You have plenty of headroom. The model needs <1 GB VRAM; the 16 GB on your
5060 Ti is overkill (in a good way — it leaves the GPU free for anything else
running alongside). The Xeon + 64 GB RAM means CPU fallback is also painless.

**One trap to avoid.** The RTX 5060 Ti is Blackwell, CUDA compute capability
**sm_120**. The default `pip install torch` wheels do **not** support sm_120 —
they'll print `NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not
compatible with the current PyTorch installation` and silently fall back to
CPU. Use the **cu128** build instead (see step 2 below).

## Setup

### 1. System dependencies

- **espeak-ng** (Kokoro's grapheme-to-phoneme fallback)
  - Windows: download installer from <https://github.com/espeak-ng/espeak-ng/releases>
  - Linux: `sudo apt-get install espeak-ng`
  - macOS: `brew install espeak-ng`
- **ffmpeg** (for MP3 encoding via pydub)
  - Windows: `choco install ffmpeg` or `winget install Gyan.FFmpeg`
  - Linux: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`

### 2. Python environment

Python **3.10–3.12** (Kokoro requires `<3.13`).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. PyTorch — the cu128 build (required for RTX 50-series)

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

Verify it sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
```

You should see `True`, `NVIDIA GeForce RTX 5060 Ti`, and a list of `sm_*`
entries that includes `sm_120`. If `sm_120` is missing from that list, the
script will warn you and fall back to CPU.

### 4. The rest

```bash
pip install -r requirements.txt
```

First run will download Kokoro weights (~330 MB) and voice packs from
Hugging Face into your HF cache. Subsequent runs are offline.

## Usage

```bash
# Quick test
python tts_to_mp3.py "Hello world, this is Kokoro speaking." -o hello.mp3

# From a file
python tts_to_mp3.py --file chapter1.txt -o chapter1.mp3

# British male voice, slightly faster
python tts_to_mp3.py --file article.txt --voice bm_george --speed 1.15 -o article.mp3

# From stdin
type article.txt | python tts_to_mp3.py - -o article.mp3      # Windows
cat  article.txt | python tts_to_mp3.py - -o article.mp3      # Linux/macOS

# List common voices
python tts_to_mp3.py --list-voices

# Force CPU
python tts_to_mp3.py "test" --device cpu -o test.mp3
```

### All flags

| Flag             | Default      | Meaning                                                              |
| ---------------- | ------------ | -------------------------------------------------------------------- |
| `text` (pos.)    | —            | Text to read. Use `-` (or pipe) to read from stdin.                  |
| `-f / --file`    | —            | Read text from a file instead of the positional argument.            |
| `-o / --output`  | `output.mp3` | Output MP3 path.                                                     |
| `-v / --voice`   | `af_heart`   | Voice name. See `--list-voices`.                                     |
| `-s / --speed`   | `1.0`        | Speed multiplier (0.25–4.0).                                         |
| `-l / --lang`    | `en-us`      | Language: `a` American, `b` British, `j` Japanese, `z` Mandarin, etc.|
| `--device`       | `auto`       | `auto`, `cuda`, or `cpu`.                                            |
| `--bitrate`      | `192k`       | MP3 bitrate.                                                         |
| `--list-voices`  | —            | Print the common voice list and exit.                                |

## Notes

- For long inputs, Kokoro already chunks internally on sentence boundaries;
  the script concatenates the chunks into a single MP3.
- If you want a different language, you'll also want to install its G2P
  extras: `pip install "misaki[ja]"` for Japanese, `pip install "misaki[zh]"`
  for Mandarin, etc.
- The Kokoro model card lists per-voice quality grades (A–F). `af_heart`,
  `af_bella`, and `bm_george` are among the best-graded English voices.

## Voice cloning — clone_tts.py

`clone_tts.py` is a second script that speaks in a **cloned voice** from a
short reference sample instead of one of Kokoro's preset voices. It uses
[Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2), a zero-shot voice
cloning model, and shares `tts_to_mp3.py`'s text preprocessing pipeline.

**License**: XTTS-v2 is distributed under Coqui's [CPML](https://coqui.ai/cpml)
— free for personal/research use, but **not** free for commercial use without
a license from Coqui (licensing@coqui.ai). `clone_tts.py` sets
`COQUI_TOS_AGREED=1`, the documented non-interactive way to accept the CPML —
only use this script for non-commercial output.

### Setup

Already covered by `pip install -r requirements.txt` (adds `coqui-tts`,
a pinned `transformers==4.57.1`, and `torchcodec`). Two things worth knowing:

- **`transformers` is pinned to 4.57.1.** Newer `transformers` (5.x) removed
  an internal function XTTS's tortoise-derived layers depend on. This pin is
  compatible with Kokoro too, so both scripts share one venv.
- **`torchaudio` doesn't actually use ffmpeg here.** Since `torchaudio` 2.9,
  its audio loader requires torchcodec, which needs ffmpeg's *shared*
  libraries (DLLs) — but this project's ffmpeg (from the "full_build" static
  package) doesn't ship those. `clone_tts.py` routes around this by
  monkeypatching `torchaudio.load` to use `soundfile` instead, which already
  reads WAV/MP3 fine. No extra install needed; if you ever see a
  `libtorchcodec` error, it means something imported `torchaudio` before
  `clone_tts.py`'s patch ran.

First run downloads the XTTS-v2 weights (~1.8 GB) into your HF cache.
Subsequent runs are offline.

### Reference sample

Record or extract 6–30 seconds of clean, single-speaker audio (WAV or MP3).
Minimal background noise/music, steady natural pace, no overlapping talkers.
More than one clip can be passed for a stronger voice print.

### Usage

```bash
# Clone from a single reference clip
python clone_tts.py "This is a test of my cloned voice." --sample voice_sample.wav

# From a file, custom output path
python clone_tts.py --file chapter1.txt --sample me.wav -o chapter1_cloned.mp3

# Multiple reference clips for a stronger voice print
python clone_tts.py --file article.txt --sample clip1.wav clip2.wav

# Non-English (see --help for the full language list)
python clone_tts.py --file article_fr.txt --sample me.wav --language fr

# Force CPU
python clone_tts.py "test" --sample me.wav --device cpu -o test.mp3
```

| Flag             | Default      | Meaning                                                              |
| ---------------- | ------------ | -----------------------------------------------------------------    |
| `text` (pos.)    | —            | Text to read. Use `-` (or pipe) to read from stdin.                  |
| `-f / --file`    | —            | Read text from a file instead of the positional argument.            |
| `--sample`       | *(required)* | One or more reference audio clips (WAV/MP3) of the voice to clone.   |
| `-o / --output`  | `output.mp3` | Output MP3 path.                                                     |
| `-l / --language`| `en`         | XTTS-v2 language code (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi). |
| `--device`       | `auto`       | `auto`, `cuda`, or `cpu`.                                            |
| `--bitrate`      | `192k`       | MP3 bitrate.                                                         |
