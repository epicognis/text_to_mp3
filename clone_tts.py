"""
clone_tts.py
============
Convert text (or a text file) into an MP3 spoken in a cloned voice, using
Coqui XTTS-v2's zero-shot voice cloning. Runs locally with optional CUDA
acceleration.

How it works
------------
  1. You supply a short reference clip of the target voice (--sample):
     6-30 seconds of clean, single-speaker speech, minimal background noise
     or music. WAV or MP3 both work.
  2. Text is read and preprocessed with the same normalisation pipeline as
     tts_to_mp3.py (Markdown stripping, abbreviation expansion, etc.).
  3. XTTS-v2 synthesises audio conditioned on the reference clip's voice
     characteristics (24 kHz).
  4. The waveform is encoded to MP3 via pydub/ffmpeg.

License note
------------
  XTTS-v2 is distributed under Coqui's CPML (non-commercial) license unless
  you've purchased a commercial license from Coqui. Running this script sets
  COQUI_TOS_AGREED=1, which is the documented way to accept the CPML
  non-interactively -- only use this for personal/non-commercial output.
  See https://coqui.ai/cpml

Input sources
-------------
    python clone_tts.py "Hello world." --sample me.wav
    python clone_tts.py --file chapter.txt --sample me.wav
    python clone_tts.py --file notes.md --sample me.wav
    python clone_tts.py --file report.pdf --sample me.wav
    cat chapter.txt | python clone_tts.py - --sample me.wav -o out.mp3

Reference sample tips
----------------------
  - 6-30 seconds is enough; more doesn't reliably help and XTTS only uses a
    short window internally.
  - Prefer a clip with steady, natural speaking pace -- avoid clips that are
    mostly silence, laughter, or overlapping speakers.
  - Clean audio (no music, minimal room echo/noise) clones noticeably better.
  - You can pass multiple clips for a more robust voice print:
      --sample clip1.wav clip2.wav clip3.wav

Languages
---------
  XTTS-v2 supports: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn,
  ja, hu, ko, hi. Pass with --language (default: en).

Examples
--------
    # Clone from a single reference clip
    python clone_tts.py "This is a test of my cloned voice." --sample voice_sample.wav

    # From a file, custom output path
    python clone_tts.py --file chapter1.txt --sample me.wav -o chapter1_cloned.mp3

    # Multiple reference clips for a stronger voice print
    python clone_tts.py --file article.txt --sample clip1.wav clip2.wav

    # 5% faster pacing
    python clone_tts.py --file article.txt --sample me.wav --speed 1.05

    # Force CPU
    python clone_tts.py "test" --sample me.wav --device cpu -o test.mp3
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment

# Reuse the text-preprocessing / input-reading pipeline from tts_to_mp3.py.
from tts_to_mp3 import preprocess_for_tts, read_input_text

# XTTS-v2 is CPML-licensed (non-commercial unless you hold a Coqui commercial
# license). This env var is the documented non-interactive way to accept the
# CPML so the first run doesn't block on an interactive prompt.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

# torchaudio >= 2.9 loads audio exclusively through torchcodec, which in turn
# needs ffmpeg's *shared* libraries (avcodec/avformat DLLs) on PATH. The
# ffmpeg build this project's README has you install (Gyan's "full_build")
# is static-only and doesn't ship those DLLs, so torchaudio.load() fails.
# soundfile (already a dependency, via libsndfile) reads WAV/MP3/FLAC fine
# without touching ffmpeg at all, so swap it in as a drop-in replacement --
# coqui-tts's speaker-embedding code only ever calls torchaudio.load().
def _load_with_soundfile(uri, frame_offset=0, num_frames=-1, normalize=True,
                          channels_first=True, format=None, buffer_size=4096,
                          backend=None):
    data, sr = sf.read(str(uri), dtype="float32", always_2d=True)  # [time, channels]
    if num_frames != -1:
        data = data[frame_offset:frame_offset + num_frames]
    elif frame_offset:
        data = data[frame_offset:]
    tensor = torch.from_numpy(data.copy())
    if channels_first:
        tensor = tensor.transpose(0, 1)  # [channels, time]
    return tensor, sr


torchaudio.load = _load_with_soundfile

XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_SAMPLE_RATE = 24_000

XTTS_LANGUAGES = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar",
    "zh-cn", "ja", "hu", "ko", "hi",
}


def pick_device(requested: str) -> str:
    """Return 'cuda' or 'cpu', warning if CUDA was requested but unavailable."""
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        if requested == "cuda":
            print("[warn] --device cuda was requested but torch.cuda.is_available() is False. "
                  "Falling back to CPU.", file=sys.stderr)
        return "cpu"
    name = torch.cuda.get_device_name(0)
    print(f"[info] Using CUDA on {name}.", file=sys.stderr)
    return "cuda"


def write_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "192k") -> None:
    """Re-encode a WAV file to MP3 via pydub/ffmpeg."""
    segment = AudioSegment.from_wav(wav_path)
    segment.export(mp3_path, format="mp3", bitrate=bitrate)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert text to an MP3 spoken in a cloned voice, using Coqui XTTS-v2.\n"
            "Text is preprocessed with the same pipeline as tts_to_mp3.py.\n"
            "Run with --help for full documentation including sample-clip guidance."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "text", nargs="?",
        help=(
            "Text to synthesise (inline string). "
            "Pass '-' to read from stdin, or omit and use --file."
        ),
    )
    parser.add_argument(
        "-f", "--file",
        metavar="PATH",
        help=(
            "Read input text from a file. Supported formats: "
            ".txt (UTF-8 plain text), .md (Markdown), .pdf (requires pymupdf). "
            "Takes priority over the positional argument."
        ),
    )
    parser.add_argument(
        "--sample", nargs="+", required=True,
        metavar="AUDIO_PATH",
        help=(
            "One or more reference audio clips (WAV/MP3) of the voice to clone. "
            "6-30 seconds of clean, single-speaker speech works best. "
            "Multiple clips build a more robust voice print."
        ),
    )
    parser.add_argument(
        "-o", "--output", default=None,
        metavar="PATH",
        help=(
            "Destination MP3 file path. Parent directories are created automatically. "
            "Defaults to the input filename with a .mp3 extension, or 'output.mp3' "
            "when reading from a string or stdin."
        ),
    )
    parser.add_argument(
        "-l", "--language", default="en",
        metavar="LANG",
        choices=sorted(XTTS_LANGUAGES),
        help="XTTS-v2 language code. (default: en)",
    )
    parser.add_argument(
        "-s", "--speed", type=float, default=1.0,
        metavar="MULTIPLIER",
        help="Playback speed multiplier. 1.0 is natural speed; 1.05 is 5%% faster. "
             "(default: 1.0)",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="Compute device: 'auto' tries CUDA then falls back to CPU, 'cuda' forces GPU, "
             "'cpu' always uses CPU. (default: auto)",
    )
    parser.add_argument(
        "--bitrate", default="192k",
        metavar="BITRATE",
        help="MP3 encoding bitrate passed to ffmpeg/pydub. (default: 192k)",
    )
    args = parser.parse_args()

    for sample_path in args.sample:
        if not Path(sample_path).exists():
            parser.error(f"Reference sample not found: {sample_path}")

    if args.output is None:
        if args.file:
            args.output = str(Path(args.file).with_suffix(".mp3"))
        else:
            args.output = "output.mp3"

    text = read_input_text(args).strip()
    if not text:
        parser.error("Input text is empty.")
    print(f"[info] Preprocessing {len(text):,} chars for TTS...", file=sys.stderr)
    text = preprocess_for_tts(text)

    device = pick_device(args.device)

    print(f"[info] Loading XTTS-v2 (device={device})... this can take a while on first run "
          f"(downloads ~1.8 GB).", file=sys.stderr)
    from TTS.api import TTS
    tts = TTS(XTTS_MODEL_NAME, progress_bar=True).to(device)

    print(f"[info] Synthesizing {len(text):,} characters cloned from "
          f"{', '.join(args.sample)}...", file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path = out_path.with_suffix(".tmp.wav")

    tts.tts_to_file(
        text=text,
        speaker_wav=args.sample,
        language=args.language,
        speed=args.speed,
        file_path=str(wav_path),
    )

    write_mp3(wav_path, out_path, bitrate=args.bitrate)
    wav_path.unlink(missing_ok=True)

    print(f"[info] Wrote {out_path.resolve()} "
          f"({out_path.stat().st_size / 1024:.1f} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
