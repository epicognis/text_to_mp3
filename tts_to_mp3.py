"""
tts_to_mp3.py
=============
Convert text (or a text file) into an MP3 using the Kokoro-82M TTS model,
running entirely locally with optional CUDA acceleration.  No API key, no
network call after the first model download.

How it works
------------
  1. Text is read from a positional argument, --file, or stdin.
  2. A preprocessing pass normalises the text for cleaner speech: Roman
     numeral outline markers are spelled out, footnote/reference numbers are
     stripped, hyphenated compounds are joined, markdown formatting is removed,
     and common abbreviations are expanded.
  3. Kokoro-82M synthesises audio in sentence-length chunks and concatenates
     them into a single waveform (24 kHz, mono, float32).
  4. The waveform is encoded to MP3 via pydub/ffmpeg at the requested bitrate.

Input sources
-------------
  Three ways to supply text (mutually exclusive; --file takes priority):

    python tts_to_mp3.py "Hello world."               # inline string
    python tts_to_mp3.py --file chapter.txt            # plain text file
    python tts_to_mp3.py --file notes.md               # Markdown file
    python tts_to_mp3.py --file report.pdf             # PDF file (requires pymupdf)
    cat chapter.txt | python tts_to_mp3.py - -o out.mp3  # from stdin

Voices
------
  Kokoro ships ~54 voices across 8 languages.  Run --list-voices to see
  the curated shortlist.  Full catalogue:
    https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices

  Voice name prefixes:
    af_*   American English female     am_*   American English male
    bf_*   British English female      bm_*   British English male
    jf_*   Japanese female             jm_*   Japanese male
    zf_*   Mandarin female             zm_*   Mandarin male

  Recommended high-quality English voices:
    af_heart  af_bella  af_nicole   (female)
    am_adam   am_michael             (male)
    bf_emma   bf_isabella            (British female)
    bm_george bm_lewis               (British male)

Language codes
--------------
  Pass a Kokoro single-letter code or a friendly alias to --lang:

    a / en-us / american    American English  (default)
    b / en-gb / british     British English
    j / ja / japanese       Japanese
    z / zh / mandarin       Mandarin Chinese
    e / es / spanish        Spanish
    f / fr / french         French
    h / hi / hindi          Hindi
    i / it / italian        Italian
    p / pt / portuguese     Portuguese

  Non-English voices need their G2P extras installed first:
    pip install "misaki[ja]"   # Japanese
    pip install "misaki[zh]"   # Mandarin

Device selection
----------------
  --device auto  (default) tries CUDA first; falls back to CPU with a warning
                 if the GPU's compute capability isn't supported by the current
                 PyTorch build.  RTX 50-series (Blackwell, sm_120) requires the
                 cu128 wheels: pip install --index-url
                 https://download.pytorch.org/whl/cu128 torch torchvision
                 torchaudio
  --device cuda  Force GPU; exits with a warning if unavailable.
  --device cpu   Always use CPU (useful for debugging or low-VRAM machines).

Text preprocessing
------------------
  Applied automatically before synthesis.  Transformations include:

  Roman numeral outlines   i. / ii) / III)              ->  one. / two. / three.
  Footnote markers         word[1] / word^1              ->  word  (stripped)
  Hyphenated compounds     well-known                    ->  well known  (no pause)
  Em / en dashes           word -- word                  ->  word, word
  Ellipsis                 ... / (Unicode)               ->  .
  MD headings              ## Heading / underlined       ->  Heading
  MD links                 [text](url)                   ->  text
  MD images                ![alt](url)                   ->  (removed)
  MD bold / italic         **bold** / *italic*           ->  bold / italic
  MD strikethrough         ~~deleted~~                   ->  (removed)
  MD blockquotes           > quoted text                 ->  quoted text
  MD tables                | col1 | col2 |               ->  col1  col2
  MD code                  ```block``` / `inline`        ->  (removed) / inline
  HTML                     <em>word</em> / <!-- ... -->  ->  word / (removed)
  Bullet markers           bullet / - / *                ->  (removed)
  URLs                     https://...                   ->  (removed)
  Currency                 $1,200 / GBP50 / EUR9         ->  1,200 dollars / 50 pounds / 9 euros
  Percentages              42%                           ->  42 percent
  Abbreviations            i.e. / e.g. / et al. / vs. / etc.  ->  expanded

Examples
--------
    # Quick test
    python tts_to_mp3.py "Hello world, this is Kokoro speaking."

    # Plain-text document, British male voice, slightly faster
    python tts_to_mp3.py --file chapter1.txt --voice bm_george --speed 1.15 -o chapter1.mp3

    # Markdown document (output defaults to notes.mp3)
    python tts_to_mp3.py --file notes.md

    # PDF document (output defaults to report.mp3)
    python tts_to_mp3.py --file report.pdf

    # Pipe from stdin (Windows)
    type article.txt | python tts_to_mp3.py - -o article.mp3

    # Pipe from stdin (Linux / macOS)
    cat article.txt | python tts_to_mp3.py - -o article.mp3

    # High-quality 320k MP3
    python tts_to_mp3.py --file essay.txt --bitrate 320k -o essay.mp3

    # Force CPU, American male voice
    python tts_to_mp3.py "Hello" --device cpu --voice am_adam -o hello.mp3

    # List available voices and exit
    python tts_to_mp3.py --list-voices
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline
from pydub import AudioSegment


# A short curated list. Kokoro ships ~54 voices total across 8 languages;
# the full list is at https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices
# Voice prefixes:  a* = American English female, am_* = American English male,
# b* = British English female, bm_* = British English male, etc.
COMMON_VOICES = [
    # American English – female
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "af_alloy", "af_aoede", "af_jessica", "af_kore", "af_nova", "af_river",
    # American English – male
    "am_adam", "am_michael", "am_eric", "am_liam", "am_onyx", "am_puck",
    # British English – female
    "bf_emma", "bf_isabella", "bf_alice", "bf_lily",
    # British English – male
    "bm_george", "bm_lewis", "bm_daniel", "bm_fable",
]

# Map common short language codes to Kokoro's single-letter lang codes.
LANG_MAP = {
    "en-us": "a", "en_us": "a", "en": "a", "american": "a",
    "en-gb": "b", "en_gb": "b", "british": "b",
    "ja": "j", "japanese": "j",
    "zh": "z", "mandarin": "z", "chinese": "z",
    "es": "e", "spanish": "e",
    "fr": "f", "french": "f",
    "hi": "h", "hindi": "h",
    "it": "i", "italian": "i",
    "pt": "p", "portuguese": "p",
}

KOKORO_SAMPLE_RATE = 24_000

# Roman numeral -> word, ordered longest-first so regex alternation is unambiguous.
_ROMAN = {
    'xx': 'twenty', 'xix': 'nineteen', 'xviii': 'eighteen',
    'xvii': 'seventeen', 'xvi': 'sixteen', 'xv': 'fifteen',
    'xiv': 'fourteen', 'xiii': 'thirteen', 'xii': 'twelve',
    'xi': 'eleven', 'x': 'ten', 'ix': 'nine', 'viii': 'eight',
    'vii': 'seven', 'vi': 'six', 'v': 'five', 'iv': 'four',
    'iii': 'three', 'ii': 'two', 'i': 'one',
}
_ROMAN_ALT = '|'.join(_ROMAN)


def preprocess_for_tts(text: str) -> str:
    """Normalize text for cleaner TTS synthesis.

    Handles plain text and full Markdown documents (.md).
    """
    # ── Markdown ─────────────────────────────────────────────────────────────
    # Fenced code blocks (``` and ~~~) — remove before anything else.
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'~~~[\s\S]*?~~~', '', text)
    # HTML comments
    text = re.sub(r'<!--[\s\S]*?-->', '', text)
    # Setext headings (text underlined with === or ---) — keep the heading text.
    text = re.sub(r'^(.+)\n[=]{2,}[ \t]*$', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'^(.+)\n[-]{2,}[ \t]*$', r'\1', text, flags=re.MULTILINE)
    # ATX headings (## Heading)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Horizontal rules (--- / *** / ___)
    text = re.sub(r'^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$', '', text, flags=re.MULTILINE)
    # Images: ![alt](url) — drop entirely; audio has no visual equivalent.
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
    # Inline links: [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    # Reference-style links: [text][ref] -> text
    text = re.sub(r'\[([^\]]+)\]\[[^\]]*\]', r'\1', text)
    # Link reference definitions: [ref]: url "title"
    text = re.sub(r'^\s*\[[^\]]+\]:\s*\S+.*$', '', text, flags=re.MULTILINE)
    # Bold-italic, bold, italic
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'\1', text)
    text = re.sub(r'(?<!\w)__(.+?)__(?!\w)', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'(?<!\w)_([^_\n]+?)_(?!\w)', r'\1', text)
    # Strikethrough: ~~deleted~~ — remove (it's content the author crossed out).
    text = re.sub(r'~~(.+?)~~', '', text)
    # Inline code: `code` -> code
    text = re.sub(r'`([^`]+?)`', r'\1', text)
    # Blockquotes: strip leading > markers
    text = re.sub(r'^(\s*>\s*)+', '', text, flags=re.MULTILINE)
    # Table separator rows (|---|---|)
    text = re.sub(r'^\s*\|[\s\-:|]+\|\s*$', '', text, flags=re.MULTILINE)
    # Table data rows: keep cell text, drop pipe chars
    text = re.sub(
        r'^\s*\|(.+)\|\s*$',
        lambda m: '  '.join(c.strip() for c in m.group(1).split('|') if c.strip()),
        text, flags=re.MULTILINE,
    )
    # Remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # ── Reference markers (not meant to be spoken) ────────────────────────────
    text = re.sub(r'[⁰¹²³⁴⁵⁶⁷⁸⁹]+', '', text)                        # Unicode superscripts
    text = re.sub(r'(?<=\w)\s*\[\^?\d+(?:[,\s]\d+)*\]', '', text)     # [1], [^2], [1,2]
    text = re.sub(r'\^\{?\d+\}?', '', text)                             # LaTeX ^1, ^{12}

    # ── Roman numeral outline markers ─────────────────────────────────────────
    # Lowercase with . or ): safe — lowercase "i." never starts an English sentence.
    text = re.sub(
        rf'^(\s*)({_ROMAN_ALT})([.)]\s)',
        lambda m: m.group(1) + _ROMAN[m.group(2)] + '. ',
        text, flags=re.MULTILINE,
    )
    # Uppercase with ) only — avoids converting the pronoun "I." at sentence start.
    text = re.sub(
        rf'^(\s*)({_ROMAN_ALT.upper()})(\)\s)',
        lambda m: m.group(1) + _ROMAN[m.group(2).lower()] + '. ',
        text, flags=re.MULTILINE,
    )

    # ── Hyphens ───────────────────────────────────────────────────────────────
    # Remove intra-word hyphens; TTS models insert an unnatural pause at them.
    text = re.sub(r'(?<=\w)-(?=\w)', ' ', text)

    # ── Punctuation normalisation ──────────────────────────────────────────────
    text = re.sub(r'\s*[—–]\s*', ', ', text)        # em/en dash -> comma
    text = re.sub(r'…|\.{3,}', '.', text)            # ellipsis -> period

    # ── Bullet markers ────────────────────────────────────────────────────────
    text = re.sub(r'^\s*[•·▪▸►●○◦‣⁃]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)

    # ── URLs ──────────────────────────────────────────────────────────────────
    text = re.sub(r'https?://\S+', '', text)

    # ── Currency / unit symbols ────────────────────────────────────────────────
    text = re.sub(r'\$\s*([\d,]+(?:\.\d+)?)', r'\1 dollars', text)
    text = re.sub(r'£\s*([\d,]+(?:\.\d+)?)', r'\1 pounds', text)
    text = re.sub(r'€\s*([\d,]+(?:\.\d+)?)', r'\1 euros', text)
    text = re.sub(r'([\d.]+)\s*%', r'\1 percent', text)
    text = re.sub(r'(?<=\s)&(?=\s)', 'and', text)

    # ── Common abbreviations that confuse TTS sentence-boundary detection ──────
    text = re.sub(r'\bi\.e\.\s*', 'that is, ',     text, flags=re.IGNORECASE)
    text = re.sub(r'\be\.g\.\s*', 'for example, ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bet al\.\s*', 'and others ',  text, flags=re.IGNORECASE)
    text = re.sub(r'\bvs\.\s*',   'versus ',       text, flags=re.IGNORECASE)
    text = re.sub(r'\betc\.\s*',  'et cetera. ',   text, flags=re.IGNORECASE)

    # ── Whitespace ────────────────────────────────────────────────────────────
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()


def pick_device(requested: str) -> str:
    """Return 'cuda' or 'cpu', loudly warning if the user's GPU isn't supported
    by the installed PyTorch (the classic RTX 50-series / sm_120 trap)."""
    if requested == "cpu":
        return "cpu"

    if not torch.cuda.is_available():
        if requested == "cuda":
            print("[warn] --device cuda was requested but torch.cuda.is_available() is False. "
                  "Falling back to CPU.", file=sys.stderr)
        return "cpu"

    # CUDA appears available — but check that PyTorch actually supports this card's
    # compute capability. RTX 50-series (Blackwell) needs the cu128 PyTorch wheels.
    try:
        major, minor = torch.cuda.get_device_capability(0)
        cap = major * 10 + minor  # e.g. 12.0 -> 120
        name = torch.cuda.get_device_name(0)
        supported = torch.cuda.get_arch_list()  # e.g. ['sm_70', 'sm_80', 'sm_90', 'sm_120']
        if f"sm_{cap}" not in supported and f"sm_{major}{minor}" not in supported:
            print(f"[warn] {name} reports compute capability sm_{cap}, but this PyTorch "
                  f"build only supports {supported}. Falling back to CPU.\n"
                  f"       For RTX 50-series cards, install the cu128 build:\n"
                  f"       pip install --index-url https://download.pytorch.org/whl/cu128 "
                  f"torch torchvision torchaudio", file=sys.stderr)
            return "cpu"
        print(f"[info] Using CUDA on {name} (sm_{cap}).", file=sys.stderr)
        return "cuda"
    except Exception as e:
        print(f"[warn] Could not probe GPU capabilities ({e}). Falling back to CPU.",
              file=sys.stderr)
        return "cpu"


def resolve_lang_code(user_value: str) -> str:
    """Accept either Kokoro's single-letter code ('a', 'b', 'j', ...) or a
    friendly alias ('en-us', 'british', etc.)."""
    if len(user_value) == 1:
        return user_value
    key = user_value.lower().replace("_", "-")
    if key in LANG_MAP:
        return LANG_MAP[key]
    raise ValueError(
        f"Unknown language '{user_value}'. Use one of "
        f"{sorted(set(LANG_MAP.values()))} or an alias like {sorted(LANG_MAP.keys())}."
    )


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (pymupdf / fitz)."""
    try:
        import fitz  # pymupdf
    except ImportError:
        raise SystemExit(
            "[error] PDF support requires pymupdf: pip install pymupdf"
        )
    doc = fitz.open(str(path))
    try:
        pages = []
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            if page_text.strip():
                pages.append(page_text)
            print(f"[info]   PDF page {page_num}/{len(doc)}: "
                  f"{len(page_text):,} chars extracted.", file=sys.stderr)
    finally:
        doc.close()
    return "\n\n".join(pages)


def read_input_text(args: argparse.Namespace) -> str:
    """Pull text from --file, stdin, or a positional string."""
    if args.file:
        path = Path(args.file)
        if path.suffix.lower() == ".pdf":
            print(f"[info] Reading PDF: {path}", file=sys.stderr)
            return _read_pdf(path)
        return path.read_text(encoding="utf-8")
    if args.text == "-" or (not sys.stdin.isatty() and not args.text):
        return sys.stdin.read()
    if args.text:
        return args.text
    raise SystemExit("No input text provided. Pass a string, --file PATH, or pipe via stdin.")


def synthesize(text: str, *, voice: str, lang_code: str, speed: float,
               device: str) -> np.ndarray:
    """Run Kokoro on `text` and return the full audio as a single float32 array."""
    print(f"[info] Loading Kokoro pipeline (lang_code={lang_code!r}, device={device})...",
          file=sys.stderr)
    t0 = time.perf_counter()
    pipeline = KPipeline(lang_code=lang_code, device=device)
    print(f"[info] Pipeline ready in {time.perf_counter() - t0:.2f}s.", file=sys.stderr)

    print(f"[info] Synthesizing {len(text):,} characters with voice={voice!r}, "
          f"speed={speed}...", file=sys.stderr)
    t0 = time.perf_counter()
    chunks: list[np.ndarray] = []
    for i, (graphemes, phonemes, audio) in enumerate(
        pipeline(text, voice=voice, speed=speed)
    ):
        # `audio` is a torch tensor on the active device — pull to CPU/numpy.
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        chunks.append(np.asarray(audio, dtype=np.float32))
        print(f"[info]   chunk {i + 1}: {len(chunks[-1]) / KOKORO_SAMPLE_RATE:.2f}s "
              f"({len(graphemes)} chars)", file=sys.stderr)

    if not chunks:
        raise RuntimeError("Kokoro produced no audio. Is the input text empty?")

    full = np.concatenate(chunks)
    duration = len(full) / KOKORO_SAMPLE_RATE
    elapsed = time.perf_counter() - t0
    print(f"[info] Done. {duration:.2f}s of audio in {elapsed:.2f}s "
          f"({duration / elapsed:.1f}x realtime).", file=sys.stderr)
    return full


def write_mp3(audio: np.ndarray, path: Path, bitrate: str = "192k") -> None:
    """Write `audio` (float32 in [-1, 1], mono, 24 kHz) to an MP3 via pydub/ffmpeg."""
    # Round-trip through an in-memory WAV — pydub doesn't read raw numpy.
    buf = io.BytesIO()
    sf.write(buf, audio, KOKORO_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    segment = AudioSegment.from_wav(buf)
    segment.export(path, format="mp3", bitrate=bitrate)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert text to an MP3 using the Kokoro-82M TTS model, running locally.\n"
            "Text is automatically preprocessed for cleaner speech before synthesis.\n"
            "Run with --help for full documentation including voice/language tables."
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
        "-o", "--output", default=None,
        metavar="PATH",
        help=(
            "Destination MP3 file path. Parent directories are created automatically. "
            "Defaults to the input filename with a .mp3 extension (e.g. chapter1.txt -> "
            "chapter1.mp3), or 'output.mp3' when reading from a string or stdin."
        ),
    )
    parser.add_argument(
        "-v", "--voice", default="af_heart",
        metavar="VOICE",
        help=(
            "Kokoro voice name. Run --list-voices to see the curated shortlist, or visit "
            "https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices for all 54+ voices. "
            "(default: af_heart)"
        ),
    )
    parser.add_argument(
        "-s", "--speed", type=float, default=1.0,
        metavar="MULTIPLIER",
        help=(
            "Playback speed multiplier. 1.0 is natural speed; "
            "0.75 is 25%% slower; 1.25 is 25%% faster. Range: 0.25 to 4.0. (default: 1.0)"
        ),
    )
    parser.add_argument(
        "-l", "--lang", default="en-us",
        metavar="LANG",
        help=(
            "Language of the input text. Accepts a Kokoro single-letter code "
            "(a=American English, b=British English, j=Japanese, z=Mandarin, "
            "e=Spanish, f=French, h=Hindi, i=Italian, p=Portuguese) "
            "or a friendly alias (en-us, en-gb, japanese, mandarin, spanish, ...). "
            "Non-English languages need their G2P extras: pip install \"misaki[ja]\". "
            "(default: en-us)"
        ),
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help=(
            "Compute device. "
            "'auto' tries CUDA first and falls back to CPU with a warning if "
            "the GPU's compute capability is unsupported (RTX 50-series requires "
            "the cu128 PyTorch wheels). "
            "'cuda' forces GPU (warns and exits if unavailable). "
            "'cpu' always uses CPU. (default: auto)"
        ),
    )
    parser.add_argument(
        "--bitrate", default="192k",
        metavar="BITRATE",
        help=(
            "MP3 encoding bitrate passed to ffmpeg/pydub. "
            "Common values: 96k (small), 128k, 192k (good quality), 320k (archival). "
            "(default: 192k)"
        ),
    )
    parser.add_argument(
        "--list-voices", action="store_true",
        help="Print the curated list of commonly used Kokoro voices and exit.",
    )
    args = parser.parse_args()

    if args.list_voices:
        print("Common Kokoro voices:")
        for v in COMMON_VOICES:
            print(f"  {v}")
        print("\nFull voice list: https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices")
        return 0

    if not (0.25 <= args.speed <= 4.0):
        parser.error("--speed must be between 0.25 and 4.0")

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

    lang_code = resolve_lang_code(args.lang)
    device = pick_device(args.device if args.device != "auto" else "cuda")

    audio = synthesize(
        text,
        voice=args.voice,
        lang_code=lang_code,
        speed=args.speed,
        device=device,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_mp3(audio, out_path, bitrate=args.bitrate)
    print(f"[info] Wrote {out_path.resolve()} "
          f"({out_path.stat().st_size / 1024:.1f} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
