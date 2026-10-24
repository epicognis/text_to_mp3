# plan.md — Local Text-to-MP3 with Kokoro-82M

## Goal

Build a small, local Python CLI that takes raw text (string, file, or stdin)
and produces an MP3 file using the open-weight **Kokoro-82M** TTS model. Must
run entirely offline after first-run model download, with GPU acceleration on
an RTX 5060 Ti.

## Hardware

- **CPU**: Intel Xeon E5-1620 v4 (4c/8t, Broadwell, 2016, no AVX-512, PCIe 3.0)
- **RAM**: 64 GB
- **Storage**: 2.73 TB HDD + 2 × ~1 TB SSDs
- **GPU**: NVIDIA RTX 5060 Ti **16 GB** — Blackwell, **compute capability sm_120**
- **OS**: Windows (adjust shell commands accordingly if Linux)

## ⚠️ Critical constraint — read first

**The RTX 5060 Ti is sm_120 and is NOT supported by the default PyTorch
wheels.** Installing torch the normal way (`pip install torch`) will produce
a runtime warning `NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is
not compatible with the current PyTorch installation` and silently fall back
to CPU.

**Mandatory install command:**

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

Verify after install:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
```

Output must include `True`, `RTX 5060 Ti`, and `sm_120` somewhere in the
arch list. If `sm_120` is missing, the install is wrong — do not proceed.

## Stack

| Layer       | Choice                          | Notes                                         |
| ----------- | ------------------------------- | --------------------------------------------- |
| TTS model   | Kokoro-82M (`kokoro` PyPI pkg)  | 82M params, Apache-2.0, 24 kHz output         |
| G2P         | `misaki` (Kokoro dependency)    | Falls back to `espeak-ng` system binary       |
| Audio I/O   | `soundfile` (libsndfile)        | WAV intermediate                              |
| MP3 encode  | `pydub` → system `ffmpeg`       | ffmpeg binary must be on PATH                 |
| Numerics    | `numpy`, `torch` (cu128 build)  | See critical constraint above                 |
| Python      | **3.10–3.12**                   | Kokoro requires `<3.13`                       |

## System prerequisites (install before pip)

1. **espeak-ng** — Kokoro's G2P fallback
   - Windows: <https://github.com/espeak-ng/espeak-ng/releases> (run installer)
   - Linux: `sudo apt-get install espeak-ng`
   - macOS: `brew install espeak-ng`

2. **ffmpeg** — required by pydub for MP3 encoding
   - Windows: `winget install Gyan.FFmpeg` (or `choco install ffmpeg`)
   - Linux: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`

Verify both are on PATH:

```bash
espeak-ng --version
ffmpeg -version
```

## Setup steps (in order)

```bash
# 1. Create venv with Python 3.10/3.11/3.12 (NOT 3.13)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Install the cu128 PyTorch build FIRST (order matters — pip resolver
#    will otherwise pull a CPU wheel as a transitive dep)
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

# 3. Verify GPU is properly detected
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"

# 4. Install the remaining dependencies
pip install -r requirements.txt
```

## Deliverables

The following files already exist and should be left as-is unless a task
below explicitly says to modify them:

- `tts_to_mp3.py` — CLI entry point (text in, MP3 out, handles device
  detection, voice/speed/lang/bitrate flags)
- `requirements.txt` — non-torch pip deps
- `README.md` — human-readable setup and usage docs

## Verification checklist

Run these in order. Each must pass before moving on.

1. **GPU visible to PyTorch with sm_120 support**
   ```bash
   python -c "import torch; print(torch.cuda.get_arch_list())"
   ```
   Must include `sm_120`.

2. **Smoke test, default voice**
   ```bash
   python tts_to_mp3.py "Hello from Kokoro." -o smoke.mp3
   ```
   - Must complete without errors.
   - `smoke.mp3` must exist, be > 5 KB, and play back as intelligible speech.
   - stderr should report `Using CUDA on NVIDIA GeForce RTX 5060 Ti (sm_120).`
   - Realtime factor should be **>10× realtime** on GPU (e.g. 2s of audio in
     <0.2s). If it's ~1×, the cu128 install is wrong — the script silently
     fell back to CPU. Re-do step 2 of setup.

3. **File input + non-default voice**
   ```bash
   echo This is a slightly longer test, with multiple sentences. It should be read with the British male voice George. > test.txt
   python tts_to_mp3.py --file test.txt --voice bm_george --speed 1.1 -o test_george.mp3
   ```
   - Output must use the British male voice.
   - Multiple sentences must flow as continuous speech (Kokoro chunks
     internally; the script concatenates).

4. **Voice listing works**
   ```bash
   python tts_to_mp3.py --list-voices
   ```
   Must print the curated voice list and exit 0.

5. **CPU fallback path works**
   ```bash
   python tts_to_mp3.py "CPU test." --device cpu -o cpu_test.mp3
   ```
   Must produce valid audio even with the GPU available (proves the fallback
   code path).

## Known gotchas

- **Python 3.13 will fail** at `pip install kokoro` with an unresolved dep
  error. Stay on 3.10–3.12.
- **First run downloads ~330 MB** of Kokoro weights + per-voice packs from
  Hugging Face into `~/.cache/huggingface/`. Subsequent runs are offline.
- **espeak-ng on Windows**: the installer doesn't always add itself to PATH.
  If `phonemizer` can't find it, set the env var
  `PHONEMIZER_ESPEAK_LIBRARY=C:\Program Files\eSpeak NG\libespeak-ng.dll`
  (adjust path).
- **PCIe 3.0 ×16**: the 5060 Ti is a PCIe 5.0 card but the Xeon E5-1620 v4
  only supports PCIe 3.0. This is fine for inference; it just means initial
  model load is slightly slower than benchmarks elsewhere.
- **VRAM**: Kokoro uses <1 GB. The remaining 15 GB on the 5060 Ti is free for
  anything else running concurrently (LLM, Whisper, etc.).

## Out of scope (do NOT add unless explicitly requested)

- PDF / EPUB / DOCX ingestion — input is **raw text only**
- Voice cloning
- Web UI / Gradio
- Streaming / real-time playback
- Batch directory processing
- Audiobook chapterization / ID3 metadata

Input is raw text. Output is a single MP3. Keep it scoped.

## Reference

- Kokoro model card: <https://huggingface.co/hexgrad/Kokoro-82M>
- Kokoro PyPI: <https://pypi.org/project/kokoro/>
- Full voice list: <https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices>
- PyTorch cu128 wheels: <https://download.pytorch.org/whl/cu128>
