#!/usr/bin/env bash
# convert_this.sh — convert a single text/markdown/PDF file to MP3.
# Usage: ./convert_this.sh <input_file>
# Output is written next to the input, same path & filename, with a .mp3 extension.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file>" >&2
    exit 1
fi

input_file="$1"

if [ ! -f "$input_file" ]; then
    echo "[error] Input file not found: $input_file" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Strip the existing extension (if any) and append .mp3, preserving the directory.
dir="$(dirname "$input_file")"
base="$(basename "$input_file")"
stem="${base%.*}"
output_file="$dir/$stem.mp3"

"$script_dir/.venv/Scripts/python.exe" "$script_dir/tts_to_mp3.py" --file "$input_file" -o "$output_file"
